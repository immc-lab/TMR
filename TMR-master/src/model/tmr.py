from typing import Dict, Optional

from omegaconf import DictConfig
from torch import Tensor
import wandb
from tqdm import tqdm
from src.data.collate import collate_text_motion
import torch
import clip
from hydra.utils import instantiate
import torch.nn as nn
import torch.nn.functional as F
from .temos import TEMOS
from src.model.pcmepp import ClosedFormSampledDistanceLoss
from .losses import InfoNCE_with_filtering
from .metrics import all_contrastive_metrics
wandb.init(project="TMR(comple)", name="TMR +CL（tokens)_weight=1+CL（mu）weight=0.1 infer(mu)")
from retrieval import retrieval



# x.T will be deprecated in pytorch
def transpose(x):
    return x.permute(*torch.arange(x.ndim - 1, -1, -1))


def get_sim_matrix(x, y):
    x_logits = torch.nn.functional.normalize(x, dim=-1)
    y_logits = torch.nn.functional.normalize(y, dim=-1)
    sim_matrix = x_logits @ transpose(y_logits)
    return sim_matrix


# Scores are between 0 and 1
def get_score_matrix(x, y):
    sim_matrix = get_sim_matrix(x, y)
    scores = sim_matrix / 2 + 0.5
    return scores


class TMR(TEMOS):
    r"""TMR: Text-to-Motion Retrieval
    Using Contrastive 3D Human Motion Synthesis
    Find more information about the model on the following website:
    https://mathis.petrovich.fr/tmr

    Args:
        motion_encoder: a module to encode the input motion features in the latent space (required).
        text_encoder: a module to encode the text embeddings in the latent space (required).
        motion_decoder: a module to decode the latent vector into motion features (required).
        vae: a boolean to make the model probabilistic (required).
        fact: a scaling factor for sampling the VAE (optional).
        sample_mean: sample the mean vector instead of random sampling (optional).
        lmd: dictionary of losses weights (optional).
        lr: learninig rate for the optimizer (optional).
        temperature: temperature of the softmax in the contrastive loss (optional).
        threshold_selfsim: threshold used to filter wrong negatives for the contrastive loss (optional).
        threshold_selfsim_metrics: threshold used to filter wrong negatives for the metrics (optional).
    """

    def __init__(
        self,
        motion_encoder: nn.Module,
        text_encoder: nn.Module,
        motion_decoder: nn.Module,
        vae: bool,
        fact: Optional[float] = None,
        sample_mean: Optional[bool] = False,
        lmd: Dict = {"recons": 1.0, "latent": 1.0e-5, "kl": 1.0e-5, "contrastive": 0.1},
        lr: float = 1e-4,
        temperature: float = 0.7,
        threshold_selfsim: float = 0.80,
        threshold_selfsim_metrics: float = 0.95,
    ) -> None:
        # Initialize module like TEMOS
        super().__init__(
            motion_encoder=motion_encoder,
            text_encoder=text_encoder,
            motion_decoder=motion_decoder,
            vae=vae,
            fact=fact,
            sample_mean=sample_mean,
            lmd=lmd,
            lr=lr,
        )
        data = {'motion_loader': {'_target_': 'src.data.motion.AMASSMotionLoader',
                                  'base_dir': 'datasets/motions/guoh3dfeats',
                                  'normalizer': {'_target_': 'src.data.motion.Normalizer',
                                                 'base_dir': 'stats/kitml/guoh3dfeats', 'eps': 1e-12}, 'fps': 20.0,
                                  'nfeats': 263}, '_target_': 'src.data.text_motion.TextMotionDataset',
                'path': 'datasets/annotations/kitml',
                'text_to_token_emb': {'_target_': 'src.data.text.TokenEmbeddings', 'path': 'datasets/annotations/kitml',
                                      'modelname': 'distilbert-base-uncased', 'preload': True},
                'text_to_sent_emb': {'_target_': 'src.data.text.SentenceEmbeddings',
                                     'path': 'datasets/annotations/kitml',
                                     'modelname': 'sentence-transformers/all-mpnet-base-v2', 'preload': True},
                'preload': True}

        # data = {'motion_loader': {'_target_': 'src.data.motion.AMASSMotionLoader',
        #                           'base_dir': 'datasets/motions/guoh3dfeats',
        #                           'normalizer': {'_target_': 'src.data.motion.Normalizer',
        #                                          'base_dir': 'stats/humanml3d/guoh3dfeats', 'eps': 1e-12},
        #                           'fps': 20.0, 'nfeats': 263}, '_target_': 'src.data.text_motion.TextMotionDataset',
        #         'path': 'datasets/annotations/humanml3d',
        #         'text_to_token_emb': {'_target_': 'src.data.text.TokenEmbeddings',
        #                               'path': 'datasets/annotations/humanml3d',
        #                               'modelname': 'distilbert-base-uncased', 'preload': True},
        #         'text_to_sent_emb': {'_target_': 'src.data.text.SentenceEmbeddings',
        #
        #                              'path': 'datasets/annotations/humanml3d',
        #                              'modelname': 'sentence-transformers/all-mpnet-base-v2',
        #                              'preload': True}, 'preload': True}
        self.dataset = instantiate(data, split="test")
        self.clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load("ViT-B/32", device=self.clip_device)
        self.proj = nn.Linear(512, 256)
        self.sentence_proj = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256))
        #添加pcmepp
        self.pcme_loss=ClosedFormSampledDistanceLoss(init_shift=5,
                                                    init_negative_scale=5,
                                                    vib_beta=0.0001,
                                                    smoothness_alpha=0.1,
                                                    prob_distance='csd',)
        #adding the contrastive loss
        self.contrastive_loss_fn = InfoNCE_with_filtering(
            temperature=temperature, threshold_selfsim=threshold_selfsim
        )
        self.threshold_selfsim_metrics = threshold_selfsim_metrics

        # store validation values to compute retrieval metrics
        # on the whole validation set
        self.validation_step_t_latents = []
        self.validation_step_m_latents = []
        self.validation_step_sent_emb = []

    def compute_loss(self, batch: Dict, return_all=False) -> Dict:
        text_x_dict = batch["text_x_dict"]
        motion_x_dict = batch["motion_x_dict"]
        #获取mot出来的特征
        # mot_feature=batch["mot_feature"]

        mask = motion_x_dict["mask"]
        ref_motions = motion_x_dict["x"]

        text=batch['text']
        #tokenized计算
        # text_tokenized = clip.tokenize(text, truncate=True).to(self.clip_device)
        # with torch.no_grad():
        #     #计算sentence_feature，dim=512
        #     text_emb = self.clip_model.encode_text(text_tokenized).float()
        # #text_emb = self.proj(text_emb)
        # text_emb = self.sentence_proj(text_emb)
        # text_emb = torch.nn.functional.normalize(text_emb, p=2, dim=1)
        # mot_feature=torch.nn.functional.normalize(mot_feature, p=2, dim=1)


        # sentence embeddings
        sent_emb = batch["sent_emb"]

        # text -> motion
        t_motions, t_latents, t_dists,t_mean_tokens_pooled_final = self(text_x_dict,mask=mask, return_all=True)

        # motion -> motion
        m_motions, m_latents, m_dists,m_mean_tokens_pooled_final= self(motion_x_dict, mask=mask, return_all=True,)

        #添加pcme++loss
        matched = torch.eye(len(batch["motion_x_dict"]["x"]))


        # Store all losses
        losses = {}
        #losses["pcme_loss"],a=self.pcme_loss(m_dists,t_dists,matched)


        # Reconstructions losses
        # fmt: off
        losses["recons"] = (
                + self.reconstruction_loss_fn(t_motions, ref_motions)  # text -> motion
                + self.reconstruction_loss_fn(m_motions, ref_motions)  # motion -> motion
        )


        # VAE losses  TokenEmbeddings
        if self.vae:
            # Create a centred normal distribution to compare with
            # logvar = 0 -> std = 1
            ref_mus = torch.zeros_like(m_dists[0])
            ref_logvar = torch.zeros_like(m_dists[1])
            ref_dists = (ref_mus, ref_logvar)

            losses["kl"] = (
                self.kl_loss_fn(t_dists, m_dists)  # text_to_motion
                + self.kl_loss_fn(m_dists, t_dists)  # motion_to_text
                + self.kl_loss_fn(m_dists, ref_dists)  # motion
               + self.kl_loss_fn(t_dists, ref_dists)  # text
            )

        # Latent manifold loss
        losses["latent"] = self.latent_loss_fn(t_latents, m_latents)

        # TMR: adding the contrastive loss
        losses["contrastive"] = self.contrastive_loss_fn(t_latents, m_latents, sent_emb)
        #tokens 对比学习
        losses["clip_contrastive"] = self.contrastive_loss_fn(t_mean_tokens_pooled_final, m_mean_tokens_pooled_final)
        #mu上的对比学习
        losses["mu_contrastive"] =self.contrastive_loss_fn(t_dists[0],m_dists[0])

        # Weighted average of the losses
        losses["loss"] = sum(
            self.lmd[x] * val for x, val in losses.items() if x in self.lmd
        )+losses["clip_contrastive"]+0.1*losses["mu_contrastive"]
        wandb.log(losses)

        # Used for the validation step
        if return_all:
            return losses, t_latents, m_latents, t_dists, m_dists

        return losses

    # def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
    #     #bs = len(batch["motion_x_dict"]["x"])
    #     # self.encode()
    #     # losses, t_latents, m_latents, t_dists, m_dists = self.compute_loss(batch,
    #     #                                                                    return_all=True)  # 返回的t_dists中是mu+text_EMBEDDING
    #     text_x_dict = batch["text_x_dict"]
    #     motion_x_dict = batch["motion_x_dict"]
    #     latent_text, t_mean_tokens_pooled_final = self.encode(text_x_dict, sample_mean=True)
    #     latent_motion, m_mean_tokens_pooled_final = self.encode(motion_x_dict, sample_mean=True)
    #     # Store the latent vectors
    #     self.validation_step_t_latents.append(latent_text)
    #     self.validation_step_m_latents.append(latent_motion)
    #     self.validation_step_sent_emb.append(batch["sent_emb"])
    #
    #
    #
    # def on_validation_epoch_end(self):
    #     # Compute contrastive metrics on the whole batch
    #     t_latents = torch.cat(self.validation_step_t_latents)
    #     m_latents = torch.cat(self.validation_step_m_latents)
    #     sent_emb = torch.cat(self.validation_step_sent_emb)
    #
    #     # Compute the similarity matrix
    #     sim_matrix = get_sim_matrix(t_latents, m_latents).cpu().numpy()
    #
    #     contrastive_metrics = all_contrastive_metrics(
    #         sim_matrix,
    #         emb=None,
    #         threshold=None,
    #     )
    #     Rsum = contrastive_metrics['t2m/R01'] + contrastive_metrics['t2m/R02'] + contrastive_metrics['t2m/R03'] + \
    #            contrastive_metrics['t2m/R05'] + contrastive_metrics['t2m/R10'] + contrastive_metrics['m2t/R01'] + \
    #            contrastive_metrics['m2t/R02'] + contrastive_metrics['m2t/R03'] + contrastive_metrics['m2t/R05'] + \
    #            contrastive_metrics['m2t/R10']
    #     contrastive_metrics['Rsum'] = Rsum
    #     wandb.log(contrastive_metrics)
    #
    #     for loss_name in sorted(contrastive_metrics):
    #         loss_val = contrastive_metrics[loss_name]
    #         self.log(
    #             f"val_{loss_name}_epoch",
    #             loss_val,
    #             on_epoch=True,
    #             on_step=False,
    #         )
    #
    #     self.validation_step_t_latents.clear()
    #     self.validation_step_m_latents.clear()
    #     self.validation_step_sent_emb.clear()
    def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
        bs = len(batch["motion_x_dict"]["x"])




    def on_validation_epoch_end(self):


        #Compute contrastive metrics on the whole batch

        datasets = {}
        results = {}

        # # data = {'motion_loader': {'_target_': 'src.data.motion.AMASSMotionLoader',
        # #                           'base_dir': 'datasets/motions/guoh3dfeats',
        # #                           'normalizer': {'_target_': 'src.data.motion.Normalizer',
        # #                                          'base_dir': 'stats/kitml/guoh3dfeats', 'eps': 1e-12}, 'fps': 20.0,
        # #                           'nfeats': 263}, '_target_': 'src.data.text_motion.TextMotionDataset',
        # #         'path': 'datasets/annotations/kitml',
        # #         'text_to_token_emb': {'_target_': 'src.data.text.TokenEmbeddings', 'path': 'datasets/annotations/kitml',
        # #                               'modelname': 'distilbert-base-uncased', 'preload': True},
        # #         'text_to_sent_emb': {'_target_': 'src.data.text.SentenceEmbeddings',
        # #                              'path': 'datasets/annotations/kitml',
        # #                              'modelname': 'sentence-transformers/all-mpnet-base-v2', 'preload': True},
        # #         'preload': True}
        #
        # data={'motion_loader': {'_target_': 'src.data.motion.AMASSMotionLoader',
        #                         'base_dir': 'datasets/motions/guoh3dfeats',
        #                         'normalizer': {'_target_': 'src.data.motion.Normalizer',
        #                                        'base_dir': 'stats/humanml3d/guoh3dfeats', 'eps': 1e-12},
        #                         'fps': 20.0, 'nfeats': 263}, '_target_': 'src.data.text_motion.TextMotionDataset',
        #       'path': 'datasets/annotations/humanml3d',
        #       'text_to_token_emb': {'_target_': 'src.data.text.TokenEmbeddings',
        #                             'path': 'datasets/annotations/humanml3d',
        #                             'modelname': 'distilbert-base-uncased', 'preload': True},
        #       'text_to_sent_emb': {'_target_': 'src.data.text.SentenceEmbeddings',
        #
        #           'path': 'datasets/annotations/humanml3d',
        #                            'modelname': 'sentence-transformers/all-mpnet-base-v2',
        #                            'preload': True}, 'preload': True}
        dataset = self.dataset
        datasets.update(
            {key: dataset for key in ["normal"]}
        )
        dataset = datasets["normal"]
        import numpy as np
        nsplit = int(np.ceil(len(dataset) / 256))
        with torch.inference_mode():
            all_data = [dataset.load_keyid(keyid) for keyid in dataset.keyids]
            all_data_splitted = np.array_split(all_data, nsplit)

            # by batch (can be too costly on cuda device otherwise)
            latent_texts = []
            latent_motions = []
            sent_embs = []
            mot_features = []
            text_embs = []
            t_mean_tokens = []
            m_mean_tokens = []
            for data in tqdm(all_data_splitted, leave=False):
                batch = collate_text_motion(data, device=self.device)

                # Text is already encoded
                text_x_dict = batch["text_x_dict"]
                # mot模型encoder出的feature
                # mot_feature = batch["mot_feature"]
                # mot_feature.to(device)

                # mot_feature_dict = mot_feature_dict["x"]
                motion_x_dict = batch["motion_x_dict"]
                # mot_feature_dict=batch["mot_feature_dict"]
                sent_emb = batch["sent_emb"]
                text = batch['text']
                # tokenized计算
                # text_tokenized = clip.tokenize(text, truncate=True).to(clip_device)
                # with torch.no_grad():
                #     # 计算sentence_feature，dim=512
                #     text_emb = clip_model.encode_text(text_tokenized).float()
                # text_emb = proj(text_emb)
                # text_emb = torch.nn.functional.normalize(text_emb, p=2, dim=1)

                #  mot_feature=torch.nn.functional.normalize(mot_feature, p=2, dim=1)

                # Encode both motion and text
                latent_text, t_mean_tokens_pooled_final = self.encode(text_x_dict, sample_mean=True)
                latent_motion, m_mean_tokens_pooled_final = self.encode(motion_x_dict, sample_mean=True)

                latent_texts.append(latent_text)
                latent_motions.append(latent_motion)
                sent_embs.append(sent_emb)
                # mot_features.append(mot_feature)
                # text_embs.append(text_emb)

                # mean（tokens）append
                t_mean_tokens.append(t_mean_tokens_pooled_final)
                m_mean_tokens.append(m_mean_tokens_pooled_final)

            t_mean_tokens = torch.cat(t_mean_tokens)
            m_mean_tokens = torch.cat(m_mean_tokens)
            latent_texts = torch.cat(latent_texts)
            latent_motions = torch.cat(latent_motions)
            sent_embs = torch.cat(sent_embs)
            # mot_features=torch.cat(mot_features)
            # text_embs = torch.cat(text_embs)
            sim_matrix = get_sim_matrix(latent_texts, latent_motions)
            returned = {
                "sim_matrix": sim_matrix.cpu().numpy(),
                "sent_emb": sent_embs.cpu().numpy(),
            }
            results.update({key: returned for key in ["normal"]})
            result = results["normal"]
            sim_matrix = result["sim_matrix"]
            contrastive_metrics = all_contrastive_metrics(sim_matrix)

            Rsum = contrastive_metrics['t2m/R01'] + contrastive_metrics['t2m/R02'] + contrastive_metrics['t2m/R03'] + \
                   contrastive_metrics['t2m/R05'] + contrastive_metrics['t2t/R10'] + contrastive_metrics['m2t/R01'] + \
                   contrastive_metrics['m2t/R02'] + contrastive_metrics['m2t/R03'] + contrastive_metrics['m2t/R05'] + \
                   contrastive_metrics['m2t/R10']
            contrastive_metrics['Rsum'] = Rsum
            for loss_name in sorted(contrastive_metrics):
                loss_val = contrastive_metrics[loss_name]
                self.log(
                    f"val_{loss_name}_epoch",
                    loss_val,
                    on_epoch=True,
                    on_step=False,
                )
            wandb.log(contrastive_metrics)

        self.validation_step_t_latents.clear()
        self.validation_step_m_latents.clear()
        self.validation_step_sent_emb.clear()