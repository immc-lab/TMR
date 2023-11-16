from typing import Dict, Optional
from torch import Tensor

import torch
import torch.nn as nn
from .temos import TEMOS
from .losses import InfoNCE_with_filtering
from .metrics import all_contrastive_metrics
import torch.nn.functional as F
from src.model.probemb import MCSoftContrastiveLoss
from torch.cuda import amp
import wandb

wandb.init(project="TMR", name="TMR+cl(w/o nega)+pcme")


# x.T will be deprecated in pytorch
def transpose(x):
    return x.permute(*torch.arange(x.ndim - 1, -1, -1))


def get_sim_matrix(x, y):
    x_logits = torch.nn.functional.normalize(x, dim=-1)
    y_logits = torch.nn.functional.normalize(y, dim=-1)
    sim_matrix = x_logits @ transpose(y_logits)
    dual_soft = 0
    #    sim_matrix = torch.nn.functional.softmax(sim_matrix,dim=1) @ torch.nn.functional.softmax(sim_matrix,dim=0)
    if (dual_soft == 0):
        return sim_matrix
    else:
        return F.softmax(sim_matrix, dim=1) * F.softmax(sim_matrix, dim=0)


# Scores are between 0 and 1
def get_score_matrix(x, y, dual_soft=0):
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
            dual_soft: int = 0,
            init_negative_scale: float = 1,
            init_shift: float = 40,
            num_samples: int = 7,
            uniform_lambda: int = 10,
            vib_beta: float = 0.00001,
            soft_contrastive: float = 0,
            reduction: str = "sum"

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
        self.proj = nn.Linear(768, 256)
        self.soft_contrastive = soft_contrastive
        self.num_samples = num_samples
        # adding the contrastive loss
        self.contrastive_loss_fn = InfoNCE_with_filtering(
            temperature=temperature, threshold_selfsim=threshold_selfsim, dual_soft=dual_soft
        )
        self.soft_contrastive_loss = MCSoftContrastiveLoss(init_negative_scale=init_negative_scale,
                                                           init_shift=init_shift, num_samples=num_samples,
                                                           uniform_lambda=uniform_lambda, vib_beta=vib_beta,
                                                           reduction=reduction)
        self.threshold_selfsim_metrics = threshold_selfsim_metrics

        # store validation values to compute retrieval metrics
        # on the whole validation set
        self.validation_step_t_latents = []
        self.validation_step_m_latents = []
        self.validation_step_sent_emb = []

    def compute_loss(self, batch: Dict, return_all=False) -> Dict:
        text_x_dict = batch["text_x_dict"]
        motion_x_dict = batch["motion_x_dict"]

        mask = motion_x_dict["mask"]
        ref_motions = motion_x_dict["x"]

        # sentence embeddings
        sent_emb = batch["sent_emb"]
        #sent_emb1 = self.proj(sent_emb)

        # text -> motion

        t_motions, t_latents, t_samples, t_logsigma, t_dists = self(text_x_dict, mask=mask, return_all=True,
                                                                    num_samples=self.num_samples)
        # t_samples(32,7,256)   t_logsigma (32,256)
        # motion -> motion
        m_motions, m_latents, m_samples, m_logsigma, m_dists = self(motion_x_dict, mask=mask, return_all=True,
                                                                    num_samples=self.num_samples)

        # Store all losses
        losses = {}
        #losses["contrastive_clip"] = self.contrastive_loss_fn(sent_emb, m_dists[0], sent_emb1)

        # Reconstructions losses
        # fmt: off
        losses["recons"] = (
                + self.reconstruction_loss_fn(t_motions, ref_motions)  # text -> motion
                + self.reconstruction_loss_fn(m_motions, ref_motions)  # motion -> motion
        )
        # fmt: on

        # VAE losses
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
        scaler = amp.GradScaler()
        # Latent manifold loss
        losses["latent"] = self.latent_loss_fn(t_latents, m_latents)
        if (self.soft_contrastive == 1):
            losses["soft_contrastive"] = self.soft_contrastive_loss(m_samples, t_samples, m_logsigma, t_logsigma)
        else:
            losses["soft_contrastive"] = 0
        # losses["soft_contrastive"] = scaler.scale(losses["soft_contrastive"]) #LOSSsuofang
        # TMR: adding the contrastive loss
        losses["contrastive"] = self.contrastive_loss_fn(t_latents, m_latents, sent_emb)

        # Weighted average of the losses
        losses["loss"] = sum(
            self.lmd[x] * val for x, val in losses.items() if x in self.lmd
        )
        wandb.log(losses)

        # Used for the validation step
        if return_all:
            return losses, t_latents, m_latents,t_samples,m_samples

        return losses

    def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
        bs = len(batch["motion_x_dict"]["x"])
        losses, t_latents, m_latents, t_samples,m_samples= self.compute_loss(batch, return_all=True)

        # Store the latent vectors
        self.validation_step_t_latents.append(t_latents)
        self.validation_step_m_latents.append(m_latents)
        self.validation_step_sent_emb.append(batch["sent_emb"])

        #self.validation_step_t_latents.append(t_samples)
        #self.validation_step_m_latents.append(m_samples)
        #



        for loss_name in sorted(losses):
            loss_val = losses[loss_name]
            self.log(
                f"val_{loss_name}",
                loss_val,
                on_epoch=True,
                on_step=True,
                batch_size=bs,
            )

        return losses["loss"]

    def on_validation_epoch_end(self):
        # Compute contrastive metrics on the whole batch
        t_latents = torch.cat(self.validation_step_t_latents)
        m_latents = torch.cat(self.validation_step_m_latents)
        sent_emb = torch.cat(self.validation_step_sent_emb)

        #pmm = MatchingProbModule(self.criterion.match_prob)
        #pmm.set_g_features(g_features)

        # Compute the similarity matrix
        sim_matrix = get_sim_matrix(t_latents, m_latents, ).cpu().numpy()

        contrastive_metrics = all_contrastive_metrics(
            sim_matrix,
            emb=sent_emb.cpu().numpy(),
            threshold=None,
        )
        Rsum = contrastive_metrics['t2m/R01'] + contrastive_metrics['t2m/R02'] + contrastive_metrics['t2m/R03'] + \
               contrastive_metrics['t2m/R05'] + contrastive_metrics['m2t/R10'] + contrastive_metrics['t2m/R01'] + \
               contrastive_metrics['m2t/R02'] + contrastive_metrics['m2t/R03'] + contrastive_metrics['m2t/R05'] + \
               contrastive_metrics['m2t/R10']
        contrastive_metrics['Rsum'] = Rsum
        wandb.log(contrastive_metrics)

        for loss_name in sorted(contrastive_metrics):
            loss_val = contrastive_metrics[loss_name]
            self.log(
                f"val_{loss_name}_epoch",
                loss_val,
                on_epoch=True,
                on_step=False,
            )

        self.validation_step_t_latents.clear()
        self.validation_step_m_latents.clear()
        self.validation_step_sent_emb.clear()
