import os
from omegaconf import DictConfig
import logging
import wandb
import hydra
import yaml
from tqdm import tqdm
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

logger = logging.getLogger(__name__)


def save_metric(path, metrics):
    strings = yaml.dump(metrics, indent=4, sort_keys=False)
    with open(path, "w") as f:
        f.write(strings)


def compute_sim_matrix(model, dataset,proj, keyids, batch_size=256):
    import torch
    import numpy as np
    from src.data.collate import collate_text_motion
    from src.model.tmr import get_sim_matrix
    import clip
    clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=clip_device)

    device = model.device

    nsplit = int(np.ceil(len(dataset) / batch_size))
    with torch.inference_mode():
        all_data = [dataset.load_keyid(keyid) for keyid in keyids]
        all_data_splitted = np.array_split(all_data, nsplit)

        # by batch (can be too costly on cuda device otherwise)  cuda
        latent_texts = []
        latent_motions = []
        sent_embs = []
        mot_features=[]
        text_embs=[]
        t_mean_tokens=[]
        m_mean_tokens=[]
        for data in tqdm(all_data_splitted, leave=False):
            batch = collate_text_motion(data, device=device)

            # Text is already encoded
            text_x_dict = batch["text_x_dict"]
            # mot模型encoder出的feature
            # mot_feature = batch["mot_feature"]
            # mot_feature.to(device)

            #mot_feature_dict = mot_feature_dict["x"]
            motion_x_dict = batch["motion_x_dict"]
            #mot_feature_dict=batch["mot_feature_dict"]
            sent_emb = batch["sent_emb"]
            text = batch['text']
            # tokenized计算
            text_tokenized = clip.tokenize(text, truncate=True).to(clip_device)
            with torch.no_grad():
                # 计算sentence_feature，dim=512
                text_emb = clip_model.encode_text(text_tokenized).float()
            text_emb = proj(text_emb)
           # text_emb = torch.nn.functional.normalize(text_emb, p=2, dim=1)

          #  mot_feature=torch.nn.functional.normalize(mot_feature, p=2, dim=1)

            # Encode both motion and text
            latent_text,t_mean_tokens_pooled_final = model.encode(text_x_dict,sample_mean=True)
            latent_motion,m_mean_tokens_pooled_final = model.encode(motion_x_dict,sample_mean=True)

            latent_texts.append(latent_text)
            latent_motions.append(latent_motion)
            sent_embs.append(sent_emb)
            # mot_features.append(mot_feature)
            text_embs.append(text_emb)

            #mean（tokens）append
            t_mean_tokens.append(t_mean_tokens_pooled_final)
            m_mean_tokens.append(m_mean_tokens_pooled_final)

        t_mean_tokens=torch.cat(t_mean_tokens)
        m_mean_tokens=torch.cat(m_mean_tokens)
        latent_texts = torch.cat(latent_texts)
        latent_motions = torch.cat(latent_motions)
        sent_embs = torch.cat(sent_embs)
        # mot_features=torch.cat(mot_features)
        text_embs=torch.cat(text_embs)
        sim_matrix = get_sim_matrix(latent_texts, latent_motions)
        #mean_tokens 相似度矩阵
        sim_matrix_tokens=get_sim_matrix(t_mean_tokens, m_mean_tokens)
        #sim_matrix=sim_matrix_tokens+sim_matrix
        #sim_matrix=F.softmax(sim_matrix, dim=1) * F.softmax(sim_matrix, dim=0)
        #sim_matrix=sim_matrix/2haidai1yighui
    returned = {
        "sim_matrix": sim_matrix_tokens.cpu().numpy(),
        "sent_emb": sent_embs.cpu().numpy(),
    }
    return returned


@hydra.main(version_base=None, config_path="configs", config_name="retrieval")
def retrieval(newcfg: DictConfig) -> None:
    protocol = newcfg.protocol
    threshold_val = newcfg.threshold
    device = newcfg.device
    run_dir = newcfg.run_dir
    ckpt_name = newcfg.ckpt
    batch_size = newcfg.batch_size

    assert protocol in ["all", "normal", "threshold", "nsim", "guo"]

    if protocol == "all":
        protocols = ["normal", "threshold", "nsim", "guo"]
    else:
        protocols = [protocol]

    save_dir = os.path.join(run_dir, "contrastive_metrics")
    os.makedirs(save_dir, exist_ok=True)

    # Load last config
    from src.config import read_config
    import src.prepare  # noqa

    cfg = read_config(run_dir)

    import pytorch_lightning as pl
    import numpy as np
    from hydra.utils import instantiate
    from src.load import load_model_from_cfg
    from src.model.metrics import all_contrastive_metrics, print_latex_metrics

    pl.seed_everything(cfg.seed)

    logger.info("Loading the model")
    model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)
    if model.sentence_proj != None:
        proj = model.sentence_proj

    datasets = {}
    results = {}
    for protocol in protocols:
        # Load the dataset if not already
        if protocol not in datasets:
            if protocol in ["normal", "threshold", "guo"]:
                dataset = instantiate(cfg.data, split="test")
                datasets.update(
                    {key: dataset for key in ["normal", "threshold", "guo"]}
                )
            elif protocol == "nsim":
                datasets[protocol] = instantiate(cfg.data, split="nsim_test")
        dataset = datasets[protocol]

        # Compute sim_matrix for each protocol
        if protocol not in results:
            if protocol in ["normal", "threshold"]:
                res = compute_sim_matrix(
                    model, dataset,proj, dataset.keyids, batch_size=batch_size
                )
                results.update({key: res for key in ["normal", "threshold"]})
            elif protocol == "nsim":
                res = compute_sim_matrix(
                    model, dataset,proj, dataset.keyids, batch_size=batch_size
                )
                results[protocol] = res
            elif protocol == "guo":
                keyids = sorted(dataset.keyids)
                N = len(keyids)

                # make batches of 32
                idx = np.arange(N)
                np.random.seed(0)
                np.random.shuffle(idx)
                idx_batches = [
                    idx[32 * i : 32 * (i + 1)] for i in range(len(keyids) // 32)
                ]

                # split into batches of 32
                # batched_keyids = [ [32], [32], [...]]
                results["guo"] = [
                    compute_sim_matrix(
                        model,
                        dataset,
                        proj,
                        np.array(keyids)[idx_batch],
                        batch_size=batch_size,
                    )
                    for idx_batch in idx_batches
                ]
        result = results[protocol]

        # Compute the metrics
        if protocol == "guo":
            all_metrics = []
            for x in result:
                sim_matrix = x["sim_matrix"]
                metrics = all_contrastive_metrics(sim_matrix, rounding=None)
                all_metrics.append(metrics)

            avg_metrics = {}
            for key in all_metrics[0].keys():
                avg_metrics[key] = round(
                    float(np.mean([metrics[key] for metrics in all_metrics])), 2
                )

            metrics = avg_metrics
            protocol_name = protocol
        else:
            sim_matrix = result["sim_matrix"]

            protocol_name = protocol
            if protocol == "threshold":
                emb = result["sent_emb"]
                threshold = threshold_val
                protocol_name = protocol + f"_{threshold}"
            else:
                emb, threshold = None, None
            metrics = all_contrastive_metrics(sim_matrix, emb, threshold=threshold)
            #wandb.log(metrics)

        print_latex_metrics(metrics)

        metric_name = f"{protocol_name}.yaml"
        path = os.path.join(save_dir, metric_name)
        save_metric(path, metrics)

        logger.info(f"Testing done, metrics saved in:\n{path}")


if __name__ == "__main__":
    retrieval()
