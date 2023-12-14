import os
import codecs as cs
import orjson  # loading faster than json
import json
import torch

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from .collate import collate_text_motion


def read_split(path, split):
    split_file = os.path.join(path, "splits", split + ".txt")
    id_list = []
    with cs.open(split_file, "r") as f:
        for line in f.readlines():
            id_list.append(line.strip())
    return id_list


def load_annotations(path, name="annotations.json"):
    json_path = os.path.join(path, name)
    with open(json_path, "rb") as ff:
        return orjson.loads(ff.read())


class TextMotionDataset(Dataset):
    def __init__(
        self,
        path: str,
        motion_loader,
        text_to_sent_emb,
        text_to_token_emb,
        split: str = "train",
        min_seconds: float = 2.0,
        max_seconds: float = 10.0,
        preload: bool = True,
        tiny: bool = False,
    ):
        if tiny:
            split = split + "_tiny"

        self.collate_fn = collate_text_motion
        self.split = split
        self.keyids = read_split(path, split)

        self.text_to_sent_emb = text_to_sent_emb
        self.text_to_token_emb = text_to_token_emb
        self.motion_loader = motion_loader

        self.min_seconds = min_seconds
        self.max_seconds = max_seconds

        # remove too short or too long annotations
        self.annotations = load_annotations(path)

        # filter annotations (min/max)
        # but not for the test set
        # otherwise it is not fair for everyone
        if "test" not in split:
            self.annotations = self.filter_annotations(self.annotations)

        self.is_training = split == "train"
        self.keyids = [keyid for keyid in self.keyids if keyid in self.annotations]
        self.nfeats = self.motion_loader.nfeats

        if preload:
            for _ in tqdm(self, desc="Preloading the dataset"):
                continue

    def __len__(self):
        return len(self.keyids)

    def __getitem__(self, index):
        keyid = self.keyids[index]
        return self.load_keyid(keyid)

    def load_keyid(self, keyid):
        annotations = self.annotations[keyid]

        # Take the first one for testing/validation
        # Otherwise take a random one
        index = 0
        if self.is_training:
            index = np.random.randint(len(annotations["annotations"]))
        #infer时返回最长的索引 性能会显著提高
        #index = max(range(len(annotations["annotations"])), key=lambda i: len(annotations["annotations"][i]['text']))
        annotation = annotations["annotations"][index]

        text = annotation["text"]
        text_x_dict = self.text_to_token_emb(text)
        motion_x_dict = self.motion_loader(
            path=annotations["path"],
            start=annotation["start"],
            end=annotation["end"],
        )
        sent_emb = self.text_to_sent_emb(text)
        #添加mot_feature dim=256
        file_name = "ids.txt"  # 文件名

        # 打开文件并写入 ID

        # mot_feature_path= '/sda/home/shihaoyu/Projects/MOT/dataset/HumanML3D/mot_encoder_feature_new/'+keyid + '.npy'
        # if os.path.exists(mot_feature_path)==False:
        #     print(keyid)
        #
        # mot_feature = np.load(mot_feature_path)
        # mot_feature = torch.from_numpy(mot_feature).to(torch.float)
        # mot_feature = mot_feature




        output = {
            "motion_x_dict": motion_x_dict,
            "text_x_dict": text_x_dict,
            "text": text,
            "keyid": keyid,
            "sent_emb": sent_emb,
            # "mot_feature":mot_feature
        }
        return output

    def filter_annotations(self, annotations):
        filtered_annotations = {}
        for key, val in annotations.items():
            annots = val.pop("annotations")
            filtered_annots = []
            for annot in annots:
                duration = annot["end"] - annot["start"]
                if self.max_seconds >= duration >= self.min_seconds:
                    filtered_annots.append(annot)

            if filtered_annots:
                val["annotations"] = filtered_annots
                filtered_annotations[key] = val

        return filtered_annotations


def write_json(data, path):
    with open(path, "w") as ff:
        ff.write(json.dumps(data, indent=4))
