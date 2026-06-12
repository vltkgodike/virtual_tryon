import os
from PIL import Image

import torch
from torch.utils.data import Dataset

from torchvision import transforms


class VITONDataset(Dataset):

    def __init__(
        self,
        root_dir,
        pair_file="/kaggle/input/datasets/adarshsingh0903/virtual-tryon-dataset/Virtual tryon data/train_pairs.txt",
        image_size=256
    ):

        self.root_dir = root_dir

        self.transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size)
            ),
            transforms.ToTensor()
        ])

        pair_path = os.path.join(
            root_dir,
            pair_file
        )

        with open(pair_path, "r") as f:

            self.pairs = [
                line.strip().split()
                for line in f.readlines()
            ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        person_name, cloth_name = self.pairs[idx]

        person_path = os.path.join(
            self.root_dir,
            "train",
            "image",
            person_name
        )

        cloth_path = os.path.join(
            self.root_dir,
            "train",
            "cloth",
            cloth_name
        )

        target_path = os.path.join(
            self.root_dir,
            "train",
            "image",
            person_name
        )

        person = Image.open(
            person_path
        ).convert("RGB")

        cloth = Image.open(
            cloth_path
        ).convert("RGB")

        target = Image.open(
            target_path
        ).convert("RGB")
        pose_path = os.path.join(
            self.root_dir,
            "train",
            "openpose_img",
            person_name.replace(
                ".jpg",
                "_rendered.png"
            )
        )

        pose = Image.open(
            pose_path
        ).convert("RGB")

        pose = self.transform(
            pose
        )
        person = self.transform(person)
        cloth = self.transform(cloth)
        target = self.transform(target)

        return {
            "person": person,
            "cloth": cloth,
            "target": target,
            "pose": pose,
            "person_name": person_name,
            "cloth_name": cloth_name
        }
    