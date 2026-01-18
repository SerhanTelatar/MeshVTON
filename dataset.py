import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class VITONDataset(Dataset):

    def __init__(self, root_dir, pairs_file, mode="train", img_size=(512, 384), load_densepose=True, load_parse=True, load_openpose=True):
        super(VITONDataset, self).__init__()

        self.root_dir = root_dir
        self.pairs_file = pairs_file
        self.mode = mode
        self.img_size = img_size
        self.load_densepose = load_densepose
        self.load_parse = load_parse
        self.load_openpose = load_openpose
        
        self.data_dir = os.path.join(root_dir, mode)
        self.image_dir = os.path.join(self.data_dir, "image")
        self.cloth_dir = os.path.join(self.data_dir, "cloth")
        self.cloth_mask_dir = os.path.join(self.data_dir, "cloth-mask")
        self.agnostic_dir = os.path.join(self.data_dir, "agnostic-v3.2")
        self.densepose_dir = os.path.join(self.data_dir, "image-densepose")
        self.parse_agnostic_dir = os.path.join(self.data_dir, "image-parse-agnostic-v3.2")
        self.parse_dir = os.path.join(self.data_dir, "image-parse-v3")
        self.openpose_img_dir = os.path.join(self.data_dir, "openpose_img")
        self.openpose_json_dir = os.path.join(self.data_dir, "openpose_json")

        self.pairs = []
        pairs_path = os.path.join(self.root_dir, pairs_file)
        with open(pairs_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        self.pairs.append((parts[0], parts[1]))

        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.transform_mask = transforms.Compose([
            transforms.Resize(img_size, interpolation= transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

        self.transform_parse = transforms.Compose([
            transforms.Resize(img_size, interpolation= transforms.InterpolationMode.NEAREST),
        ])

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):

        image_name, cloth_name = self.pairs[idx]

        image_name = os.path.splitext(image_name)[0]
        cloth_name = os.path.splitext(cloth_name)[0]

        result = {
            "image_name": image_name,
            "cloth_name": cloth_name
        }

        image_path = os.path.join(self.image_dir, image_name + ".jpg")
        image = Image.open(image_path).convert("RGB")
        result["image"] = self.transform(image)

        cloth_path = os.path.join(self.cloth_dir, cloth_name + ".jpg")
        image = Image.open(cloth_path).convert("RGB")
        result["cloth"] = self.transform(image)

        cloth_mask_path = os.path.join(self.cloth_mask_dir, cloth_name + ".jpg")
        if os.path.exists(cloth_mask_path):
            cloth_mask = Image.open(cloth_mask_path).convert("L")
            result["cloth_mask"] = self.transform_mask(cloth_mask)

        agnostic_path = os.path.join(self.agnostic_dir, image_name + ".jpg")
        if os.path.exists(agnostic_path):
            agnostic = Image.open(agnostic_path).convert("RGB")
            result["agnostic"] = self.transform(agnostic)

        if self.load_parse:
            parse_path = os.path.join(self.parse_dir, image_name + ".png")
            if os.path.exists(parse_path):
                parse = Image.open(parse_path)
                parse = self.transform_parse(parse)
                result["parse"] = torch.from_numpy(np.array(parse)).long()


            parse_agnostic_path = os.path.join(self.parse_agnostic_dir, image_name + ".png")
            if os.path.exists(parse_agnostic_path):
                parse_agnostic = Image.open(parse_agnostic_path)
                parse_agnostic = self.transform_parse(parse_agnostic)
                result["parse_agnostic"] = torch.from_numpy(np.array(parse_agnostic)).long()

        if self.load_densepose:
            densepose_path = os.path.join(self.densepose_dir, image_name + ".jpg")
            if os.path.exists(densepose_path):
                densepose = Image.open(densepose_path).convert("RGB")
                result["densepose"] = self.transform(densepose)

        if self.load_openpose:
            openpose_image_path = os.path.join(self.openpose_img_dir, image_name + "_rendered.png")
            if os.path.exists(openpose_image_path):
                openpose_image = Image.open(openpose_image_path).convert("RGB")
                result["openpose_image"] = self.transform(openpose_image)

            openpose_json_path = os.path.join(self.openpose_json_dir, image_name + "_keypoints.json")
            if os.path.exists(openpose_json_path):
                with open(openpose_json_path, "r") as f:
                    openpose_data = json.load(f)
                    if openpose_data.get("people"):
                        keypoints = openpose_data["people"][0].get("pose_keypoints_2d", [])
                        keypoints = np.array(keypoints).reshape(-1,3)
                        result["pose_keypoints"] = torch.from_numpy(keypoints).float()

        return result









