"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os, pathlib
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from collections import defaultdict
import math
import logging
import json
import torchvision.transforms as transforms
import custom_datasets.edge_utils as edge_utils

num_classes = 20
ignore_label = 0
#current_path = pathlib.Path.absolute()
#parent_path = current_path.parent.absolute()
list_paths = {'train':'train.lst','val':"val.lst",'test':'test.lst'}



palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
           153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
           255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


class Rellis(data.Dataset):

    def __init__(self, mode, datadir, image_transforms, sliding_crop=None,
                 cv_split=None, eval_mode=False,
                 eval_scales=None, eval_flip=False):
        self.mode = mode
        self.image_transforms= image_transforms
        self.sliding_crop = sliding_crop
        self.cv_split = cv_split
        self.eval_mode = eval_mode
        self.eval_scales = eval_scales
        self.eval_flip = eval_flip

        self.root = datadir
        self.list_path = list_paths[mode]
        # List containing all the image
        self.img_list = [line.strip().split() for line in open(self.root+self.list_path)]
        # Read them with our own method
        self.files = self.read_files()
        if len(self.files) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

        self.mean_std = ([0.54218053, 0.64250553, 0.56620195], [0.54218052, 0.64250552, 0.56620194])
        self.label_mapping = {0: 0,
                              1: 0,
                              3: 1,
                              4: 2,
                              5: 3,
                              6: 4,
                              7: 5,
                              8: 6,
                              9: 7,
                              10: 8,
                              12: 9,
                              15: 10,
                              17: 11,
                              18: 12,
                              19: 13,
                              23: 14,
                              27: 15,
                              29: 1,
                              30: 1,
                              31: 16,
                              32: 4,
                              33: 17,
                              34: 18}
    def read_files(self):
        "Return: A list of {img, label, name, weight} turple"
        files = []
        for item in self.img_list:
            image_path, label_path = item
            image_path = self.root + image_path
            label_path = self.root + label_path
            name = os.path.splitext(os.path.basename(label_path))[0]
            files.append({
                "img": image_path,
                "label": label_path,
                "name": name,
                "weight": 1
            })
        return files

    def _eval_get_item(self, img, mask, scales, flip_bool):
        "Method of getting item for evaluation mode"
        return_imgs = []
        for flip in range(int(flip_bool)+1):
            imgs = []
            if flip :
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            for scale in scales:
                w,h = img.size
                target_w, target_h = int(w * scale), int(h * scale)
                resize_img =img.resize((target_w, target_h))
                tensor_img = transforms.ToTensor()(resize_img)
                final_tensor = transforms.Normalize(*self.mean_std)(tensor_img)
                imgs.append(tensor_img)
            return_imgs.append(imgs)
        return return_imgs, mask

    def convert_label(self, label, inverse=False):

        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __getitem__(self, index):
        print("getting an item")
        item = self.files[index]
        img_name = item["name"]
        img_path = item["img"]
        label_path = item["label"]
        img = Image.open(img_path).convert('RGB')  # Using PIL.Image to open RGB images

        mask = np.array(Image.open(label_path))  #PIL.image opened the labels and converted into numpy array
        mask = mask[:,:]  # try delete this line and see what will happen

        mask_copy = self.convert_label(mask)

        # Metrics(mIoU) evaluation
        if self.eval_mode:
            return self._eval_get_item(img, mask_copy, self.eval_scales, self.eval_flip), img_name

        # Convert mask(numpy array) to PIL Image
        mask = Image.fromarray(mask_copy.astype(np.uint8))

        # Image Transformation using torchvision.transforms
        if self.image_transforms is not None:
            img = self.image_transforms(img)  ## img is a tensor now

        # Convert mask to tensor as well
        mask = torch.from_numpy(np.array(mask, dtype=np.int32)).long()

        if self.mode == 'test':
            return img, mask, img_name, item['img']

        _edgemap = mask.numpy()
        _edgemap = edge_utils.mask_to_onehot(_edgemap, num_classes)
        _edgemap = edge_utils.onehot_to_binary_edges(_edgemap, 2, num_classes)
        edgemap = torch.from_numpy(_edgemap).float()

        return img, mask#, edgemap, img_name

    def __len__(self):
        return len(self.files)
