import os
import torch.utils.data
from PIL import Image
import cv2
import numpy as np
import torchvision.transforms as transforms

class DDSM(torch.utils.data.Dataset):
    def __init__(self, root, view_laterality, task, split, transform):
        self.root = root
        self.view_laterality = view_laterality
        self.task = task
        self.split = split
        self.transform = transform

        def process_line(line):
            series_id, lcc_img_id, lcc_breast_birads, lcc_breast_density, rcc_img_id, rcc_breast_birads, rcc_breast_density, lmlo_img_id, lmlo_breast_birads, lmlo_breast_density, rmlo_img_id, rmlo_breast_birads, rmlo_breast_density = line.strip().split(',')
            if self.task == 'density':
                label = int(lcc_breast_density)
            elif self.task == 'birads':
                label = int(lcc_breast_birads)

            return series_id, lcc_img_id, lmlo_img_id, rcc_img_id, rmlo_img_id, label

        with open(os.path.join(self.root, '{}_{}.txt'.format(self.split, self.task)), 'r') as f:
            self.image_list = list(map(process_line, f.readlines()))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        series_id, lcc_img_id, lmlo_img_id, rcc_img_id, rmlo_img_id, label = self.image_list[idx]
    
        lcc_img = Image.open(os.path.join(self.root, 'images', lcc_img_id )).convert('RGB')
        lcc_img = self.transform(lcc_img)
        lmlo_img = Image.open(os.path.join(self.root, 'images',lmlo_img_id)).convert('RGB')
        lmlo_img = self.transform(lmlo_img)
        rcc_img = Image.open(os.path.join(self.root, 'images', rcc_img_id )).convert('RGB')
        rcc_img = self.transform(rcc_img)
        rmlo_img = Image.open(os.path.join(self.root, 'images', rmlo_img_id )).convert('RGB')
        rmlo_img = self.transform(rmlo_img)
        
        return lcc_img, lmlo_img, rcc_img, rmlo_img, label
