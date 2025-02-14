from torch.utils.data import Dataset
import numpy as np
import os
from scipy import interpolate
import json
import csv
import torch
from pathlib import Path
import torchvision.transforms as transforms
from scipy.interpolate import interp1d
from typing import Callable, Optional, Tuple, Union
# from glob import glob
import os
from PIL import Image

class image_eeg_dataset(Dataset):
    def __init__(self, eeg_path, image_path, repeat=1):
        super().__init__()
        loaded = torch.load(eeg_path, weights_only=False)
        self.data = loaded['dataset']
        self.images = loaded['images']
        self.labels = loaded['labels']
        self.imagenet = image_path
        self.data_len = 440
        self.num_segments = 11  # We want 11 segments
        self.segment_size = 220  # Each segment is of size 220
        self.step_size = (self.data_len - self.segment_size) // (self.num_segments - 1)  # Compute overlap
        self.repeat = repeat
        self.size = len(self.data) * repeat
        # print(self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        i = index // self.repeat
        eeg = self.data[i]['eeg'].float().t()
        eeg = eeg[20:460, :]  # Crop to timesteps 20-460
        eeg = torch.from_numpy(np.array(eeg)).float()
        eeg = eeg.permute(1, 0)  # Shape: [128, 440]
        
        # Split into 11 segments of size 220
        eeg_segments = []
        labels =[]
        image_name = []
        for j in range(self.num_segments):
            start_idx = j * self.step_size
            end_idx = start_idx + self.segment_size
            eeg_segments.append(eeg[:, start_idx:end_idx])
            labels.append(torch.tensor(self.data[i]["label"]).long())
            image_name.append(str(self.data[i]["image"])+"_"+str(self.data[i]["subject"]))
        
        # Concatenate all segments along the time dimension
        eeg = torch.stack(eeg_segments, dim=0)  # Shape: [11,128, 220]
        eeg = eeg.unsqueeze(2)

        label = torch.stack(labels,dim=0)
        return eeg, label, image_name

    
# image_eeg_dataset = image_eeg_dataset('../data/eeg_14_70_std.pth', '../data/imageNet_images')
# image_eeg_dataset.__getitem__(0)
# # img = Image.open('./DreamDiffusion/datasets/imageNet_images/n03452741/n03452741_17620.JPEG').convert('RGB')
# # print(img)
# for i in image_eeg_dataset:
#     print(i['eeg'].shape)
#     break