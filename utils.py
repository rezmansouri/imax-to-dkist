import os
import torch
import random
import cv2 as cv
import numpy as np
from PIL import Image
from scipy.special import softmax
import torchvision.transforms.v2 as Ttorch
from scipy.ndimage import gaussian_filter

# 'Intergranular lane' : 0,
# 'Uniform-shape granules': 1,
# 'Granules with dots' : 2,
# 'Granules with a lane' : 3,
# 'Complex-shape granules' : 4

class SunriseDataset(torch.utils.data.Dataset):
    def __init__(self, aug_data_root, file_names, barlow=False):
        super(SunriseDataset, self).__init__()
        self.barlow = barlow
        self.X, self.y = [], []
        for file_name in file_names:
            data_path = os.path.join(aug_data_root, file_name + '.npz')
            data = np.load(data_path)
            self.X.extend(list(data['X']))
            self.y.extend(list(data['y']))

    def __getitem__(self, idx):
        if self.barlow:
            return self.X[idx]
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)


if __name__ == '__main__':
    train_dataset = SunriseDataset(
        'data/augmented', ['1', '2'])
