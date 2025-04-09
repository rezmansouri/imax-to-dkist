import os
import torch
import cv2 as cv
import numpy as np
from PIL import Image
from tqdm import trange, tqdm
from scipy.special import softmax
from scipy.ndimage import gaussian_filter
import torchvision.transforms.v2 as Ttorch


class SequentialTransformation(torch.nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def forward(self):
        super().forward()

    def __call__(self, img, pmap, ind):
        t_list = [img]
        for t, transform in enumerate(self.transforms):
            if t == 1:
                rotation, c = transform(t_list[-1], pmap, ind)
                t_list.append(rotation)
            else:
                t_list.append(transform(t_list[-1]))

        return t_list[-1], c


class SRS_crop(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = int(size)

    def forward(self, img, pmap, ind):
        pmap_flatten = pmap.flatten()
        counter = np.arange(len(pmap_flatten))
        pos = np.random.choice(counter, 10, p=pmap_flatten)
        for p in pos:
            c = (int(ind[p, 0]),
                 int(ind[p, 1]))
            img_raw = img.cpu().detach().numpy()

            half_size = self.size // 2
            cy, cx = c[0], c[1]

            cr_image_1 = img_raw[1, cx-half_size:cx +
                                 half_size, cy-half_size:cy+half_size]

            if np.any(cr_image_1 == 0):
                continue

            cr_image_0 = img_raw[0, cx-half_size:cx +
                                 half_size, cy-half_size:cy+half_size]

            return torch.Tensor(np.array([cr_image_0, cr_image_1]), device='cpu'), c
        raise ValueError('Wrong mask')


def all_rotate(image, mask):
    degrees = trange(0, 359, 5)
    images, masks = [], []
    for degree in degrees:
        M = cv.getRotationMatrix2D((384, 384), degree, 1)
        image_rotated = cv.warpAffine(
            image, M, (768, 768), flags=cv.INTER_LANCZOS4)
        mask_rotated = cv.warpAffine(
            mask, M, (768, 768), flags=cv.INTER_NEAREST)
        images.append(image_rotated)
        masks.append(mask_rotated)
    return images, masks


def create_dataset(images_path, masks_path, file_names, n=3750, size=128):

    sequential_transform = SequentialTransformation([Ttorch.ToTensor(),
                                                     SRS_crop(
        size),
        Ttorch.RandomHorizontalFlip(
        p=0.5),
        Ttorch.RandomVerticalFlip(
        p=0.5)
    ])

    images = []
    masks = []
    weight_maps = []
    print('Creating dataset...')
    index_list = []
    for image_name in file_names:
        image_path = os.path.join(images_path, image_name + '.png')
        mask_path = os.path.join(masks_path, image_name + '.npy')
        image = np.array(Image.open(image_path))
        mask = np.load(mask_path)
        mask_plus = np.asarray(mask, dtype=np.float32) + 1

        images_rotated, masks_rotated = all_rotate(image, mask_plus)

        for (image, mask) in tqdm(zip(images_rotated, masks_rotated), leave=False):
            image = image / image.max()
            weight_map = np.zeros_like(mask).astype(np.float32)
            weight_map[mask == 1] = 0.0267  # 1#1#1#
            weight_map[mask == 2] = 1.0000  # 5#10#1#
            weight_map[mask == 3] = 0.3206  # 1#10#1#
            weight_map[mask == 4] = 0.8704  # 5#10#1#
            weight_map[mask == 5] = 0.0286  # 1#1#1#

            frame_is, frame_js = np.where(mask == 0)

            weight_map = softmax(gaussian_filter(weight_map, sigma=14))

            weight_map[frame_is, frame_js] = 0

            half_size = size // 2
            weight_map[:half_size, :] = 0
            weight_map[-half_size:, :] = 0
            weight_map[:, :half_size] = 0
            weight_map[:, -half_size:] = 0

            for i, j in zip(frame_is, frame_js):
                xmin = max(0, i-half_size)
                xmax = min(768, i+half_size)
                ymin = max(0, j-half_size)
                ymax = min(768, j+half_size)
                weight_map[xmin:xmax, ymin:ymax] = 0

            non_zero = np.count_nonzero(weight_map)
            weight_map[weight_map !=
                       0] += (1 - np.sum(weight_map)) / non_zero

            weight_maps.append(weight_map)
            index_list.append(
                np.array(list(np.ndindex(weight_map.shape))))

            images.append(image)
            masks.append(mask.astype(np.int8))

    X = []
    y = []
    step = n // len(images)
    for i in trange(n, leave=False):
        idx = i // step
        if idx >= len(images):
            idx = np.random.randint(low=0, high=len(images))
        image = images[idx]
        mask = masks[idx]

        weight_map = weight_maps[idx]
        index_l = index_list[idx]
        img_t, c = sequential_transform(
            np.array([image, mask]).transpose(), weight_map, index_l)

        image = img_t[0].unsqueeze(0)
        mask = img_t[1].type(torch.int64)
        X.append(image)
        y.append(mask-1)
    return X, y


if __name__ == '__main__':
    for i in range(1, 9):
        X_train, y_train = create_dataset(
            'data/original/images', 'data/original/masks', str(i))
        np.savez_compressed(f'{i}.npz', X=np.array(
            X_train), y=np.array(y_train))
