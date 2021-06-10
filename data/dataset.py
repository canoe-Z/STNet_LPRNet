#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import random
import time

from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import *
from tqdm import tqdm

from data.ccpd import imgs, ImgType
from model.lprnet import CHARS
from utils.general import load_cache, save_cache

logger = logging.getLogger(__name__)
CHARS_DICT = {char: i for i, char in enumerate(CHARS)}


class LPRDataSet(Dataset):
    def __init__(self, img_size, start_rate, size_rate, cache=False):
        data_set = imgs[ImgType.base_selected] + imgs[ImgType.blur_selected] + \
            imgs[ImgType.fn_selected] + imgs[ImgType.rotate_selected]
        random.seed(73)  # 随意设置，42 也成，12345678 也成，只要训练的过程中不变即可
        random.shuffle(data_set)

        total_img_count = len(data_set)
        start = int(total_img_count * start_rate)
        end = min(total_img_count - 1, start +
                  int(total_img_count * size_rate))
        self.data_set = data_set[start:end]
        self.len = len(self.data_set)
        self.img_size = img_size

        if cache:
            self.images = []
            since = time.time()
            self.preload(start_rate, size_rate)
            logger.info("Cache %d images used %.2fs." %
                        (len(self.images), time.time() - since))

            del self.data_set
            del self.img_size
        else:
            self.images = None

    def preload(self, start_rate, size_rate):
        image_count = len(self)

        cache_file_name = 'cache_dataset_%.6f_%.6f.pt' % (
            start_rate, size_rate)
        images = load_cache(cache_file_name, 'dataset')
        if images and len(images) == image_count:
            self.images = images
            logger.info("Use cache file %s." % cache_file_name)
            return

        logger.info("Cache file %s not found, start load images..." %
                    cache_file_name)
        pbar = tqdm(range(image_count), desc='Load images')
        for idx in pbar:
            image = self.load_img(idx)
            self.images.append(image)

        save_cache(cache_file_name, self.images, 'dataset')

    def load_img(self, idx):
        img = self.data_set[idx]
        image = Image.open(img.src)

        # 裁剪
        image.crop(img.rect)

        # 缩放
        image = image.resize(self.img_size)

        # to tensor
        tran = transforms.ToTensor()
        image = tran(image)

        label = []
        for c in img.label:
            label.append(CHARS_DICT[c])

        return image, label

    def __getitem__(self, index):
        if self.images:
            (image, label) = self.images[index]
        else:
            (image, label) = self.load_img(index)

        return image, label, len(label)

    def __len__(self):
        return self.len


def collate_fn(batch):
    images = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        image, label, length = sample
        images.append(image)
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten()

    return torch.stack(images, 0), torch.from_numpy(labels), lengths
