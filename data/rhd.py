import os
import cv2
import pdb
import json
import glob
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import urllib.request
from PIL import Image
from collections import Counter

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def create_dataset_csv(d='./dataset/RHD_published_v2'):
    train_dir, test_dir = os.path.join(
        d, 'training'), os.path.join(d, 'evaluation')
    train_hd, test_hd = pickle.load(open(os.path.join(train_dir, 'anno_training.pickle'), 'rb')), pickle.load(
        open(os.path.join(test_dir, 'anno_evaluation.pickle'), 'rb'))

    items = []
    for k, v in train_hd.items():
        v = v['uv_vis']
        vl, vr = v[:21], v[21:]
        visl, visr = [x[-1] for x in vl], [x[-1] for x in vr]
        if np.sum(visl) == 21:
            items.append(['{:05d}.png'.format(
                k), list(vl[:, :2].reshape(-1)), 0])

    for k, v in test_hd.items():
        v = v['uv_vis']
        vl, vr = v[:21], v[21:]
        visl, visr = [x[-1] for x in vl], [x[-1] for x in vr]
        if np.sum(visl) == 21:
            items.append(['{:05d}.png'.format(
                k), list(vl[:, :2].reshape(-1)), 2])

    csv = pd.DataFrame(items, columns=['id', 'hd', 'mode'])
    val = csv[csv['mode'] == 0].sample(2000)
    csv.at[val.index, 'mode'] = 1

    csv.to_csv('./dataset/RHD_published_v2/data.csv', index=False)


class RHDLoader(Dataset):
    def __init__(self, opt, mode='train'):
        self.opt = opt
        self.mode = mode

        data = pd.read_csv(self.opt.rhd_csv)
        if mode == 'train':
            data = data[data['mode'] == 0]
        elif mode == 'val':
            data = data[data['mode'] == 1]
        else:
            data = data[data['mode'] == 2]

        data.reset_index(drop=True, inplace=True)
        self.data = data
        self.trans = self.get_trans()

    # feat loader
    def __getitem__(self, idx):
        item = self.data.loc[idx]

        id = item['id']
        hd = item['hd']

        imgp = os.path.join('/'.join(self.opt.rhd_csv.split('/')
                                     [:-1]), 'training' if self.mode != 'test' else 'evaluation', 'color', id)
        img = cv2.imread(imgp)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hd = np.array(eval(hd)).reshape(-1, 2)

        img, hd = self.reshape_img_hd(img, hd, rot=random.randint(0, 3))
        img = Image.fromarray(img)
        img = self.trans(img)

        return img, hd, imgp

    def __len__(self):
        return len(self.data)

    def reshape_img_hd(self, img, hd, rot=0):
        h, w, c = img.shape
        hd = hd / np.expand_dims(np.array([h, w]), 0)
        hd *= 128

        img = cv2.resize(img, (128, 128))
        if self.mode == 'train':
            for _ in range(rot):
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            if rot == 0:
                pass
            elif rot == 1:
                hd = np.expand_dims(np.array([0, 128]), 0) + hd[:,::-1] * np.expand_dims(np.array([1, -1]), 0)
            elif rot == 2:
                hd = np.expand_dims(np.array([128, 128]), 0) - hd
            elif rot == 3:
                hd = np.expand_dims(np.array([128, 0]), 0) + hd[:,::-1] * np.expand_dims(np.array([-1, 1]), 0)

            # for x, y in hd:
            #     img = cv2.circle(img, (int(x), int(y)), 1, (255, 0, 0), 2)

            # cv2.imshow('image',img)
            # cv2.waitKey(0)

        return img, hd

    def get_trans(self):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if self.mode == 'train':
            return transforms.Compose([
            	transforms.ColorJitter(brightness=0.5, saturation=(0.5, 1.5)),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])


if __name__ == '__main__':
    create_dataset_csv()
