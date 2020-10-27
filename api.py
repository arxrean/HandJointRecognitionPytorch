import os
import cv2
import pdb
import glob
import random
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from option import get_parser
import utils


def trans():
	normalize = transforms.Normalize(
			mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	
	return transforms.Compose([
		transforms.ToTensor(),
		normalize,
	])


def api_get_joint(opt, imgp):
	img = cv2.imread(imgp)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, (128, 128))
	img = Image.fromarray(img)
	img = trans()(img).unsqueeze(0)

	if opt.gpu:
		img = img.cuda()

	with torch.no_grad():
		out = encode(img)
		out = middle(out)
		out = decode(out).squeeze(0).cpu().numpy()

	return out


def draw_circle(imgp, hd):
	img = cv2.imread(imgp)
	img = utils.draw_hd(img, hd)

	return img


if __name__ == '__main__':
	save_root = '/home/kzy/Code/GuitarChordRecognition/dataset/kd'
	visual_root = '/home/kzy/Code/GuitarChordRecognition/dataset/visual'

	imgs = glob.glob(
		'/home/kzy/Code/GuitarChordRecognition/dataset/crop/*/*')
	opt = get_parser()

	encode, middle, decode = utils.get_model(opt)
	if opt.gpu:
		encode, middle, decode = encode.cuda(), middle.cuda(), decode.cuda()

	best_pth = torch.load(os.path.join(
		'./save', opt.name, 'check', 'best.pth.tar'), map_location='cpu')
	encode.load_state_dict(best_pth['encode'])
	middle.load_state_dict(best_pth['middle'])
	decode.load_state_dict(best_pth['decode'])

	random.shuffle(imgs)
	for imgp in imgs[:100]:
		jot = api_get_joint(opt, imgp)
		d = imgp.split('/')[-2]
		name = imgp.split('/')[-1].split('.')[0]
		out = os.path.join(save_root, d, name+'.npy')
		if not os.path.exists(os.path.join(save_root, d)):
			os.mkdir(os.path.join(save_root, d))
		np.save(out, jot)

		img_draw = draw_circle(imgp, jot)
		if not os.path.exists(os.path.join(visual_root, d)):
			os.mkdir(os.path.join(visual_root, d))
		out = os.path.join(visual_root, d, imgp.split('/')[-1])
		cv2.imwrite(out, img_draw)

