import os
import cv2
import pdb
import math
import random
import shutil
import numpy as np

import torch


def init_log_dir(opt):
	if os.path.exists(os.path.join('./save', opt.name)):
		shutil.rmtree(os.path.join('./save', opt.name))

	os.mkdir(os.path.join('./save', opt.name))
	with open(os.path.join('./save', opt.name, 'options.txt'), "a") as f:
		for k, v in vars(opt).items():
			f.write('{} -> {}\n'.format(k, v))
			print('{} -> {}\n'.format(k, v))

	os.mkdir(os.path.join('./save', opt.name, 'check'))
	os.mkdir(os.path.join('./save', opt.name, 'imgs'))
	os.mkdir(os.path.join('./save', opt.name, 'tb'))


def get_dataset(options):
	if options.dataset == 'rhd':
		from data.rhd import RHDLoader
		dataset_train = RHDLoader(options, mode='train')
		dataset_val = RHDLoader(options, mode='val')
		dataset_test = RHDLoader(options, mode='test')
	else:
		raise

	return (dataset_train, dataset_val, dataset_test)


def get_model(options):
	if options.encode == 'resnet152':
		from model.resnet import ResNetBackbone
		encode = ResNetBackbone(options)
	else:
		raise

	if options.middle == 'pass':
		from model.head import Pass
		middle = Pass(options)
	else:
		raise

	if options.decode == 'pass':
		from model.head import Pass
		decode = Pass(options)
	elif options.decode == 'fc':
		from model.head import FC
		decode = FC(options)
	else:
		raise

	return encode, middle, decode


def draw_hd(img, hds, size=(128, 128)):
	img = cv2.resize(img, size)
	assert len(hds.shape) == 2
	assert hds.shape[1] == 2

	if hds[0][0] < 1:
		hds *= 128
		
	for x, y in hds:
		image = cv2.circle(img, (int(x), int(y)), 1, (255, 0, 0), 2)

	return image
