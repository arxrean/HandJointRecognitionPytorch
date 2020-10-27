import os
import cv2
import pdb
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from option import get_parser
import utils


def train(opt):
	utils.init_log_dir(opt)
	writer = SummaryWriter('./save/{}/tb'.format(opt.name))

	encode, middle, decode = utils.get_model(opt)
	if opt.gpu:
		encode, middle, decode = encode.cuda(), middle.cuda(), decode.cuda()

	loss_func = nn.CrossEntropyLoss()

	optimizer = optim.Adam([{'params': encode.parameters()},
							{'params': middle.parameters()},
							{'params': decode.parameters()}], opt.base_lr, weight_decay=opt.weight_decay)

	# k fold
	test_auc = []

	trainset, valset, _ = utils.get_dataset(opt)
	trainloader = DataLoader(trainset, batch_size=opt.batch_size,
							 shuffle=True, num_workers=opt.num_workers, drop_last=True)
	valloader = DataLoader(valset, batch_size=opt.batch_size,
						   shuffle=False, num_workers=opt.num_workers)

	best_val_loss = 1e9
	for epoch in range(opt.epoches):
		encode, middle, decode = encode.train(), middle.train(), decode.train()
		for step, pack in enumerate(trainloader):
			img = pack[0]
			hd = pack[1]

			if opt.gpu:
				img, hd = img.cuda(), hd.cuda()

			out = encode(img)
			out = middle(out)
			out = decode(out)

			loss = torch.sum((out-hd)**2)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			writer.add_scalar('train/loss', loss.item(), epoch*len(trainloader)+step)
			print('epoch:{} step:{}/{} train loss:{}'.format(epoch, step, len(trainloader), loss.item()))

		if epoch % opt.val_interval == 0:
			encode, middle, decode = encode.eval(), middle.eval(), decode.eval()
			with torch.no_grad():
				res, gt = [], []
				for step, pack in enumerate(valloader):
					img = pack[0]
					hd = pack[1]
					if opt.gpu:
						img = img.cuda()

					out = encode(img)
					out = middle(out)
					out = decode(out).cpu().numpy()

					res.append(out)
					gt.append(hd.numpy())

				res = np.concatenate(res, 0)
				gt = np.concatenate(gt, 0)
				loss = np.sum((res-gt)**2)
				writer.add_scalar('val/loss', loss, epoch)
				print('epoch:{} val loss:{}'.format(epoch, loss.item()))
				if loss <= best_val_loss:
					best_val_loss = loss
					torch.save({'encode': encode.module.state_dict() if opt.gpus else encode.state_dict(),
								'middle': middle.module.state_dict() if opt.gpus else middle.state_dict(),
								'decode': decode.module.state_dict() if opt.gpus else decode.state_dict(),
								'optimizer': optimizer}, os.path.join('./save', opt.name, 'check', 'best.pth.tar'))

				torch.save({'encode': encode.module.state_dict() if opt.gpus else encode.state_dict(),
								'middle': middle.module.state_dict() if opt.gpus else middle.state_dict(),
								'decode': decode.module.state_dict() if opt.gpus else decode.state_dict(),
								'optimizer': optimizer}, os.path.join('./save', opt.name, 'check', 'epoch_{}.pth.tar'.format(epoch)))

			writer.flush()
		
	writer.close()


def test(opt):
	encode, middle, decode = utils.get_model(opt)
	if opt.gpu:
		encode, middle, decode = encode.cuda(), middle.cuda(), decode.cuda()

	best_pth = torch.load(os.path.join(
			'./save', opt.name, 'check', 'best.pth.tar'), map_location='cpu')
	encode.load_state_dict(best_pth['encode'])
	middle.load_state_dict(best_pth['middle'])
	decode.load_state_dict(best_pth['decode'])

	encode, middle, decode = encode.eval(), middle.eval(), decode.eval()

	# k fold
	test_auc = []

	_, valset, testset = utils.get_dataset(opt)
	valloader = DataLoader(valset, batch_size=opt.batch_size,
						   shuffle=False, num_workers=opt.num_workers)
	testloader = DataLoader(testset, batch_size=opt.batch_size,
						   shuffle=False, num_workers=opt.num_workers)

	with torch.no_grad():
		res, gt, imgp = [], [], []
		for step, pack in enumerate(testloader):
			img = pack[0]
			hd = pack[1]
			ps = pack[2]
			if opt.gpu:
				img = img.cuda()

			out = encode(img)
			out = middle(out)
			out = decode(out).cpu().numpy()

			res.append(out)
			gt.append(hd.numpy())
			imgp.extend(ps)
			break

		res = np.concatenate(res, 0)
		gt = np.concatenate(gt, 0)
		loss = np.sum((res-gt)**2)
		print(loss)

		for x, p in zip(res, imgp):
			img = cv2.imread(p)
			h, w, c = img.shape
			img = utils.draw_hd(img, x)
			cv2.imwrite('./save/{}/imgs/{}'.format(opt.name, p.split('/')[-1]), img)





if __name__ == '__main__':
	opt = get_parser()
	train(opt)
	# test(opt)
