from __future__ import division
import os
import numpy as np
# from imageio import imread
from skimage import io
import torch.utils.data as data
from datasets import transforms
np.random.seed(0)


class CustomDataset(data.Dataset):
	def __init__(self, args, root, split='train'):
		self.root = os.path.join(root)
		self.split = split
		self.args = args

	def _getInputPath(self, index):  # TODO
		img_path = os.path.join(self.args.data_dir, 'Images', str(index))
		return img_path

	def __getitem__(self, index):  # TODO
		img_dir = self._getInputPath(index)
		if self.args.template_debug:  # if debugging the template, don't read an image
			img = np.empty((100, 100, 3)).astype(np.float32) / 255.0
		else:
			img = io.imread(img_dir, as_gray=self.args.grayscale).astype(np.float32) / 255.0

		h, w, c = img.shape
		crop_h, crop_w = self.args.crop_h, self.args.crop_w
		if self.args.rescale:
			sc_h = np.random.randint(crop_h, h)
			sc_w = np.random.randint(crop_w, w)
			img = transforms.rescale(img, sc_h, sc_w)

		if self.args.crop:
			img = transforms.random_crop(img, [crop_h, crop_w])

		if self.args.color_aug:
			img = (img * np.random.uniform(1, 3)).clip(0, 2)

		if self.args.noise_aug:
			img = transforms.random_noise_aug(img, self.args.noise)

		if self.args.template_debug:
			label = 1
		else:
			pass  # FIXME

		item = {'img': img, 'label': label}  # TODO

		return item

	def __len__(self):
		# return len(self.XX)
		return 1  # FIXME!




