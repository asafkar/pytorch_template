from skimage.transform import resize
import random
import numpy as np


def rescale(inputs, h, w):
	in_h, in_w, _ = inputs.shape
	if h != in_h or w != in_w:
		inputs = resize(inputs, (h, w))
	return inputs


def random_crop(inputs, size):
	h, w, _ = inputs.shape
	c_h, c_w = size
	if h == c_h and w == c_w:
		return inputs
	x1 = random.randint(0, w - c_w)
	y1 = random.randint(0, h - c_h)
	inputs = inputs[y1: y1 + c_h, x1: x1 + c_w]
	return inputs


def random_noise_aug(inputs, noise_level=0.05):
	noise = np.random.random(inputs.shape)
	noise = (noise - 0.5) * noise_level
	inputs += noise
	return inputs

