import argparse
import os
import torch


class BaseOptions(object):
	def __init__(self):
		self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	def initialize(self):
		self.parser.add_argument('--template_debug', default='True')  # used for debugging template! FIXME

		# Training Dataset
		self.parser.add_argument('--dataset', default='custom_dataset')  # TODO
		self.parser.add_argument('--data_dir', default='../data/processed/custom_dataset')  # TODO

		# Training Data and Preprocessing Arguments - some examples...
		self.parser.add_argument('--rescale', default=True, action='store_false')
		self.parser.add_argument('--crop', default=True, action='store_false')
		self.parser.add_argument('--int_aug', default=True, action='store_false')
		self.parser.add_argument('--noise_aug', default=True, action='store_false')
		self.parser.add_argument('--noise', default=0.05, type=float)
		self.parser.add_argument('--color_aug', default=True, action='store_false')
		self.parser.add_argument('--grayscale', default=False, action='store_false')


		# Device Arguments
		self.parser.add_argument('--cuda', default=True, action='store_false')
		self.parser.add_argument('--multi_gpu', default=False, action='store_true')
		self.parser.add_argument('--workers', default=0, type=int)  # FIXME! doesn't work on windows, set to 0

		self.parser.add_argument('--seed', default=0, type=int)

		# Model Arguments
		self.parser.add_argument('--model', default='custom_model')  # TODO
		self.parser.add_argument('--resume', default=None)  # used to resume training from a certain checkpoint
		self.parser.add_argument('--save_interval', default=1, type=int)  # save model every x epoch
		self.parser.add_argument('--start_epoch', default=1, type=int)
		self.parser.add_argument('--epochs', default=20, type=int)

		# Displaying Arguments
		self.parser.add_argument('--train_disp', default=20, type=int)
		self.parser.add_argument('--train_save', default=200, type=int)
		self.parser.add_argument('--val_interval', default=3, type=int)  # validate every x epoch
		self.parser.add_argument('--val_disp', default=1, type=int)
		self.parser.add_argument('--val_save', default=1, type=int)
		self.parser.add_argument('--max_train_iter', default=-1, type=int)
		self.parser.add_argument('--max_val_iter', default=-1, type=int)
		self.parser.add_argument('--max_test_iter', default=-1, type=int)
		self.parser.add_argument('--train_save_n', default=4, type=int)
		self.parser.add_argument('--test_save_n', default=4, type=int)

		# Log Arguments
		self.parser.add_argument('--log_dir', default='../reports/logs/')
		self.parser.add_argument('--debug', default=False, action='store_true')
		self.parser.add_argument('--dump_log_to_file', default=True, action='store_false')
		self.parser.add_argument('--save_split', default=False, action='store_true')

	def parse(self):
		self.args = self.parser.parse_args()
		return self.args


class TrainOptions(BaseOptions):
	def __init__(self):
		super(TrainOptions, self).__init__()
		self.initialize()

	def initialize(self):
		BaseOptions.initialize(self)

		# Training Arguments
		self.parser.add_argument('--solver', default='adam', help='adam|sgd')
		self.parser.add_argument('--milestones', default=[5, 10, 15, 20, 25], nargs='+', type=int)
		self.parser.add_argument('--init_lr', default=1e-3, type=float)
		self.parser.add_argument('--lr_decay', default=0.5, type=float)
		self.parser.add_argument('--beta_1', default=0.9, type=float, help='adam')
		self.parser.add_argument('--beta_2', default=0.999, type=float, help='adam')
		self.parser.add_argument('--momentum', default=0.9, type=float, help='sgd')
		self.parser.add_argument('--batch', default=4, type=int)
		self.parser.add_argument('--val_batch', default=8, type=int)

		# default size for cropping, if crop is True
		self.parser.add_argument('--crop_h', default=32, type=int)
		self.parser.add_argument('--crop_w', default=32, type=int)

		# Loss Arguments
		self.parser.add_argument('--normal_loss', default='ce', help='mse|ce')
		self.parser.add_argument('--normal_w', default=1)

		# model Arguments
		self.parser.add_argument('--use_batchnorm', default=True)





	def parse(self):
		BaseOptions.parse(self)
		return self.args


class RunOptions(BaseOptions):
	def __init__(self):
		super(RunOptions, self).__init__()
		self.initialize()

	def initialize(self):
		BaseOptions.initialize(self)
		self.parser.add_argument('--run_model', default=True, action='store_false')
		self.parser.add_argument('--epochs', default=1, type=int)
		self.parser.add_argument('--test_batch', default=1, type=int)
		self.parser.add_argument('--test_disp', default=1, type=int)
		self.parser.add_argument('--test_save', default=1, type=int)

	def parse(self):
		BaseOptions.parse(self)
		self.setDefault()
		return self.args
