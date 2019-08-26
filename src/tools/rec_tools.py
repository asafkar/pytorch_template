from collections import OrderedDict
import numpy as np
import os
import torch


class Records(object):
	"""
	This class holds all information relevant to give ability to continue training the model
	after stopping.
	This includes the current: iteration, loss, accuracy, 
	"""
	def __init__(self, log_dir, records=None):
		if records is None:
			self.records = OrderedDict()
		else:
			self.records = records
		self.iter_rec = OrderedDict()
		self.log_dir = log_dir
		self.classes = ['loss', 'acc', 'err', 'ratio']

	@staticmethod
	def _check_if_in_dict(a_dict, key, sub_type='dict'):  # create a new dict/list if doesn't exist
		if key not in a_dict.keys():
			if sub_type == 'dict':
				a_dict[key] = OrderedDict()
			if sub_type == 'list':
				a_dict[key] = []

	# add current loss/accuracy to records
	def update_iter(self, split, keys, values):
		self._check_if_in_dict(self.iter_rec, split, 'dict')
		for k, v in zip(keys, values):
			self._check_if_in_dict(self.iter_rec[split], k, 'list')
			self.iter_rec[split][k].append(v)

	def _save_iter_record(self, epoch, reset=True):
		for split in self.iter_rec.keys():
			self._check_if_in_dict(self.records, split, 'dict')
			for key in self.iter_rec[split].keys():
				self._check_if_in_dict(self.records[split], key, 'dict')
				self._check_if_in_dict(self.records[split][key], epoch, 'list')
				self.records[split][key][epoch].append(np.mean(self.iter_rec[split][key]))
		if reset:
			self.iter_rec.clear()

	# insert an updated learning rate
	def insert_record(self, split, key, epoch, lr):
		self._check_if_in_dict(self.records, split, 'dict')
		self._check_if_in_dict(self.records[split], key, 'dict')
		self._check_if_in_dict(self.records[split][key], epoch, 'list')
		self.records[split][key][epoch].append(lr)

	def epoch_record_to_str(self, split, epoch):
		rec_strs = ''
		for c in self.classes:
			strs = ''
			self._check_if_in_dict(self.records, split, 'dict')
			for k in self.records[split].keys():
				if (c in k.lower()) and (epoch in self.records[split][k].keys()):
					strs += '{}: {:.3f}| '.format(k, np.mean(self.records[split][k][epoch]))
			if strs != '':
				rec_strs += '\t [{}] {}\n'.format(c.upper(), strs)
		return rec_strs

	def iteration_rec_to_str(self, split, epoch):
		rec_strs = ''
		for c in self.classes:
			strs = ''
			for k in self.iter_rec[split].keys():
				if (c in k.lower()):
					strs += '{}: {:.3f}| '.format(k, np.mean(self.iter_rec[split][k]))
			if strs != '':
				rec_strs += '\t [{}] {}\n'.format(c.upper(), strs)
		self._save_iter_record(epoch)
		return rec_strs


def load_records(path, optimizer):
	if os.path.isfile(path):
		records = torch.load(path[:-8] + '_rec' + path[-8:])
		optimizer.load_state_dict(records['optimizer'])
		start_epoch = records['epoch'] + 1
		records = records['records']
		print("=> loaded Records")
	else:
		raise Exception("=> no checkpoint found at '{}'".format(path))
	return records, start_epoch
