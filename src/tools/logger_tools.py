import time
import os
import datetime

"""" 
Logger is a class for printing information to the screen, and saving it to 
file, in order to be able to analyze later. 
"""


class Logger(object):
	def __init__(self, args):
		self.times = {'init': time.time()}
		self.args = args
		args.log = self

		if args.resume and os.path.isfile(args.resume):
			log_root = os.path.join(os.path.dirname(os.path.dirname(args.resume)), 'resume')
			args.log_dir = log_root  # update this if resuming
		else:
			log_root = args.log_dir
		for sub_dir in ['train', 'val']:
			make_dirs([os.path.join(log_root, sub_dir)])
		args.chkpnt_dir = os.path.join(log_root, 'train')  # update checkpoint dir

		date_now = datetime.datetime.now()
		dir_name = '%d-%d' % (date_now.month, date_now.day)
		dir_name += ',DEBUG' if args.debug else ''

		file_dir = os.path.join(args.log_dir, '%s' % dir_name)
		self.log_file = open(file_dir, 'w')

	def write_log(self, str_):
		print('%s' % str_)
		if self.args.dump_log_to_file:
			self.log_file.write('%s\n' % str_)
			self.log_file.flush()

	def print_iteration_summary(self, opt):
		epoch, iters, batch = opt['epoch'], opt['iters'], opt['batch']
		strs = ' | {}'.format(str.upper(opt['split']))
		strs += ' Iter [{}/{}] Epoch [{}/{}]'.format(iters, batch, epoch, self.args.epochs)
		if opt['split'] == 'train':
			strs += ' LR [{}]'.format(opt['recorder'].records[opt['split']]['lr'][epoch][0])
		self.write_log(strs)
		if 'recorder' in opt.keys():
			self.write_log(opt['recorder'].iteration_rec_to_str(opt['split'], epoch))

	def print_epoch_summary(self, opt):
		split = opt['split']
		epoch = opt['epoch']
		self.write_log('---------- {} Epoch {} Summary -----------'.format(str.upper(split), epoch))
		self.write_log(opt['recorder'].epoch_record_to_str(split, epoch))


def make_dirs(dir_list):
	for dir_ in dir_list:
		if not os.path.exists(dir_):
			os.makedirs(dir_)



