import time, os, datetime
import matplotlib.pyplot as plt


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
            time_elapsed, time_total = self.getTimeInfo(epoch, iters, batch)  # Buggy for test
            strs += ' Clock [{:.2f}h/{:.2f}h]'.format(time_elapsed, time_total)
            strs += ' LR [{}]'.format(opt['recorder'].records[opt['split']]['lr'][epoch][0])
        self.write_log(strs)
        if 'timer' in opt.keys():
            self.write_log(opt['timer'].timeToString())
        if 'recorder' in opt.keys():
            self.write_log(opt['recorder'].iterRecToString(opt['split'], epoch))

    def print_epoch_summary(self, opt):
        split = opt['split']
        epoch = opt['epoch']
        self.write_log('---------- {} Epoch {} Summary -----------'.format(str.upper(split), epoch))
        self.write_log(opt['recorder'].epochRecToString(split, epoch))


def make_dirs(dir_list):
    for dir in dir_list:
        if not os.path.exists(dir):
            os.makedirs(dir)



