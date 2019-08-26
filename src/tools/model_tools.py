import os
import torch
import torch.nn as nn


# this function splits the sample data from data loader to relevant categories,
# i.e. sentences and labels, images and categories, etc...
# TODO implement this according to data
def parse_input(args, sample):
    if args.template_debug:  # FIXME
        img, label = sample['img'], sample['label']
        if args.cuda:
            img, label = img.cuda(), label.cuda()
        return img, label
    else:
        pass


# Loads the models state_dict.
# Epoch is stored in Records (which is loaded in main_flow)
# Loss / Optimizer is defined in args, and not loaded here (FIXME maybe should be?)
def load_checkpoint(path, model, cuda=True):
    if cuda:
        checkpoint = torch.load(path)  # assume NN is always trained on GPU. Else may need to add map_location="cuda:0")
    else:
        checkpoint = torch.load(path, map_location=torch.device('cpu'))  # load GPU trained net to CPU
    model.load_state_dict(checkpoint['state_dict'])


def save_checkpoint(save_path, epoch=-1, model=None, optimizer=None, records=None, args=None):
    state = {'state_dict': model.state_dict(), 'model': args.model}
    records = {'epoch': epoch, 'optimizer': optimizer.state_dict(), 'records': records}
    torch.save(state,   os.path.join(save_path, 'chkpnt_{}.pt'.format(epoch)))
    torch.save(records, os.path.join(save_path, 'chkpnt_{}_rec.pt'.format(epoch)))


# TODO - add blocks which are repeated here. Example:
def fc(args, cin):
    if args.normal_loss == 'mse':
        return nn.Linear(cin, 1)
    else:
        return nn.Linear(cin, 2)



