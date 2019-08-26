# Content here is designed to load custom Loss, optimizer, scheduler etc. according to args
import torch
import os


class Criterion(object):
    def __init__(self, args):
        self.normal_loss = args.normal_loss
        self.normal_w = args.normal_w
        self.loss = None
        if args.normal_loss == 'mse':
            self.loss_criterion = torch.nn.MSELoss()
        elif args.normal_loss == 'ce':
            self.loss_criterion = torch.nn.CrossEntropyLoss()
        else:
            raise Exception("Unknown/Unsupported Loss '{}'".format(args.normal_loss))
        if args.cuda:
            self.loss_criterion = self.loss_criterion.cuda()

    def forward(self, output, target):
        if self.normal_loss == 'mse':
            self.loss = self.normal_w * self.loss_criterion(output, target.float())
        elif self.normal_loss == 'ce':
            self.loss = self.normal_w * self.loss_criterion(output, target.long())
        return {'loss': self.loss.item()}

    def backward(self):
        self.loss.backward()


# initiates an optimizer according to the settings in the arguments
def get_optimizer(args, params):
    if args.solver == 'adam':
        optimizer = torch.optim.Adam(params, args.init_lr, betas=(args.beta_1, args.beta_2))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(params, args.init_lr, momentum=args.momentum)
    else:
        raise Exception("Unknown/Unsupported Optimizer {}".format(args.solver))
    return optimizer


def config_optimizer(args, model):
    optimizer = get_optimizer(args, model.parameters())
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
            milestones=args.milestones, gamma=args.lr_decay, last_epoch=args.start_epoch-2)
    return optimizer, scheduler
