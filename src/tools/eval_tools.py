
# TODO implement according to data
import torch


def calc_acc(args, outputs, truths):
	truths = truths.float()
	if args.normal_loss == 'mse':
		acc = (outputs == truths).sum().item()/len(truths)  # FIXME
	elif args.normal_loss == 'ce':
		# predicted, _ = torch.max(outputs, 1)
		predicted = outputs.argmax(1).float()
		acc = (predicted == truths).sum().item()/len(truths)  # FIXME
		# acc = (outputs == truths).sum()
	res = {'acc': acc}  # return as dictionary
	return res

