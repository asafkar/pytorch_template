
# TODO implement according to data


def calc_acc(outputs, truths):
	acc = (outputs == truths).sum().item()  # FIXME
	res = {'acc': acc.item()}  # return as dictionary
	return res

