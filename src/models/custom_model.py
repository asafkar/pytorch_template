# This function creates a custom user selected model according to arguments
from tools import model_tools


def build_model(args):
	if args.model == 'custom_model':
		from models.user_model import SomeModule
		model = SomeModule(args, 32)
	else:
		raise Exception("Unknown/Unsupported Model '{}'".format(args.model))

	if args.cuda:
		model = model.cuda()

	if args.resume:
		print("Resume training from checkpoint {}".format(args.resume))
		model_tools.load_checkpoint(args.resume, model, cuda=args.cuda)
	print(model)
	return model

