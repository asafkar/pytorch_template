from tools import model_tools


# train for one epoch
def train(model, args, data_loader, criterion, optimizer, logger, epoch, recorder):
	model.train()  # Set model to training mode

	# TODO inputs, labels are an example. Change accordingly
	for iteration, sample in enumerate(data_loader):
		inputs, labels = model_tools.parse_input(args, sample)
		outputs = model(inputs)  # pass inputs through model (forward pass)
		optimizer.zero_grad()
		loss = criterion(outputs, labels)
		criterion.backward()

		recorder.updateIter('train', loss.keys(), loss.values())
		optimizer.step()

		if (iteration + 1) % args.train_disp == 0:
			current_stat = {'split': 'train', 'epoch': epoch, 'iters': (iteration + 1), 'batch': len(data_loader),
					'recorder': recorder}
			logger.print_iteration_summary(current_stat)

	current_stat = {'split': 'train', 'epoch': epoch, 'recorder': recorder}
	logger.print_epoch_summary(current_stat)

