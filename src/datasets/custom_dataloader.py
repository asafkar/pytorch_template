import torch.utils.data


def custom_dataloader(args):
	if args.dataset == 'custom_dataset':  # TODO
		from datasets.custom_dataset import CustomDataset
		train_set = CustomDataset(args, args.data_dir, 'train')
		val_set = CustomDataset(args, args.data_dir, 'val')
	else:
		raise Exception('Unknown dataset: {}'.format(args.dataset))

	train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch, num_workers=args.workers,
												pin_memory=args.cuda, shuffle=True)
	test_loader = torch.utils.data.DataLoader(val_set, batch_size=args.val_batch, num_workers=args.workers,
												pin_memory=args.cuda, shuffle=True)
	return train_loader, test_loader
