import torch
from tools import logger_tools, rec_tools, model_tools
from tools import train_tools, test_tools, solver_tools

import options
from datasets import custom_dataloader
from models import custom_model


def main_sequence(args, log):
    train_loader, val_loader = custom_dataloader.custom_dataloader(args)
    model = custom_model.build_model(args)  # build a custom model according to arguments
    optimizer, scheduler = solver_tools.config_optimizer(args, model)

    if args.resume:
        records, start_epoch = rec_tools.load_records(args.resume, optimizer)
        args.start_epoch = start_epoch
    else:
        records = None

    criterion = solver_tools.Criterion(args)
    recorder = rec_tools.Records(args.log_dir, records)

    for epoch in range(args.start_epoch, args.epochs+1):
        scheduler.step()
        recorder.insertRecord('train', 'lr', epoch, scheduler.get_lr()[0])

        train_tools.train(model, args, train_loader, criterion, optimizer, log, epoch, recorder)
        if epoch % args.save_interval == 0:
            model_tools.save_checkpoint(args.chkpnt_dir, epoch, model, optimizer, recorder.records, args)

        if epoch % args.val_interval == 0:
            test_tools.test(model, args, val_loader, 'val', log, epoch, recorder)


if __name__ == '__main__':
    arguments = options.TrainOptions().parse()
    log_ = logger_tools.Logger(arguments)
    torch.manual_seed(arguments.seed)
    main_sequence(arguments, log_)

