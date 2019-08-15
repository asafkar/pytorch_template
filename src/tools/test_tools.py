import os
import torch
from tools import model_tools, eval_tools


def get_itervals(args, split):
    args_var = vars(args)
    disp_intv = args_var[split+'_disp']
    save_intv = args_var[split+'_save']
    return disp_intv, save_intv


def test(model, args, loader, split, log, epoch, recorder):
    model.eval()

    disp_interval, save_interval = get_itervals(args, split)
    with torch.no_grad():
        for ii, sample in enumerate(loader):
            inputs, labels = model_tools.parse_input(args, sample)
            outputs = model(inputs)  # pass inputs through model (forward pass)

            acc = eval_tools.calc_acc(outputs.data, labels)  # TODO
            recorder.updateIter('train', acc.keys(), acc.values())

            iters = ii + 1
            if iters % disp_interval == 0:
                current_stat = {'split': split, 'epoch': epoch, 'iters': iters, 'batch': len(loader),
                                'recorder': recorder}
                log.print_iteration_summary(current_stat)

    current_stat = {'split': split, 'epoch': epoch, 'recorder': recorder}
    log.print_epoch_summary(current_stat)
