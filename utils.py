import sys
from texttable import Texttable
import numpy as np
import torch
import torch.nn as nn


def logger(info):
    fold, epoch = info['fold'], info['epoch']
    if epoch == 1 or epoch % 10 == 0:
        train_acc, test_acc = info['train_acc'], info['test_acc']
        print('{:02d}/{:03d}: Train Acc: {:.3f}, Test Accuracy: {:.3f}'.format(
            fold, epoch, train_acc, test_acc))
    sys.stdout.flush()

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())