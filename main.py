from itertools import product
import sys
import argparse
from utils import logger
from datasets import get_dataset
from train_eval import cross_validation_with_val_set
from param_parser import parameter_parser
from utils import tab_printer
import torch
import random
import numpy as np

args = parameter_parser()
#tab_printer(args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if not args.no_cuda:
    torch.cuda.manual_seed(args.seed)

def create_n_filter_triples(dataset, feat_str, gfn_add_ak3=True,
                            gfn_reall=False, reddit_odeg10=True,
                            dd_odeg10_ak1=True):
    triples_filtered = []

    if gfn_add_ak3:
        feat_str += '+ak3'

    if reddit_odeg10 and dataset in [
            'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K']:
        feat_str = feat_str.replace('odeg100', 'odeg10')

    if dd_odeg10_ak1 and dataset in ['DD']:
        feat_str = feat_str.replace('odeg100', 'odeg10')
        feat_str = feat_str.replace('ak3', 'ak1')
    triples_filtered.append((dataset, feat_str))
    return triples_filtered


def run_exp_lib(dataset_feat_net_triples):
    results = []
    sys.stdout.flush()

    for (dataset_name, feat_str) in dataset_feat_net_triples:
        sys.stdout.flush()
        dataset = get_dataset(
            dataset_name, sparse=True, feat_str=feat_str, root=args.data_root)
        
        max_node_num = max(dataset.data.num_nodes)
        print('Data: {}, Max Node Num: {}'.format(dataset_name, max_node_num))

        train_acc, acc, std, duration = cross_validation_with_val_set(
            args,
            dataset,
            max_node_num=max_node_num,
            folds=10,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            weight_decay=args.weight_decay,
            epoch_select=args.epoch_select,
            with_eval_mode=args.with_eval_mode,
            logger=logger)

def main():
    dataset = args.dataset
    feat_str = 'deg+odeg100'
    run_exp_lib(create_n_filter_triples(dataset, feat_str))

if __name__ == '__main__':
    main()
