import sys
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch import tensor
from capsgnn import CapsGNN
from torch.optim import Adam, SGD
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cross_validation_with_val_set(args,
                                  dataset,
                                  max_node_num,
                                  folds,
                                  epochs,
                                  batch_size,
                                  lr,
                                  lr_decay_factor,
                                  lr_decay_step_size,
                                  weight_decay,
                                  epoch_select,
                                  with_eval_mode=True,
                                  logger=None):
    assert epoch_select in ['val_min', 'test_max'], epoch_select

    val_losses, train_accs, test_accs, durations = [], [], [], []
    for fold, (train_idx, test_idx, val_idx) in enumerate(
            zip(*k_fold(dataset, folds, epoch_select))):

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]

        train_loader = DenseLoader(train_dataset, batch_size, shuffle=True)
        val_loader = DenseLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DenseLoader(test_dataset, batch_size, shuffle=False)

        model = CapsGNN(args, dataset.num_features, dataset.num_classes, max_node_num).to(device)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()
        writer = SummaryWriter(args.log_path)

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train(
                args, model, optimizer, train_loader, device, max_node_num, epoch, writer)
            train_accs.append(train_acc)
            val_loss, _ = eval_loss(args, model, val_loader, device, max_node_num, with_eval_mode)
            val_losses.append(val_loss)
            test_accs.append(eval_acc(
                model, test_loader, device, max_node_num, with_eval_mode))
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_accs[-1],
                'val_loss': val_losses[-1],
                'test_acc': test_accs[-1],
            }

            if logger is not None:
                logger(eval_info)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)
    
    duration = tensor(durations)
    train_acc, test_acc = tensor(train_accs), tensor(test_accs)
    val_loss = tensor(val_losses)

    train_acc = train_acc.view(folds, epochs)
    test_acc = test_acc.view(folds, epochs)
    val_loss = val_loss.view(folds, epochs)
    if epoch_select == 'test_max':  # take a single epoch that yields best test results across 10 folds
        _, selected_epoch = test_acc.mean(dim=0).max(dim=0)
        selected_epoch_rep = selected_epoch.repeat(folds)
    else:  # take epoch that yields min val loss for each fold individually.
        _, selected_epoch_rep = val_loss.min(dim=1)
    
    # The criteria used in GMN and STRUCPOOL
    test_acc_epoch_max = torch.max(test_acc, dim=1)[0]
    test_acc_epoch_mean = test_acc_epoch_max.mean().item()
    test_acc_epoch_std = test_acc_epoch_max.std().item()
    ########################################

    test_acc = test_acc[torch.arange(folds, dtype=torch.long), selected_epoch_rep]
    train_acc_mean = train_acc[:, -1].mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()
    duration_mean = duration.mean().item()
    
    print('Train Acc: {:.2f}, Test Acc: {:.2f} {:.2f}, Test Acc (C*): {:.2f} {:.2f}, Duration: {:.3f}'.
          format(train_acc_mean*100, test_acc_mean*100, test_acc_std*100, test_acc_epoch_mean*100, test_acc_epoch_std*100, duration_mean))
    sys.stdout.flush()

    return train_acc_mean, test_acc_mean, test_acc_std, duration_mean


def k_fold(dataset, folds, epoch_select):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx))

    if epoch_select == 'test_max':
        # in this situation, val_invices = test_indices
        val_indices = [test_indices[i] for i in range(folds)]
    else:
        val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i].long()] = 0
        train_mask[val_indices[i].long()] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(args, model, optimizer, loader, device, max_node_num, epoch, writer):
    model.train()

    total_loss = 0
    correct = 0
    for batch_iter, data in enumerate(loader):
        optimizer.zero_grad()
        data = data.to(device)
        features_batch, adj_batch, mask_batch = data.x, data.adj, data.mask
        batch = torch.tensor([i for i in range(mask_batch.size(0)) for t in range(torch.sum(mask_batch[i]))]).to(device)
        
        out, reconstruction_loss = model(features_batch, adj_batch, mask_batch, batch, data.y)
        loss_ce = margin_loss(out, data.y.view(-1))
        loss = loss_ce + args.theta*reconstruction_loss
        
        pred = out.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        
        writer.add_scalar('CE loss', loss_ce, (epoch - 1) * len(loader) + (batch_iter + 1))
        writer.add_scalar('Reconstruction loss', args.theta*reconstruction_loss, (epoch - 1) * len(loader) + (batch_iter + 1))
        optimizer.step()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def eval_acc(model, loader, device, max_node_num, with_eval_mode):
    if with_eval_mode:
        model.eval()
    
    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            features_batch, adj_batch, mask_batch = data.x, data.adj, data.mask
            batch = torch.tensor([i for i in range(mask_batch.size(0)) for t in range(torch.sum(mask_batch[i]))]).to(device)
            out, _ = model(features_batch, adj_batch, mask_batch, batch, data.y)
            pred = out.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(args, model, loader, device, max_node_num, with_eval_mode):
    if with_eval_mode:
        model.eval()

    loss = 0
    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            features_batch, adj_batch, mask_batch = data.x, data.adj, data.mask
            batch = torch.tensor([i for i in range(mask_batch.size(0)) for t in range(torch.sum(mask_batch[i]))]).to(device)
            out, reconstruction_loss = model(features_batch, adj_batch, mask_batch, batch, data.y)
            pred = out.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        loss += (margin_loss(out, data.y.view(-1)).item() + \
                args.theta*reconstruction_loss)
  
    return loss / len(loader.dataset), correct / len(loader.dataset)

def margin_loss(scores, target, loss_lambda=0.5):
    target = F.one_hot(target, scores.size(1))
    v_mag = scores

    zero = torch.zeros(1)
    zero = zero.cuda()
    m_plus = 0.9
    m_minus = 0.1

    max_l = torch.max(m_plus - v_mag, zero)**2
    max_r = torch.max(v_mag - m_minus, zero)**2
    T_c = target

    L_c = T_c * max_l + loss_lambda * (1.0 - T_c) * max_r
    L_c = L_c.sum(dim=1)
    L_c = L_c.mean()
    return L_c