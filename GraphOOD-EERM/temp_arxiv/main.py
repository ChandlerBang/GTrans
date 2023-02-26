import argparse
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_scatter import scatter

from logger import Logger, SimpleLogger
from dataset import load_nc_dataset
from data_utils import normalize, gen_normalized_adjs, evaluate, evaluate_whole_graph, eval_acc, eval_rocauc, eval_f1, to_sparse_tensor, load_fixed_splits
from parse import parse_method_base, parse_method_ours, parser_add_main_args

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
fix_seed(0)

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()

# #TODO
# if args.gnn == 'gcn':
#     args.hidden_channels = 128
#     args.num_layers=2
#     args.lr=0.01
#     args.epochs=500
#     args.dropout=0.5
print(args)

from data_utils import get_gpu_memory_map
mem_st = get_gpu_memory_map()

device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

def get_dataset(dataset, year=None):
    ### Load and preprocess data ###
    if dataset == 'ogb-arxiv':
        dataset = load_nc_dataset(args.data_dir, 'ogb-arxiv', year=year)
    else:
        raise ValueError('Invalid dataname')

    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)

    dataset.n = dataset.graph['num_nodes']
    dataset.c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    dataset.d = dataset.graph['node_feat'].shape[1]

    dataset.graph['edge_index'], dataset.graph['node_feat'] = \
        dataset.graph['edge_index'], dataset.graph['node_feat']

    return dataset

if args.dataset == 'ogb-arxiv':
    tr_year, val_year, te_years = [[1950, 2011]], [[2011, 2014]], [[2014, 2016], [2016, 2018], [2018, 2020]]
    dataset_tr = get_dataset(dataset='ogb-arxiv', year=tr_year[0])
    dataset_val = get_dataset(dataset='ogb-arxiv', year=val_year[0])
    datasets_te = [get_dataset(dataset='ogb-arxiv', year=te_years[i]) for i in range(len(te_years))]
else:
    raise ValueError('Invalid dataname')

print(f"Train num nodes {dataset_tr.n} | target nodes {dataset_tr.test_mask.sum()} | num classes {dataset_tr.c} | num node feats {dataset_tr.d}")
print(f"Val num nodes {dataset_val.n} | target nodes {dataset_val.test_mask.sum()} | num classes {dataset_val.c} | num node feats {dataset_val.d}")

for i in range(len(te_years)):
    dataset_te = datasets_te[i]
    print(f"Test {i} num nodes {dataset_te.n} | target nodes {dataset_te.test_mask.sum()} | num classes {dataset_te.c} | num node feats {dataset_te.d}")

### Load method ###
if args.method == 'erm':
    model = parse_method_base(args, dataset_tr, device)
else:
    model = parse_method_ours(args, dataset_tr, device)

# using rocauc as the eval function
criterion = nn.NLLLoss()
eval_func = eval_acc

logger = Logger(args.runs, args)

model.train()
print('MODEL:', model)
print('DATASET:', args.dataset)

### Training loop ###
for run in range(args.runs):
    # fix_seed(run)
    if run > 0:
        fix_seed(run)
    model.reset_parameters()
    if args.method == 'erm':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.method == 'eerm':
        optimizer_gnn = torch.optim.AdamW(model.gnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_aug = torch.optim.AdamW(model.gl.parameters(), lr=args.lr_a)
    best_val = float('-inf')
    import time; st = time.time()
    for epoch in range(args.epochs):
        model.train()
        if args.method == 'erm':
            optimizer.zero_grad()
            loss = model(dataset_tr, criterion)
            loss.backward()
            optimizer.step()
        elif args.method == 'eerm':
            model.gl.reset_parameters()
            for m in range(args.T):
                Var, Mean, Log_p = model(dataset_tr, criterion)
                outer_loss = Var + args.beta * Mean
                reward = Var.detach()
                inner_loss = - reward * Log_p
                if m == 0:
                    optimizer_gnn.zero_grad()
                    outer_loss.backward()
                    optimizer_gnn.step()
                optimizer_aug.zero_grad()
                inner_loss.backward()
                optimizer_aug.step()

        if epoch == 19:
            print('time consumed:', time.time()-st)
            assert False

        accs, test_outs = evaluate_whole_graph(args, model, dataset_tr, dataset_val, datasets_te, eval_func)
        logger.add_result(run, accs)

        if epoch % args.display_step == 0:
            if args.method == 'erm':
                print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * accs[0]:.2f}%, '
                  f'Valid: {100 * accs[1]:.2f}%, ')
                test_info = ''
                for test_acc in accs[2:]:
                    test_info += f'Test: {100 * test_acc:.2f}% '
                print(test_info)
            elif args.method == 'eerm':
                print(f'Epoch: {epoch:02d}, '
                      f'Mean Loss: {Mean:.4f}, '
                      f'Var Loss: {Var:.4f}, '
                      f'Train: {100 * accs[0]:.2f}%, '
                      f'Valid: {100 * accs[1]:.2f}%, ')
                test_info = ''
                for test_acc in accs[2:]:
                    test_info += f'Test: {100 * test_acc:.2f}% '
                print(test_info)
        # # if (epoch+1) % 100 == 0:
        # if epoch + 1== 10:
        #     print(f'Total running for {epoch+1} epochs: {time.time() - st:.2f}s')
        #     from data_utils import get_gpu_memory_map
        #     gpu_mem = get_gpu_memory_map()
        #     print('Mem used: %s MB'% (int(gpu_mem[args.device])-int(mem_st[args.device])))

    print(f'Total running for {args.epochs} epochs: {time.time() - st:.2f}s')
    logger.print_statistics(run)
    y_te = []
    for i, dataset in enumerate(datasets_te):
        y_te.append(dataset.label[dataset.test_mask])
        if i == len(datasets_te) - 1:
            acc_te = eval_func(torch.cat(y_te, dim=0), torch.cat(test_outs, dim=0))
    print(f'Flatten Test: {100*acc_te:.2f}')


### Save results ###
results = logger.print_statistics()
filename = f'./results/{args.dataset}.csv'
print(f"Saving results to {filename}")
with open(f"{filename}", 'a+') as write_obj:
    # sub_dataset = f'{args.sub_dataset},' if args.sub_dataset else ''
    log = f"{args.method}," + f"{args.gnn},"
    for i in range(results.shape[1]):
        r = results[:, i]
        log += f"{r.mean():.3f} ± {r.std():.3f},"
    write_obj.write(log + f"\n")
