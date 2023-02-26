import numpy as np
from models import *
import torch.nn.functional as F
import torch
import deeprobust.graph.utils as utils
from torch.nn.parameter import Parameter
from tqdm import tqdm
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from copy import deepcopy
from utils import reset_args
import random
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix, dropout_adj, is_undirected, to_undirected

class FeatAgent:

    def __init__(self, data_all, args):
        self.device = 'cuda'
        self.args = args
        self.data_all = data_all
        self.model = self.pretrain_model()

    def initialize_as_ori_feat(self, feat):
        self.delta_feat.data.copy_(feat)

    def learn_graph(self, data):
        args = self.args
        args = self.args
        self.data = data
        nnodes = data.graph['node_feat'].shape[0]
        d = data.graph['node_feat'].shape[1]
        delta_feat = Parameter(torch.FloatTensor(nnodes, d).to(self.device))
        self.delta_feat = delta_feat
        delta_feat.data.fill_(1e-7)
        self.optimizer_feat = torch.optim.Adam([delta_feat], lr=args.lr_feat)

        model = self.model
        for param in model.parameters():
            param.requires_grad = False
        model.eval() # should set to eval

        feat, labels = data.graph['node_feat'].to(self.device), data.label.to(self.device)#.squeeze()
        edge_index = data.graph['edge_index'].to(self.device)
        self.edge_index, self.feat, self.labels = edge_index, feat, labels

        for it in tqdm(range(args.epochs)):
            self.optimizer_feat.zero_grad()
            loss = self.test_time_loss(model, feat+delta_feat, edge_index)

            loss.backward()
            if it % 100 == 0:
                print(f'Epoch {it}: {loss}')

            self.optimizer_feat.step()
            if args.debug==2:
                output = model.predict(feat+delta_feat, edge_index)
                print('Test:', self.evaluate_single(model, output, labels, data))


        with torch.no_grad():
            loss = self.test_time_loss(model, feat+delta_feat, edge_index)
        print(f'Epoch {it+1}: {loss}')

        output = model.predict(feat+delta_feat, edge_index)
        print('Test on transformed graph:')
        if args.dataset == 'elliptic':
            return self.evaluate_single(model, output, labels, data), output[data.mask], labels[data.mask]
        else:
            return self.evaluate_single(model, output, labels, data), output, labels

    def augment(self, strategy='dropedge', p=0.5, edge_index=None, edge_weight=None):
        model = self.model
        if hasattr(self, 'delta_feat'):
            delta_feat = self.delta_feat
            feat = self.feat + delta_feat
        else:
            feat = self.feat
        if strategy == 'shuffle':
            idx = np.random.permutation(feat.shape[0])
            shuf_fts = feat[idx, :]
            output = model.get_embed(shuf_fts, edge_index, edge_weight)
        if strategy == "dropedge":
            edge_index, edge_weight = dropout_adj(edge_index, edge_weight, p=p)
            output = model.get_embed(feat, edge_index, edge_weight)
        if strategy == "dropnode":
            feat = self.feat + self.delta_feat
            mask = torch.cuda.FloatTensor(len(feat)).uniform_() > p
            feat = feat * mask.view(-1, 1)
            output = model.get_embed(feat, edge_index, edge_weight)
        if strategy == "dropmix":
            feat = self.feat + self.delta_feat
            mask = torch.cuda.FloatTensor(len(feat)).uniform_() > p
            feat = feat * mask.view(-1, 1)
            edge_index, edge_weight = dropout_adj(edge_index, edge_weight, p=p)
            output = model.get_embed(feat, edge_index, edge_weight)

        if strategy == "dropfeat":
            feat = F.dropout(self.feat, p=p) + self.delta_feat
            output = model.get_embed(feat, edge_index, edge_weight)
        if strategy == "featnoise":
            mean, std = 0, p
            noise = torch.randn(feat.size()) * std + mean
            feat = feat + noise.to(feat.device)
            output = model.get_embed(feat, edge_index)
        return output

    def test_time_loss(self, model, feat, edge_index, edge_weight=None, mode='train'):
        args = self.args
        loss = 0
        if 'LC' in args.loss: # label constitency
            if mode == 'eval': # random seed setting
                random.seed(args.seed)
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                torch.cuda.manual_seed(args.seed)
            if args.strategy == 'dropedge':
                # output1 = self.augment(strategy=args.strategy, p=0.5, edge_index=edge_index, edge_weight=edge_weight)
                output1 = self.augment(strategy=args.strategy, p=0.05, edge_index=edge_index, edge_weight=edge_weight) #TODO
            if args.strategy == 'dropnode':
                output1 = self.augment(strategy=args.strategy, p=0.05, edge_index=edge_index, edge_weight=edge_weight)
            if args.strategy == 'rwsample':
                output1 = self.augment(strategy=args.strategy, edge_index=edge_index, edge_weight=edge_weight)
            output2 = self.augment(strategy='dropedge', p=0.0, edge_index=edge_index, edge_weight=edge_weight)
            output3 = self.augment(strategy='shuffle', edge_index=edge_index, edge_weight=edge_weight)
            if args.margin != -1:
                loss = inner(output1, output2) - inner_margin(output2, output3, margin=args.margin)
            else:
                loss = inner(output1, output2) - inner(output2, output3)

        if 'recon' in args.loss: # data reconstruction
            model = self.model
            delta_feat = self.delta_feat
            feat = self.feat + delta_feat
            output2 = model.get_embed(feat, edge_index, edge_weight)
            loss += inner(output2[edge_index[0]], output2[edge_index[1]])

        if args.loss == "train":
            train_mask = self.data.train_mask
            loss = F.nll_loss(output[train_mask], labels[train_mask])

        if args.loss == "test":
            model, data = self.model, self.data
            output = model.forward(feat, edge_index, edge_weight)
            y = data.label.to(self.device)
            if self.args.dataset == 'elliptic':
                loss = model.sup_loss(y[data.mask], output[data.mask])
            elif args.dataset == 'ogb-arxiv':
                loss = model.sup_loss(y[data.test_mask], output[data.test_mask])
            else:
                loss = model.sup_loss(y, output)

        if "entropy" in args.loss:
            model, data = self.model, self.data
            if hasattr(self, 'delta_feat'):
                delta_feat = self.delta_feat
                feat = self.feat + delta_feat
            else:
                feat = self.feat
            batch_size = 1000
            output = model.forward(feat, edge_index, edge_weight)
            sampled = np.random.permutation(np.arange(len(output))[: batch_size])
            loss += softmax_entropy(output[sampled]).mean(0)
        if args.loss == 'dae':
            if hasattr(self, 'delta_feat'):
                delta_feat = self.delta_feat
                feat = self.feat + delta_feat
            else:
                feat = self.feat
            loss = model.get_loss_masked_features(feat, edge_index, edge_weight)

        return loss

    def pretrain_model(self, verbose=True):
        data_all = self.data_all
        args = self.args
        device = self.device
        if type(data_all[0]) is not list:
            feat, labels = data_all[0].graph['node_feat'], data_all[0].label
            edge_index = data_all[0].graph['edge_index']
        else:
            feat, labels = data_all[0][0].graph['node_feat'], data_all[0][0].label
            edge_index = data_all[0][0].graph['edge_index']
        # reset_args(args)
        if args.model == "GCN" or args.model == "GCNSLAPS":
            save_mem = False
            model = GCN(nfeat=feat.shape[1], nhid=args.hidden, dropout=args.dropout, nlayers=args.nlayers,
                        weight_decay=args.weight_decay, with_bn=True, lr=args.lr, save_mem=save_mem,
                        nclass=max(labels).item()+1, device=device, args=args).to(device)

        elif args.model == "GAT":
            model = GAT(nfeat=feat.shape[1], nhid=32, heads=4, lr=args.lr, nlayers=args.nlayers,
                  nclass=labels.max().item() + 1, with_bn=True, weight_decay=args.weight_decay,
                  dropout=0.0, device=device, args=args).to(device)
        elif args.model == "GCNII":
            model = GCNII(nfeat=feat.shape[1], nhid=32, lr=args.lr, nlayers=10,
                  nclass=labels.max().item() + 1, weight_decay=args.weight_decay,
                  dropout=0.0, device=device, args=args).to(device)
        elif args.model == "SAGE":
            if args.dataset == "fb100":
                model = SAGE2(feat.shape[1], 32, max(labels).item()+1, num_layers=args.nlayers,
                        dropout=0.0, lr=0.01, weight_decay=args.weight_decay,
                        device=device, args=args, with_bn=args.with_bn).to(device)
            else:
                model = SAGE(feat.shape[1], 32, max(labels).item()+1, num_layers=args.nlayers,
                        dropout=0.0, lr=0.01, weight_decay=args.weight_decay, device=device,
                        args=args, with_bn=args.with_bn).to(device)
        elif args.model == "GPR":
            model = GPRGNN(feat.shape[1], 32, max(labels).item()+1, dropout=0.0,
                    lr=0.01, weight_decay=args.weight_decay, device=device, args=args).to(device)
        elif args.model == "APPNP":
            if args.dataset == 'ogb-arxiv':
                model = APPNP(nfeat=feat.shape[1], nhid=args.hidden, dropout=0.2, nlayers=4,
                            with_bn=True, lr=0.01,
                            weight_decay=0, nclass=max(labels).item()+1, device=device, args=args).to(device)
            else:
                model = APPNP(nfeat=feat.shape[1], nhid=args.hidden, dropout=0.5, nlayers=10,
                        weight_decay=args.weight_decay,
                        nclass=max(labels).item()+1, device=device, args=args).to(device)
        else:
            raise NotImplementedError
        if verbose: print(model)

        import os.path as osp
        if args.ood:
            filename = f'saved/{args.dataset}_{args.model}_s{args.seed}.pt'
            if args.model == "GCNSLAPS":
                filename = f'saved/{args.dataset}_GCN_s{args.seed}.pt'
        else:
            filename = f'saved_no_ood/{args.dataset}_{args.model}_s{args.seed}.pt'
        if args.debug and osp.exists(filename):
            model.load_state_dict(torch.load(filename, map_location=self.device))
        else:
            train_iters = 500 if args.dataset == 'ogb-arxiv' else 200
            model.fit_inductive(data_all, train_iters=train_iters, patience=500, verbose=True)
            if args.debug:
                torch.save(model.state_dict(), filename)
        if args.model == "GCNSLAPS":
            assert args.debug > 0
            model.setup_dae(feat.shape[1], nhid=args.hidden, nclass=feat.shape[1])
            model.train_dae(feat, edge_index, None)

        if verbose: self.evaluate(model)
        return model

    def evaluate_single(self, model, output, labels, test_data, verbose=True):
        eval_func = model.eval_func
        if self.args.dataset in ['ogb-arxiv']:
            acc_test = eval_func(labels[test_data.test_mask], output[test_data.test_mask])
        elif self.args.dataset in ['cora', 'amazon-photo', 'twitch-e', 'fb100']:
            acc_test = eval_func(labels, output)
        elif self.args.dataset in ['elliptic']:
            acc_test = eval_func(labels[test_data.mask], output[test_data.mask])
        else:
            raise NotImplementedError
        if verbose:
            print('Test:', acc_test)
        return acc_test

    def evaluate(self, model):
        model.eval()
        accs = []
        y_te, out_te = [], []
        y_te_all, out_te_all = [], []
        for ii, test_data in enumerate(self.data_all[2]):
            x, edge_index = test_data.graph['node_feat'], test_data.graph['edge_index']
            x, edge_index = x.to(self.device), edge_index.to(self.device)
            output = model.predict(x, edge_index)

            labels = test_data.label.to(self.device) #.squeeze()
            eval_func = model.eval_func
            if self.args.dataset in ['ogb-arxiv']:
                acc_test = eval_func(labels[test_data.test_mask], output[test_data.test_mask])
                accs.append(acc_test)
                y_te_all.append(labels[test_data.test_mask])
                out_te_all.append(output[test_data.test_mask])
            elif self.args.dataset in ['cora', 'amazon-photo', 'twitch-e', 'fb100']:
                acc_test = eval_func(labels, output)
                accs.append(acc_test)
                y_te_all.append(labels)
                out_te_all.append(output)
            elif self.args.dataset in ['elliptic']:
                acc_test = eval_func(labels[test_data.mask], output[test_data.mask])
                y_te.append(labels[test_data.mask])
                out_te.append(output[test_data.mask])
                y_te_all.append(labels[test_data.mask])
                out_te_all.append(output[test_data.mask])
                if ii % 4 == 0 or ii == len(self.data_all[2]) - 1:
                    acc_te = eval_func(torch.cat(y_te, dim=0), torch.cat(out_te, dim=0))
                    accs += [float(f'{acc_te:.2f}')]
                    y_te, out_te = [], []
            else:
                raise NotImplementedError
        print('Test accs:', accs)
        acc_te = eval_func(torch.cat(y_te_all, dim=0), torch.cat(out_te_all, dim=0))
        print(f'flatten test: {acc_te}')

    def get_perf(self, output, labels, mask):
        loss = F.nll_loss(output[mask], labels[mask])
        acc = utils.accuracy(output[mask], labels[mask])
        print("loss= {:.4f}".format(loss.item()),
              "accuracy= {:.4f}".format(acc.item()))
        return loss.item(), acc.item()

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from **logits**."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.jit.script
def entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from **log_softmax**."""
    return -(x * torch.log(x+1e-15)).sum(1)

def compare_models(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True

def sim(t1, t2):
    # cosine similarity
    t1 = t1 / (t1.norm(dim=1).view(-1,1) + 1e-15)
    t2 = t2 / (t2.norm(dim=1).view(-1,1) + 1e-15)
    return (t1 * t2).sum(1)

def inner(t1, t2):
    t1 = t1 / (t1.norm(dim=1).view(-1,1) + 1e-15)
    t2 = t2 / (t2.norm(dim=1).view(-1,1) + 1e-15)
    return (1-(t1 * t2).sum(1)).mean()

def inner_margin(t1, t2, margin):
    t1 = t1 / (t1.norm(dim=1).view(-1,1) + 1e-15)
    t2 = t2 / (t2.norm(dim=1).view(-1,1) + 1e-15)
    return F.relu(1-(t1 * t2).sum(1)-margin).mean()

def diff(t1, t2):
    t1 = t1 / (t1.norm(dim=1).view(-1,1) + 1e-15)
    t2 = t2 / (t2.norm(dim=1).view(-1,1) + 1e-15)
    return 0.5*((t1-t2)**2).sum(1).mean()

