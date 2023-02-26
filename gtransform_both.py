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
import torch.optim as optim
from copy import deepcopy
from utils import reset_args
from gtransform_adj import EdgeAgent
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix, dropout_adj, is_undirected, to_undirected
from gtransform_adj import *

class GraphAgent(EdgeAgent):

    def __init__(self, data_all, args):
        self.device = 'cuda'
        self.args = args
        self.data_all = data_all
        self.model = self.pretrain_model()

    def learn_graph(self, data):
        print('====learning on this graph===')
        args = self.args
        self.setup_params(data)
        args = self.args
        model = self.model
        model.eval() # should set to eval

        self.max_final_samples = 5

        from utils import get_gpu_memory_map
        mem_st = get_gpu_memory_map()
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
        self.edge_weight = torch.ones(self.edge_index.shape[1]).to(self.device)

        n_perturbations = int(args.ratio * self.edge_index.shape[1] //2)
        print('n_perturbations:', n_perturbations)
        self.sample_random_block(n_perturbations)

        self.perturbed_edge_weight.requires_grad = True
        self.optimizer_adj = torch.optim.Adam([self.perturbed_edge_weight], lr=args.lr_adj)
        edge_index, edge_weight = edge_index, None

        for it in tqdm(range(args.epochs//(args.loop_feat+args.loop_adj))):
            for loop_feat in range(args.loop_feat):
                self.optimizer_feat.zero_grad()
                loss = self.test_time_loss(model, feat+delta_feat, edge_index, edge_weight)
                loss.backward()

                if loop_feat == 0:
                    print(f'Epoch {it}, Loop Feat {loop_feat}: {loss.item()}')

                self.optimizer_feat.step()
                if args.debug==2 or args.debug==3:
                    output = model.predict(feat+delta_feat, edge_index, edge_weight)
                    print('Debug Test:', self.evaluate_single(model, output, labels, data, verbose=0))

            new_feat = (feat+delta_feat).detach()
            for loop_adj in range(args.loop_adj):
                self.perturbed_edge_weight.requires_grad = True
                edge_index, edge_weight  = self.get_modified_adj()
                if torch.cuda.is_available() and self.do_synchronize:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                loss = self.test_time_loss(model, new_feat, edge_index, edge_weight)

                gradient = grad_with_checkpoint(loss, self.perturbed_edge_weight)[0]
                if not args.existing_space:
                    if torch.cuda.is_available() and self.do_synchronize:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                if loop_adj == 0:
                    print(f'Epoch {it}, Loop Adj {loop_adj}: {loss.item()}')

                with torch.no_grad():
                    self.update_edge_weights(n_perturbations, it, gradient)
                    self.perturbed_edge_weight = self.project(
                        n_perturbations, self.perturbed_edge_weight, self.eps)
                    del edge_index, edge_weight #, logits
                    if not args.existing_space:
                        if it < self.epochs_resampling - 1:
                            self.resample_random_block(n_perturbations)
                if it < self.epochs_resampling - 1:
                    self.perturbed_edge_weight.requires_grad = True
                    self.optimizer_adj = torch.optim.Adam([self.perturbed_edge_weight], lr=args.lr_adj)

            # edge_index, edge_weight = self.sample_final_edges(n_perturbations, data)
            if args.loop_adj != 0:
                edge_index, edge_weight  = self.get_modified_adj()
                edge_weight = edge_weight.detach()

        print(f'Epoch {it+1}: {loss}')
        gpu_mem = get_gpu_memory_map()
        print(f'Mem used: {int(gpu_mem[args.gpu_id])-int(mem_st[args.gpu_id])}MB')

        if args.loop_adj != 0:
            edge_index, edge_weight = self.sample_final_edges(n_perturbations, data)

        with torch.no_grad():
            loss = self.test_time_loss(model, feat+delta_feat, edge_index, edge_weight)
        print('final loss:', loss.item())
        output = model.predict(feat+delta_feat, edge_index, edge_weight)
        print('Test:')

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
        if strategy == "rwsample":
            import augmentor as A
            if self.args.dataset in ['twitch-e', 'elliptic']:
                walk_length = 1
            else:
                walk_length = 10
            aug = A.RWSampling(num_seeds=1000, walk_length=walk_length)
            x = self.feat + self.delta_feat
            x2, edge_index2, edge_weight2 = aug(x, edge_index, edge_weight)
            output = model.get_embed(x2, edge_index2, edge_weight2)

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

def inner(t1, t2):
    t1 = t1 / (t1.norm(dim=1).view(-1,1) + 1e-15)
    t2 = t2 / (t2.norm(dim=1).view(-1,1) + 1e-15)
    return (1-(t1 * t2).sum(1)).mean()

def diff(t1, t2):
    t1 = t1 / (t1.norm(dim=1).view(-1,1) + 1e-15)
    t2 = t2 / (t2.norm(dim=1).view(-1,1) + 1e-15)
    return 0.5*((t1-t2)**2).sum(1).mean()


