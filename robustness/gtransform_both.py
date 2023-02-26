"""learn edge indices only for existing edges!!!"""
import numpy as np
import torch.nn.functional as F
import torch
import deeprobust.graph.utils as utils
from torch.nn.parameter import Parameter
from tqdm import tqdm
from gtransform_feat import add_feature_noise, add_feature_noise_test, mask2indices, l21norm
from gtransform_adj_small import EdgeAgent
import torch_sparse
from torch_sparse import coalesce
import math
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix, dropout_adj


class GraphAgent(EdgeAgent):

    def __init__(self, data, args, model=None):
        self.device = 'cuda'
        self.args = args

        self.data = data
        if model is None:
            self.model = self.pretrain_model()
        else:
            self.model =  model

        if args.noise_feature > 0:
            self.clean_feat = data.x.clone()
            self.feat_noise, self.noisy_nodes = add_feature_noise_test(data,
                    args.noise_feature, args.seed)
        if args.noise_structure > 0:
            add_structure_noise(data, args.noise_structure)

        output = self.model.predict()
        labels = data.y.to(self.device)
        if args.noise_feature > 0:
            print("After noise, test set results:", self.get_perf(output, labels, data.test_mask, verbose=0)[1])
            print('Abnomral test nodes:', self.get_perf(output, labels, self.noisy_nodes, verbose=0)[1])
            # print('Normal test nodes:', self.get_perf(output, labels, data.test_mask & (~self.noisy_nodes), verbose=0)[1])

        for param in self.model.parameters():
            param.requires_grad = False

        nnodes = data.x.shape[0]
        d = data.x.shape[1]

        self.n, self.d = nnodes, nnodes

        self.make_undirected = True
        self.max_final_samples = 20
        self.search_space_size = 10_000_000
        self.eps = 1e-7

        self.modified_edge_index: torch.Tensor = None
        self.perturbed_edge_weight: torch.Tensor = None
        if self.make_undirected:
            self.n_possible_edges = self.n * (self.n - 1) // 2
        else:
            self.n_possible_edges = self.n ** 2

        lr_factor = args.lr_adj
        self.lr_factor = lr_factor * max(math.log2(self.n_possible_edges / self.search_space_size), 1.)
        self.epochs_resampling = self.args.epochs
        self.with_early_stopping = True
        self.do_synchronize = True

        self.delta_feat = Parameter(torch.FloatTensor(nnodes, d).to(self.device))
        self.delta_feat.data.fill_(0)
        self.optimizer_feat = torch.optim.Adam([self.delta_feat], lr=args.lr_feat)

    def learn_graph(self, edge_index_=None, edge_weight=None):
        data = self.data
        args = self.args
        model = self.model
        model.eval() # should set to eval
        delta_feat = self.delta_feat

        edge_index, feat, labels = data.edge_index, data.x, data.y
        if edge_index_ is not None:
            edge_index = edge_index_
        self.edge_index = edge_index

        if args.loss == 'attack':
            with torch.no_grad():
                output = model.forward(feat, self.edge_index)
                pred = output.argmax(1)
            gt_labels = labels
            labels = labels.clone() # to avoid shallow copy
            labels[~data.train_mask] = pred[~data.train_mask]

        if edge_index is not None:
            self.edge_index = edge_index

        self.edge_weight = torch.ones(self.edge_index.shape[1]).to(self.device)

        n_perturbations = int(args.ratio * self.edge_index.shape[1] //2)
        print('n_perturbations:', n_perturbations)
        self.sample_random_block(n_perturbations)

        self.perturbed_edge_weight.requires_grad = True
        self.optimizer_adj = torch.optim.Adam([self.perturbed_edge_weight], lr=args.lr_adj)
        best_loss_val = 100
        best_acc_val = 0
        # alternating optimizing structure and feature
        for it in tqdm(range(args.epochs//(args.loop_adj+args.loop_feat))):
            for j in range(args.loop_feat):
                self.optimizer_feat.zero_grad()
                new_feat = feat+delta_feat
                loss = self.test_time_loss(model, new_feat, edge_index, edge_weight)

                loss.backward()
                if it % 100 == 0:
                    print(f'Epoch {it}: {loss}')

                self.optimizer_feat.step()

                if args.debug == 2:
                    with torch.no_grad():
                        new_feat = feat+delta_feat
                        output = model.forward(new_feat, edge_index, edge_weight) #
                    print("After noise, test set results:", self.get_perf(output, labels, data.test_mask, verbose=0)[1])
                    print('Validation:', self.get_perf(output, labels, data.val_mask, verbose=0)[1])
                    print('Abnomral test nodes:', self.get_perf(output, labels, self.noisy_nodes, verbose=0)[1])
                    print('Normal test nodes:', self.get_perf(output, labels, data.test_mask & (~self.noisy_nodes), verbose=0)[1])

                with torch.no_grad():
                    new_feat = (feat+delta_feat).detach()
                    output = model.forward(new_feat, edge_index, edge_weight)
                val_mask = self.data.val_mask
                acc_val = utils.accuracy(output[val_mask], labels[val_mask])
                if best_acc_val < acc_val:
                    best_acc_val = acc_val
                    best_feat = new_feat
                    best_delta_feat = delta_feat.detach().clone()
                    best_edge_index = self.modified_edge_index.clone().cpu()
                    best_edge_weight_diff = self.perturbed_edge_weight.detach().clone().cpu()

            new_feat = (feat + delta_feat).detach()
            for i in range(args.loop_adj):
                self.perturbed_edge_weight.requires_grad = True
                edge_index, edge_weight  = self.get_modified_adj()
                if torch.cuda.is_available() and self.do_synchronize:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                loss = self.test_time_loss(model, new_feat, edge_index, edge_weight)
                gradient = grad_with_checkpoint(loss, self.perturbed_edge_weight)[0]

                if torch.cuda.is_available() and self.do_synchronize:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                if it % 10 == 0:
                    print(f'Epoch {it}: {loss}')

                with torch.no_grad():
                    self.update_edge_weights(n_perturbations, it, gradient)
                    self.perturbed_edge_weight = self.project(
                        n_perturbations, self.perturbed_edge_weight, self.eps)

                    del edge_index, edge_weight

                    if not args.existing_space:
                        if it < self.epochs_resampling - 1:
                            self.resample_random_block(n_perturbations)

                    with torch.no_grad():
                        edge_index, edge_weight = self.get_modified_adj()
                        output = model.forward(new_feat, edge_index, edge_weight)
                    val_mask = self.data.val_mask
                    acc_val = utils.accuracy(output[val_mask], labels[val_mask])
                    if best_acc_val < acc_val:
                        best_acc_val = acc_val
                        best_feat = new_feat
                        best_delta_feat = delta_feat.detach().clone()
                        best_edge_index = self.modified_edge_index.clone().cpu()
                        best_edge_weight_diff = self.perturbed_edge_weight.detach().clone().cpu()

                self.perturbed_edge_weight.requires_grad = True
                self.optimizer_adj = torch.optim.Adam([self.perturbed_edge_weight], lr=args.lr_adj)
            if args.loop_adj != 0:
                edge_index, edge_weight = self.get_modified_adj()
                edge_weight = edge_weight.detach()


        self.modified_edge_index = best_edge_index.to(self.device)
        self.perturbed_edge_weight = best_edge_weight_diff.to(self.device)
        self.delta_feat = best_delta_feat
        new_feat = best_feat

        # edge_index, edge_weight = self.sample_final_edges(n_perturbations)[0], None

        if args.loop_adj !=0 :
            edge_index, edge_weight = self.sample_final_edges(n_perturbations)

        output = model.forward(new_feat, edge_index, edge_weight)
        loss = self.test_time_loss(model, delta_feat+feat, edge_index, edge_weight)
        print('final loss:', loss.item())

        if args.noise_feature > 0:
            print("After noise, test set results:", self.get_perf(output, labels, data.test_mask, verbose=0)[1])
            print('Validation:', self.get_perf(output, labels, data.val_mask, verbose=0)[1])
            print('Abnomral test nodes:', self.get_perf(output, labels, self.noisy_nodes, verbose=0)[1])
            print('Normal test nodes:', self.get_perf(output, labels, data.test_mask & (~self.noisy_nodes), verbose=0)[1])
        else:
            print('Test:', self.get_perf(output, labels, data.test_mask, verbose=0)[1])
            print('Validation:', self.get_perf(output, labels, data.val_mask, verbose=0)[1])
        return new_feat, edge_index, edge_weight

    def augment(self, strategy='dropedge', p=0.5, edge_index=None, edge_weight=None):
        model = self.model
        if hasattr(self, 'delta_feat'):
            delta_feat = self.delta_feat
            feat = self.data.x + delta_feat
        else:
            feat = self.data.x
        if strategy == 'shuffle':
            idx = np.random.permutation(feat.shape[0])
            shuf_fts = feat[idx, :]
            output = model.get_embed(shuf_fts, edge_index, edge_weight)
        if strategy == "dropedge":
            edge_index, edge_weight = dropout_adj(edge_index, edge_weight, p=p)
            output = model.get_embed(feat, edge_index, edge_weight)
        if strategy == "dropnode":
            feat = self.data.x + self.delta_feat
            mask = torch.cuda.FloatTensor(len(feat)).uniform_() > p
            feat = feat * mask.view(-1, 1)
            output = model.get_embed(feat, edge_index, edge_weight)
        if strategy == "dropmix":
            feat = self.data.x + self.delta_feat
            mask = torch.cuda.FloatTensor(len(feat)).uniform_() > p
            feat = feat * mask.view(-1, 1)
            edge_index, edge_weight = dropout_adj(edge_index, edge_weight, p=p)
            output = model.get_embed(feat, edge_index, edge_weight)
        if strategy == "dropfeat":
            feat = F.dropout(self.data.x, p=p) + self.delta_feat
            output = model.get_embed(feat, edge_index, edge_weight)
        if strategy == "featnoise":
            mean, std = 0, p
            noise = torch.randn(feat.size()) * std + mean
            feat = feat + noise.to(feat.device)
            output = model.get_embed(feat, edge_index)
        return output

def grad_with_checkpoint(outputs, inputs):
    inputs = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
    for input in inputs:
        if not input.is_leaf:
            input.retain_grad()
    torch.autograd.backward(outputs)

    grad_outputs = []
    for input in inputs:
        grad_outputs.append(input.grad.clone())
        input.grad.zero_()
    return grad_outputs
