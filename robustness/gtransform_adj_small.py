"""learn edge indices only for existing edges!!!"""
import numpy as np
import torch.nn.functional as F
import torch
import deeprobust.graph.utils as utils
from torch.nn.parameter import Parameter
from tqdm import tqdm
from gtransform_feat import FeatAgent, add_feature_noise, add_structure_noise
import torch_sparse
from torch_sparse import coalesce
import math
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix, dropout_adj, is_undirected, to_undirected


class EdgeAgent(FeatAgent):

    def __init__(self, data, args, model=None):
        self.device = 'cuda'
        self.args = args

        self.data = data
        if model is None:
            self.model = self.pretrain_model()
        else:
            self.model =  model

        if args.noise_feature > 0:
            add_feature_noise(data, args.noise_feature)
        if args.noise_structure > 0:
            add_structure_noise(data, args.noise_structure)

        output = self.model.predict()
        labels = data.y.to(self.device)
        print("After noise, test set results:")
        self.get_perf(output, labels, data.test_mask)

        for param in self.model.parameters():
            param.requires_grad = False

        nnodes = data.x.shape[0]
        d = data.x.shape[1]

        self.n, self.d = nnodes, nnodes

        self.make_undirected = True
        self.max_final_samples = 10
        self.search_space_size = 10_000_000
        self.eps = 1e-7

        self.modified_edge_index: torch.Tensor = None
        self.perturbed_edge_weight: torch.Tensor = None
        if self.make_undirected:
            self.n_possible_edges = self.n * (self.n - 1) // 2
        else:
            self.n_possible_edges = self.n ** 2  # We filter self-loops later

        lr_factor = args.lr_adj
        self.lr_factor = lr_factor * max(math.log2(self.n_possible_edges / self.search_space_size), 1.)
        self.epochs_resampling = self.args.epochs
        self.with_early_stopping = True
        self.do_synchronize = True

    def learn_graph(self, edge_index=None, edge_weight=None):
        data = self.data
        args = self.args
        model = self.model
        model.eval() # should set to eval

        self.edge_index, feat, labels = data.edge_index, data.x, data.y

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
        for it in tqdm(range(args.epochs)):
            self.perturbed_edge_weight.requires_grad = True
            edge_index, edge_weight  = self.get_modified_adj()
            if torch.cuda.is_available() and self.do_synchronize:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            loss = self.test_time_loss(model, feat, edge_index, edge_weight)
            gradient = grad_with_checkpoint(loss, self.perturbed_edge_weight)[0]

            if torch.cuda.is_available() and self.do_synchronize:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            if it % 10 == 0:
                print(f'Epoch {it}: {loss}')

            with torch.no_grad():
                self.update_edge_weights(n_perturbations, it, gradient)
                if True:
                    self.perturbed_edge_weight = self.project(
                    n_perturbations, self.perturbed_edge_weight, self.eps, bisec=False)
                else:
                    self.perturbed_edge_weight = self.project(
                    n_perturbations, self.perturbed_edge_weight, self.eps)

                del edge_index, edge_weight
                edge_index, edge_weight = self.get_modified_adj()
                output = model.predict(feat, edge_index, edge_weight)
                loss_val = F.nll_loss(output[data.val_mask], labels[data.val_mask])
                if best_loss_val > loss_val:
                    best_loss_val = loss_val
                    best_edge_index = self.modified_edge_index.clone().cpu()
                    best_edge_weight_diff = self.perturbed_edge_weight.detach().clone().cpu()
                    print('best_loss_val: %s' % best_loss_val)

                acc_train = utils.accuracy(output[data.train_mask], labels[data.train_mask])
                acc_test = utils.accuracy(output[data.test_mask], labels[data.test_mask])
                acc_val = utils.accuracy(output[data.val_mask], labels[data.val_mask])
                print(acc_train.item(), acc_val.item(), acc_test.item())

            self.perturbed_edge_weight.requires_grad = True
            self.optimizer_adj = torch.optim.Adam([self.perturbed_edge_weight], lr=args.lr_adj)

        if self.args.loss != 'attack':
            self.modified_edge_index = best_edge_index.to(self.device)
            self.perturbed_edge_weight = best_edge_weight_diff.to(self.device)

        if args.loss == 'attack':
            edge_index, edge_weight = self.sample_final_edges(n_perturbations)
        else:
            edge_index, edge_weight = self.sample_final_edges(n_perturbations)
        output = model.predict(feat, edge_index, edge_weight) # TODO: fix for noisy graph setting
        loss = self.test_time_loss(model, feat, edge_index, edge_weight)
        print('final loss:', loss.item())

        if args.loss == 'attack':
            print('Test:')
            self.get_perf(output, gt_labels, data.test_mask)
            print('Validatoin:')
            self.get_perf(output, gt_labels, data.val_mask)
        else:
            print('Test:')
            self.get_perf(output, labels, data.test_mask)
            print('Validatoin:')
            self.get_perf(output, labels, data.val_mask)
        return edge_index, edge_weight

    def augment(self, strategy='dropedge', p=0.5, edge_index=None, edge_weight=None):
        model = self.model
        feat = self.data.x
        if strategy == 'shuffle':
            idx = np.random.permutation(feat.shape[0])
            shuf_fts = feat[idx, :]
            output = model.get_embed(shuf_fts, edge_index, edge_weight)
        if strategy == "dropedge":
            edge_index, edge_weight = dropout_adj(edge_index, edge_weight, p=p)
            output = model.get_embed(feat, edge_index, edge_weight)
        if strategy == "dropfeat":
            feat = F.dropout(feat, p=p)
            output = model.get_embed(feat, edge_index, edge_weight)
        if strategy == "featnoise":
            mean, std = 0, p
            noise = torch.randn(feat.size()) * std + mean
            feat = feat + noise.to(feat.device)
            output = model.get_embed(feat, edge_index)
        return output

    def sample_random_block(self, n_perturbations):
        if self.args.loss != 'attack':
            edge_index = self.edge_index.clone()
            edge_index = edge_index[:, edge_index[0] < edge_index[1]]
            row, col = edge_index[0], edge_index[1]
            edge_index_id = (2*self.n - row-1)*row//2 + col - row -1 # // is important to get the correct result
            edge_index_id = edge_index_id.long()
            self.current_search_space = edge_index_id
            self.modified_edge_index = linear_to_triu_idx(self.n, self.current_search_space)
            self.perturbed_edge_weight = torch.full_like(
                self.current_search_space, self.eps, dtype=torch.float32, requires_grad=True
            )
            return

        else:
            for _ in range(self.max_final_samples):
                self.current_search_space = torch.randint(
                    self.n_possible_edges, (self.search_space_size,), device=self.device)
                self.current_search_space = torch.unique(self.current_search_space, sorted=True)
                if self.make_undirected:
                    self.modified_edge_index = linear_to_triu_idx(self.n, self.current_search_space)
                else:
                    self.modified_edge_index = linear_to_full_idx(self.n, self.current_search_space)
                    is_not_self_loop = self.modified_edge_index[0] != self.modified_edge_index[1]
                    self.current_search_space = self.current_search_space[is_not_self_loop]
                    self.modified_edge_index = self.modified_edge_index[:, is_not_self_loop]

                self.perturbed_edge_weight = torch.full_like(
                    self.current_search_space, self.eps, dtype=torch.float32, requires_grad=True
                )
                if self.current_search_space.size(0) >= n_perturbations:
                    return
            raise RuntimeError('Sampling random block was not successfull. Please decrease `n_perturbations`.')

    @torch.no_grad()
    def sample_final_edges(self, n_perturbations):
        descent = False if self.args.loss == 'attack' else True
        best_loss = float('Inf') if descent else -float('Inf')
        perturbed_edge_weight = self.perturbed_edge_weight.detach()
        # TODO: potentially convert to assert
        perturbed_edge_weight[perturbed_edge_weight <= self.eps] = 0

        _, feat, labels = self.edge_index, self.data.x, self.data.y
        if hasattr(self, 'delta_feat'):
            delta_feat = self.delta_feat
            feat = (feat + delta_feat).detach()

        for i in range(self.max_final_samples):
            sampled_edges = torch.bernoulli(perturbed_edge_weight).float()
            if self.args.loss == 'attack':
                if sampled_edges.sum() > n_perturbations:
                    n_samples = sampled_edges.sum()
                    print(f'{i}-th sampling: too many samples {n_samples}')
                    continue
            self.perturbed_edge_weight = sampled_edges

            edge_index, edge_weight = self.get_modified_adj()
            with torch.no_grad():
                output = self.model.forward(feat, edge_index, edge_weight)
                loss = -utils.accuracy(output[self.data.val_mask], labels[self.data.val_mask]).item()
            if descent:
                if best_loss > loss:
                    best_loss = loss
                    print('best_loss:', best_loss)
                    best_edges = self.perturbed_edge_weight.clone().cpu()
            else:
                if best_loss < loss:
                    best_loss = loss
                    print('best_loss:', best_loss)
                    best_edges = self.perturbed_edge_weight.clone().cpu()

        # Recover best sample
        self.perturbed_edge_weight.data.copy_(best_edges.to(self.device))

        edge_index, edge_weight = self.get_modified_adj()
        edge_mask = edge_weight == 1

        allowed_perturbations = 2 * n_perturbations if self.make_undirected else n_perturbations
        edges_after_attack = edge_mask.sum()
        clean_edges = self.edge_index.shape[1]
        if self.args.loss == 'attack':
            assert (edges_after_attack >= clean_edges - allowed_perturbations
                    and edges_after_attack <= clean_edges + allowed_perturbations), \
                f'{edges_after_attack} out of range with {clean_edges} clean edges and {n_perturbations} pertutbations'
        return edge_index[:, edge_mask], edge_weight[edge_mask]

    def resample_random_block(self, n_perturbations: int):
        if self.args.loss != 'attack':
            self.modified_edge_index = self.edge_index.clone()
            self.perturbed_edge_weight = torch.full_like(
                self.modified_edge_index[0], self.eps, dtype=torch.float32, requires_grad=True
            )
            return

        else:
            self.keep_heuristic = 'WeightOnly'
            if self.keep_heuristic == 'WeightOnly':
                sorted_idx = torch.argsort(self.perturbed_edge_weight)
                idx_keep = (self.perturbed_edge_weight <= self.eps).sum().long()
                # Keep at most half of the block (i.e. resample low weights)
                if idx_keep < sorted_idx.size(0) // 2:
                    idx_keep = sorted_idx.size(0) // 2
            else:
                raise NotImplementedError('Only keep_heuristic=`WeightOnly` supported')

            sorted_idx = sorted_idx[idx_keep:]
            self.current_search_space = self.current_search_space[sorted_idx]
            self.modified_edge_index = self.modified_edge_index[:, sorted_idx]
            self.perturbed_edge_weight = self.perturbed_edge_weight[sorted_idx]

            # Sample until enough edges were drawn
            for i in range(self.max_final_samples):
                n_edges_resample = self.search_space_size - self.current_search_space.size(0)
                lin_index = torch.randint(self.n_possible_edges, (n_edges_resample,), device=self.device)

                self.current_search_space, unique_idx = torch.unique(
                    torch.cat((self.current_search_space, lin_index)),
                    sorted=True,
                    return_inverse=True
                )

                if self.make_undirected:
                    self.modified_edge_index = linear_to_triu_idx(self.n, self.current_search_space)
                else:
                    self.modified_edge_index = linear_to_full_idx(self.n, self.current_search_space)

                # Merge existing weights with new edge weights
                perturbed_edge_weight_old = self.perturbed_edge_weight.clone()
                self.perturbed_edge_weight = torch.full_like(self.current_search_space, self.eps, dtype=torch.float32)
                self.perturbed_edge_weight[
                    unique_idx[:perturbed_edge_weight_old.size(0)]
                ] = perturbed_edge_weight_old

                if not self.make_undirected:
                    is_not_self_loop = self.modified_edge_index[0] != self.modified_edge_index[1]
                    self.current_search_space = self.current_search_space[is_not_self_loop]
                    self.modified_edge_index = self.modified_edge_index[:, is_not_self_loop]
                    self.perturbed_edge_weight = self.perturbed_edge_weight[is_not_self_loop]

                if self.current_search_space.size(0) > n_perturbations:
                    return
            raise RuntimeError('Sampling random block was not successfull. Please decrease `n_perturbations`.')


    def project(self, n_perturbations, values, eps, inplace=False, bisec=True):
        if not inplace:
            values = values.clone()

        if not bisec:
            values.data.copy_(torch.clamp(
                values, min=eps, max=1 - eps
            ))
            return values

        if torch.clamp(values, 0, 1).sum() > n_perturbations:
            left = (values - 1).min()
            right = values.max()
            miu = bisection(values, left, right, n_perturbations)
            values.data.copy_(torch.clamp(
                values - miu, min=eps, max=1 - eps
            ))
        else:
            values.data.copy_(torch.clamp(
                values, min=eps, max=1 - eps
            ))
        return values

    def get_modified_adj(self):
        if self.make_undirected:
            modified_edge_index, modified_edge_weight = to_symmetric(
                self.modified_edge_index, self.perturbed_edge_weight, self.n
            )
        else:
            modified_edge_index, modified_edge_weight = self.modified_edge_index, self.perturbed_edge_weight
        edge_index = torch.cat((self.edge_index.to(self.device), modified_edge_index), dim=-1)
        edge_weight = torch.cat((self.edge_weight.to(self.device), modified_edge_weight))

        edge_index, edge_weight = torch_sparse.coalesce(edge_index, edge_weight, m=self.n, n=self.n, op='sum')

        # Allow removal of edges
        edge_weight[edge_weight > 1] = 2 - edge_weight[edge_weight > 1]
        return edge_index, edge_weight

    def update_edge_weights(self, n_perturbations, epoch, gradient):
        self.optimizer_adj.zero_grad()
        if self.args.loss == "attack":
            self.perturbed_edge_weight.grad = -gradient
        else:
            self.perturbed_edge_weight.grad = gradient
        self.optimizer_adj.step()
        self.perturbed_edge_weight.data[self.perturbed_edge_weight < self.eps] = self.eps

    def _update_edge_weights(self, n_perturbations, epoch, gradient):
        lr_factor = n_perturbations / self.n / 2 * self.lr_factor
        lr = lr_factor / np.sqrt(max(0, epoch - self.epochs_resampling) + 1)
        if self.args.loss == 'attack':
            self.perturbed_edge_weight.data.add_(lr * gradient)
        else:
            self.perturbed_edge_weight.data.add_(-lr * gradient)
        self.perturbed_edge_weight.data[self.perturbed_edge_weight < self.eps] = self.eps
        return None


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from **logits**."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.jit.script
def entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from **log_softmax**."""
    return -(torch.exp(x) * x).sum(1)


def to_symmetric(edge_index, edge_weight, n, op='mean'):
    symmetric_edge_index = torch.cat(
        (edge_index, edge_index.flip(0)), dim=-1
    )

    symmetric_edge_weight = edge_weight.repeat(2)

    symmetric_edge_index, symmetric_edge_weight = coalesce(
        symmetric_edge_index,
        symmetric_edge_weight,
        m=n,
        n=n,
        op=op
    )
    return symmetric_edge_index, symmetric_edge_weight

def linear_to_full_idx(n: int, lin_idx: torch.Tensor) -> torch.Tensor:
    row_idx = lin_idx // n
    col_idx = lin_idx % n
    return torch.stack((row_idx, col_idx))

def linear_to_triu_idx(n: int, lin_idx: torch.Tensor) -> torch.Tensor:
    row_idx = (
        n
        - 2
        - torch.floor(torch.sqrt(-8 * lin_idx.double() + 4 * n * (n - 1) - 7) / 2.0 - 0.5)
    ).long()
    col_idx = (
        lin_idx
        + row_idx
        + 1 - n * (n - 1) // 2
        + (n - row_idx) * ((n - row_idx) - 1) // 2
    )
    return torch.stack((row_idx, col_idx))


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

def bisection(edge_weights, a, b, n_perturbations, epsilon=1e-5, iter_max=1e5):
    def func(x):
        return torch.clamp(edge_weights - x, 0, 1).sum() - n_perturbations

    miu = a
    for i in range(int(iter_max)):
        miu = (a + b) / 2
        # Check if middle point is root
        if (func(miu) == 0.0):
            break
        # Decide the side to repeat the steps
        if (func(miu) * func(a) < 0):
            b = miu
        else:
            a = miu
        if ((b - a) <= epsilon):
            break
    return miu

def homophily(adj, labels):
    edge_index = adj.nonzero()
    homo = (labels[edge_index[0]] == labels[edge_index[1]])
    return np.mean(homo.numpy())

