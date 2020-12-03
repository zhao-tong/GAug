import os
import sys
import time
import pickle
import warnings
import numpy as np
import networkx as nx
import scipy.sparse as sp
import dgl
from dgl import DGLGraph
import torch
from collections import defaultdict
from sklearn.preprocessing import normalize

from utils import sparse_to_tuple

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CUR_DIR)

class DataLoader():
    def __init__(self, args):
        self.args = args
        self.dataset = args.dataset

        if self.dataset == 'zkc':
            self.load_data_zkc()
        elif self.dataset in ('cora', 'citeseer', 'pubmed'):
            self.load_data(self.dataset)
        else:
            self.load_data_binary(self.dataset)
        self.mask_test_edges(args.val_frac, args.test_frac, args.no_mask)
        self.normalize_adj()
        self.to_pyt_sp()

    def load_data_zkc(self):
        edges = [[1, 0], [2, 0], [2, 1], [3, 0], [3, 1], [3, 2], [4, 0],
                [5, 0], [6, 0], [6, 4], [6, 5], [7, 0], [7, 1], [7, 2],
                [7, 3], [8, 0], [8, 2], [9, 2], [10, 0], [10, 4], [10, 5],
                [11, 0], [12, 0], [12, 3], [13, 0], [13, 1], [13, 2], [13, 3],
                [16, 5], [16, 6], [17, 0], [17, 1], [19, 0], [19, 1], [21, 0],
                [21, 1], [25, 23], [25, 24], [27, 2], [27, 23], [27, 24], [28, 2],
                [29, 23], [29, 26], [30, 1], [30, 8], [31, 0], [31, 24], [31, 25],
                [31, 28], [32, 2], [32, 8], [32, 14], [32, 15], [32, 18], [32, 20],
                [32, 22], [32, 23], [32, 29], [32, 30], [32, 31], [33, 8], [33, 9],
                [33, 13], [33, 14], [33, 15], [33, 18], [33, 19], [33, 20], [33, 22],
                [33, 23], [33, 26], [33, 27], [33, 28], [33, 29], [33, 30], [33, 31], [33, 32]]
        edges = np.asarray(edges)
        row = edges.T[0]
        col = edges.T[1]
        adj_mat = sp.csr_matrix((np.ones_like(row), (row, col)), shape=(34, 34))
        adj_mat = adj_mat + adj_mat.T
        features = torch.eye(34)
        features = sp.coo_matrix(features.numpy())
        self.adj_orig = adj_mat
        self.features_orig = features

    def load_data(self, dataset):
        # load the data: x, tx, allx, graph
        names = ['x', 'tx', 'allx', 'graph']
        objects = []
        for n in names:
            with open(f'{BASE_DIR}/data/citation_networks/ind.{dataset}.{n}', 'rb') as f:
                objects.append(pickle.load(f, encoding='latin1'))
        x, tx, allx, graph = tuple(objects)
        test_idx_reorder = []
        for line in open(f'{BASE_DIR}/data/citation_networks/ind.{dataset}.test.index'):
            test_idx_reorder.append(int(line.strip()))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        if adj.diagonal().sum() > 0:
            adj = sp.coo_matrix(adj)
            adj.setdiag(0)
            adj.eliminate_zeros()
            adj = sp.csr_matrix(adj)
        self.adj_orig = adj
        self.features_orig = normalize(features, norm='l1', axis=1)

    def load_data_binary(self, dataset):
        adj = pickle.load(open(f'{BASE_DIR}/graphs/{dataset}_adj.pkl', 'rb'))
        if adj.diagonal().sum() > 0:
            adj = sp.coo_matrix(adj)
            adj.setdiag(0)
            adj.eliminate_zeros()
            adj = sp.csr_matrix(adj)
        features = pickle.load(open(f'{BASE_DIR}/graphs/{dataset}_features.pkl', 'rb'))
        if isinstance(features, torch.Tensor):
            features = features.numpy()
        features = sp.csr_matrix(features)
        self.adj_orig = adj
        if dataset == 'ppi':
            features = features.toarray()
            m = features.mean(axis=0)
            s = features.std(axis=0, ddof=0, keepdims=True) + 1e-12
            features -= m
            features /= s
            self.features_orig = sp.csr_matrix(features)
        else:
            self.features_orig = normalize(features, norm='l1', axis=1)

    def mask_test_edges(self, val_frac, test_frac, no_mask):
        adj = self.adj_orig
        assert adj.diagonal().sum() == 0

        adj_triu = sp.triu(adj)
        edges = sparse_to_tuple(adj_triu)[0]
        edges_all = sparse_to_tuple(adj)[0]
        num_test = int(np.floor(edges.shape[0] * test_frac))
        num_val = int(np.floor(edges.shape[0] * val_frac))

        all_edge_idx = list(range(edges.shape[0]))
        np.random.shuffle(all_edge_idx)
        val_edge_idx = all_edge_idx[:num_val]
        test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
        test_edges = edges[test_edge_idx]
        val_edges = edges[val_edge_idx]
        if no_mask:
            train_edges = edges
        else:
            train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

        def ismember(a, b, tol=5):
            rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
            return np.any(rows_close)

        test_edges_false = []
        while len(test_edges_false) < len(test_edges):
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], edges_all):
                continue
            if test_edges_false:
                if ismember([idx_j, idx_i], np.array(test_edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(test_edges_false)):
                    continue
            test_edges_false.append([idx_i, idx_j])

        val_edges_false = []
        while len(val_edges_false) < len(val_edges):
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], train_edges):
                continue
            if ismember([idx_j, idx_i], train_edges):
                continue
            if ismember([idx_i, idx_j], val_edges):
                continue
            if ismember([idx_j, idx_i], val_edges):
                continue
            if val_edges_false:
                if ismember([idx_j, idx_i], np.array(val_edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(val_edges_false)):
                    continue
            val_edges_false.append([idx_i, idx_j])

        # assert ~ismember(test_edges_false, edges_all)
        # assert ~ismember(val_edges_false, edges_all)
        # assert ~ismember(val_edges, test_edges)
        # if not no_mask:
        #     assert ~ismember(val_edges, train_edges)
        #     assert ~ismember(test_edges, train_edges)

        # Re-build adj matrix
        adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
        self.adj_train = adj_train + adj_train.T
        self.adj_label = adj_train + sp.eye(adj_train.shape[0])
        # NOTE: these edge lists only contain single direction of edge!
        self.val_edges = val_edges
        self.val_edges_false = np.asarray(val_edges_false)
        self.test_edges = test_edges
        self.test_edges_false = np.asarray(test_edges_false)

    def normalize_adj(self):
        adj_ = sp.coo_matrix(self.adj_train)
        adj_.setdiag(1)
        rowsum = np.array(adj_.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        self.adj_norm = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()

    def to_pyt_sp(self):
        adj_norm_tuple = sparse_to_tuple(self.adj_norm)
        adj_label_tuple = sparse_to_tuple(self.adj_label)
        features_tuple = sparse_to_tuple(self.features_orig)
        self.adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm_tuple[0].T),
                                                torch.FloatTensor(adj_norm_tuple[1]),
                                                torch.Size(adj_norm_tuple[2]))
        self.adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label_tuple[0].T),
                                                torch.FloatTensor(adj_label_tuple[1]),
                                                torch.Size(adj_label_tuple[2]))
        self.features = torch.sparse.FloatTensor(torch.LongTensor(features_tuple[0].T),
                                                torch.FloatTensor(features_tuple[1]),
                                                torch.Size(features_tuple[2]))

