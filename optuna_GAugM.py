import os
import time
import copy
import torch
import pickle
import logging
import argparse
import numpy as np
import scipy.sparse as sp
from collections import Counter
import optuna
import torch

from models.GCN_dgl import GCN
from models.GAT_dgl import GAT
from models.GSAGE_dgl import GraphSAGE
from models.JKNet_dgl import JKNet

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser(description='single')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--gnn', type=str, default='gcn')
parser.add_argument('--i', type=str, default='1')
parser.add_argument('--gpu', type=str, default='-1')
parser.add_argument('--eval_orig', type=int, default=0)
parser.add_argument('--nlayers', type=int, default=-1)
parser.add_argument('--add_train', type=int, default=-1)
args = parser.parse_args()

gpu = args.gpu
if gpu == '-1':
    gpuid = -1
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    gpuid = 0

def sample_graph_det(adj_orig, A_pred, remove_pct, add_pct):
    if remove_pct == 0 and add_pct == 0:
        return copy.deepcopy(adj_orig)
    orig_upper = sp.triu(adj_orig, 1)
    n_edges = orig_upper.nnz
    edges = np.asarray(orig_upper.nonzero()).T
    if remove_pct:
        n_remove = int(n_edges * remove_pct / 100)
        pos_probs = A_pred[edges.T[0], edges.T[1]]
        e_index_2b_remove = np.argpartition(pos_probs, n_remove)[:n_remove]
        mask = np.ones(len(edges), dtype=bool)
        mask[e_index_2b_remove] = False
        edges_pred = edges[mask]
    else:
        edges_pred = edges

    if add_pct:
        n_add = int(n_edges * add_pct / 100)
        # deep copy to avoid modifying A_pred
        A_probs = np.array(A_pred)
        # make the probabilities of the lower half to be zero (including diagonal)
        A_probs[np.tril_indices(A_probs.shape[0])] = 0
        # make the probabilities of existing edges to be zero
        A_probs[edges.T[0], edges.T[1]] = 0
        all_probs = A_probs.reshape(-1)
        e_index_2b_add = np.argpartition(all_probs, -n_add)[-n_add:]
        new_edges = []
        for index in e_index_2b_add:
            i = int(index / A_probs.shape[0])
            j = index % A_probs.shape[0]
            new_edges.append([i, j])
        edges_pred = np.concatenate((edges_pred, new_edges), axis=0)
    adj_pred = sp.csr_matrix((np.ones(len(edges_pred)), edges_pred.T), shape=adj_orig.shape)
    adj_pred = adj_pred + adj_pred.T
    return adj_pred

def test_gaugm(trial):
    ds = args.dataset
    gnn = args.gnn
    eval_orig = args.eval_orig
    t = time.time()
    tvt_nids = pickle.load(open(f'data/graphs/{ds}_tvt_nids.pkl', 'rb'))
    adj_orig = pickle.load(open(f'data/graphs/{ds}_adj.pkl', 'rb'))
    features = pickle.load(open(f'data/graphs/{ds}_features.pkl', 'rb'))
    labels = pickle.load(open(f'data/graphs/{ds}_labels.pkl', 'rb'))
    if sp.issparse(features):
        features = torch.FloatTensor(features.toarray())
    A_pred = pickle.load(open(f'data/edge_probabilities/{ds}_graph_{args.i}_logits.pkl', 'rb'))
    if ds == 'cora' and args.add_train > 0:
        if args.add_train < 20:
            new_trainids = []
            cnt = Counter()
            for i in tvt_nids[0]:
                if cnt[labels.numpy()[i]] < args.add_train:
                    new_trainids.append(i)
                    cnt[labels.numpy()[i]] += 1
            tvt_nids[0] = np.array(new_trainids)
        else:
            tvt_nids[0] = np.concatenate((tvt_nids[0], np.arange(640, 640+args.add_train)))
    # sample the graph
    remove_pct = trial.suggest_int('remove_pct', 0, 80)
    add_pct = trial.suggest_int('add_pct', 0, 80)
    adj_pred = sample_graph_det(adj_orig, A_pred, remove_pct, add_pct)
    if gnn == 'gcn':
        GNN = GCN
    elif gnn == 'gat':
        GNN = GAT
    elif gnn == 'gsage':
        GNN = GraphSAGE
    elif gnn == 'jknet':
        GNN = JKNet
    accs = []
    for _ in range(30):
        if eval_orig > 0:
            if args.nlayers > 0:
                model = GNN(adj_pred, copy.deepcopy(adj_orig), features, labels, tvt_nids, print_progress=False, cuda=gpuid, epochs=200, n_layers=args.nlayers)
            else:
                model = GNN(adj_pred, copy.deepcopy(adj_orig), features, labels, tvt_nids, print_progress=False, cuda=gpuid, epochs=200)
        else:
            if args.nlayers > 0:
                model = GNN(adj_pred, adj_pred, features, labels, tvt_nids, print_progress=False, cuda=gpuid, epochs=200, n_layers=args.nlayers)
            else:
                model = GNN(adj_pred, adj_pred, features, labels, tvt_nids, print_progress=False, cuda=gpuid, epochs=200)
        acc, _, _ = model.fit()
        accs.append(acc)
    acc = np.mean(accs)
    std = np.std(accs)
    # print results
    ev = 'e-orig' if eval_orig else 'e-pred'
    trial.suggest_categorical('dataset', [ds])
    trial.suggest_categorical('i', [args.i])
    trial.suggest_categorical('gnn', [gnn])
    trial.suggest_categorical('eval_orig', [eval_orig])
    return acc

if __name__ == "__main__":
    logging.info('start')
    study = optuna.create_study(direction='maximize')
    study.optimize(test_gaugm, n_trials=200)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))





