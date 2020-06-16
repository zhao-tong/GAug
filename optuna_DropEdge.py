import os
import pickle
import argparse
import numpy as np
import optuna
import torch
import scipy.sparse as sp
from models.GCN_dgl import GCN
from models.GAT_dgl import GAT
from models.GSAGE_dgl import GraphSAGE
from models.JKNet_dgl import JKNet

parser = argparse.ArgumentParser(description='single')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--gnn', type=str, default='gcn')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()


gpu = args.gpu
if gpu == '-1':
    cuda = -1
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    cuda = 0

def objective(trial):
    ds = args.dataset
    gnn = args.gnn
    tvt_nids = pickle.load(open(f'data/graphs/{ds}_tvt_nids.pkl', 'rb'))
    adj_orig = pickle.load(open(f'data/graphs/{ds}_adj.pkl', 'rb'))
    features = pickle.load(open(f'data/graphs/{ds}_features.pkl', 'rb'))
    labels = pickle.load(open(f'data/graphs/{ds}_labels.pkl', 'rb'))
    if sp.issparse(features):
        features = torch.FloatTensor(features.toarray())
    dropedge = trial.suggest_uniform('dropedge', 0.01, 1)
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
        model = GNN(adj_orig, adj_orig, features, labels, tvt_nids, print_progress=False, cuda=0, dropedge=dropedge)
        acc, _, _ = model.fit()
        accs.append(acc)
    acc = np.mean(accs)
    std = np.std(accs)
    trial.suggest_categorical('dataset', [ds])
    trial.suggest_categorical('gnn', [gnn])
    trial.suggest_uniform('acc', acc, acc)
    trial.suggest_uniform('std', std, std)
    return acc

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

