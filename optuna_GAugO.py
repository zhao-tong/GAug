import os
import pickle
import argparse
import numpy as np
from collections import Counter
from models.GAug import GAug
import torch
import optuna
import scipy.sparse as sp

parser = argparse.ArgumentParser(description='single')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--gnn', type=str, default='gcn')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--layers', type=int, default=-1)
parser.add_argument('--add_train', type=int, default=-1)
parser.add_argument('--feat_norm', type=str, default='row')
args = parser.parse_args()

ds = args.dataset
gnn = args.gnn
layer_type = args.gnn
gpu = args.gpu

if gpu == '-1':
    cuda = -1
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    cuda = 0

jk = False
if gnn == 'jknet':
    layer_type = 'gsage'
    jk = True

def objective(trial):
    tvt_nids = pickle.load(open(f'data/graphs/{ds}_tvt_nids.pkl', 'rb'))
    adj_orig = pickle.load(open(f'data/graphs/{ds}_adj.pkl', 'rb'))
    features = pickle.load(open(f'data/graphs/{ds}_features.pkl', 'rb'))
    labels = pickle.load(open(f'data/graphs/{ds}_labels.pkl', 'rb'))
    if sp.issparse(features):
        features = torch.FloatTensor(features.toarray())
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

    lr = 0.005 if layer_type == 'gat' else 0.01
    if args.layers > 0:
        n_layers = args.layers
    else:
        n_layers = 1
        if jk:
            n_layers = 3
    feat_norm = args.feat_norm
    if ds == 'ppi':
        feat_norm = 'col'
    elif ds in ('blogcatalog', 'flickr'):
        feat_norm = 'none'
    change_frac = trial.suggest_discrete_uniform('alpha', 0, 1, 0.01)
    beta = trial.suggest_discrete_uniform('beta', 0.0, 4.0, 0.1)
    temp = trial.suggest_discrete_uniform('temp', 0.1, 2.1, 0.1)
    warmup = trial.suggest_int('warmup', 0, 10)
    pretrain_ep = trial.suggest_discrete_uniform('pretrain_ep', 5, 300, 5)
    pretrain_nc = trial.suggest_discrete_uniform('pretrain_nc', 5, 300, 5)
    accs = []
    for _ in range(30):
        model = GAug(adj_orig, features, labels, tvt_nids, cuda=cuda, gae=True, beta=beta, temperature=temp, warmup=int(warmup), gnnlayer_type=layer_type, jknet=jk, lr=lr, n_layers=n_layers, log=False, alpha=change_frac, feat_norm=feat_norm)
        acc = model.fit(pretrain_ep=int(pretrain_ep), pretrain_nc=int(pretrain_nc))
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
    study.optimize(objective, n_trials=400)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

