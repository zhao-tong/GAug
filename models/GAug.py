import gc
import logging
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
from itertools import combinations
from sklearn.metrics import roc_auc_score, average_precision_score
import pickle

class GAug(object):
    def __init__(self, adj_matrix, features, labels, tvt_nids, cuda=-1, hidden_size=128, emb_size=32, n_layers=1, epochs=200, seed=-1, lr=1e-2, weight_decay=5e-4, dropout=0.5, gae=False, beta=0.5, temperature=0.2, log=True, name='debug', warmup=3, gnnlayer_type='gcn', jknet=False, alpha=1, sample_type='add_sample', feat_norm='row'):
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = epochs
        self.gae = gae
        self.beta = beta
        self.warmup = warmup
        self.feat_norm = feat_norm
        # create a logger, logs are saved to GAug-[name].log when name is not None
        if log:
            self.logger = self.get_logger(name)
        else:
            # disable logger if wanted
            # logging.disable(logging.CRITICAL)
            self.logger = logging.getLogger()
        # config device (force device to cpu when cuda is not available)
        if not torch.cuda.is_available():
            cuda = -1
        self.device = torch.device(f'cuda:{cuda}' if cuda>=0 else 'cpu')
        # log all parameters to keep record
        all_vars = locals()
        self.log_parameters(all_vars)
        # fix random seeds if needed
        if seed > 0:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # load data
        self.load_data(adj_matrix, features, labels, tvt_nids, gnnlayer_type)
        # setup the model
        self.model = GAug_model(self.features.size(1),
                                hidden_size,
                                emb_size,
                                self.out_size,
                                n_layers,
                                F.relu,
                                dropout,
                                self.device,
                                gnnlayer_type,
                                temperature=temperature,
                                gae=gae,
                                jknet=jknet,
                                alpha=alpha,
                                sample_type=sample_type)

    def load_data(self, adj_matrix, features, labels, tvt_nids, gnnlayer_type):
        """ preprocess data """
        # features (torch.FloatTensor)
        if isinstance(features, torch.FloatTensor):
            self.features = features
        else:
            self.features = torch.FloatTensor(features)
        # normalize feature matrix if needed
        if self.feat_norm == 'row':
            self.features = F.normalize(self.features, p=1, dim=1)
        elif self.feat_norm == 'col':
            self.features = self.col_normalization(self.features)
        elif self.feat_norm == 'rowcol':
            self.features = F.normalize(self.features, p=1, dim=1)
            self.features = self.col_normalization(self.features)
        elif self.feat_norm == 'colrow':
            self.features = self.col_normalization(self.features)
            self.features = F.normalize(self.features, p=1, dim=1)
        # original adj_matrix for training vgae (torch.FloatTensor)
        assert sp.issparse(adj_matrix)
        if not isinstance(adj_matrix, sp.coo_matrix):
            adj_matrix = sp.coo_matrix(adj_matrix)
        adj_matrix.setdiag(1)
        self.adj_orig = scipysp_to_pytorchsp(adj_matrix).to_dense()
        # normalized adj_matrix used as input for ep_net (torch.sparse.FloatTensor)
        degrees = np.array(adj_matrix.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(degrees, -0.5).flatten())
        adj_norm = degree_mat_inv_sqrt @ adj_matrix @ degree_mat_inv_sqrt
        self.adj_norm = scipysp_to_pytorchsp(adj_norm)
        # adj_matrix used as input for nc_net (torch.sparse.FloatTensor)
        if gnnlayer_type == 'gcn':
            self.adj = scipysp_to_pytorchsp(adj_norm)
        elif gnnlayer_type == 'gsage':
            adj_matrix_noselfloop = sp.coo_matrix(adj_matrix)
            # adj_matrix_noselfloop.setdiag(0)
            # adj_matrix_noselfloop.eliminate_zeros()
            adj_matrix_noselfloop = sp.coo_matrix(adj_matrix_noselfloop / adj_matrix_noselfloop.sum(1))
            self.adj = scipysp_to_pytorchsp(adj_matrix_noselfloop)
        elif gnnlayer_type == 'gat':
            # self.adj = scipysp_to_pytorchsp(adj_matrix)
            self.adj = torch.FloatTensor(adj_matrix.todense())
        # labels (torch.LongTensor) and train/validation/test nids (np.ndarray)
        if len(labels.shape) == 2:
            labels = torch.FloatTensor(labels)
        else:
            labels = torch.LongTensor(labels)
        self.labels = labels
        self.train_nid = tvt_nids[0]
        self.val_nid = tvt_nids[1]
        self.test_nid = tvt_nids[2]
        # number of classes
        if len(self.labels.size()) == 1:
            self.out_size = len(torch.unique(self.labels))
        else:
            self.out_size = labels.size(1)
        # sample the edges to evaluate edge prediction results
        # sample 10% (1% for large graph) of the edges and the same number of no-edges
        if labels.size(0) > 5000:
            edge_frac = 0.01
        else:
            edge_frac = 0.1
        adj_matrix = sp.csr_matrix(adj_matrix)
        n_edges_sample = int(edge_frac * adj_matrix.nnz / 2)
        # sample negative edges
        neg_edges = []
        added_edges = set()
        while len(neg_edges) < n_edges_sample:
            i = np.random.randint(0, adj_matrix.shape[0])
            j = np.random.randint(0, adj_matrix.shape[0])
            if i == j:
                continue
            if adj_matrix[i, j] > 0:
                continue
            if (i, j) in added_edges:
                continue
            neg_edges.append([i, j])
            added_edges.add((i, j))
            added_edges.add((j, i))
        neg_edges = np.asarray(neg_edges)
        # sample positive edges
        nz_upper = np.array(sp.triu(adj_matrix, k=1).nonzero()).T
        np.random.shuffle(nz_upper)
        pos_edges = nz_upper[:n_edges_sample]
        self.val_edges = np.concatenate((pos_edges, neg_edges), axis=0)
        self.edge_labels = np.array([1]*n_edges_sample + [0]*n_edges_sample)

    def pretrain_ep_net(self, model, adj, features, adj_orig, norm_w, pos_weight, n_epochs):
        """ pretrain the edge prediction network """
        optimizer = torch.optim.Adam(model.ep_net.parameters(),
                                     lr=self.lr)
        model.train()
        for epoch in range(n_epochs):
            adj_logits = model.ep_net(adj, features)
            loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, adj_orig, pos_weight=pos_weight)
            if not self.gae:
                mu = model.ep_net.mean
                lgstd = model.ep_net.logstd
                kl_divergence = 0.5/adj_logits.size(0) * (1 + 2*lgstd - mu**2 - torch.exp(2*lgstd)).sum(1).mean()
                loss -= kl_divergence
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            adj_pred = torch.sigmoid(adj_logits.detach()).cpu()
            ep_auc, ep_ap = self.eval_edge_pred(adj_pred, self.val_edges, self.edge_labels)
            self.logger.info('EPNet pretrain, Epoch [{:3}/{}]: loss {:.4f}, auc {:.4f}, ap {:.4f}'
                        .format(epoch+1, n_epochs, loss.item(), ep_auc, ep_ap))

    def pretrain_nc_net(self, model, adj, features, labels, n_epochs):
        """ pretrain the node classification network """
        optimizer = torch.optim.Adam(model.nc_net.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        # loss function for node classification
        if len(self.labels.size()) == 2:
            nc_criterion = nn.BCEWithLogitsLoss()
        else:
            nc_criterion = nn.CrossEntropyLoss()
        best_val_acc = 0.
        for epoch in range(n_epochs):
            model.train()
            nc_logits = model.nc_net(adj, features)
            # losses
            loss = nc_criterion(nc_logits[self.train_nid], labels[self.train_nid])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                nc_logits_eval = model.nc_net(adj, features)
            val_acc = self.eval_node_cls(nc_logits_eval[self.val_nid], labels[self.val_nid])
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = self.eval_node_cls(nc_logits_eval[self.test_nid], labels[self.test_nid])
                self.logger.info('NCNet pretrain, Epoch [{:2}/{}]: loss {:.4f}, val acc {:.4f}, test acc {:.4f}'
                            .format(epoch+1, n_epochs, loss.item(), val_acc, test_acc))
            else:
                self.logger.info('NCNet pretrain, Epoch [{:2}/{}]: loss {:.4f}, val acc {:.4f}'
                            .format(epoch+1, n_epochs, loss.item(), val_acc))

    def fit(self, pretrain_ep=200, pretrain_nc=20):
        """ train the model """
        # move data to device
        adj_norm = self.adj_norm.to(self.device)
        adj = self.adj.to(self.device)
        features = self.features.to(self.device)
        labels = self.labels.to(self.device)
        adj_orig = self.adj_orig.to(self.device)
        model = self.model.to(self.device)
        # weights for log_lik loss when training EP net
        adj_t = self.adj_orig
        norm_w = adj_t.shape[0]**2 / float((adj_t.shape[0]**2 - adj_t.sum()) * 2)
        pos_weight = torch.FloatTensor([float(adj_t.shape[0]**2 - adj_t.sum()) / adj_t.sum()]).to(self.device)
        # pretrain VGAE if needed
        if pretrain_ep:
            self.pretrain_ep_net(model, adj_norm, features, adj_orig, norm_w, pos_weight, pretrain_ep)
        # pretrain GCN if needed
        if pretrain_nc:
            self.pretrain_nc_net(model, adj, features, labels, pretrain_nc)
        # optimizers
        optims = MultipleOptimizer(torch.optim.Adam(model.ep_net.parameters(),
                                                    lr=self.lr),
                                   torch.optim.Adam(model.nc_net.parameters(),
                                                    lr=self.lr,
                                                    weight_decay=self.weight_decay))
        # get the learning rate schedule for the optimizer of ep_net if needed
        if self.warmup:
            ep_lr_schedule = self.get_lr_schedule_by_sigmoid(self.n_epochs, self.lr, self.warmup)
        # loss function for node classification
        if len(self.labels.size()) == 2:
            nc_criterion = nn.BCEWithLogitsLoss()
        else:
            nc_criterion = nn.CrossEntropyLoss()
        # keep record of the best validation accuracy for early stopping
        best_val_acc = 0.
        patience_step = 0
        # train model
        for epoch in range(self.n_epochs):
            # update the learning rate for ep_net if needed
            if self.warmup:
                optims.update_lr(0, ep_lr_schedule[epoch])

            model.train()
            nc_logits, adj_logits = model(adj_norm, adj_orig, features)
            # nc_logits, adj_logits, adj_new = model(adj_norm, adj_orig, features)
            # # TODO: tmp saving for experiments, delete the following line and adj_new later
            # pickle.dump(adj_new, open(f'results/gaug_adjs/ep{epoch+1}.pkl', 'wb'))

            # losses
            loss = nc_loss = nc_criterion(nc_logits[self.train_nid], labels[self.train_nid])
            ep_loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, adj_orig, pos_weight=pos_weight)
            loss += self.beta * ep_loss
            optims.zero_grad()
            loss.backward()
            optims.step()
            # validate (without dropout)
            model.eval()
            with torch.no_grad():
                nc_logits_eval = model.nc_net(adj, features)
            val_acc = self.eval_node_cls(nc_logits_eval[self.val_nid], labels[self.val_nid])
            adj_pred = torch.sigmoid(adj_logits.detach()).cpu()
            ep_auc, ep_ap = self.eval_edge_pred(adj_pred, self.val_edges, self.edge_labels)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = self.eval_node_cls(nc_logits_eval[self.test_nid], labels[self.test_nid])
                self.logger.info('Epoch [{:3}/{}]: ep loss {:.4f}, nc loss {:.4f}, ep auc: {:.4f}, ep ap {:.4f}, val acc {:.4f}, test acc {:.4f}'
                            .format(epoch+1, self.n_epochs, ep_loss.item(), nc_loss.item(), ep_auc, ep_ap, val_acc, test_acc))
                patience_step = 0
            else:
                self.logger.info('Epoch [{:3}/{}]: ep loss {:.4f}, nc loss {:.4f}, ep auc: {:.4f}, ep ap {:.4f}, val acc {:.4f}'
                            .format(epoch+1, self.n_epochs, ep_loss.item(), nc_loss.item(), ep_auc, ep_ap, val_acc))
                patience_step += 1
                if patience_step == 100:
                    self.logger.info('Early stop!')
                    break
        # get final test result without early stop
        with torch.no_grad():
            nc_logits_eval = model.nc_net(adj, features)
        test_acc_final = self.eval_node_cls(nc_logits_eval[self.test_nid], labels[self.test_nid])
        # log both results
        self.logger.info('Final test acc with early stop: {:.4f}, without early stop: {:.4f}'
                    .format(test_acc, test_acc_final))
        # release RAM and GPU memory
        del adj, features, labels, adj_orig
        torch.cuda.empty_cache()
        gc.collect()
        return test_acc

    def log_parameters(self, all_vars):
        """ log all variables in the input dict excluding the following ones """
        del all_vars['self']
        del all_vars['adj_matrix']
        del all_vars['features']
        del all_vars['labels']
        del all_vars['tvt_nids']
        self.logger.info(f'Parameters: {all_vars}')

    @staticmethod
    def eval_edge_pred(adj_pred, val_edges, edge_labels):
        logits = adj_pred[val_edges.T]
        logits = np.nan_to_num(logits)
        roc_auc = roc_auc_score(edge_labels, logits)
        ap_score = average_precision_score(edge_labels, logits)
        return roc_auc, ap_score

    @staticmethod
    def eval_node_cls(nc_logits, labels):
        """ evaluate node classification results """
        if len(labels.size()) == 2:
            preds = torch.round(torch.sigmoid(nc_logits))
            tp = len(torch.nonzero(preds * labels))
            tn = len(torch.nonzero((1-preds) * (1-labels)))
            fp = len(torch.nonzero(preds * (1-labels)))
            fn = len(torch.nonzero((1-preds) * labels))
            pre, rec, f1 = 0., 0., 0.
            if tp+fp > 0:
                pre = tp / (tp + fp)
            if tp+fn > 0:
                rec = tp / (tp + fn)
            if pre+rec > 0:
                fmeasure = (2 * pre * rec) / (pre + rec)
        else:
            preds = torch.argmax(nc_logits, dim=1)
            correct = torch.sum(preds == labels)
            fmeasure = correct.item() / len(labels)
        return fmeasure

    @staticmethod
    def get_lr_schedule_by_sigmoid(n_epochs, lr, warmup):
        """ schedule the learning rate with the sigmoid function.
        The learning rate will start with near zero and end with near lr """
        factors = torch.FloatTensor(np.arange(n_epochs))
        factors = ((factors / factors[-1]) * (warmup * 2)) - warmup
        factors = torch.sigmoid(factors)
        # range the factors to [0, 1]
        factors = (factors - factors[0]) / (factors[-1] - factors[0])
        lr_schedule = factors * lr
        return lr_schedule

    @staticmethod
    def get_logger(name):
        """ create a nice logger """
        logger = logging.getLogger(name)
        # clear handlers if they were created in other runs
        if (logger.hasHandlers()):
            logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        # create console handler add add to logger
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        # create file handler add add to logger when name is not None
        if name is not None:
            fh = logging.FileHandler(f'GAug-{name}.log')
            fh.setFormatter(formatter)
            fh.setLevel(logging.DEBUG)
            logger.addHandler(fh)
        return logger

    @staticmethod
    def col_normalization(features):
        """ column normalization for feature matrix """
        features = features.numpy()
        m = features.mean(axis=0)
        s = features.std(axis=0, ddof=0, keepdims=True) + 1e-12
        features -= m
        features /= s
        return torch.FloatTensor(features)


class GAug_model(nn.Module):
    def __init__(self,
                 dim_feats,
                 dim_h,
                 dim_z,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 device,
                 gnnlayer_type,
                 temperature=1,
                 gae=False,
                 jknet=False,
                 alpha=1,
                 sample_type='neigh'):
        super(GAug_model, self).__init__()
        self.device = device
        self.temperature = temperature
        self.gnnlayer_type = gnnlayer_type
        self.alpha = alpha
        self.sample_type=sample_type
        # edge prediction network
        self.ep_net = VGAE(dim_feats, dim_h, dim_z, activation, gae=gae)
        # node classification network
        if jknet:
            self.nc_net = GNN_JK(dim_feats, dim_h, n_classes, n_layers, activation, dropout, gnnlayer_type=gnnlayer_type)
        else:
            self.nc_net = GNN(dim_feats, dim_h, n_classes, n_layers, activation, dropout, gnnlayer_type=gnnlayer_type)

    def sample_adj(self, adj_logits):
        """ sample an adj from the predicted edge probabilities of ep_net """
        edge_probs = adj_logits / torch.max(adj_logits)
        # sampling
        adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temperature, probs=edge_probs).rsample()
        # making adj_sampled symmetric
        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T
        return adj_sampled

    def sample_adj_no_sample(self, adj_logits):
        edge_probs = adj_logits / torch.max(adj_logits)
        # making adj_sampled symmetric
        adj_sampled = edge_probs.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T
        return adj_sampled

    def sample_adj_add_bernoulli(self, adj_logits, adj_orig, alpha):
        edge_probs = adj_logits / torch.max(adj_logits)
        edge_probs = alpha*edge_probs + (1-alpha)*adj_orig
        # sampling
        adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temperature, probs=edge_probs).rsample()
        # making adj_sampled symmetric
        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T
        return adj_sampled

    def sample_adj_add_round(self, adj_logits, adj_orig, alpha):
        edge_probs = adj_logits / torch.max(adj_logits)
        edge_probs = alpha*edge_probs + (1-alpha)*adj_orig
        # sampling
        adj_sampled = RoundNoGradient.apply(edge_probs)
        # making adj_sampled symmetric
        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T
        return adj_sampled

    def sample_adj_random(self, adj_logits):
        adj_rand = torch.rand(adj_logits.size())
        adj_rand = adj_rand.triu(1)
        adj_rand = torch.round(adj_rand)
        adj_rand = adj_rand + adj_rand.T
        return adj_rand

    def sample_adj_block(self, adj_logits, adj_orig, change_frac):
        edge_probs = adj_logits / torch.max(adj_logits)
        indexes = torch.LongTensor(list(combinations(np.random.choice(adj_orig.size(1), int(adj_orig.size(1)*change_frac), replace=False), 2))).T
        edge_probs = adj_logits[indexes[0], indexes[1]]
        edges_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temperature, probs=edge_probs).rsample()
        # avoid modifying the original adj
        adj_new = adj_orig.index_put((indexes[0], indexes[1]), edges_sampled)
        adj_new.index_put_((indexes[1], indexes[0]), edges_sampled)
        return adj_new

    def sample_adj_edge(self, adj_logits, adj_orig, change_frac):
        adj = adj_orig.to_dense() if adj_orig.is_sparse else adj_orig
        n_edges = adj.nonzero().size(0)
        n_change = int(n_edges * change_frac / 2)
        # take only the upper triangle
        edge_probs = adj_logits.triu(1)
        edge_probs = edge_probs - torch.min(edge_probs)
        edge_probs = edge_probs / torch.max(edge_probs)
        adj_inverse = 1 - adj
        # get edges to be removed
        mask_rm = edge_probs * adj
        nz_mask_rm = mask_rm[mask_rm>0]
        if len(nz_mask_rm) > 0:
            n_rm = len(nz_mask_rm) if len(nz_mask_rm) < n_change else n_change
            thresh_rm = torch.topk(mask_rm[mask_rm>0], n_rm, largest=False)[0][-1]
            mask_rm[mask_rm > thresh_rm] = 0
            mask_rm = CeilNoGradient.apply(mask_rm)
            mask_rm = mask_rm + mask_rm.T
        # remove edges
        adj_new = adj - mask_rm
        # get edges to be added
        mask_add = edge_probs * adj_inverse
        nz_mask_add = mask_add[mask_add>0]
        if len(nz_mask_add) > 0:
            n_add = len(nz_mask_add) if len(nz_mask_add) < n_change else n_change
            thresh_add = torch.topk(mask_add[mask_add>0], n_add, largest=True)[0][-1]
            mask_add[mask_add < thresh_add] = 0
            mask_add = CeilNoGradient.apply(mask_add)
            mask_add = mask_add + mask_add.T
        # add edges
        adj_new = adj_new + mask_add
        return adj_new

    def normalize_adj(self, adj):
        if self.gnnlayer_type == 'gcn':
            # adj = adj + torch.diag(torch.ones(adj.size(0))).to(self.device)
            adj.fill_diagonal_(1)
            # normalize adj with A = D^{-1/2} @ A @ D^{-1/2}
            D_norm = torch.diag(torch.pow(adj.sum(1), -0.5)).to(self.device)
            adj = D_norm @ adj @ D_norm
        elif self.gnnlayer_type == 'gat':
            # adj = adj + torch.diag(torch.ones(adj.size(0))).to(self.device)
            adj.fill_diagonal_(1)
        elif self.gnnlayer_type == 'gsage':
            # adj = adj + torch.diag(torch.ones(adj.size(0))).to(self.device)
            adj.fill_diagonal_(1)
            adj = F.normalize(adj, p=1, dim=1)
        return adj

    def forward(self, adj, adj_orig, features):
        adj_logits = self.ep_net(adj, features)
        if self.sample_type == 'rand':
            adj_new = self.sample_adj_block(adj_logits, adj_orig, self.alpha)
        elif self.sample_type == 'edge':
            adj_new = self.sample_adj_edge(adj_logits, adj_orig, self.alpha)
        elif self.sample_type == 'add_round':
            adj_new = self.sample_adj_add_round(adj_logits, adj_orig, self.alpha)
        elif self.sample_type == 'no_sample':
            adj_new = self.sample_adj_no_sample(adj_logits)
        elif self.sample_type == 'rand':
            adj_new = self.sample_adj_random(adj_logits)
        elif self.sample_type == 'add_sample':
            if self.alpha == 1:
                adj_new = self.sample_adj(adj_logits)
            else:
                adj_new = self.sample_adj_add_bernoulli(adj_logits, adj_orig, self.alpha)
        adj_new_normed = self.normalize_adj(adj_new)
        nc_logits = self.nc_net(adj_new_normed, features)
        # # TODO: remove adj_new
        # return nc_logits, adj_logits, adj_new
        return nc_logits, adj_logits


class VGAE(nn.Module):
    """ GAE/VGAE as edge prediction model """
    def __init__(self, dim_feats, dim_h, dim_z, activation, gae=False):
        super(VGAE, self).__init__()
        self.gae = gae
        self.gcn_base = GCNLayer(dim_feats, dim_h, 1, None, 0, bias=False)
        self.gcn_mean = GCNLayer(dim_h, dim_z, 1, activation, 0, bias=False)
        self.gcn_logstd = GCNLayer(dim_h, dim_z, 1, activation, 0, bias=False)

    def forward(self, adj, features):
        # GCN encoder
        hidden = self.gcn_base(adj, features)
        self.mean = self.gcn_mean(adj, hidden)
        if self.gae:
            # GAE (no sampling at bottleneck)
            Z = self.mean
        else:
            # VGAE
            self.logstd = self.gcn_logstd(adj, hidden)
            gaussian_noise = torch.randn_like(self.mean)
            sampled_Z = gaussian_noise*torch.exp(self.logstd) + self.mean
            Z = sampled_Z
        # inner product decoder
        adj_logits = Z @ Z.T
        return adj_logits


class GNN(nn.Module):
    """ GNN as node classification model """
    def __init__(self, dim_feats, dim_h, n_classes, n_layers, activation, dropout, gnnlayer_type='gcn'):
        super(GNN, self).__init__()
        heads = [1] * (n_layers + 1)
        if gnnlayer_type == 'gcn':
            gnnlayer = GCNLayer
        elif gnnlayer_type == 'gsage':
            gnnlayer = SAGELayer
        elif gnnlayer_type == 'gat':
            gnnlayer = GATLayer
            if dim_feats in (50, 745, 12047): # hard coding n_heads for large graphs
                heads = [2] * n_layers + [1]
            else:
                heads = [8] * n_layers + [1]
            dim_h = int(dim_h / 8)
            dropout = 0.6
            activation = F.elu
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(gnnlayer(dim_feats, dim_h, heads[0], activation, 0))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(gnnlayer(dim_h*heads[i], dim_h, heads[i+1], activation, dropout))
        # output layer
        self.layers.append(gnnlayer(dim_h*heads[-2], n_classes, heads[-1], None, dropout))

    def forward(self, adj, features):
        h = features
        for layer in self.layers:
            h = layer(adj, h)
        # return F.log_softmax(h, dim=1)
        return h


class GNN_JK(nn.Module):
    """ GNN with JK design as a node classification model """
    def __init__(self, dim_feats, dim_h, n_classes, n_layers, activation, dropout, gnnlayer_type='gcn'):
        super(GNN_JK, self).__init__()
        heads = [1] * (n_layers + 1)
        if gnnlayer_type == 'gcn':
            gnnlayer = GCNLayer
        elif gnnlayer_type == 'gsage':
            gnnlayer = SAGELayer
        elif gnnlayer_type == 'gat':
            gnnlayer = GATLayer
            heads = [8] * n_layers + [1]
            dim_h = int(dim_h / 8)
            activation = F.elu
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(gnnlayer(dim_feats, dim_h, heads[0], activation, 0))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(gnnlayer(dim_h*heads[i], dim_h, heads[i+1], activation, dropout))
        # output layer
        self.layer_output = nn.Linear(dim_h*n_layers*heads[-2], n_classes)

    def forward(self, adj, features):
        h = features
        hs = []
        for layer in self.layers:
            h = layer(adj, h)
            hs.append(h)
        # JK-concat design
        h = torch.cat(hs, 1)
        h = self.layer_output(h)
        # return F.log_softmax(h, dim=1)
        return h


class GCNLayer(nn.Module):
    """ one layer of GCN """
    def __init__(self, input_dim, output_dim, n_heads, activation, dropout, bias=True):
        super(GCNLayer, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.activation = activation
        if bias:
            self.b = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.b = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0
        self.init_params()

    def init_params(self):
        """ Initialize weights with xavier uniform and biases with all zeros """
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.0)

    def forward(self, adj, h):
        if self.dropout:
            h = self.dropout(h)
        x = h @ self.W
        x = adj @ x
        if self.b is not None:
            x = x + self.b
        if self.activation:
            x = self.activation(x)
        return x


class SAGELayer(nn.Module):
    """ one layer of GraphSAGE with gcn aggregator """
    def __init__(self, input_dim, output_dim, n_heads, activation, dropout, bias=True):
        super(SAGELayer, self).__init__()
        self.linear_neigh = nn.Linear(input_dim, output_dim, bias=False)
        # self.linear_self = nn.Linear(input_dim, output_dim, bias=False)
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0
        self.init_params()

    def init_params(self):
        """ Initialize weights with xavier uniform and biases with all zeros """
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.0)

    def forward(self, adj, h):
        if self.dropout:
            h = self.dropout(h)
        x = adj @ h
        x = self.linear_neigh(x)
        # x_neigh = self.linear_neigh(x)
        # x_self = self.linear_self(h)
        # x = x_neigh + x_self
        if self.activation:
            x = self.activation(x)
        # x = F.normalize(x, dim=1, p=2)
        return x


class GATLayer(nn.Module):
    """ one layer of GAT """
    def __init__(self, input_dim, output_dim, n_heads, activation, dropout, bias=True):
        super(GATLayer, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.activation = activation
        self.n_heads = n_heads
        self.attn_l = nn.Linear(output_dim, self.n_heads, bias=False)
        self.attn_r = nn.Linear(output_dim, self.n_heads, bias=False)
        self.attn_drop = nn.Dropout(p=0.6)
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0
        if bias:
            self.b = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.b = None
        self.init_params()

    def init_params(self):
        """ Initialize weights with xavier uniform and biases with all zeros """
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.0)

    def forward(self, adj, h):
        if self.dropout:
            h = self.dropout(h)
        x = h @ self.W # torch.Size([2708, 128])
        # calculate attentions, both el and er are n_nodes by n_heads
        el = self.attn_l(x)
        er = self.attn_r(x) # torch.Size([2708, 8])
        if isinstance(adj, torch.sparse.FloatTensor):
            nz_indices = adj._indices()
        else:
            nz_indices = adj.nonzero().T
        attn = el[nz_indices[0]] + er[nz_indices[1]] # torch.Size([13264, 8])
        attn = F.leaky_relu(attn, negative_slope=0.2).squeeze()
        # reconstruct adj with attentions, exp for softmax next
        attn = torch.exp(attn) # torch.Size([13264, 8]) NOTE: torch.Size([13264]) when n_heads=1
        if self.n_heads == 1:
            adj_attn = torch.zeros(size=(adj.size(0), adj.size(1)), device=adj.device)
            adj_attn.index_put_((nz_indices[0], nz_indices[1]), attn)
        else:
            adj_attn = torch.zeros(size=(adj.size(0), adj.size(1), self.n_heads), device=adj.device)
            adj_attn.index_put_((nz_indices[0], nz_indices[1]), attn) # torch.Size([2708, 2708, 8])
            adj_attn.transpose_(1, 2) # torch.Size([2708, 8, 2708])
        # edge softmax (only softmax with non-zero entries)
        adj_attn = F.normalize(adj_attn, p=1, dim=-1)
        adj_attn = self.attn_drop(adj_attn)
        # message passing
        x = adj_attn @ x # torch.Size([2708, 8, 128])
        if self.b is not None:
            x = x + self.b
        if self.activation:
            x = self.activation(x)
        if self.n_heads > 1:
            x = x.flatten(start_dim=1)
        return x # torch.Size([2708, 1024])


class MultipleOptimizer():
    """ a class that wraps multiple optimizers """
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

    def update_lr(self, op_index, new_lr):
        """ update the learning rate of one optimizer
        Parameters: op_index: the index of the optimizer to update
                    new_lr:   new learning rate for that optimizer """
        for param_group in self.optimizers[op_index].param_groups:
            param_group['lr'] = new_lr


class RoundNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g


class CeilNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.ceil()

    @staticmethod
    def backward(ctx, g):
        return g


def scipysp_to_pytorchsp(sp_mx):
    """ converts scipy sparse matrix to pytorch sparse matrix """
    if not sp.isspmatrix_coo(sp_mx):
        sp_mx = sp_mx.tocoo()
    coords = np.vstack((sp_mx.row, sp_mx.col)).transpose()
    values = sp_mx.data
    shape = sp_mx.shape
    pyt_sp_mx = torch.sparse.FloatTensor(torch.LongTensor(coords.T),
                                         torch.FloatTensor(values),
                                         torch.Size(shape))
    return pyt_sp_mx




