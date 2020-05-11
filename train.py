from __future__ import division
from __future__ import print_function
import scipy
import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import scipy.sparse as sp
from utils_nhop_neighbours import load_data, accuracy, multi_labels_nll_loss
from models import ADSF, RWR_process


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--dataset', type=str, default='citeseer', help='DataSet of model')
parser.add_argument('--no-sparse', action='store_true', default=False, help='Not use sparse matrix')  # 缺少args.no_sparse
# 实验名称，用于生成.pkl文件夹
parser.add_argument('--experiment', type=str, default='GAT', help='Name of current experiment.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)
if not args.cuda:
    exit()
args.sparse = not args.no_sparse  # args.sparse: default True, RWR

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def preprocess_features(features, dataset_str):  # 将features按行归一化，features为coo稀疏矩阵格式
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))  # 每行求和
    r_inv = (1 / rowsum).flatten()  # 求倒数，展开，原代码为r_inv = np.power(rowsum, -1).flatten()，无法运行-1次方
    r_inv[np.isinf(r_inv)] = 0.  # 处理nan
    r_mat_inv = sp.diags(r_inv)  # 构建稀疏的对角矩阵
    features = r_mat_inv.dot(features)
    if dataset_str == 'citeseer':
        features = features.todense()
    return features# , sparse_to_tuple(features)


def sparse_to_tuple(sparse_mx):  # 稀疏矩阵features变元组
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()  # coords是稀疏矩阵中位置信息
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):  # 不会进来
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)  # 得到(位置，值，shape)的元组

    return sparse_mx


# Load data
adj, features, idx_train, idx_val, idx_test, train_mask, val_mask, test_mask, labels, nclass, adj_ad = load_data(args.dataset, args.sparse)  # features为coo稀疏矩阵
features = preprocess_features(features, args.dataset)  # 归一化，得到实矩阵features和元组spars
features = np.array(features, dtype=np.float32)
# features = scipy.sparse.csr_matrix(features)  # 稀疏矩阵features
#
# features = features.astype(np.float32)
features = torch.FloatTensor(features)

if args.sparse:
    model = RWR_process(nfeat=features.shape[1],
                        nhid=args.hidden,
                        nclass=nclass,
                        dropout=args.dropout,
                        nheads=args.nb_heads,
                        alpha=args.alpha,
                        adj_ad=adj_ad,
                        adj=adj,
                        dataset_str=args.dataset)
else:
    model = ADSF(nfeat=features.shape[1],
                 nhid=args.hidden,
                 nclass=nclass,
                 dropout=args.dropout,
                 nheads=args.nb_heads,
                 alpha=args.alpha,
                 adj_ad=adj_ad,
                 adj=adj)

if args.cuda:
    print("Using {} gpu".format(torch.cuda.device_count()))
    if torch.cuda.device_count() > 1:  # 多卡
        model = torch.nn.DataParallel(model)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    adj_ad = adj_ad.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
features, adj, labels = Variable(features), Variable(adj), Variable(labels)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features)
    loss_train = multi_labels_nll_loss(output[idx_train], labels[idx_train])  # softmax+nllloss
    acc_train, preds = accuracy(output[idx_train], labels[idx_train], args.cuda)
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features)
    
    loss_val = multi_labels_nll_loss(output[idx_val], labels[idx_val])
    acc_val, preds = accuracy(output[idx_val], labels[idx_val], args.cuda)

    file_handle1 = open('./{}/auc.txt'.format(args.experiment), mode='a')
    print("epoch: {:04d}, acc_val: {:.4f}, loss_val: {:.4f}, time: {:.4f}s".format(epoch, acc_val, loss_val.item(), time.time() - t), file=file_handle1)
    file_handle1.close()

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val),
          'time: {:.4f}s'.format(time.time() - t))
    
    return loss_val.data.item()


# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch))
    torch.save(model.state_dict(), './{}/{}.pkl'.format(args.experiment, epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break
    files = glob.glob('./{}/*.pkl'.format(args.experiment))
    for file in files:
        epoch_nb = int(file.split('/')[-1].split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)
files = glob.glob('./{}/*.pkl'.format(args.experiment))
for file in files:
    epoch_nb = int(file.split('/')[-1].split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
# print('Loading {}th epoch'.format(best_epoch))
# model.load_state_dict(torch.load('./{}/{}.pkl'.format(args.experiment, best_epoch)))
