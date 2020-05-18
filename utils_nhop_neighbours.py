import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import torch
import pickle
import os


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def encode_onehot(labels):
    classes = set()
    for label in labels:
        classes |= set(label)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array([np.sum([classes_dict.get(l) for l in label], axis=0) for label in labels],
                             dtype=np.int32)
    return labels_onehot, len(classes)


def structural_interaction(ri_index, ri_all, g):
    """structural interaction between the structural fingerprints for citeseer"""
    for i in range(g.shape[0]):
        for j in range(g.shape[0]):
            intersection = set(ri_index[i]).intersection(set(ri_index[j]))  # 返回两个集合交集
            union = set(ri_index[i]).union(set(ri_index[j]))  # 并集
            intersection = list(intersection)
            union = list(union)
            intersection_ri_alli = []
            intersection_ri_allj = []
            union_ri_alli = []
            union_ri_allj = []
            g[i][j] = 0
            if len(intersection) == 0:
                g[i][j] = 0.0001
                break
            else:
                for k in range(len(intersection)):
                    intersection_ri_alli.append(ri_all[i][ri_index[i].tolist().index(intersection[k])])
                    intersection_ri_allj.append(ri_all[j][ri_index[j].tolist().index(intersection[k])])
                union_rest = set(union).difference(set(intersection))
                union_rest = list(union_rest)
                if len(union_rest) == 0:
                    g[i][j] = 0.0001
                    break
                else:
                    for k in range(len(union_rest)):
                        if union_rest[k] in ri_index[i]:
                            union_ri_alli.append(ri_all[i][ri_index[i].tolist().index(union_rest[k])])
                        else:
                            union_ri_allj.append(ri_all[j][ri_index[j].tolist().index(union_rest[k])])
                k_max = max(intersection_ri_allj, intersection_ri_alli)
                k_min = min(intersection_ri_allj, intersection_ri_alli)
                union_ri_allj = k_max + union_ri_allj
                union_num = np.sum(np.array(union_ri_allj), axis=0)
                inter_num = np.sum(np.array(k_min), axis=0)
                g[i][j] = inter_num / union_num
    return g


def load_data(dataset_str, sparse):
    """Load data."""
    # x: scipy.sparse.csr.csr_matrix训练集特征
    # y: numpy.array独热训练集标签
    # tx: 测特，tx和allx拼起来作为feature
    # ty: 测标
    # allx: 有label+无label的特征，是x的超集
    # ally: allx的特征，没有标签的数据的y值:[0,0,0,0,0,0,0]
    # graph: 图，格式如{index: [index of neighbour nodes]}
    # test: 测试集id
    if dataset_str == 'citeseer':
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/{}/ind.{}.{}".format(dataset_str, dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("data/{}/ind.{}.test.index".format(dataset_str, dataset_str))
        test_idx_range = np.sort(test_idx_reorder)


        # citeseer测试数据集中有一些孤立的点，在test.index中没有对应的索引，这部分孤立点特征和标签设置为全0，得到新的tx和ty
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), tx.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), ty.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()  # allx+tx是整个feature
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        nclass = labels.shape[1]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

    else:  # FB15K237, WN18RR
        idx_features_labels = np.genfromtxt("./data/{}/{}.content".format(dataset_str, dataset_str), dtype=np.dtype(str))
        features = np.array(idx_features_labels[:, 2:-1], dtype=np.float32)

        labels = list(map(lambda x: x.split(','), idx_features_labels[:, -1]))
        labels, nclass = encode_onehot(labels)

        idx = np.array(idx_features_labels[:, 1], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("./data/{}/{}.cites".format(dataset_str, dataset_str), dtype=np.int32)
        # 将每组关系中的entity用索引表示
        edges = np.array(list(map(idx_map.get, edges_unordered[:, :2].flatten())), dtype=np.int32).reshape(
            edges_unordered[:, :2].shape)
        # 构建图的邻接矩阵，用坐标形式的稀疏矩阵表示，非对称邻接矩阵
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

        # build symmetric adjacency matrix
        # 将非对称邻接矩阵转变为对称邻接矩阵，有向图变为无向图
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        idx_train = range(len(idx_map) // 10 * 8)
        idx_val = range(len(idx_map) // 10 * 8, len(idx_map) // 10 * 9)
        idx_test = range(len(idx_map) // 10 * 9, len(idx_map))

    adj = adj.astype(np.float32)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = torch.FloatTensor(np.array(adj.todense()))
    adj_delta = adj

    features = preprocess_features(features, dataset_str)
    # caculate n-hop neighbors
    if os.path.exists('data/{}/dijskra_{}.pkl'.format(dataset_str, dataset_str)):
        fw = open('data/{}/dijskra_{}.pkl'.format(dataset_str, dataset_str), 'rb')
        adj_delta = pickle.load(fw)
        fw.close()
    else:
        G = nx.DiGraph()  # 创建有向图
        # 原代码inf = pickle.load(open('data/citeseer/adj_citeseer.pkl', 'rb')),for i in range(len(inf)):，for j in range(len(inf[i])):,G.add_edge(i, inf[i][j], weight=1)没有文件，应当是图的连通性文件，
        if dataset_str == 'citeseer':
            for i in graph.keys():
                for j in graph[i]:
                    G.add_edge(i, j, weight=1)
        else:
            for i in edges:
                G.add_edge(i[0], i[1], weight=1)
        for i in range(features.shape[0]):  # 原代码缩进有问题
            for j in range(features.shape[0]):
                try:
                    rs = nx.astar_path_length(G, i, j)  # A星算法求两点距离
                except nx.NetworkXNoPath:
                    rs = 0
                if rs == 0:
                    length = 0
                else:
                    # print(rs)
                    length = rs  # 原代码为length = len(rs)
                adj_delta[i][j] = length  # adj_delta为任意两点间最短距离，不连通则为0
        a = open("data/{}/dijskra_{}.pkl".format(dataset_str, dataset_str), 'wb')  # 写入
        pickle.dump(adj_delta, a)

    if not sparse:  # adsf
        fw = open('data/{}/ri_index_c_0.5_{}_highorder_1_x_abs.pkl'.format(dataset_str, dataset_str), 'rb')
        ri_index = pickle.load(fw)
        fw.close()

        fw = open('data/{}/ri_all_c_0.5_{}_highorder_1_x_abs.pkl'.format(dataset_str, dataset_str), 'rb')
        ri_all = pickle.load(fw)
        fw.close()
        # Evaluate structural interaction between the structural fingerprints of node i and j
        adj_delta = structural_interaction(ri_index, ri_all, adj_delta)  # 构建结构信息

    features = np.array(features, dtype=np.float32)
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return adj, features, idx_train, idx_val, idx_test, labels, nclass, adj_delta


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = (1 / np.power(rowsum, 0.5)).flatten()  # 原代码为r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def preprocess_features(mx, dataset_str):  # 将features按行归一化，features为coo稀疏矩阵格式
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(mx.sum(1), dtype=np.float32)  # 每行求和
    r_inv = np.power(rowsum, -1).flatten()  # 求倒数
    r_inv[np.isinf(r_inv)] = 0.  # 处理nan
    r_mat_inv = sp.diags(r_inv)  # 构建稀疏的对角矩阵
    mx = r_mat_inv.dot(mx)
    if dataset_str == 'citeseer':
        mx = mx.todense()
    return mx


def accuracy(output, labels, is_cuda):
    label_1_num = torch.sum(labels, dim=1, dtype=torch.int8)  # labels每行1的个数
    cnt = labels.sum().item()  # labels所有1的个数
    indices = torch.sort(output, descending=True)[1]  # output按行排序，得到下标
    preds = torch.zeros_like(labels, dtype=torch.int8)
    if is_cuda:
        preds = preds.cuda()
    for i in range(label_1_num.shape[0]):
        predict_1_index = indices[i][:label_1_num[i]]
        preds[i][predict_1_index] = 1
    correct = preds.type_as(labels).mul(labels).sum()
    return correct.item() / cnt, preds


def multi_labels_nll_loss(output, labels):
    # labels和output按位点乘，结果相加，除以labels中1的总数，作为适用于多标签的nll_loss。
    loss = -labels.type_as(output).mul(output).sum()
    n = labels.sum().item()
    return loss / n
