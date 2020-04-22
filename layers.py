import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
np.set_printoptions(threshold=np.inf)


class StructuralFingerprintLayer(nn.Module):
    """
    adaptive structural fingerprint layer
    """

    def __init__(self, in_features, out_features, dropout, alpha, adj_ad, concat=True):
        super(StructuralFingerprintLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.adj_ad = adj_ad
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # 均匀分布
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.W_si = nn.Parameter(torch.zeros(size=(1, 1)))
        nn.init.xavier_uniform_(self.W_si.data, gain=1.414)
        self.W_ei = nn.Parameter(torch.zeros(size=(1, 1)))
        nn.init.xavier_uniform_(self.W_ei.data, gain=1.414)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)  # h * w
        N = h.size()[0]
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # leakyrelu(h * w * a)，即leakyrelu的eij
        s = self.adj_ad  # sij
        e = e.cuda()
        s = s.cuda()

        # combine sij and eij
        e = abs(self.W_ei) * e + abs(self.W_si) * s  # aij=前半部分+后半部分（均未softmax）

        zero_vec = -9e15 * torch.ones_like(e)
        k_vec = -9e15 * torch.ones_like(e)
        adj = adj.cuda()  # adj为图连通性
        np.set_printoptions(threshold=np.inf)
        attention = torch.where(adj > 0, e, zero_vec)  # 第一个参数是条件，第二个参数是满足时的值，第三个参数时不满足时的值
        attention = F.softmax(attention, dim=1)  # alpha
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)  # h=alpha * W * h
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
