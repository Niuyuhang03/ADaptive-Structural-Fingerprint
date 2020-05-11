import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import StructuralFingerprintLayer
from rwr_process import RWRLayer


class ADSF(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, adj_ad, adj):
        """version of ADSF."""
        super(ADSF, self).__init__()
        self.dropout = dropout
        self.attentions = [StructuralFingerprintLayer(nfeat, nhid, dropout=dropout, alpha=alpha, adj_ad=adj_ad, adj=adj, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)  # 按attention_i名使用layer，似乎未用到
        self.out_att = StructuralFingerprintLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, adj_ad=adj_ad, adj=adj, concat=False)

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x))
        return F.log_softmax(x, dim=1)


class RWR_process(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, adj_ad, adj, dataset_str):
        """version of RWR_process."""
        super(RWR_process, self).__init__()
        self.dropout = dropout
        self.attentions = [RWRLayer(nfeat, nhid, dropout=dropout, alpha=alpha, adj_ad=adj_ad, adj=adj, dataset_str=dataset_str, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = RWRLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, adj_ad=adj_ad, adj=adj, dataset_str=dataset_str, concat=False)

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x))
        return F.log_softmax(x, dim=1)
