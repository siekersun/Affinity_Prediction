# hypergraph_conv_example.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

class HypergraphConv(nn.Module):
    """
    超图卷积层（谱域实现）
    X' = Dv^{-1/2} * H * W * De^{-1} * H^T * Dv^{-1/2} * X * Theta
    使用稠密矩阵实现（适合小型/中型超图）
    """
    def __init__(self, in_feats, out_feats, use_bias=True):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.fc = nn.Sequential(
            nn.Linear(out_feats, out_feats, bias=use_bias),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            # nn.Linear(out_feats, out_feats, bias=use_bias),
            nn.Linear(out_feats, out_feats, bias=use_bias),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
        )
        self.act = nn.LeakyReLU(0.2)
        self.dp = nn.Dropout(0.3)
        self.De_linear_now = nn.Sequential(
            nn.Linear(out_feats, out_feats, bias=use_bias),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            # nn.Linear(out_feats, out_feats, bias=use_bias),
        )
        self.fuse_fc = nn.Sequential(
            nn.Linear(out_feats, out_feats, bias=use_bias),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            # nn.Linear(out_feats, out_feats, bias=use_bias),
        )
        # self.De_linear_next = nn.Sequential(
        #     nn.Linear(out_feats, out_feats, bias=use_bias),
        #     nn.LeakyReLU(0.1),
        #     nn.Dropout(0.3)
        # )

    @staticmethod
    def build_normalization(H, W=None):
        """
        构建归一化系数
        H: (N, E)
        W: (E,) 可选超边权重
        """
        N, E = H.shape
        device = H.device
        dtype = H.dtype

        # 超边权重
        if W is None:
            W = torch.ones(E, dtype=dtype, device=device)
        else:
            W = W.clone().to(device).to(dtype).reshape(E)

        # 超边度 De
        De = H.sum(dim=0)  # (E,)
        De_inv_now = torch.zeros_like(De) 
        mask_edge = De > 0
        De_inv_now[mask_edge] = 1.0 / De[mask_edge]  # 空超边置0

        # 节点度 Dv
        Dv = (H * W.unsqueeze(0)).sum(dim=1)  # (N,)
        Dv_inv_sqrt = torch.zeros_like(Dv)
        mask_node = Dv > 0
        Dv_inv_sqrt[mask_node] = 1.0 / torch.sqrt(Dv[mask_node])  # 空节点置0

        return W, De_inv_now, Dv_inv_sqrt

    def forward(self, X, H, e_x_pre, De_pre, i, W=None):
        """
        X: (N, F_in)
        H: (N, E) incidence matrix (0/1) or floats
        W: (E,) optional hyperedge weights
        returns: (N, F_out)
        """
        # prepare normalization terms
        W_vec, De_inv_now, Dv_inv_sqrt = self.build_normalization(H, W)
     
        edge_scale = W_vec * De_inv_now  # (E,)

        e_x_now = H.T @ (Dv_inv_sqrt.unsqueeze(1) * X)

        Prop = edge_scale.unsqueeze(1) * e_x_now

        e_x_now = self.De_linear_now(Prop)

        if e_x_pre is not None:
            e_x_now = self.fuse_fc(e_x_now + e_x_pre)

        # normalize with Dv^{-1/2}
        Prop = (Dv_inv_sqrt.unsqueeze(1) * H) @ e_x_now

        # # apply to features
        # support = Prop @ X  # (N, F_in)
        out = self.fc(Prop)  # (N, F_out)
        # out = self.act(out)
        # out = self.dp(out)
        return out, e_x_now, De_inv_now

