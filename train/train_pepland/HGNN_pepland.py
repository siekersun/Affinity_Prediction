import torch
import torch.nn as nn
import torch.nn.functional as F

class HypergraphConv(nn.Module):
    """
    原始超图卷积层（Zhou 2007）
    X' = Dv^{-1/2} H W De^{-1} H^T Dv^{-1/2} X Theta
    """

    def __init__(self, in_feats, out_feats, use_bias=True):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.theta = nn.Linear(in_feats, out_feats, bias=use_bias)
        self.act = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(out_feats, out_feats)

    @staticmethod
    def build_normalization(H, W=None):
        """
        构建归一化系数:
        H: (N, E)
        W: (E,) 超边权重，可为空
        """
        N, E = H.shape
        device = H.device
        dtype = H.dtype

        # 默认所有超边权重为 1
        if W is None:
            W = torch.ones(E, dtype=dtype, device=device)
        else:
            W = W.to(device).to(dtype).reshape(E)

        # De：超边度，每条超边包含多少节点
        De = H.sum(dim=0)  # (E,)
        De_inv = torch.zeros_like(De)
        mask_e = De > 0
        De_inv[mask_e] = 1.0 / De[mask_e]

        # Dv：节点度
        Dv = (H * W.unsqueeze(0)).sum(dim=1)  # (N,)
        Dv_inv_sqrt = torch.zeros_like(Dv)
        mask_v = Dv > 0
        Dv_inv_sqrt[mask_v] = 1.0 / torch.sqrt(Dv[mask_v])

        return W, De_inv, Dv_inv_sqrt

    def forward(self, X, H, W=None):
        """
        X: (N, F_in)
        H: (N, E)
        W: (E,)
        return: (N, F_out)
        """
        # 计算归一化
        W_vec, De_inv, Dv_inv_sqrt = self.build_normalization(H, W)

        # 线性变换 X Θ
        X_theta = self.theta(X)  # (N, F_out)

        # Dv^{-1/2} X Θ
        X_hat = Dv_inv_sqrt.unsqueeze(1) * X_theta

        # H^T * (Dv^{-1/2} X Θ)
        HtX = H.T @ X_hat  # (E, F_out)

        # W * De^{-1}
        edge_scale = (W_vec * De_inv).unsqueeze(1)  # (E, 1)

        # scale edges
        HtX = edge_scale * HtX  # (E, F_out)

        # H * (...)
        out = H @ HtX  # (N, F_out)

        # final left normalize: Dv^{-1/2}
        out = Dv_inv_sqrt.unsqueeze(1) * out

        out = self.act(out)
        out = self.dropout(out)
        out = self.fc(out)

        return out
