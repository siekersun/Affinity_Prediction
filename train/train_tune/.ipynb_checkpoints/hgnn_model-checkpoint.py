import torch
import torch.nn as nn


class CrossLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.w = nn.Parameter(torch.empty(int(in_dim), int(out_dim)))
        self.b = nn.Parameter(torch.zeros(int(out_dim)))
        
        # 统一初始化
        nn.init.xavier_uniform_(self.w)
        nn.init.zeros_(self.b)          # 也可

    def forward(self, x0, xl):
        return x0 * (xl @self.w + self.b) + xl

class DCNLayer(nn.Module):
    def __init__(self, input_dim, out_dim, num_layers=3):
        super().__init__()
        self.cross_layers = nn.ModuleList()
        self.cross_layers.append(CrossLayer(input_dim, out_dim))
        for _ in range(num_layers-1):
            self.cross_layers.append(CrossLayer(out_dim, out_dim))

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 2*out_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2*out_dim, out_dim)
        )

    def forward(self, x):
        x0 = x
        xl = x
        for layer in self.cross_layers:
            xl = layer(x0, xl)
        x_mlp = self.mlp(x0)
        return xl, x_mlp


# 超图卷积层定义
class HypergraphConv(nn.Module):
    """原始超图卷积层"""
    def __init__(self, in_feats, out_feats, e_length  = 500, use_bias=True):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.theta = nn.Linear(in_feats, out_feats, bias=use_bias)
        self.W = nn.Parameter(torch.ones(e_length))
        self.act = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.5)


    def build_normalization(self, H, W=None):
        """构建归一化系数"""
        N, E = H.shape
        device = H.device
        dtype = H.dtype
        if W is None:
            # W = torch.ones(E, dtype=dtype, device=device)
            W = self.W.to(dtype)
        else:
            W = W.to(device).to(dtype).reshape(E)
        
        # 超边度
        De = H.sum(dim=0)
        De_inv = torch.zeros_like(De)
        mask_e = De > 0
        De_inv[mask_e] = 1.0 / De[mask_e]

        # 节点度
        Dv = H.sum(dim=1)
        Dv_inv_sqrt = torch.zeros_like(Dv)
        mask_v = Dv > 0
        Dv_inv_sqrt[mask_v] = 1.0 / torch.sqrt(Dv[mask_v])

        return W, De_inv, Dv_inv_sqrt
    
    def forward(self, X, H, combin_H, W=None):
        """前向传播"""

        W_vec, De_inv, Dv_inv_sqrt = self.build_normalization(H, W)
        
        X_theta = self.theta(X)
        X_hat = Dv_inv_sqrt.unsqueeze(1) * X_theta
        HtX = H.T @ X_hat
        edge_scale = (W_vec * De_inv).unsqueeze(1)
        HtX = edge_scale * HtX
        out = H @ HtX
        out = Dv_inv_sqrt.unsqueeze(1) * out
        
        out = self.act(out)
        out = self.dropout(out)
        
        return out

# HGNN层定义
class HGNNLayer(nn.Module):
    def __init__(self, embed_dim, out_dim, layer_num=2, e_length=500):
        super(HGNNLayer, self).__init__()
        self.layer_num = layer_num
        self.conv = nn.ModuleList()
        self.conv.append(HypergraphConv(embed_dim, out_dim, e_length=e_length))
        for _ in range(layer_num-1):
            self.conv.append(HypergraphConv(out_dim, out_dim, e_length=e_length))
        
        self.res = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
        )
    
    def forward(self, x, H, combin_H):
        resx = x
        for i in range(self.layer_num):
            x = self.conv[i](x, H, combin_H)
        x = self.res(resx) + x
        return x

# HGNN模型定义
class HGNN(nn.Module):
    """用于微调的HGNN模型"""
    def __init__(self, in_dim=200, hidden_dim=400, out_dim=1, nlayers=1, e_length=300, dp=0.5):
        super().__init__()
        
        self.hgnn_encoder = nn.ModuleList()
        for i in range(nlayers):
            if i == 0:
                self.hgnn_encoder.append(HGNNLayer(in_dim, hidden_dim, layer_num=2, e_length=e_length))
            else:
                self.hgnn_encoder.append(HGNNLayer(hidden_dim, hidden_dim, layer_num=2, e_length=e_length))

        # self.combin_H = nn.Parameter(torch.full((1024, e_length), 1e-6))
        # self.combin_H = nn.Parameter(torch.eye(e_length))
        self.combin_H = nn.Linear(253, e_length)
        self.actH = nn.ReLU()

        self.dcn = DCNLayer(int(e_length / 2), int(e_length / 2))

        self.ln = nn.LayerNorm(253)
        
        self.emb = nn.Parameter(torch.empty(7924, int(e_length / 2)))

        self.predict = nn.Linear(hidden_dim, out_dim)

        nn.init.xavier_uniform_(self.emb)
    
    def forward(self, x, bit_fp, Chem_fp):

        # H = torch.cat([bit_fp, count_fp], dim=1)
        H = Chem_fp
        H = H@self.emb
        H, H_mlp = self.dcn(H)
        H = torch.cat([H, H_mlp], dim=1)
        # H = H @ self.combin_H
        # H = self.actH(H @ self.combin_H)
        # H = self.combin_H(H)

        for layer in self.hgnn_encoder:
            x = layer(x, H, self.combin_H)
            
        out = self.predict(x)
        return out