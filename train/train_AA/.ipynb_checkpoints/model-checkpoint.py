import torch
import torch.nn as nn
# from .HGNN_ori import HypergraphConv
from .HGNN import HypergraphConv



# =========================
# 3️⃣ 模型定义 (修改: 移除 smiles_f 相关输入和层)
# =========================
class TransformerRegressor(nn.Module):
    def __init__(self, emb_dim, hidden_dim=256, nhead=8, nlayers=1):
        super().__init__()
        
        self.fc = nn.Sequential(
            # nn.Linear(emb_dim, 512),
            # nn.LeakyReLU(0.1),
            # nn.Dropout(0.3),
            # nn.Linear(512, hidden_dim),
            nn.Linear(emb_dim, hidden_dim),
            # nn.Dropout(0.3),
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        self.fc_smiles = nn.Sequential(
            nn.Linear(1042, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3), # 添加 Dropout 以防过拟合
            nn.Linear(512, hidden_dim),
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.hgnn = nn.ModuleList()
        for _ in range(1):
            self.hgnn.append(HypergraphConv(hidden_dim, hidden_dim))
        
        self.res_fc = nn.Linear(hidden_dim, hidden_dim)

        # 预测层
        self.predict = nn.Linear(hidden_dim, 1)

    def forward(self, x, smiles_x, mask, e_x, De, i): # 移除 smiles_f
        # 创建 key_padding_mask (Transformer 需要 True/False 的布尔掩码)
        key_padding_mask = (mask == 0) 
        
        # Transformer 编码
        out = self.encoder(self.fc(x), src_key_padding_mask=key_padding_mask)
        
        pooled = (out * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        # pooled = self.ln1(pooled)
        # HGNN 部分
        H = smiles_x[:,-1024:]

        # pooled = torch.mean(x, dim=1)

        all_x = pooled
        x = all_x
        for conv in self.hgnn:
            x, e_x, De = conv(x, H, e_x, De, i)
            # x = conv(x, H)
        x = self.res_fc(all_x) + x

        # 预测
        y = self.predict(x).squeeze(-1)
        return y, e_x,De