import torch
import torch.nn as nn
from .HGNN_ori import HypergraphConv
# from .HGNN import HypergraphConv



# =========================
# 3️⃣ 模型定义 (修改: 移除 smiles_f 相关输入和层)
# =========================
class TransformerRegressor(nn.Module):
    def __init__(self, emb_dim, hidden_dim=256, nhead=8, nlayers=2, dp=0.5):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            # nn.LeakyReLU(0.1),
            # nn.Dropout(0.3),
            # nn.Linear(512, hidden_dim),
            # nn.LeakyReLU(0.1),
            # nn.Dropout(0.6),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.LeakyReLU(0.1),
            # nn.Dropout(0.3),
            # nn.Linear(hidden_dim, hidden_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc_smiles = nn.Sequential(
            nn.Linear(253, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(dp),
            nn.Linear(512, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dp),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.xfc = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(dp),
            nn.Linear(512, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dp),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # self.hgnn = nn.ModuleList()
        # for _ in range(1):
        #     self.hgnn.append(HypergraphConv(2*hidden_dim, hidden_dim))
        
        # self.res_fc = nn.Linear(2*hidden_dim, hidden_dim)

        # 预测层
        self.predict = nn.Linear(2*hidden_dim, 1)

    def forward(self, x, smiles_x, mask, e_x, De, i, pool='mean'): # 移除 smiles_f
        # 创建 key_padding_mask (Transformer 需要 True/False 的布尔掩码)
        key_padding_mask = (mask == 0) 
        
        # Transformer 编码
        out = self.encoder(self.fc(x), src_key_padding_mask=key_padding_mask)
        # out = out + self.res_fc(x)
        if pool == 'mean':
            pooled = (out * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        elif pool == 'max':
            out_pad = out.masked_fill(mask.unsqueeze(-1) == 0, -1e9)  
            pooled = out_pad.max(dim=1)[0]                            
        # pooled = self.ln1(pooled)
        # HGNN 部分
        # H = smiles_x[:,-1024:]

        pooled = self.xfc(pooled)
        
        smiles_x = self.fc_smiles(smiles_x)
        H = smiles_x

        all_x = torch.concat([pooled, smiles_x], dim=-1)
        x = all_x
        # for conv in self.hgnn:
        #     # x, e_x, De = conv(x, H, e_x, De, i)
        #     x = conv(x, H)
        # x = self.res_fc(all_x) + x

        # 预测
        y = self.predict(x).squeeze(-1)
        return y, e_x,De