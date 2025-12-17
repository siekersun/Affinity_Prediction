
from .HGNN_pepland import HypergraphConv
import torch
import torch.nn as nn


class HGNNLayer(nn.Module):
    def __init__(self, embed_dim, out_dim, layer_num = 1):
        super(HGNNLayer, self).__init__()
        self.layer_num = layer_num
        self.conv = nn.ModuleList()
        for _ in range(layer_num):
            self.conv.append(HypergraphConv(embed_dim, embed_dim))
        
        self.res = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            # nn.LeakyReLU(0.1),
            # nn.Dropout(0.5),
        )

    def forward(self, x, H):
        resx = x
        for i in range(self.layer_num):
            x = self.conv[i](x, H)
        x = self.res(resx) + x
        return x

class FCLayer(nn.Module):
    def __init__(self, in_dim, out_dim, max_dim=200, dp=0.5):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, max_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dp),
            nn.Linear(max_dim, out_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dp),
            # nn.Linear(out_dim, max_dim),
            # nn.LeakyReLU(0.1),
            # nn.Dropout(dp),
            # nn.Linear(max_dim, out_dim),
            # nn.LeakyReLU(0.1),
            # nn.Dropout(dp),
            # nn.Linear(out_dim, max_dim),
            # nn.LeakyReLU(0.1),
            # nn.Dropout(dp),
            # nn.Linear(max_dim, out_dim),
            # nn.LeakyReLU(0.1),
            # nn.Dropout(dp),
            nn.Linear(out_dim, out_dim)
        )
        self.resfc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)+ self.resfc(x)
# =========================
# 3️⃣ 模型定义
# =========================
class TransformerRegressor(nn.Module):
    def __init__(self, emb_dim, hidden_dim=100, nhead=8, nfc=3, nlayers=1, dp=0.5):
        super().__init__()

        self.nlayers = nlayers
        self.nfc = nfc
        self.x_fc = nn.Sequential(
            nn.Linear(emb_dim, 1*hidden_dim),
        )

        self.x_encoder = nn.ModuleList()
        # self.x_encoder.append(FCLayer(emb_dim, hidden_dim))
        for _ in range(self.nfc):
            self.x_encoder.append(FCLayer(1*hidden_dim, 1*hidden_dim, dp=dp))
        
        self.smiles_encoder = nn.ModuleList()
        # self.smiles_encoder.append(FCLayer(253, hidden_dim))
        for _ in range(self.nfc):
            self.smiles_encoder.append(FCLayer(1*hidden_dim, 1*hidden_dim, dp=dp))

        self.interact_encoder = nn.ModuleList()
        # self.interact_encoder.append(FCLayer(253, hidden_dim))
        for _ in range(self.nfc):
            self.interact_encoder.append(FCLayer(1*hidden_dim, 1*hidden_dim, dp=dp))

        self.ln = nn.ModuleList()
        for _ in range(self.nfc):
            self.ln.append(nn.LayerNorm(hidden_dim))

        # self.hgnn_encoder = nn.ModuleList()
        # for _ in range(nlayers):
        #     self.hgnn_encoder.append(HGNNLayer(3*hidden_dim, 3*hidden_dim, layer_num=nlayers))
        
        self.res_x = nn.Linear(emb_dim, hidden_dim)
        self.res_smiles = nn.Linear(253, hidden_dim)
        self.res_fc2 = nn.Sequential(
            nn.Linear(3*hidden_dim, 3*hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dp))
        self.smiles_fc = nn.Sequential(
            nn.Linear(253, 1*hidden_dim),
        )
        
        self.fc = nn.ModuleList()
        for _ in range(self.nfc):
            self.fc.append(FCLayer(3*hidden_dim, 3*hidden_dim, dp=dp))

        # 预测层
        self.predict = nn.Linear(3*hidden_dim, 1)

        

    def forward(self, x, smiles_x ): 
        
        H = smiles_x
        smiles_emb = self.smiles_fc(smiles_x)

        x_emb = self.x_fc(x)

        interact = smiles_emb*x_emb

        # smiles_emb = smiles_x
        # x_emb = x
        for i in range(self.nfc):
            x_emb = self.x_encoder[i](x_emb)
            smiles_emb = self.smiles_encoder[i](smiles_emb)
            interact = self.interact_encoder[i](self.ln[i](interact + x_emb*smiles_emb))


        # all_x = torch.concat([x_emb, smiles_emb], dim=-1)
        all_x = torch.concat([interact, interact, interact], dim=-1)
        # all_x = interact

        # for i in range(self.nfc):
        #     all_x = self.fc[i](all_x)

        # all_res = all_x
        # for i in range(self.nlayers):
        #     all_x = self.hgnn_encoder[i](all_x, H)

        # all_x = self.res_fc2(all_res) + all_x
        # 预测
        
        y = self.predict(all_x).squeeze(-1)
        return y