import torch
import torch.nn as nn



class CrossLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn((int(in_dim), int(out_dim))))
        self.b = nn.Parameter(torch.zeros(out_dim))

        nn.init.xavier_uniform_(self.w)

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
            nn.Linear(input_dim, 2*input_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2*input_dim, input_dim)
        )

    def forward(self, x):
        x0 = x
        xl = x
        for layer in self.cross_layers:
            xl = layer(x0, xl)
        x_mlp = self.mlp(x0)
        return xl, x_mlp


# =========================
# 3️⃣ 模型定义
# =========================
class TransformerRegressor(nn.Module):
    def __init__(self, emb_dim, hidden_dim=100, nhead=8, nfc=15, nlayers=1, dp=0.5):
        super().__init__()

        self.nlayers = nlayers
        self.nfc = nfc
        self.x_fc = nn.Sequential(
            nn.Linear(emb_dim, 1*hidden_dim),
        )

        self.ln = nn.ModuleList()
        for _ in range(self.nfc):
            self.ln.append(nn.LayerNorm(hidden_dim))

        self.dcn_encoder = DCNLayer(hidden_dim, hidden_dim)

        # self.smiles_fc = nn.Sequential(
        #     nn.Linear(253, hidden_dim)
        # )
        self.smiles_fc = DCNLayer(253, hidden_dim)
    

        # 预测层
        self.predict = nn.Linear(2*hidden_dim, 1)

        

    def forward(self, x, smiles_x ): 

        # smiles_emb = self.smiles_fc(smiles_x)
        # smiles_emb, smiles_mlp = self.dcn_encoder(smiles_emb)

        smiles_emb, smiles_mlp = self.smiles_fc(smiles_x)
        
        x = torch.cat([smiles_mlp, smiles_emb], dim=-1)
        y = self.predict(x).squeeze(-1)
        return y