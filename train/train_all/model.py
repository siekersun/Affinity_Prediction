
from .HGNN import HypergraphConv
import torch
import torch.nn as nn


class HGNNLayer(nn.Module):
    def __init__(self, embed_dim, out_dim, layer_num = 1):
        super(HGNNLayer, self).__init__()
        self.layer_num = layer_num
        self.conv = nn.ModuleList()
        for _ in range(layer_num):
            self.conv.append(HypergraphConv(embed_dim, out_dim))
        
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
        max_dim = 2*out_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, max_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dp),
            nn.Linear(max_dim, out_dim),
            # nn.LeakyReLU(0.1),
            # nn.Dropout(dp),
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
            # nn.Linear(out_dim, out_dim)
        )
        self.resfc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)+ self.resfc(x)
# =========================
# 3️⃣ 模型定义
# =========================
class TransformerRegressor(nn.Module):
    def __init__(self, mon_dim=128, Chem_emb=768, bit_emb=1024, count_emb=253, hidden_dim=128, nhead=4, n_trans=1, nfc=10, nlayers=3, dp=0.5):
        super().__init__()

        self.nlayers = nlayers
        self.nfc = nfc

        self.ln_mon = nn.LayerNorm(mon_dim)
        self.ln_Chem = nn.LayerNorm(Chem_emb)
        self.ln_count = nn.LayerNorm(count_emb)

        self.x_fc = nn.Sequential(nn.Linear(Chem_emb, 1*hidden_dim),)
        self.smiles_fc = nn.Sequential(nn.Linear(count_emb, 1*hidden_dim),)
        self.trans_res = nn.Sequential(nn.Linear(mon_dim, mon_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=mon_dim, nhead=nhead, batch_first=True)
        self.trans_encoder = nn.ModuleList()
        self.trans_encoder.append(nn.TransformerEncoder(encoder_layer, num_layers=n_trans))
        for _ in range(self.nfc-1):
            self.trans_encoder.append(FCLayer(1*hidden_dim, 1*hidden_dim, dp=dp))

        self.x_encoder = nn.ModuleList()
        self.x_encoder.append(FCLayer(Chem_emb, hidden_dim))
        for _ in range(self.nfc-1):
            self.x_encoder.append(FCLayer(1*hidden_dim, 1*hidden_dim, dp=dp))
        
        self.smiles_encoder = nn.ModuleList()
        self.smiles_encoder.append(FCLayer(count_emb, hidden_dim))
        for _ in range(self.nfc-1):
            self.smiles_encoder.append(FCLayer(1*hidden_dim, 1*hidden_dim, dp=dp))

        # self.interact_encoder = nn.ModuleList()
        # # self.interact_encoder.append(FCLayer(253, hidden_dim))
        # for _ in range(self.nfc):
        #     self.interact_encoder.append(FCLayer(1*hidden_dim, 1*hidden_dim, dp=dp))

        self.hgnn_encoder = nn.ModuleList()
        for _ in range(nlayers):
            self.hgnn_encoder.append(HGNNLayer(3*hidden_dim, 3*hidden_dim, layer_num=nlayers))
        

        self.res_fc2 = nn.Sequential(
            nn.Linear(3*hidden_dim, 3*hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dp))
        
        self.fc = nn.ModuleList()
        for _ in range(self.nfc):
            self.fc.append(FCLayer(3*hidden_dim, 3*hidden_dim, dp=dp))

        # 预测层
        # self.predict = nn.Linear(3*hidden_dim, 1)
        self.predict = nn.Linear(2*hidden_dim + mon_dim, 1)

        

    def forward(self, mon_emb, bit_fp, count_fp, Chem_emb, mask): 
        H = bit_fp

        # mon_emb = self.ln_mon(mon_emb)
        # count_fp = self.ln_count(count_fp)
        # Chem_emb = self.ln_Chem(Chem_emb)

        key_padding_mask = (mask == 0) 
        
        # Transformer 语义 编码
        out = self.trans_encoder[0](mon_emb, src_key_padding_mask=key_padding_mask) + self.trans_res(mon_emb)
        trans_emb = (out * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)

        # 前馈 结构及字符 编码
        # x_emb = self.x_fc(Chem_emb)
        # smiles_emb = self.smiles_fc(count_fp)
        x_emb = Chem_emb
        smiles_emb = count_fp
        # interact = smiles_emb*x_emb
        for i in range(self.nfc):
            if i > 0:
                trans_emb = self.trans_encoder[i](trans_emb)
            x_emb = self.x_encoder[i](x_emb)
            smiles_emb = self.smiles_encoder[i](smiles_emb)
            # interact = self.interact_encoder[i]((interact + x_emb*smiles_emb)/2)


        all_x = torch.concat([x_emb, smiles_emb, trans_emb], dim=-1)

        # for i in range(self.nfc):
        #     all_x = self.fc[i](all_x)

        # all_res = all_x
        # for i in range(self.nlayers):
        #     all_x = self.hgnn_encoder[i](all_x, H)

        # all_x = self.res_fc2(all_res) + all_x
        # 预测
        
        y = self.predict(all_x)
        return y