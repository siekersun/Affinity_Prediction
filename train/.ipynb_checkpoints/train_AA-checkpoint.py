# =========================
# 0️⃣ 导入依赖
# =========================
# %matplotlib inline
import os
import ast
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm

print("Imports loaded")

# =========================
# 1️⃣ 数据加载与处理 (修改)
# =========================
train_csv_path = '/nas1/develop/lixiang/data/split_train_test/train_data.csv'
test_csv_path = '/nas1/develop/lixiang/data/split_train_test/test_data.csv'

print("Loading training data...")
data_train = pd.read_csv(train_csv_path)

print("Loading testing data...")
data_test = pd.read_csv(test_csv_path)

# 提取序列和标签
seq_train_full = data_train['PepetideID'].to_list()
y_train_full = np.log(data_train.iloc[:, 1].astype('float32')).values.reshape(-1, 1)

seq_test = data_test['PepetideID'].to_list()
y_test = np.log(data_test.iloc[:, 1].astype('float32')).values.reshape(-1, 1)

del data_train, data_test # 释放内存

GPepT_emb = pd.read_csv('../GPepT_emb.csv')

print(f"Full Train data loaded: {len(seq_train_full)} samples")
print(f"Test data loaded: {len(seq_test)} samples")


# ---- 划分训练/验证索引 (从 full train data 中划分 1/10 作为验证集) ----
idx_full_train = np.arange(len(seq_train_full))

# 使用 stratify 确保划分均匀
idx_train, idx_valid, y_train, y_valid = train_test_split(
    idx_full_train, y_train_full, test_size=0.1, random_state=42, # 1/10 = 0.1
    stratify=pd.qcut(y_train_full.ravel(), 10, duplicates='drop')
)

# 根据索引划分
seq_train = np.array(seq_train_full, dtype=object)[idx_train]
seq_valid = np.array(seq_train_full, dtype=object)[idx_valid]

print(f'Train: {seq_train.shape}, Valid: {seq_valid.shape}, Test: {len(seq_test)}')


# =========================
# 2️⃣ 数据集与 DataLoader
# =========================
class TensorDataset(Dataset):
    def __init__(self, seq, y, GPepT_emb):
        NATURAL_AA = set('ACDEFGHIKLMNPQRSTVWY')
        
        self.mon_emb = {
            row['token']: np.array(ast.literal_eval(row['embedding']), dtype=float)
            for _, row in GPepT_emb.iterrows() if row['token'] in NATURAL_AA
        }
        self.seq = seq
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        seq_i = self.seq[idx]
        y_i = self.y[idx]
        x_emb = np.stack([self.mon_emb[aa] for aa in seq_i], axis=0)
        x_emb = torch.from_numpy(x_emb).float()
        return x_emb, y_i

def collate_fn(batch):
    xs, ys = zip(*batch)
    lengths = [x.size(0) for x in xs]
    max_len = max(lengths)

    # Padding
    padded = []
    for x in xs:
        pad_len = max_len - x.size(0)
        if pad_len > 0:
            x = torch.cat([x, torch.zeros(pad_len, x.size(1))], dim=0)
        padded.append(x)

    xs = torch.stack(padded)
    ys = torch.stack(ys)
    mask = torch.tensor([[1]*l + [0]*(max_len-l) for l in lengths])
    
    # 仅返回 peptide embedding, mask, 和 label
    return xs, mask, ys 

BATCH = 1024
# 注意：seq_test 和 y_test 在前面已经被处理成 list 和 numpy 数组，可以直接使用
train_loader = DataLoader(TensorDataset(seq_train, y_train, GPepT_emb), batch_size=BATCH, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(TensorDataset(seq_valid, y_valid, GPepT_emb), batch_size=BATCH, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(TensorDataset(seq_test, y_test, GPepT_emb), batch_size=BATCH, shuffle=False, collate_fn=collate_fn)

print(f"Batch size: {BATCH}")

# =========================
# 3️⃣ 模型定义 (修改: 移除 smiles_f 相关输入和层)
# =========================
class TransformerRegressor(nn.Module):
    def __init__(self, emb_dim, hidden_dim=256, nhead=8, nlayers=3):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        
        # 简化全连接层
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3), # 添加 Dropout 以防过拟合
        )

        # 预测层
        self.predict = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask): # 移除 smiles_f
        # 创建 key_padding_mask (Transformer 需要 True/False 的布尔掩码)
        key_padding_mask = (mask == 0) 
        
        # Transformer 编码
        out = self.encoder(x, src_key_padding_mask=key_padding_mask)
        
        # Masked Pooling (Sequence-to-vector 转换)
        # pooled = 序列中非零元素的和 / 序列中非零元素的数量
        pooled = (out * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)

        # 预测
        y = self.predict(self.fc(pooled)).squeeze(-1)
        return y

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
# GPepT embedding dimension is 1280
model = TransformerRegressor(emb_dim=1280).to(device) 
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-4)
scheduler = CyclicLR(
    optimizer,
    base_lr=1e-6, 
    max_lr=1e-5, 
    step_size_up=8, 
    mode='exp_range', 
    gamma=0.95,
    cycle_momentum=False 
)

# =========================
# 4️⃣ 训练 + 早停 (修改: 移除 smiles_f)
# =========================
EPOCH = 64
best_val_loss = float('inf')
save_path = 'best_transformer_AAmodel.pth'
patience = 8
counter = 0

for epoch in range(EPOCH):
    print(f"\nEpoch {epoch+1}/{EPOCH}")

    # ---- Training ----
    model.train()
    total_loss = 0
    # 注意：collate_fn 现在只返回 x, mask, y
    for x, mask, y in tqdm(train_loader, desc="Training"): 
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        optimizer.zero_grad()
        # 注意：model.forward 现在只接收 x, mask
        pred = model(x, mask) 
        loss = criterion(pred, y.view(-1))
        loss.backward()
        optimizer.step()
        # scheduler.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    print(f"Train Loss: {avg_train_loss:.4f}")

    # ---- Validation ----
    model.eval()
    val_loss = 0
    with torch.no_grad():
        # 注意：collate_fn 现在只返回 x, mask, y
        for x, mask, y in tqdm(valid_loader, desc="Validation"): 
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            # 注意：model.forward 现在只接收 x, mask
            pred = model(x, mask) 
            val_loss += criterion(pred, y.view(-1)).item()
    avg_val_loss = val_loss / len(valid_loader)
    print(f"Valid Loss: {avg_val_loss:.4f}")

    # ---- Early stopping ----
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), save_path)
        print(f"Saved best model at epoch {epoch+1} with val loss {best_val_loss:.4f}")
        counter = 0
    else:
        counter += 1
        print(f"No improvement for {counter} epoch(s)")
        if counter >= patience:
             print(f"Early stopping triggered after {counter} epochs")
             break

# =========================
# 5️⃣ 测试 & 评估指标 (修改: 移除 smiles_f)
# =========================
model.load_state_dict(torch.load(save_path, map_location=device))
model.eval()

preds, trues = [], []

with torch.no_grad():
    for x, mask, y in tqdm(test_loader, desc="Testing"):
        x, mask = x.to(device), mask.to(device)
        # 注意：model.forward 现在只接收 x, mask
        pred = model(x, mask).cpu().numpy()
        preds.append(pred)
        trues.append(y.numpy())

preds = np.concatenate(preds).ravel()
trues = np.concatenate(trues).ravel()

rmse = np.sqrt(mean_squared_error(trues, preds))
mae  = mean_absolute_error(trues, preds)
r2   = r2_score(trues, preds)
rho_s, _ = stats.spearmanr(trues, preds)
rho_p, _ = stats.pearsonr(trues, preds)

print("\n--- Test Set Performance ---")
print("Test RMSE:", rmse)
print("Test MAE :", mae)
print("R2      :", r2)
print("Spearman:", rho_s)
print("Pearson :", rho_p)