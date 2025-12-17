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
# 1️⃣ 数据加载与处理
# =========================
csv_path = '/nas1/develop/lixiang/data/class_AA/PA20231102001_AA.csv'
print("Loading data...")
data = pd.read_csv(csv_path)

# 提取序列和标签
seq = data['PepetideID'].to_list()
seq = np.array(seq, dtype=object)
y = np.log(data.iloc[:, 1].astype('float32')).values.reshape(-1, 1)

del data  # 释放内存

# ---- 划分训练/验证/测试索引 ----
idx = np.arange(len(seq))

idx_train, idx_tmp, idx_y_train, idx_y_tmp = train_test_split(
    idx, y, test_size=0.2, random_state=42,
    stratify=pd.qcut(y.ravel(), 10, duplicates='drop')
)
idx_valid, idx_test, _, _ = train_test_split(
    idx_tmp, idx_y_tmp, test_size=0.5, random_state=42,
    stratify=pd.qcut(idx_y_tmp.ravel(), 10, duplicates='drop')
)

# 根据索引划分
seq_train, seq_valid, seq_test = seq[idx_train], seq[idx_valid], seq[idx_test]
y_train, y_valid, y_test = y[idx_train], y[idx_valid], y[idx_test]

print(f'Train: {seq_train.shape}, Valid: {seq_valid.shape}, Test: {seq_test.shape}')

# =========================
# 2️⃣ 数据集与 DataLoader
# =========================
class TensorDataset(Dataset):
    def __init__(self, seq, y, emb_path='../GPepT_emb.csv'):
        NATURAL_AA = set('ACDEFGHIKLMNPQRSTVWY')
        GPepT_emb = pd.read_csv(emb_path)
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
    return xs, mask, ys

BATCH = 1024
train_loader = DataLoader(TensorDataset(seq_train, y_train), batch_size=BATCH, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(TensorDataset(seq_valid, y_valid), batch_size=BATCH, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(TensorDataset(seq_test, y_test), batch_size=BATCH, shuffle=False, collate_fn=collate_fn)

print(f"Batch size: {BATCH}")

# =========================
# 3️⃣ 模型定义
# =========================
class TransformerRegressor(nn.Module):
    def __init__(self, emb_dim, hidden_dim=256, nhead=8, nlayers=3):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, mask):
        key_padding_mask = (mask == 0)
        out = self.encoder(x, src_key_padding_mask=key_padding_mask)
        pooled = (out * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        y = self.fc(pooled).squeeze(-1)
        return y

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
model = TransformerRegressor(1280).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-4)
scheduler = CyclicLR(
    optimizer,
    base_lr=1e-6,        # 学习率的最小值 (min_lr)
    max_lr=1e-4,       # 学习率的最大值 (max_lr)
    step_size_up=8, # 学习率从 min_lr 上升到 max_lr 所需的迭代次数
    mode='exp_range',   # 调度模式：'triangular' (三角波)
    gamma=0.95,
    cycle_momentum=False # 如果使用带有动量的优化器 (如 Adam)，通常设置为 False
)

# =========================
# 4️⃣ 训练 + 早停
# =========================
EPOCH = 64
best_val_loss = float('inf')
save_path = 'best_transformer_model.pth'
patience = 8
counter = 0

for epoch in range(EPOCH):
    print(f"\nEpoch {epoch+1}/{EPOCH}")

    # ---- Training ----
    model.train()
    total_loss = 0
    for x, mask, y in tqdm(train_loader, desc="Training"):
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x, mask)
        loss = criterion(pred, y.view(-1))
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    print(f"Train Loss: {avg_train_loss:.4f}")

    # ---- Validation ----
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, mask, y in tqdm(valid_loader, desc="Validation"):
            x, mask, y = x.to(device), mask.to(device), y.to(device)
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
        # if counter >= patience:
        #     print(f"Early stopping triggered after {counter} epochs")
        #     break

# =========================
# 5️⃣ 测试 & 评估指标
# =========================
model.load_state_dict(torch.load(save_path, map_location=device))
model.eval()

preds, trues = [], []

with torch.no_grad():
    for x, mask, y in tqdm(test_loader, desc="Testing"):
        x, mask = x.to(device), mask.to(device)
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

print("Test RMSE:", rmse)
print("Test MAE :", mae)
print("R2      :", r2)
print("Spearman:", rho_s)
print("Pearson :", rho_p)
