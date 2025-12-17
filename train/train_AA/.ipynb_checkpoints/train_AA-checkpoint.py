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
import h5py
import torch.nn as nn
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from .model import TransformerRegressor
from .AA_dataloader import get_dataloaders

print("Imports loaded")

# =========================
# 1️⃣ 数据加载与处理 (修改)
# =========================
train_csv_path = '/nas1/develop/lixiang/data/split_train_test/train_data.csv'
valid_csv_path = '/nas1/develop/lixiang/data/split_train_test/test_data_numW1.csv'
test_csv_path = '/nas1/develop/lixiang/data/split_train_test/test_data_numW2.csv'
h5_file = '/nas1/develop/lixiang/data/class_AA/smiles_embed.h5'

train_csv_path = '/nas1/develop/lixiang/data/split_train_test/train_data.csv'
valid_csv_path = '/nas1/develop/lixiang/data/split_train_test/test_data_numW1.csv'
test_csv_path = '/nas1/develop/lixiang/data/split_train_test/test_data_numW2.csv'
h5_file = '/nas1/develop/lixiang/data/class_AA/smiles_embed.h5'
GPepT_emb_path = '../GPepT_emb.csv'


train_loader, valid_loader, test_loader = get_dataloaders(
    train_csv_path,
    valid_csv_path,
    test_csv_path,
    h5_file,
    GPepT_emb_path,
    mix=False
)

print("DataLoaders ready")

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
# GPepT embedding dimension is 1280
model = TransformerRegressor(emb_dim=1280).to(device) 
criterion = nn.MSELoss()
lr =  5e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=20,      # 一个周期长度
    T_mult=1,     # 周期倍增（你可以不要倍增就写 1）
    eta_min=1e-6  # 最低学习率
)

# =========================
# 4️⃣ 训练 + 早停 (修改: 移除 smiles_f)
# =========================
EPOCH = 100
best_val_loss = float('inf')
save_path = 'train_AA/save/best_transformer_AAmodel.pth'
patience = 20
counter = 0
best_e_x = None
De = torch.zeros(1024).to(device)

for epoch in range(EPOCH):
    print(f"\nEpoch {epoch+1}/{EPOCH}")

    # ---- Training ----
    model.train()
    total_loss = 0
    i=epoch
    if epoch==0:
        e_x_in = None
    else:
        e_x_in = e_x
    e_x = e_x_in
    # 注意：collate_fn 现在只返回 x, mask, y
    for x, smiles_f, mask, y in tqdm(train_loader, desc="Training"): 
        x, smiles_f, mask, y = x.to(device), smiles_f.to(device), mask.to(device), y.to(device)
        optimizer.zero_grad()
        # 注意：model.forward 现在只接收 x, mask
        if e_x is None:
            pred, e_x, De = model(x, smiles_f, mask, None, De, epoch )
        else:
            pred, e_x, De = model(x, smiles_f, mask, e_x.detach(), De.detach(), epoch) 
        loss = criterion(pred, y.view(-1))
        # if epoch!=0:
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    avg_train_loss = total_loss / len(train_loader)
    print(f"Train Loss: {avg_train_loss:.4f}")

    # ---- Validation ----
    model.eval()
    val_loss = 0
    with torch.no_grad():
        # 注意：collate_fn 现在只返回 x, mask, y
        for x, smiles_f, mask, y in tqdm(valid_loader, desc="Validation"): 
            x, smiles_f, mask, y = x.to(device), smiles_f.to(device), mask.to(device), y.to(device)
            # 注意：model.forward 现在只接收 x, mask
            pred, e_x, De = model(x, smiles_f, mask, e_x.detach(), De.detach(), i) 
            val_loss += criterion(pred, y.view(-1)).item()
    avg_val_loss = val_loss / len(valid_loader)
    print(f"Valid Loss: {avg_val_loss:.4f}")

    # ---- Early stopping ----
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        if e_x_in is not None:
            best_e_x = e_x_in.detach()
            np.save('train_AA/save/best_e_x_AA.npy', best_e_x.cpu().numpy())
        torch.save(model.state_dict(), save_path)
        print(f"Saved best model at epoch {epoch+1} with val loss {best_val_loss:.4f}")
        counter = 0
    else:
        counter += 1
        print(f"No improvement for {counter} epoch(s)")
        if counter == patience//2:
            e_x_in = None  # 重置 e_x_in 以尝试跳出局部最优
        if counter >= patience:
             print(f"Early stopping triggered after {counter} epochs")
             break

# =========================
# 5️⃣ 测试 & 评估指标 (修改: 移除 smiles_f)
# =========================
model.load_state_dict(torch.load(save_path, map_location=device))
model.eval()
e_x = torch.from_numpy(np.load('train_AA/save/best_e_x_AA.npy')).to(device)

preds, trues = [], []

def evaluate(loader, name):
    preds, trues = [], []
    with torch.no_grad():
        for x, smiles_f, mask, y in tqdm(loader, desc=f"Evaluating {name}"):
            x, smiles_f, mask = x.to(device), smiles_f.to(device), mask.to(device)
            pred, _, _ = model(x, smiles_f, mask, e_x, 0, 0) 
            pred = pred.cpu().numpy()
            preds.append(pred)
            trues.append(y.numpy())

    preds = np.concatenate(preds).ravel()
    trues = np.concatenate(trues).ravel()

    rmse = np.sqrt(mean_squared_error(trues, preds))
    mae  = mean_absolute_error(trues, preds)
    r2   = r2_score(trues, preds)
    rho_s, _ = stats.spearmanr(trues, preds)
    rho_p, _ = stats.pearsonr(trues, preds)

    print(f"\n===== {name} Performance =====")
    print("RMSE   :", rmse)
    print("MAE    :", mae)
    print("R2     :", r2)
    print("Spearman:", rho_s)
    print("Pearson :", rho_p)

    return trues, preds, rho_p

tr_train, pr_train, r_train = evaluate(train_loader, "Train")
tr_valid, pr_valid, r_valid = evaluate(valid_loader, "Validation")
tr_test,  pr_test,  r_test  = evaluate(test_loader, "Test")

# =========================
# 6️⃣ 画 Pearson 图（测试集）
# =========================
k, b = np.polyfit(tr_test, pr_test, 1)
x_min, x_max = min(tr_test), max(tr_test)
y_min, y_max = min(pr_test), max(pr_test)

# 画散点
plt.figure(figsize=(6, 6))
plt.scatter(tr_test, pr_test, alpha=0.6)

# 画拟合直线
x_line = np.linspace(x_min, x_max, 100)
plt.plot(x_line, k * x_line + b, linewidth=2)

# 设置坐标轴范围
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title(f"Pearson Scatter Plot (Test)\nPearson r = {r_test:.4f}")
plt.grid(True)
plt.tight_layout()
plt.savefig("pearson_scatter_test_AA.png", dpi=300)