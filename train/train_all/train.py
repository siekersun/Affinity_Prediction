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
from .dataloader import get_dataloaders

print("Imports loaded")

# =========================
# 1️⃣ 数据加载与处理 (修改)
# =========================

train_csv_path = '/nas1/develop/lixiang/data/split_train_test/train_data.csv'
valid_csv_path = '/nas1/develop/lixiang/data/split_train_test/test_data_numW1.csv'
test_csv_path = '/nas1/develop/lixiang/data/split_train_test/test_data_numW2.csv'
bit_fp_file = '/nas1/develop/lixiang/data/class_AA/smiles_embed.h5'
count_fp_file = '/nas1/develop/lixiang/data/class_AA/seq_smiles_fp_matrix.npz'
GPepT_emb_path = '../GPepT_emb_64d.csv'
# GPepT_emb_path = '../GPepT_emb.csv'
ChemBERTa_path = '/nas1/develop/lixiang/data/class_AA/ChemBERT_embed.h5'

train_loader, valid_loader, test_loader = get_dataloaders(
    train_csv_path,
    valid_csv_path,
    test_csv_path,
    bit_fp_file,
    count_fp_file,
    GPepT_emb_path,
    ChemBERTa_path,
    mix=False
)

print("DataLoaders ready")

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
# GPepT embedding dimension is 1280
model = TransformerRegressor().to(device) 

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total params: {total_params}")
print(f"Trainable params: {trainable_params}")


criterion = nn.MSELoss()
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = CyclicLR(
    optimizer,
    base_lr=0.1*lr,
    max_lr=lr,
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
save_path = 'train_all/save/best_model.pth'
patience = 10
counter = 0

for epoch in range(EPOCH):
    print(f"\nEpoch {epoch+1}/{EPOCH}")

    # ---- Training ----
    model.train()
    total_loss = 0
    
    for mon_emb, bit_fp, count_fp, Chem_emb, mask, y in tqdm(train_loader, desc="Training"): 
        mon_emb, bit_fp, count_fp, Chem_emb, mask, y = mon_emb.to(device), bit_fp.to(device), count_fp.to(device), Chem_emb.to(device), mask.to(device), y.to(device)
        optimizer.zero_grad()
        
        pred= model(mon_emb, bit_fp, count_fp, Chem_emb, mask) 

        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    # scheduler.step()
    avg_train_loss = total_loss / len(train_loader)
    print(f"Train Loss: {avg_train_loss:.4f}")

    # ---- Validation ----
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for mon_emb, bit_fp, count_fp, Chem_emb, mask, y in tqdm(valid_loader, desc="Validation"): 
            mon_emb, bit_fp, count_fp, Chem_emb, mask, y = mon_emb.to(device), bit_fp.to(device), count_fp.to(device), Chem_emb.to(device), mask.to(device), y.to(device)
            
            pred= model(mon_emb, bit_fp, count_fp, Chem_emb, mask)  
            val_loss += criterion(pred, y).item()
    avg_val_loss = val_loss / len(valid_loader)
    print(f"Valid Loss: {avg_val_loss:.4f}")

    # ---- Early stopping ----
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), save_path)
        print(f"⭐ Saved best model at epoch {epoch+1} with val loss {best_val_loss:.4f}")
        counter = 0
    else:
        counter += 1
        print(f"No improvement for {counter} epoch(s)")
        if counter >= patience:
             print(f"⛔ Early stopping triggered after {counter} epochs")
             break

# =========================
# 5️⃣ 测试 & 评估指标 (修改: 移除 smiles_f)
# =========================
model.load_state_dict(torch.load(save_path, map_location=device))
model.eval()

preds, trues = [], []

def evaluate(loader, name):
    preds, trues = [], []
    with torch.no_grad():
        for mon_emb, bit_fp, count_fp, Chem_emb, mask, y in tqdm(loader, desc="Training"): 
            mon_emb, bit_fp, count_fp, Chem_emb, mask = mon_emb.to(device), bit_fp.to(device), count_fp.to(device), Chem_emb.to(device), mask.to(device)
            
            pred= model(mon_emb, bit_fp, count_fp, Chem_emb, mask)
            preds.append(pred.cpu().numpy())
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