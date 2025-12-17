# =========================
# 0️⃣ 导入依赖
# =========================
# %matplotlib inline
import os
import ast
import json
import h5py
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
from .model_pepland import TransformerRegressor
from .pepland_dataloader import get_dataloaders

print("Imports loaded")

# =========================
# 1️⃣ 数据加载与处理 (修改)
# =========================
train_csv_path = '/nas1/develop/lixiang/data/split_train_test/train_data.csv'
valid_csv_path = '/nas1/develop/lixiang/data/split_train_test/test_data_numW1.csv'
test_csv_path = '/nas1/develop/lixiang/data/split_train_test/test_data_numW2.csv'
# test_csv_path = '/nas1/develop/lixiang/data/split_train_test/test_data_numW3plus.csv'
seq_smiles_fp_matrix_path = '/nas1/develop/lixiang/data/class_AA/seq_smiles_fp_matrix.npz'
# h5_file = '/nas1/develop/lixiang/data/class_AA/smiles_embed.h5'
# pepland_path = '/nas1/develop/lixiang/data/class_AA/seq_peplandembed.h5'
pepland_path = '/nas1/develop/lixiang/data/class_AA/ChemBERT_embed.h5'

train_loader, valid_loader, test_loader = get_dataloaders(
    train_csv_path,
    valid_csv_path,
    test_csv_path,
    seq_smiles_fp_matrix_path,
    pepland_path,
    mix=False
)



device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
model = TransformerRegressor(emb_dim=768, hidden_dim=100).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total params: {total_params}")
print(f"Trainable params: {trainable_params}")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = CyclicLR(
    optimizer,
    base_lr=1e-4,        # 学习率的最小值 (min_lr)
    max_lr=1e-3,       # 学习率的最大值 (max_lr)
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
save_path = 'train_pepland/save/best_model.pth'
patience = 10
counter = 0

for epoch in range(EPOCH):
    print(f"\nEpoch {epoch+1}/{EPOCH}")

    # ---- Training ----
    model.train()
    total_loss = 0
    for x, smiles_f, y in tqdm(train_loader, desc="Training"):
        x, smiles_f, y = x.to(device), smiles_f.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x, smiles_f)
        loss = criterion(pred, y.view(-1))
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
        for x, smiles_f, y in tqdm(valid_loader, desc="Validation"):
            x, smiles_f, y = x.to(device), smiles_f.to(device), y.to(device)
            pred = model(x, smiles_f)
            val_loss += criterion(pred, y.view(-1)).item()
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


model.load_state_dict(torch.load(save_path, map_location=device))
model.eval()


def evaluate(loader, name):
    preds, trues = [], []
    with torch.no_grad():
        for x, smiles_f, y in tqdm(loader, desc=f"Evaluating {name}"):
            x, smiles_f = x.to(device), smiles_f.to(device)
            pred = model(x, smiles_f).cpu().numpy()
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
tr_valid, pr_valid, r_valid = evaluate(valid_loader, "Valid")
tr_test,  pr_test,  r_test  = evaluate(test_loader, "Test")


# =========================
# 6️⃣ 画 Pearson 图（测试集）
# =========================
def plot_pearson(trues, preds, rho, name, filename, point_size=15):
    k, b = np.polyfit(trues, preds, 1)
    x_min, x_max = min(trues-1), max(trues+1)
    y_min, y_max = min(preds-1), max(preds+1)

    plt.figure(figsize=(6, 6))
    # 散点
    plt.scatter(trues[:1000], preds[:1000], alpha=0.6, s=point_size)
    # 拟合直线
    x_line = np.linspace(x_min, x_max, 100)
    plt.plot(x_line, k * x_line + b, linewidth=2, color='red')
    # 对角线（y=x）
    min_val = min(x_min, y_min)
    max_val = max(x_max, y_max)
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='gray', linewidth=1)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{name} Pearson Scatter Plot\nPearson r = {rho:.4f}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")


# =========================
# 画图
# =========================
plot_pearson(tr_train, pr_train, r_train, "Train", "pearson_scatter_train.png", point_size=10)
# plot_pearson(tr_valid, pr_valid, r_valid, "Valid", "pearson_scatter_valid.png", point_size=10)
# plot_pearson(tr_test, pr_test, r_test, "Test",  "pearson_scatter_test.png",  point_size=10)
