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
from train_pepland.model_pepland import TransformerRegressor
from train_pepland.pepland_dataloader import get_dataloaders

print("Imports loaded")

# =========================
# 1️⃣ 数据加载与处理 (修改)
# =========================
train_csv_path = '/nas1/develop/lixiang/data/split_train_test/train_data.csv'
valid_csv_path = '/nas1/develop/lixiang/data/split_train_test/test_data_numW1.csv'
test_csv_path = '/nas1/develop/lixiang/data/split_train_test/test_data_numW2.csv'
h5_file = '/nas1/develop/lixiang/data/class_AA/smiles_embed.h5'
pepland_path = '/nas1/develop/lixiang/data/class_AA/seq_peplandembed.h5'

loader = get_dataloaders(
    train_csv_path,
    valid_csv_path,
    test_csv_path,
    h5_file,
    pepland_path,
)

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
model = TransformerRegressor(emb_dim=300, hidden_dim=100).to(device)
criterion = nn.MSELoss()
lr = 1e-6
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = CyclicLR(
    optimizer,
    base_lr=lr,        # 学习率的最小值 (min_lr)
    max_lr=1e-5,       # 学习率的最大值 (max_lr)
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
save_path = 'train_pepland/best_transformer_model.pth'
patience = 10
counter = 0

for epoch in range(EPOCH):
    print(f"\nEpoch {epoch+1}/{EPOCH}")

    # -----------------------------
    # 1️⃣ Training phase
    # -----------------------------
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x, smiles_f, y, split_id in tqdm(loader, desc="Training"):
        x = x.to(device)
        smiles_f = smiles_f.to(device)
        y = y.to(device)
        split_id = split_id.to(device)

        mask_train = (split_id == 0)
        if mask_train.sum() == 0:
            continue

        optimizer.zero_grad()
        pred = model(x, smiles_f)
        loss = criterion(pred[mask_train], y[mask_train].view(-1))

        loss.backward()
        optimizer.step()

        # 累加总 loss 和训练样本数
        total_loss += loss.item() * mask_train.sum().item()
        total_samples += mask_train.sum().item()

    avg_train_loss = total_loss / total_samples
    print(f"Train Loss: {avg_train_loss:.4f}")

    # -----------------------------
    # 2️⃣ Validation phase
    # -----------------------------
    model.eval()
    val_loss = 0.0
    total_val_samples = 0

    with torch.no_grad():
        for x, smiles_f, y, split_id in tqdm(loader, desc="Validation"):
            x = x.to(device)
            smiles_f = smiles_f.to(device)
            y = y.to(device)
            split_id = split_id.to(device)

            # 验证 batch mask
            mask_valid = (split_id == 1)
            n_valid = mask_valid.sum().item()
            if n_valid == 0:
                continue

            pred = model(x, smiles_f)

            val_loss += criterion(pred[mask_valid], y[mask_valid].view(-1)).item() * n_valid
            total_val_samples += n_valid

    avg_val_loss = val_loss / total_val_samples
    print(f"Valid Loss: {avg_val_loss:.4f}")

    # -----------------------------
    # 3️⃣ Early stopping
    # -----------------------------
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
# Testing phase
# =========================
model.load_state_dict(torch.load(save_path))  # 加载最佳模型
model.eval()

test_loss = 0.0
total_test_samples = 0
all_pred = []
all_true = []

with torch.no_grad():
    for x, smiles_f, y, split_id in tqdm(loader, desc="Testing"):
        x = x.to(device)
        smiles_f = smiles_f.to(device)
        y = y.to(device)
        split_id = split_id.to(device)

        # 测试 batch mask
        mask_test = (split_id == 2)
        n_test = mask_test.sum().item()
        if n_test == 0:
            continue

        pred = model(x, smiles_f)

        # 计算测试损失
        test_loss += criterion(pred[mask_test], y[mask_test].view(-1)).item() * n_test
        total_test_samples += n_test

        # 保存测试预测和真实值
        all_pred.append(pred[mask_test].cpu())
        all_true.append(y[mask_test].cpu())

avg_test_loss = test_loss / total_test_samples
print(f"Test Loss: {avg_test_loss:.4f}")

# 拼接结果
preds = torch.cat(all_pred).numpy().ravel()
trues = torch.cat(all_true).numpy().ravel()

rmse = np.sqrt(mean_squared_error(trues, preds))
mae  = mean_absolute_error(trues, preds)
r2   = r2_score(trues, preds)
rho_s, _ = stats.spearmanr(trues, preds)
rho_p, _ = stats.pearsonr(trues, preds)

print(f"\n===== Performance =====")
print("RMSE   :", rmse)
print("MAE    :", mae)
print("R2     :", r2)
print("Spearman:", rho_s)
print("Pearson :", rho_p)



# =========================
# 6️⃣ 画 Pearson 图（测试集）
# =========================
def plot_pearson(trues, preds, rho, name, filename, point_size=15):
    k, b = np.polyfit(trues, preds, 1)
    x_min, x_max = min(trues-1), max(trues+1)
    y_min, y_max = min(preds-1), max(preds+1)

    plt.figure(figsize=(6, 6))
    # 散点
    plt.scatter(trues, preds, alpha=0.6, s=point_size)
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
plot_pearson(trues, preds, rho_p, "Test", "pearson_scatter_test.png", point_size=10)