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
smiles_fp = '/nas1/develop/lixiang/data/class_AA/seq_smiles_fp_matrix.npz'
# smiles_fp = '/nas1/develop/lixiang/data/class_AA/smiles_embed.h5'
# pepland_path = '/nas1/develop/lixiang/data/class_AA/seq_peplandembed.h5'
pepland_path = '/nas1/develop/lixiang/data/class_AA/ChemBERT_embed.h5'

loader = get_dataloaders(
    train_csv_path,
    valid_csv_path,
    test_csv_path,
    smiles_fp,
    pepland_path,
    mix=True
)

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
model = TransformerRegressor(emb_dim=768, hidden_dim=256).to(device)
criterion = nn.MSELoss()
lr = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = CyclicLR(
    optimizer,
    base_lr=lr,        # 学习率的最小值 (min_lr)
    max_lr=1e-2,       # 学习率的最大值 (max_lr)
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
save_path = 'train_pepland/save/best_transformer_model.pth'
patience = 10
counter = 0

for epoch in range(EPOCH):
    print(f"\nEpoch {epoch+1}/{EPOCH}")

    # -----------------------------
    # TRAIN MODE
    # -----------------------------
    model.train()
    train_loss_sum = 0.0
    train_samples = 0

    # -----------------------------
    # VALID MODE
    # -----------------------------
    val_loss_sum = 0.0
    val_samples = 0


    for x, smiles_f, y, split_id in tqdm(loader, desc="Training"):
        x = x.to(device)
        smiles_f = smiles_f.to(device)
        y = y.to(device)
        split_id = split_id.to(device)

        pred = model(x, smiles_f)

        mask_train = (split_id == 0)
        # if mask_train.any():
        #     model.train()
        optimizer.zero_grad()

        loss_train = criterion(pred[mask_train],y[mask_train].view(-1))
        loss_train.backward()
        optimizer.step()

        train_loss_sum += loss_train.item() * mask_train.sum().item()
        train_samples += mask_train.sum().item()

        # ------------------------------------------------------
        # 3) VALID
        # ------------------------------------------------------
        mask_valid = (split_id == 1)
        # if mask_valid.any():
        #     model.eval()
        #     with torch.no_grad():
        loss_valid = criterion(pred[mask_valid],y[mask_valid].view(-1))

        val_loss_sum += loss_valid.item() * mask_valid.sum().item()
        val_samples += mask_valid.sum().item()

    # scheduler.step()
    # ------------------------------------------------------
    # 4) 计算 epoch 平均 loss
    # ------------------------------------------------------
    avg_train_loss = train_loss_sum / (train_samples + 1e-8)
    avg_val_loss   = val_loss_sum / (val_samples + 1e-8)

    print(f"Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_val_loss:.4f}")

    # ------------------------------------------------------
    # 5) Early Stopping
    # ------------------------------------------------------
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), save_path)
        
        print(f"⭐ Saved best model at epoch {epoch+1}, val loss = {best_val_loss:.4f}")
        counter = 0
    else:
        counter += 1
        print(f"No improvement for {counter} epoch(s)")
        if counter >= patience:
            print(f"⛔ Early stopping triggered at epoch {epoch+1}")
            break

# =========================
# Testing phase
# =========================
model.load_state_dict(torch.load(save_path))  # 加载最佳模型
model.eval()

def evaluate_mega(loader, model, criterion, device, e_x_path=None):
    """
    一次 forward 完成 Train/Valid/Test 评估
    e_x_path: 训练时保存的 best_e_x，用于初始化
    """
    model.eval()
    all_pred = {0: [], 1: [], 2: []}  # train/valid/test
    all_true = {0: [], 1: [], 2: []}
    total_samples = {0: 0, 1: 0, 2: 0}
    losses = {0: 0.0, 1: 0.0, 2: 0.0}

    with torch.no_grad():
        for x, smiles_f, y, split_id in tqdm(loader, desc="Evaluation"):
            x = x.to(device)
            smiles_f = smiles_f.to(device)
            y = y.to(device)
            split_id = split_id.to(device)

            # forward 只跑一次
            pred = model(x, smiles_f)

            # 根据 split_id 分片
            for i in range(3):
                mask_i = (split_id == i)
                if mask_i.any():
                    n_samples = mask_i.sum().item()
                    total_samples[i] += n_samples
                    losses[i] += criterion(pred[mask_i], y[mask_i].view(-1)).item() * n_samples
                    all_pred[i].append(pred[mask_i].cpu())
                    all_true[i].append(y[mask_i].cpu())

    # 计算指标
    results = {}
    for i, name in zip([0,1,2], ["Train","Valid","Test"]):
        preds = torch.cat(all_pred[i]).numpy().ravel()
        trues = torch.cat(all_true[i]).numpy().ravel()
        avg_loss = losses[i] / total_samples[i]

        rmse = np.sqrt(mean_squared_error(trues, preds))
        mae  = mean_absolute_error(trues, preds)
        r2   = r2_score(trues, preds)
        rho_s, _ = stats.spearmanr(trues, preds)
        rho_p, _ = stats.pearsonr(trues, preds)

        print(f"\n===== {name} Performance =====")
        print(f"Loss   : {avg_loss:.4f}")
        print(f"RMSE   : {rmse:.4f}")
        print(f"MAE    : {mae:.4f}")
        print(f"R2     : {r2:.4f}")
        print(f"Spearman: {rho_s:.4f}")
        print(f"Pearson : {rho_p:.4f}")

        results[name] = {"trues": trues, "preds": preds, "loss": avg_loss,
                         "rmse": rmse, "mae": mae, "r2": r2,
                         "spearman": rho_s, "pearson": rho_p}

    return results

# 调用方式
results = evaluate_mega(loader, model, criterion, device, e_x_path='train_AA/save/best_e_x_AA_pand.npy')

tr_train, pr_train, r_train = results["Train"]["trues"], results["Train"]["preds"], results["Train"]["pearson"]
tr_valid, pr_valid, r_valid = results["Valid"]["trues"], results["Valid"]["preds"], results["Valid"]["pearson"]
tr_test,  pr_test,  r_test  = results["Test"]["trues"], results["Test"]["preds"], results["Test"]["pearson"]

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