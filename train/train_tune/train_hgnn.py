# 导入依赖
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CyclicLR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import h5py
import pandas as pd
import random
from tqdm import tqdm
from .hgnn_dataloader import get_dataloaders
from .hgnn_model import HGNN

# 设置设备
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

# 数据路径
embeddings_dir = '/nas1/develop/lixiang/data/class_AA/all_embeddings.h5'
# smiles_file = '/nas1/develop/lixiang/data/class_AA/seq_smiles_fp_matrix.npz'
smiles_file = '/nas1/develop/lixiang/data/class_AA/smiles_embed.h5'
train_csv_path = '/nas1/develop/lixiang/data/split_train_test/train_data.csv'
valid_csv_path = '/nas1/develop/lixiang/data/split_train_test/test_data_numW1.csv'
test_csv_path = '/nas1/develop/lixiang/data/split_train_test/test_data_numW2.csv'

# 主函数
def main():

    def set_seed(seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    set_seed(42)

    train_loader, valid_loader, test_loader = get_dataloaders(
        train_csv_path,
        valid_csv_path,
        test_csv_path,
        # bit_fp_file,
        smiles_file,
        embeddings_dir,
        mix=False
    )

    
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    model = HGNN().to(device)
    
    # 7. 设置优化器和损失函数
    criterion = nn.MSELoss()
    lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CyclicLR(
        optimizer,
        base_lr=1e-4,        # 学习率的最小值 (min_lr)
        max_lr=1e-3,       # 学习率的最大值 (max_lr)
        step_size_up=8, # 学习率从 min_lr 上升到 max_lr 所需的迭代次数
        step_size_down=4,
        mode='exp_range',   # 调度模式：'triangular' (三角波)
        gamma=0.95,
        cycle_momentum=False # 如果使用带有动量的优化器 (如 Adam)，通常设置为 False
    )
    
    # 8. 训练循环
    EPOCH = 64
    best_val_loss = float('inf')
    save_path = 'train_tune/save/hgnn_best_model.pth'
    patience = 10
    counter = 0
    
    for epoch in range(EPOCH):
        print(f"\nEpoch {epoch+1}/{EPOCH}")
        
        # 训练阶段
        model.train()
        total_train_loss = 0
        
        for emb, H, y in tqdm(train_loader, desc="训练进度"):
            emb, H, y = emb.to(device), H.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(emb, H)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        scheduler.step()
        
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"训练损失: {avg_train_loss:.6f}")
        
        # 验证阶段
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for emb, H, y in tqdm(valid_loader, desc="验证进度"):
                emb, H, y = emb.to(device), H.to(device), y.to(device)
                pred = model(emb, H)
                loss = criterion(pred, y)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(valid_loader)
        print(f"验证损失: {avg_val_loss:.6f}")
        
        # 早停机制
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"⭐ 模型已保存到 {save_path}")
            counter = 0
        else:
            counter += 1
            print(f"未改善轮数: {counter}/{patience}")
            if counter >= patience:
                print(f"⛔ 早停: {patience} 轮没有改善")
                break
    
    # 9. 测试最佳模型
    model.load_state_dict(torch.load(save_path))
    model.eval()
    
    def evaluate_model(model, dataloader, dataset_name):
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for emb, H, y in tqdm(dataloader, desc=f"{dataset_name} 进度"):
                emb, H, y = emb.to(device), H.to(device), y.to(device)
                pred = model(emb, H)
                all_preds.append(pred.cpu().numpy())
                all_labels.append(y.cpu().numpy())
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # 计算性能指标
        mse = mean_squared_error(all_labels, all_preds)
        mae = mean_absolute_error(all_labels, all_preds)
        r2 = r2_score(all_labels, all_preds)
        spearmanr = stats.spearmanr(all_labels.flatten(), all_preds.flatten())[0]
        pearsonr = stats.pearsonr(all_labels.flatten(), all_preds.flatten())[0]
        
        print(f"\n{dataset_name} 结果:")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R²: {r2:.6f}")
        print(f"Spearman R: {spearmanr:.6f}")
        print(f"Pearson R: {pearsonr:.6f}")
    
    # 加载最佳模型
    model.load_state_dict(torch.load(save_path))
    
    # 评估所有数据集
    evaluate_model(model, train_loader, "训练集")
    evaluate_model(model, valid_loader, "验证集")
    evaluate_model(model, test_loader, "测试集")


if __name__ == "__main__":
    main()
