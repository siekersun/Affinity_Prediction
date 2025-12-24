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
smiles_count = '/nas1/develop/lixiang/data/class_AA/seq_smiles_fp_matrix.npz'
smiles_bit = '/nas1/develop/lixiang/data/class_AA/smiles_embed.h5'
smiles_chem = '/nas1/develop/lixiang/data/class_AA/ChemBERT_fp.h5'
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

    # 使用混合数据加载方式
    mixed_loader = get_dataloaders(
        train_csv_path,
        valid_csv_path,
        test_csv_path,
        smiles_bit,
        smiles_count,
        embeddings_dir,
        smiles_chem,
        mix=True
    )

    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    model = HGNN().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params: {total_params}")
    print(f"Trainable params: {trainable_params}")
    
    # 设置优化器和损失函数
    criterion = nn.MSELoss()
    lr = 5e-4
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CyclicLR(
        optimizer,
        base_lr=0.5*lr,        # 学习率的最小值 (min_lr)
        max_lr=lr,       # 学习率的最大值 (max_lr)
        step_size_up=8, # 学习率从 min_lr 上升到 max_lr 所需的迭代次数
        step_size_down=4,
        mode='exp_range',   # 调度模式：'triangular' (三角波)
        gamma=0.995,
        cycle_momentum=False # 如果使用带有动量的优化器 (如 Adam)，通常设置为 False
    )
    
    # 训练循环
    EPOCH = 400
    best_val_loss = float('inf')
    save_path = 'train_tune/save/hgnn_best_interact_model.pth'
    patience = 10
    counter = 0
    
    for epoch in range(EPOCH):
        print(f"\nEpoch {epoch+1}/{EPOCH}")
        
        # 训练和验证阶段（一次遍历loader完成）
        model.train()
        total_train_loss = 0
        total_val_loss = 0
        
        for emb, bit_fp, count_fp, y, split_id in tqdm(mixed_loader, desc="训练和验证进度"):
            emb, bit_fp, count_fp, y, split_id = emb.to(device), bit_fp.to(device), count_fp.to(device), y.to(device), split_id.to(device)
            
            # 分离训练集和验证集数据
            train_mask = (split_id == 0)
            val_mask = (split_id == 1)
            
            # 确保有训练数据和验证数据
            if not train_mask.any():
                continue
            
            optimizer.zero_grad()
            
            # 对整个批次进行前向传播
            pred = model(emb, bit_fp, count_fp)
            
            # 计算训练集损失
            train_pred = pred[train_mask]
            train_y = y[train_mask]
            train_loss = criterion(train_pred, train_y)
            
            # 反向传播仅使用训练集损失
            train_loss.backward()
            optimizer.step()
            
            total_train_loss += train_loss.item()
            
            # 计算验证集损失（无梯度）
            if val_mask.any():
                val_pred = pred[val_mask]
                val_y = y[val_mask]
                val_loss = criterion(val_pred, val_y)
                total_val_loss += val_loss.item()
        scheduler.step()

        # 计算平均损失
        avg_train_loss = total_train_loss / len(mixed_loader)
        avg_val_loss = total_val_loss / len(mixed_loader)
        
        print(f"训练损失: {avg_train_loss:.6f}")
        print(f"验证损失: {avg_val_loss:.6f}")
        
        # 早停机制
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"⭐ 模型已保存到 {save_path}")
            counter = 0
        else:
            counter += 1
            print(f"未改善: {counter}/{patience}")
            if counter >= patience:
                print(f"⛔ 早停: {patience} 轮没有改善")
                break
    
    # 定义评估函数（用于测试集评估）
    def evaluate_all_datasets(model, dataloader):
        model.eval()
        # np.savetxt('train_tune/save/combin_H.txt', model.combin_H.cpu().detach().numpy())
        
        # 为每个数据集准备存储列表
        datasets = {
            "训练集": {"preds": [], "labels": [], "id": 0},
            "验证集": {"preds": [], "labels": [], "id": 1},
            "测试集": {"preds": [], "labels": [], "id": 2}
        }
        
        with torch.no_grad():
            for emb, bit_fp, count_fp, y, split_id in tqdm(dataloader, desc="评估所有数据集进度"):
                emb, bit_fp, count_fp, y, split_id = emb.to(device), bit_fp.to(device), count_fp.to(device), y.to(device), split_id.to(device)
                
                # 对整个批次进行前向传播
                pred = model(emb, bit_fp, count_fp)
                
                # 根据split_id将结果分配到不同的数据集
                for dataset_name, dataset_info in datasets.items():
                    mask = (split_id == dataset_info["id"])
                    if mask.any():
                        dataset_info["preds"].append(pred[mask].cpu().numpy())
                        dataset_info["labels"].append(y[mask].cpu().numpy())
        
        # 计算并打印每个数据集的指标
        for dataset_name, dataset_info in datasets.items():
            preds = dataset_info["preds"]
            labels = dataset_info["labels"]
            
            if not preds:
                print(f"\n{dataset_name} 没有数据")
                continue
            
            # 合并所有批次的结果
            all_preds = np.concatenate(preds, axis=0)
            all_labels = np.concatenate(labels, axis=0)
            
            # 计算性能指标
            mse = mean_squared_error(all_labels, all_preds)
            mae = mean_absolute_error(all_labels, all_preds)
            r2 = r2_score(all_labels, all_preds)
            spearmanr = stats.spearmanr(all_labels.flatten(), all_preds.flatten())[0]
            pearsonr = stats.pearsonr(all_labels.flatten(), all_preds.flatten())[0]
            
            # 打印结果
            print(f"\n{dataset_name} 结果:")
            print(f"MSE: {mse:.6f}")
            print(f"MAE: {mae:.6f}")
            print(f"R²: {r2:.6f}")
            print(f"Spearman R: {spearmanr:.6f}")
            print(f"Pearson R: {pearsonr:.6f}")
    
    # 加载最佳模型
    model.load_state_dict(torch.load(save_path))
    model.eval()
    
    # 评估所有数据集（只遍历一次dataloader）
    evaluate_all_datasets(model, mixed_loader)

if __name__ == "__main__":
    main()