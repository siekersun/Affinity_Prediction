# 导入依赖
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import h5py

# 添加train目录到路径中，解决ModuleNotFoundError问题
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 从train_pepland模块导入模型
from train_pepland.model_pepland import TransformerRegressor

# 设置设备
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

# 数据路径
seq_smiles_fp_matrix_path = '/nas1/develop/lixiang/data/class_AA/seq_smiles_fp_matrix.npz'
pepland_path = '/nas1/develop/lixiang/data/class_AA/ChemBERT_embed.h5'

# 加载数据
data = np.load(seq_smiles_fp_matrix_path)
smiles_x = data['X']  # 第一种输入特征：SMILES指纹
seqs_smiles = data['seq_ids']  # 唯一标识符
data.close()

with h5py.File(pepland_path, "r") as f:
    seqs_pepland = f["seq"][:].astype(str)
    x = f["embed"][:]


print(seqs_smiles[:5])

# 验证序列一致性
assert (seqs_smiles == seqs_pepland).all(), "Sequences mismatch in data files!"

# 定义自定义数据集
class EmbeddingDataset(Dataset):
    def __init__(self, x, smiles_x, seq_ids):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.smiles_x = torch.tensor(smiles_x, dtype=torch.float32)
        self.seq_ids = seq_ids
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.smiles_x[idx], self.seq_ids[idx]

# 创建数据集和数据加载器（批次处理）
dataset = EmbeddingDataset(x, smiles_x, seqs_smiles)
batch_size = 128  # 可根据GPU内存调整
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

# 加载模型
model = TransformerRegressor(emb_dim=768).to(device)
model.load_state_dict(torch.load('/data/home/lixiang/software/GPepT/train/train_pepland/save/best_model_best.pth'))
model.eval()

# 使用nn.Identity()替换预测层，使模型直接输出嵌入编码
model.predict = nn.Identity()

# 定义函数来提取编码（批次处理）
def extract_embeddings(model, dataloader, device):
    all_embeddings = []
    all_seq_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            x_batch, smiles_x_batch, seq_ids_batch = batch
            x_batch, smiles_x_batch = x_batch.to(device), smiles_x_batch.to(device)
            
            # 输入模型计算嵌入编码
            emb = model(x_batch, smiles_x_batch)
            
            # 保存编码和序列ID
            all_embeddings.append(emb.cpu().numpy())
            all_seq_ids.extend(seq_ids_batch)
    
    # 合并所有批次的结果
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    return all_embeddings, all_seq_ids

# 提取编码
all_embeddings, all_seq_ids = extract_embeddings(model, dataloader, device)

# 保存编码到文件
save_dir = '/nas1/develop/lixiang/data/class_AA/'
os.makedirs(save_dir, exist_ok=True)

# 将序列ID转换为字节字符串，以便HDF5可以处理
all_seq_ids_bytes = [seq_id.encode('utf-8') for seq_id in all_seq_ids]

# 保存所有嵌入编码和对应的序列ID
with h5py.File(os.path.join(save_dir, 'all_embeddings.h5'), 'w') as f:
    # 保存序列ID（使用变长字符串类型）
    dt = h5py.special_dtype(vlen=str)
    f.create_dataset('seq_ids', data=all_seq_ids, dtype=dt)
    # 保存嵌入编码
    f.create_dataset('embeddings', data=all_embeddings)

print(f"编码提取完成并保存到文件!")
print(f"总共有 {len(all_embeddings)} 个嵌入编码")
print(f"嵌入编码形状: {all_embeddings.shape}")