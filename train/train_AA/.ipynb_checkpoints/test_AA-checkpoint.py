# =========================
# 0️⃣ 导入依赖
# =========================
import os
import ast
import numpy as np
import pandas as pd
import torch
import h5py
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from .model import TransformerRegressor
from .utils.getsmiles_f import compute_features


def collate_fn(batch):
    xs, smiles_fs= zip(*batch)
    lengths = [x.size(0) for x in xs]
    max_len = max(lengths)

    padded = []
    for x in xs:
        pad_len = max_len - x.size(0)
        if pad_len > 0:
            x = torch.cat([x, torch.zeros(pad_len, x.size(1))], dim=0)
        padded.append(x)

    xs = torch.stack(padded)
    smiles_fs = torch.tensor(np.stack(smiles_fs), dtype=torch.float32)
    mask = torch.tensor([[1]*l + [0]*(max_len-l) for l in lengths])

    return xs, smiles_fs, mask
# =========================
# 1️⃣ 数据加载与处理
# =========================
print('Preparing test data...')
test_csv_path  = 'train_AA/iGB3externel.csv'
GPepT_emb_path = '../GPepT_emb.csv'
data_test = pd.read_csv(test_csv_path)
GPepT_emb = pd.read_csv(GPepT_emb_path)
print('Test data loaded')

# compute smiles features as numpy, then convert to tensor later
smiles_f = np.stack([
    np.array([int(c) for c in compute_features(sm)[-1]], dtype=np.float32)
    for sm in data_test['Smiles'].to_list()
])

print(smiles_f[0])
print("Computed smiles features:", smiles_f.shape)

seq_emb = []
for seq in data_test['Sequences']:
    emb = []
    for mon in seq.split('-'):
        rows = GPepT_emb[GPepT_emb['token'] == mon]
        emb_str = rows['embedding'].values[0]
        emb.append(ast.literal_eval(emb_str))
    # convert to torch tensor (seq_len, emb_dim)
    seq_emb.append(torch.tensor(np.array(emb), dtype=torch.float32))

print("Computed sequence embeddings:", len(seq_emb))

# build batch (ensure smilesf rows are converted to tensors when collating)
batch = list(zip(seq_emb, smiles_f))

print("Data prepared:", len(batch), "samples")

# device selection - safer
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

model = TransformerRegressor(emb_dim=1280).to(device)

save_path = 'train_AA/save/best_transformer_AAmodel.pth'
e_x = torch.from_numpy(np.load('train_AA/save/best_e_x_AA.npy')).to(device)
model.load_state_dict(torch.load(save_path, map_location=device))
model.eval()
print("Model loaded:", save_path)

# collate and move inputs to device
x, smiles_f, mask = collate_fn(batch)
x, smiles_f, mask = x.to(device), smiles_f.to(device), mask.to(device)
pred, _, _ = model(x, smiles_f, mask, e_x.detach(), torch.zeros(1, device=device), 0)

print("Predictions:")
print(pred.detach().cpu().numpy())
