# =========================
# 0️⃣ 导入依赖
# =========================
import os
import ast
import json
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
from transformers import AutoModelWithLMHead, AutoTokenizer, pipeline, AutoModel

print("Imports loaded")

# =========================
# 1️⃣ 数据加载与处理
# =========================
test_csv_path  = '/data/home/lixiang/software/GPepT/train/train_pepland/iGB3externel.csv'
# h5_file        = '/nas1/develop/lixiang/data/class_AA/smiles_embed.h5'
smiles_file = '/nas1/develop/lixiang/data/class_AA/seq_smiles_fp_matrix.npz'
# pepland_path = '/nas1/develop/lixiang/data/class_AA/seq_peplandembed.h5'

print("Loading testing data...")
data_test = pd.read_csv(test_csv_path)
seq_smiles = data_test['Smiles'].to_list()
print("Loading smiles data...")
data = np.load(smiles_file)
hash_to_idx=json.loads(data['hash_to_idx'].item())
data.close()

model = AutoModel.from_pretrained("seyonec/PubChem10M_SMILES_BPE_240k", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_240k", local_files_only=True)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    masked = token_embeddings * mask
    summed = masked.sum(1)
    counts = mask.sum(1)
    return summed / counts


inputs = tokenizer(seq_smiles, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

x = mean_pooling(outputs, inputs['attention_mask'])

print(x.shape)
print(hash_to_idx.keys())

smiles_x = np.zeros((len(data_test), 253), dtype=np.float32)

Credibility = []

from rdkit import Chem
from rdkit.Chem import AllChem
for i, smi in enumerate(seq_smiles):
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprint(mol, radius=2, useChirality=True)
    fp_dict = fp.GetNonzeroElements()
    n = 0
    m = 0
    for k, v in fp_dict.items():
        if hash_to_idx.get(str(k), -1) != -1:
            smiles_x[i, hash_to_idx[str(k)]] = v
            m += 1
        else:
            # print(f"Hash {k} not found in hash_to_idx!, 请注意检查第{i+1}个分子")
            n += 1
            # break
    Credibility.append((m-n)/253)

BATCH = 2837

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
model = TransformerRegressor(emb_dim=768, hidden_dim=100).to(device)

model_path = 'train_pepland/save/best_model_best111.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

x, smiles_x = torch.tensor(x).to(device), torch.tensor(smiles_x).to(device)
pred = model(x, smiles_x)
data_test['predict'] = pred.cpu().detach().numpy()
data_test['Credibility'] = Credibility

data_test.to_csv('/data/home/lixiang/software/GPepT/train/train_pepland/Predict/predict_pepland.csv')

