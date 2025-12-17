# Final Mega-Batch Training + Validation + Test Prediction Pipeline + Fine‑tuning
# -----------------------------------------
# Includes:
#  ✓ compute_features 保留全部输出
#  ✓ GPepT 序列嵌入
#  ✓ 3000 → 2000 train + 1000 valid + external
#  ✓ Mega‑batch 建立（变长 peptide padded）
#  ✓ 训练微调（train_mask 反向传播，valid/test 不反传）
#  ✓ 最终模型保存 + external 预测

import os
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from .model import TransformerRegressor
from .utils.getsmiles_f import compute_features
import h5py

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ================================================================
# collate_fn: pad variable length peptide embeddings
# ================================================================
def collate_fn(batch):
    xs, fps = zip(*batch)
    lengths = [x.size(0) for x in xs]
    max_len = max(lengths)

    padded = []
    for x in xs:
        pad_len = max_len - x.size(0)
        if pad_len > 0:
            x = torch.cat([x, torch.zeros(pad_len, x.size(1))], dim=0)
        padded.append(x)
    xs = torch.stack(padded)

    fps = torch.tensor(np.stack([np.asarray(f, dtype=np.float32) for f in fps]), dtype=torch.float32)
    mask = torch.tensor([[1]*l + [0]*(max_len-l) for l in lengths], dtype=torch.float32)
    return xs, fps, mask

# ================================================================
# GPepT embedding
# ================================================================
def get_GPeTemb_from_seqs(seq_list, GPepT_emb_path='../GPepT_emb.csv', emb_dim=1280):
    GPepT_emb = pd.read_csv(GPepT_emb_path)
    seqs_emb = []
    for seq in tqdm(seq_list, desc="GPepT emb"):
        emb = []
        for mon in seq:
            rows = GPepT_emb[GPepT_emb['token'] == mon]
            if not rows.empty:
                mon_emb = np.array(ast.literal_eval(rows.iloc[0]['embedding']), dtype=np.float32)
            else:
                print(f"Warning: token {mon} not found in GPepT embedding. Using zero vector.")
                mon_emb = np.zeros(emb_dim, dtype=np.float32)
            emb.append(mon_emb)
        seqs_emb.append(torch.tensor(np.stack(emb, axis=0), dtype=torch.float32))
    return seqs_emb

# ================================================================
# 1. Load data
# ================================================================
valid_csv_path = "/nas1/develop/lixiang/data/split_train_test/test_data_numW2.csv"
external_csv   = "train_AA/iGB3externel.csv"
h5_file        = "/nas1/develop/lixiang/data/class_AA/smiles_embed.h5"
GPepT_emb_path = '../GPepT_emb.csv'

# labeled pool
df_pool = pd.read_csv(valid_csv_path)
y_full = df_pool.iloc[:, 1].astype(np.float32).values

# external data
df_ext = pd.read_csv(external_csv)
seqs_ext = df_ext['Sequences'].str.split('-').tolist()
smiles_ext = df_ext['Smiles'].astype(str).tolist()

# ================================================================
# 2. sorted‑y sample 3000 → 2000 train + 1000 valid
# ================================================================
sorted_idx = np.argsort(y_full)
step = max(1, len(sorted_idx)//3000)
sample_idx = sorted_idx[::step][:3000]
if len(sample_idx) < 3000:
    sample_idx = sorted_idx

np.random.shuffle(sample_idx)
train_idx = sample_idx[:2000]
valid_idx = sample_idx[2000:3000]

df_train = df_pool.iloc[train_idx].reset_index(drop=True)
df_valid = df_pool.iloc[valid_idx].reset_index(drop=True)

y_train = np.log10(df_train.iloc[:, 1].astype(np.float32).values)
y_valid = np.log10(df_valid.iloc[:, 1].astype(np.float32).values)

# ================================================================
# 3. Load fp from H5
# ================================================================
with h5py.File(h5_file, "r") as f:
    seqs_h5 = f["seq"][:].astype(str)
    fp_h5   = f["fp"][:]

h5_map = {s:i for i,s in enumerate(seqs_h5)}
def lookup_fp(keys):
    return [fp_h5[h5_map[k]] for k in keys]

train_keys = df_train['PepetideID'].astype(str).tolist()
valid_keys = df_valid['PepetideID'].astype(str).tolist()

fp_train = lookup_fp(train_keys)
fp_valid = lookup_fp(valid_keys)

# ================================================================
# 4. GPepT embeddings
# ================================================================
gp_train = get_GPeTemb_from_seqs(df_train['PepetideID'].astype(str).tolist(), GPepT_emb_path)
gp_valid = get_GPeTemb_from_seqs(df_valid['PepetideID'].astype(str).tolist(), GPepT_emb_path)
gp_ext   = get_GPeTemb_from_seqs(seqs_ext, GPepT_emb_path)

# ================================================================
# 5. compute_features for external
# ================================================================
fp_ext = []
for sm in tqdm(smiles_ext, desc="compute_features EXT"):
    full = compute_features(sm)
    bit = np.array([int(c) for c in full[-1]], dtype=np.float32)
    fp_ext.append(bit)

# ================================================================
# 6. Build mega‑batch
# ================================================================
mega_seq_list = gp_train + gp_valid + gp_ext
mega_fp_list  = fp_train + fp_valid + fp_ext
n_train, n_valid, n_ext = len(gp_train), len(gp_valid), len(gp_ext)

train_mask = torch.tensor([1]*n_train + [0]*(n_valid+n_ext), dtype=torch.float32)
valid_mask = torch.tensor([0]*n_train + [1]*n_valid + [0]*n_ext, dtype=torch.float32)

mega_data = list(zip(mega_seq_list, mega_fp_list))
loader = DataLoader(mega_data, batch_size=len(mega_data), shuffle=False, collate_fn=collate_fn)

# ================================================================
# 7. Load model
# ================================================================
model = TransformerRegressor(emb_dim=1280).to(device)
e_x = torch.from_numpy(np.load("train_AA/Predict/best_e_x_AA_pand.npy")).to(device)
model.load_state_dict(torch.load("train_AA/Predict/best_expand_AAmodel.pth", map_location=device))

# ================================================================
# 8. Fine‑tuning setup
# ================================================================
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
criterion = nn.MSELoss()
EPOCHS = 100

train_targets = torch.tensor(y_train, dtype=torch.float32).to(device)
valid_targets = torch.tensor(y_valid, dtype=torch.float32).to(device)

# ================================================================
# 9. Train loop (mega‑batch each epoch)
# ================================================================
(xs, fps, seq_mask) = next(iter(loader))
xs, fps, seq_mask = xs.to(device), fps.to(device), seq_mask.to(device)
train_mask = train_mask.to(device)
valid_mask = valid_mask.to(device)

best_val = float("inf")
best_epoch = -1

# for ep in range(EPOCHS):
#     model.train()
#     optimizer.zero_grad()

#     pred_all, _, _ = model(xs, fps, seq_mask, e_x.detach(),
#                            torch.zeros(1, device=device), 0)
#     pred_all = pred_all.squeeze(-1)

#     # only train on train_mask
#     pred_train = pred_all[:n_train]
#     loss = criterion(pred_train, train_targets)
#     loss.backward()
#     optimizer.step()

#     # validation (no grad)
#     with torch.no_grad():
#         pred_valid = pred_all[n_train:n_train+n_valid]
#         val_loss = criterion(pred_valid, valid_targets)

#     print(f"Epoch {ep+1}/{EPOCHS}  Train={loss.item():.5f}  Valid={val_loss.item():.5f}")

#     # -------------------------
#     #   ⭐ Save best model
#     # -------------------------
#     if val_loss.item() < best_val:
#         best_val = val_loss.item()
#         best_epoch = ep
#         torch.save(model.state_dict(),
#                    "train_AA/finetuned/AAmodel_finetuned_best.pth")
#         print(f"  >>> New best model saved at epoch {ep+1}, val={best_val:.5f}")

# ================================================================
# 10. Final forward + external prediction
# ================================================================
model.eval()
model.load_state_dict(torch.load("train_AA/finetuned/AAmodel_finetuned_best.pth",
                                 map_location=device))
with torch.no_grad():
    pred_all,_,_ = model(xs, fps, seq_mask, e_x, torch.zeros(1, device=device), 0)
    pred_all = pred_all.squeeze().cpu().numpy()

pred_train = pred_all[:n_train]
pred_valid = pred_all[n_train:n_train+n_valid]
pred_ext   = pred_all[n_train+n_valid:]

# save external
df_ext['Predict_tune'] = pred_ext
df_ext.to_csv("train_AA/Predict/iGB3_external_pred_expand.csv", index=False)
print("External predictions saved to train_AA/Predict/iGB3_external_pred_expand.csv")
# ================================================================
