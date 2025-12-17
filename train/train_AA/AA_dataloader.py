import h5py
import numpy as np
import pandas as pd
import torch
import json
import ast
from torch.utils.data import Dataset, DataLoader

def load_embeddings(file_smiles, GPepT_emb_path):
    """Load desc / fp embeddings."""
    # with h5py.File(file_smiles, "r") as f:
    #     seqs_smiles = f["seq"][:].astype(str)
    #     fp = f["fp"][:]
    data = np.load(file_smiles)
    fp=data['X']
    seqs_smiles=data['seq_ids']
    hash_to_idx=json.loads(data['hash_to_idx'].item())
    data.close()
    # sequence → index
    seq_to_idx = {s: i for i, s in enumerate(seqs_smiles)}
    GPepT_emb = pd.read_csv(GPepT_emb_path)

    return seq_to_idx, fp, hash_to_idx, GPepT_emb

def load_split_data(csv_path, seq_to_idx, fp, log_target=True, median_center=False):
    """Load a CSV split and return embeddings + labels."""
    df = pd.read_csv(csv_path)

    seq_ids = df["PepetideID"].tolist()
    y = df.iloc[:, 1].astype("float32").values.reshape(-1, 1)

    if log_target:
        y = np.log10(y)

    fp_split = fp[[seq_to_idx[s] for s in seq_ids]]

    return seq_ids, y, fp_split


class TensorDataset(Dataset):
    def __init__(self, seq, y, fp, GPepT_emb):
        NATURAL_AA = set('ACDEFGHIKLMNPQRSTVWY')
        
        self.mon_emb = {
            row['token']: np.array(ast.literal_eval(row['embedding']), dtype=float)
            for _, row in GPepT_emb.iterrows() if row['token'] in NATURAL_AA
        }
        self.seq = seq
        self.y = torch.tensor(y, dtype=torch.float32)
        self.fp = fp

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        seq_i = self.seq[idx]
        smiles_f = torch.from_numpy(self.fp[idx]).float()
        y_i = self.y[idx]
        x_emb = np.stack([self.mon_emb[aa] for aa in seq_i], axis=0)
        x_emb = torch.from_numpy(x_emb).float()
        return x_emb, smiles_f, y_i

class DynamicMegaBatchLoader:
    def __init__(self, train_ds, valid_ds, test_ds, n_batches=1181, collate_fn=None):
        self.n_train = len(train_ds)
        self.n_valid = len(valid_ds)
        self.n_test  = len(test_ds)

        self.n_batches = n_batches

        # 每个 split 的 batch size（向下取整）
        self.train_bs = max(0, self.n_train // self.n_batches)
        self.valid_bs = max(0, self.n_valid // self.n_batches)
        self.test_bs  = max(0, self.n_test  // self.n_batches)

        if (self.train_bs == 0) or (self.valid_bs == 0) or (self.test_bs == 0):
            raise ValueError(
                f"One of split batch sizes is 0 (train_bs={self.train_bs}, "
                f"valid_bs={self.valid_bs}, test_bs={self.test_bs}). "
                "Reduce n_batches or provide larger datasets."
            )

        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.test_ds  = test_ds

        if collate_fn is None:
            raise ValueError("collate_fn must be provided")
        self.collate_fn = collate_fn

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        train_idx = torch.randperm(self.n_train)
        valid_idx = torch.randperm(self.n_valid)
        test_idx  = torch.randperm(self.n_test)

        for i in range(self.n_batches):
            t_start = i * self.train_bs
            v_start = i * self.valid_bs
            te_start = i * self.test_bs

            t_idx = train_idx[t_start : t_start + self.train_bs]
            v_idx = valid_idx[v_start : v_start + self.valid_bs]
            te_idx = test_idx[te_start : te_start + self.test_bs]

            # list of tuples: [(x,sf,y), ...]
            train_batch = [self.train_ds[j] for j in t_idx]
            valid_batch = [self.valid_ds[j] for j in v_idx]
            test_batch  = [self.test_ds[j]  for j in te_idx]

            # ===== 推荐：一次性合并后调用 collate_fn =====
            combined = train_batch + valid_batch + test_batch  # 保证顺序：train, valid, test
            xs, sf, mask, ys = self.collate_fn(combined)

            # split_id：与 combined 顺序对应
            split_id = torch.cat([
                torch.zeros(self.train_bs, dtype=torch.long),
                torch.ones(self.valid_bs, dtype=torch.long),
                torch.full((self.test_bs,), 2, dtype=torch.long)
            ])

            yield xs, sf, mask, ys, split_id


def collate_fn(batch):
    xs, smiles_fs, ys = zip(*batch)    
    lengths = [x.size(0) for x in xs]
    max_len = max(lengths)

    # Padding
    padded = []
    for x in xs:
        pad_len = max_len - x.size(0)
        if pad_len > 0:
            x = torch.cat([x, torch.zeros(pad_len, x.size(1))], dim=0)
        padded.append(x)

    xs = torch.stack(padded)
    smiles_fs = torch.stack(smiles_fs)
    ys = torch.stack(ys)
    mask = torch.tensor([[1]*l + [0]*(max_len-l) for l in lengths])
    
    # 仅返回 peptide embedding, mask, 和 label
    return xs, smiles_fs, mask, ys 

def get_dataloaders(
    train_csv,
    valid_csv,
    test_csv,
    file_smiles,
    GPepT_emb_path,
    mix= False
):
    """Main function: return three DataLoaders."""

    # load embeddings
    seq_to_idx, fp, hash_to_idx, GPepT_emb = load_embeddings(file_smiles, GPepT_emb_path)

    # load each split
    seq_train, y_train, fp_train = load_split_data(
        train_csv, seq_to_idx, fp
    )
    seq_valid, y_valid, fp_valid = load_split_data(
        valid_csv, seq_to_idx, fp
    )
    seq_test, y_test, fp_test = load_split_data(
        test_csv, seq_to_idx, fp
    )

    train_dataset = TensorDataset(seq_train, y_train, fp_train, GPepT_emb)
    valid_dataset = TensorDataset(seq_valid, y_valid, fp_valid, GPepT_emb)
    test_dataset  = TensorDataset(seq_test, y_test, fp_test, GPepT_emb)
    
    mega_loader = DynamicMegaBatchLoader(train_dataset, valid_dataset, test_dataset, n_batches=1181,collate_fn=collate_fn)
    BATCH = 2837
    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, collate_fn=collate_fn, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH, shuffle=False, collate_fn=collate_fn, drop_last=False)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH, shuffle=False, collate_fn=collate_fn, drop_last=False)

    if mix:
        return mega_loader
    else:
        return train_loader, valid_loader, test_loader
