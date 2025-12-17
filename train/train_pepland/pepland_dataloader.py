import h5py
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def load_embeddings(smiles, h5_file_pepland):
    """Load desc / fp / pepland embeddings."""
    # with h5py.File(smiles, "r") as f:
    #     seqs_smiles = f["seq"][:].astype(str)
    #     desc = f["desc"][:]
    #     fp = f["fp"][:]

    data = np.load(smiles)
    fp=data['X']
    seqs_smiles=data['seq_ids']
    hash_to_idx=json.loads(data['hash_to_idx'].item())
    data.close()
    with h5py.File(h5_file_pepland, "r") as f:
        seqs_pepland = f["seq"][:].astype(str)
        pepland = f["embed"][:]
    print(seqs_smiles[:5])
    # sequence → index
    seq_to_idx = {s: i for i, s in enumerate(seqs_smiles)}

    # sanity check
    assert (seqs_smiles == seqs_pepland).all(), "Sequences mismatch in h5 files!"

    return seq_to_idx, fp, hash_to_idx, pepland

def load_split_data(csv_path, seq_to_idx, H, hash_to_idx, pepland, log_target=True, median_center=False):
    """Load a CSV split and return embeddings + labels."""
    df = pd.read_csv(csv_path)

    seq_ids = df["PepetideID"].tolist()
    y = df.iloc[:, 1].astype("float32").values.reshape(-1, 1)

    if log_target:
        y = np.log10(y)

    # desc_split = desc[[seq_to_idx[s] for s in seq_ids]]
    desc_split = None
    H = H[[seq_to_idx[s] for s in seq_ids]]
    pepland_split = pepland[[seq_to_idx[s] for s in seq_ids]]

    return y, desc_split, H, pepland_split


class SimpleDataset(Dataset):
    """
    将 desc、fp、pepland、y 封装成 PyTorch Dataset
    每次 __getitem__ 返回 (x, smiles_f, y)
    """
    def __init__(self, fp, pepland, y):
        self.fp   = torch.tensor(fp, dtype=torch.float32)
        self.pepland = torch.tensor(pepland, dtype=torch.float32)
        self.y    = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        smiles_f =self.fp[idx]
        x = self.pepland[idx]
        y = self.y[idx]
        return x, smiles_f, y
    

class DynamicMegaBatchLoader:
    """
    自动计算 batch_size，保证 train/valid/test 每个 mega-batch 对齐。
    """
    def __init__(self, train_ds, valid_ds, test_ds, n_batches = 1181):
        """
        train_ds, valid_ds, test_ds: Dataset，返回 (x, sf, y)
        max_train_bs: 可选，最大训练 batch_size
        """

        # 根据长度计算 batch_size
        self.n_train = len(train_ds)
        self.n_valid = len(valid_ds)
        self.n_test  = len(test_ds)

        # 以训练集为参考 batch数
        # 默认每个 mega-batch 至少包含一个训练样本
        self.n_batches = n_batches

        # 动态计算每个 split 的 batch_size，向下取整保证对齐
        self.train_bs = self.n_train // self.n_batches
        self.valid_bs = self.n_valid // self.n_batches
        self.test_bs  = self.n_test  // self.n_batches

        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.test_ds  = test_ds

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        train_indices = torch.randperm(self.n_train)
        valid_indices = torch.randperm(self.n_valid)
        test_indices  = torch.randperm(self.n_test)
        for i in range(self.n_batches):
            # 切片索引
            t_start = i * self.train_bs
            v_start = i * self.valid_bs
            te_start = i * self.test_bs

            t_end  = t_start + self.train_bs
            v_end  = v_start + self.valid_bs
            te_end = te_start + self.test_bs

            # 取数据
            train_batch = [self.train_ds[j] for j in train_indices[t_start:t_end]]
            valid_batch = [self.valid_ds[j] for j in valid_indices[v_start:v_end]]
            test_batch  = [self.test_ds[j]  for j in test_indices[te_start:te_end]]

            # 拼接函数
            def stack(batch):
                x, sf, y = zip(*batch)
                return torch.stack(x), torch.stack(sf), torch.stack(y)

            xb_tr, sf_tr, y_tr = stack(train_batch)
            xb_v,  sf_v,  y_v  = stack(valid_batch)
            xb_te, sf_te, y_te = stack(test_batch)

            # 合并 mega-batch
            x  = torch.cat([xb_tr, xb_v, xb_te], dim=0)
            sf = torch.cat([sf_tr, sf_v, sf_te], dim=0)
            y  = torch.cat([y_tr, y_v, y_te], dim=0)

            # split_id
            split_id = torch.cat([
                torch.zeros(self.train_bs, dtype=torch.long),
                torch.ones(self.valid_bs, dtype=torch.long),
                torch.full((self.test_bs,), 2, dtype=torch.long)
            ])

            yield x, sf, y, split_id

def get_dataloaders(
    train_csv,
    valid_csv,
    test_csv,
    smiles,
    h5_pepland,
    mix = True
):
    """Main function: return three DataLoaders."""

    # load embeddings
    # seq_to_idx, desc, fp, pepland = load_embeddings(smiles, h5_pepland)
    seq_to_idx, H, hash_to_idx, pepland = load_embeddings(smiles, h5_pepland)

    y_train, _, fp_train, pepland_train = load_split_data(
        train_csv, seq_to_idx, H, hash_to_idx, pepland
    )
    y_valid, _, fp_valid, pepland_valid = load_split_data(
        valid_csv, seq_to_idx, H, hash_to_idx, pepland
    )
    y_test, _, fp_test, pepland_test = load_split_data(
        test_csv, seq_to_idx, H, hash_to_idx, pepland
    )
    
    # y_all = np.concatenate([y_train, y_valid, y_test],0)
    # y_all = y_all-np.median(y_all)
    # y_train = y_all[:len(y_train)]
    # y_valid = y_all[len(y_train):len(y_train)+len(y_valid)]
    # y_test = y_all[len(y_train)+len(y_valid):]

    # # 先把原始数据拆开
    # fp_train_new, fp_valid_new, \
    # pepland_train_new, pepland_valid_new, \
    # y_train_new, y_valid_new = train_test_split(
    #     fp_train, pepland_train, y_train,
    #     test_size=0.1,      # 20 % 做验证
    #     random_state=42,    # 固定随机种子，可复现
    # )

    # # 重新封装
    # train_dataset = SimpleDataset(fp_train_new, pepland_train_new, y_train_new)
    # valid_dataset = SimpleDataset(fp_valid_new, pepland_valid_new, y_valid_new)

    train_dataset = SimpleDataset(fp_train, pepland_train, y_train)
    valid_dataset = SimpleDataset(fp_valid, pepland_valid, y_valid)
    test_dataset  = SimpleDataset(fp_test,  pepland_test,  y_test)


    if mix:
        mega_loader = DynamicMegaBatchLoader(train_dataset, valid_dataset, test_dataset)
        return mega_loader
    else:
        BATCH = 2837
        train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH, shuffle=False, drop_last=False)
        test_loader  = DataLoader(test_dataset,  batch_size=BATCH, shuffle=False, drop_last=False)
        return train_loader, valid_loader, test_loader