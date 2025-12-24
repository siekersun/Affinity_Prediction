from rdkit import Chem
from rdkit.Chem import ResonanceMolSupplier
from rdkit.Chem.MolStandardize import rdMolStandardize

AA_BACKBONE_SMARTS = Chem.MolFromSmarts(
    "[NX3;!$(NC=O)]-[C;X4]-[C;X3](=O)[O;X1,X2]"
)

def find_backbone_sites_strict(mol: Chem.Mol):
    matches = mol.GetSubstructMatches(AA_BACKBONE_SMARTS)
    if len(matches) >= 1:
        n_idx = matches[0][0]
        c_idx = matches[0][2]
        return n_idx, c_idx
    return None, None


def is_backbone_carboxyl(atom: Chem.Atom) -> bool:
    """
    判断该 atom（C）是否是主链羧基碳
    判据：
      C(=O)-O
      |
      Cα —— N
    """
    if atom.GetAtomicNum() != 6:
        return False

    o_dbl = False
    o_sgl = False
    alpha_c = None

    for bond in atom.GetBonds():
        nbr = bond.GetOtherAtom(atom)

        if nbr.GetAtomicNum() == 8:
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                o_dbl = True
            elif bond.GetBondType() == Chem.BondType.SINGLE:
                o_sgl = True

        elif nbr.GetAtomicNum() == 6:
            alpha_c = nbr

    if not (o_dbl and o_sgl and alpha_c):
        return False

    # α-碳必须再连一个 N
    for nbr in alpha_c.GetNeighbors():
        if nbr.GetAtomicNum() == 7:
            return True

    return False


def find_backbone_sites_loose(mol: Chem.Mol):
    # ---------- N 端 ----------
    n_idx = None
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 7 and atom.GetTotalNumHs() > 0:
            n_idx = atom.GetIdx()
            break

    # ---------- C 端（主链羧基） ----------
    c_idx = None
    for atom in mol.GetAtoms():
        if is_backbone_carboxyl(atom):
            c_idx = atom.GetIdx()
            break

    return n_idx, c_idx


def connect_residues(mol1: Chem.Mol, mol2: Chem.Mol) -> Chem.Mol:
    rw = Chem.RWMol(Chem.CombineMols(mol1, mol2))
    offset = mol1.GetNumAtoms()

    # 优先严格主链
    n2, _ = find_backbone_sites_strict(mol2)
    _, c1 = find_backbone_sites_strict(mol1)

    if n2 is None or c1 is None:
        n2, _ = find_backbone_sites_loose(mol2)
        _, c1 = find_backbone_sites_loose(mol1)

    if n2 is None or c1 is None:
        raise RuntimeError("无法找到可连接的主链 N / C 位点")

    n2 += offset

    # ---------- 找 C 端羧基的 –OH ----------
    o_remove = None
    c_atom = rw.GetAtomWithIdx(c1)

    for bond in c_atom.GetBonds():
        nbr_idx = bond.GetOtherAtomIdx(c1)
        nbr = rw.GetAtomWithIdx(nbr_idx)

        if nbr.GetAtomicNum() == 8 and bond.GetBondType() == Chem.BondType.SINGLE:
            o_remove = nbr_idx
            break

    if o_remove is None:
        raise RuntimeError("未找到羧基单键氧")

    # ---------- 建肽键 ----------
    rw.AddBond(c1, n2, Chem.BondType.SINGLE)
    rw.RemoveAtom(o_remove)

    Chem.SanitizeMol(rw)
    return rw.GetMol()


def build_peptide_from_smiles(smiles_list):
    if not smiles_list:
        raise ValueError("smiles_list 为空")

    mols = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"SMILES 解析失败: {smi}")
        Chem.SanitizeMol(mol)
        mols.append(mol)

    peptide = mols[0]
    for next_mol in mols[1:]:
        peptide = connect_residues(peptide, next_mol)

    return Chem.MolToSmiles(
        peptide,
        canonical=True,
        isomericSmiles=True
    )


# smiles_list = [
#     'C(C(=O)O)N',
#     'C(C[C@@H](C(=O)O)N)CNC(=N)N',
#     'C(C(=O)O)N',
#     'OC(=O)C[C@@H](C(=O)O)N',
#     'C[C@H]([C@@H](C(=O)O)N)O',
#     'C1C[C@H](NC1)C(=O)O',
# ]
# # smiles_list = [canonical_smiles(smi) for smi in smiles_list]
# print(build_peptide_from_smiles(smiles_list))