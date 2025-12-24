import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges


def safe_call(func, *args, default=None):
    """Call a function if exists, else return default."""
    if func is None:
        return default
    try:
        return func(*args)
    except:
        return default

def get_morgan_fp(mol, n_bits=1024):
    from rdkit.Chem import AllChem
    # Morgan FP 会自动编码手性（如果 mol 中有手性）
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return fp.ToBitString()

def get_stasis(mol):
    rings = [r for r in mol.GetRingInfo().AtomRings() 
             if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
    sizes = [len(r) for r in rings]
    if sizes:
        return [max(sizes), np.mean(sizes), min(sizes)]
    else:
        return [0,0,0]

def compute_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Gasteiger charges
    try:
        ComputeGasteigerCharges(mol)
        charges = [
            float(a.GetProp('_GasteigerCharge'))
            if a.HasProp('_GasteigerCharge') else 0.0
            for a in mol.GetAtoms()
        ]
        charge_mean = sum(charges)/len(charges)
        charge_sum = sum(charges)
        charge_max = max(charges)
        charge_min = min(charges)
        charge_median = np.median(charges)
    except:
        charge_mean, charge_sum = None, None

    aromatic_atoms = sum(a.GetIsAromatic() for a in mol.GetAtoms())

    # Basic descriptors
    MolWt = Descriptors.MolWt(mol)
    MolLogP=Crippen.MolLogP(mol)
    MolMR=Crippen.MolMR(mol)
    TPSA=Descriptors.TPSA(mol)
    HeavyAtomCount=Descriptors.HeavyAtomCount(mol)
    NumValenceElectrons=Descriptors.NumValenceElectrons(mol)
    HBA=Lipinski.NumHAcceptors(mol)
    HBD=Lipinski.NumHDonors(mol)
    NumRotatableBonds=Lipinski.NumRotatableBonds(mol)
    AromaticAtoms=aromatic_atoms
    AromaticRings = get_stasis(mol)

    feats = [
        MolWt, MolLogP, MolMR, TPSA, HeavyAtomCount,
        NumValenceElectrons, HBA, HBD, NumRotatableBonds,
        AromaticAtoms, AromaticRings[0], AromaticRings[1], AromaticRings[2],
        charge_mean, charge_sum, charge_max, charge_min, charge_median
    ]

    # Morgan Fingerprint (含手性信息)
    fp = get_morgan_fp(mol)
    feats.append(fp)

    return feats