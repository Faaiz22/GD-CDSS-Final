# gd_cdss/chem_utils.py

"""
RDKit chemistry utilities extracted from notebook.
Handles molecular feature computation, validation, and drug-likeness scoring.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import BRICS
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from typing import Optional, Dict, List


# Initialize PAINS filter catalog
try:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    PAINS_CATALOG = FilterCatalog(params)
except Exception:
    PAINS_CATALOG = None


def safe_mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    """
    Safely convert SMILES string to RDKit Mol object.
    
    Args:
        smiles: SMILES string
        
    Returns:
        RDKit Mol object or None if invalid
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Try without sanitization
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is not None:
                Chem.SanitizeMol(mol)
        return mol
    except Exception:
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is not None:
                Chem.SanitizeMol(mol)
            return mol
        except Exception:
            return None


def smiles_is_valid(smiles: str) -> bool:
    """
    Check if SMILES string is valid.
    
    Args:
        smiles: SMILES string
        
    Returns:
        True if valid, False otherwise
    """
    return safe_mol_from_smiles(smiles) is not None


def compute_morgan_fingerprint(mol: Chem.Mol, 
                               radius: int = 2,
                               nBits: int = 2048) -> Optional[np.ndarray]:
    """
    Compute Morgan fingerprint for a molecule.
    
    Args:
        mol: RDKit Mol object
        radius: Fingerprint radius
        nBits: Number of bits
        
    Returns:
        Fingerprint as numpy array or None
    """
    if mol is None:
        return None
    
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        arr = np.zeros((nBits,), dtype=int)
        from rdkit import DataStructs
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return None


def compute_rdkit_2d_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    """
    Compute RDKit 2D molecular descriptors.
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        Dictionary of descriptor values
    """
    if mol is None:
        return {
            'MolLogP': np.nan,
            'TPSA': np.nan,
            'NumHAcceptors': np.nan,
            'NumHDonors': np.nan,
            'NumRotatableBonds': np.nan,
            'RingCount': np.nan,
            'MolWt': np.nan
        }
    
    try:
        return {
            'MolLogP': Descriptors.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'RingCount': Descriptors.RingCount(mol),
            'MolWt': Descriptors.ExactMolWt(mol)
        }
    except Exception:
        return {k: np.nan for k in ['MolLogP', 'TPSA', 'NumHAcceptors', 
                                     'NumHDonors', 'NumRotatableBonds', 
                                     'RingCount', 'MolWt']}


def compute_physchem_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    """
    Compute physicochemical descriptors from RDKit mol.
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        Dictionary of physchem descriptors
    """
    if mol is None:
        return {
            "Flexibility": 0.0,
            "Lipophilicity": 0.0,
            "Size": 0.0,
            "Polarity": 0.0,
            "Insolubility": 0.0,
            "Insaturation": 0.0
        }
    
    try:
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        
        return {
            "Flexibility": float(Descriptors.NumRotatableBonds(mol)),
            "Lipophilicity": float(logp),
            "Size": float(Descriptors.MolWt(mol)),
            "Polarity": float(tpsa),
            "Insolubility": float(logp - tpsa / 100.0),
            "Insaturation": float(Descriptors.FractionCSP3(mol))
        }
    except Exception:
        return {
            "Flexibility": 0.0,
            "Lipophilicity": 0.0,
            "Size": 0.0,
            "Polarity": 0.0,
            "Insolubility": 0.0,
            "Insaturation": 0.0
        }


def compute_3d_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    """
    Compute 3D ETKDG-based shape descriptors.
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        Dictionary of 3D descriptors
    """
    if mol is None:
        return {
            "asphericity": 0.0,
            "eccentricity": 0.0,
            "inertial_shape_factor": 0.0,
            "radius_of_gyration": 0.0,
            "spherocity_index": 0.0,
            "mol_wt_3d": 0.0,
            "heavy_atoms_3d": 0.0
        }
    
    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)
        
        return {
            "asphericity": float(rdMolDescriptors.CalcAsphericity(mol)),
            "eccentricity": float(rdMolDescriptors.CalcEccentricity(mol)),
            "inertial_shape_factor": float(rdMolDescriptors.CalcInertialShapeFactor(mol)),
            "radius_of_gyration": float(rdMolDescriptors.CalcRadiusOfGyration(mol)),
            "spherocity_index": float(rdMolDescriptors.CalcSpherocityIndex(mol)),
            "mol_wt_3d": float(rdMolDescriptors.CalcExactMolWt(mol)),
            "heavy_atoms_3d": float(mol.GetNumHeavyAtoms())
        }
    except Exception:
        return {
            "asphericity": 0.0,
            "eccentricity": 0.0,
            "inertial_shape_factor": 0.0,
            "radius_of_gyration": 0.0,
            "spherocity_index": 0.0,
            "mol_wt_3d": 0.0,
            "heavy_atoms_3d": 0.0
        }


def calculate_qed(mol: Chem.Mol) -> float:
    """
    Calculate QED (Quantitative Estimate of Drug-likeness) score.
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        QED score (0-1)
    """
    if mol is None:
        return 0.0
    
    try:
        return float(QED.qed(mol))
    except Exception:
        return 0.0


def calculate_sa_score(mol: Chem.Mol) -> float:
    """
    Calculate synthetic accessibility score (approximation).
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        SA score (1-10, lower is easier to synthesize)
    """
    if mol is None:
        return 10.0
    
    try:
        # Try using official SA score if available
        from rdkit.Chem.Scoring.Scoring import CalculateSAScore
        return float(CalculateSAScore(mol))
    except Exception:
        # Fallback to heuristic approximation
        try:
            atoms = mol.GetNumHeavyAtoms()
            rings = len(Chem.GetSSSR(mol))
            logp = Descriptors.MolLogP(mol)
            sa = 5 + 0.05 * atoms + 0.1 * rings - 0.2 * logp
            return float(min(max(sa, 1.0), 10.0))
        except Exception:
            return 5.0


def check_pains(mol: Chem.Mol) -> bool:
    """
    Check if molecule contains PAINS (Pan-Assay Interference Compounds) substructures.
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        True if PAINS detected, False otherwise
    """
    if mol is None or PAINS_CATALOG is None:
        return False
    
    try:
        return PAINS_CATALOG.HasMatch(mol)
    except Exception:
        return False


def calculate_tanimoto_similarity(smiles1: str, 
                                  smiles2: str,
                                  nBits: int = 2048) -> float:
    """
    Calculate Tanimoto similarity between two molecules.
    
    Args:
        smiles1: First SMILES string
        smiles2: Second SMILES string
        nBits: Fingerprint size
        
    Returns:
        Tanimoto similarity (0-1)
    """
    mol1 = safe_mol_from_smiles(smiles1)
    mol2 = safe_mol_from_smiles(smiles2)
    
    if mol1 is None or mol2 is None:
        return 0.0
    
    try:
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=nBits)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=nBits)
        from rdkit import DataStructs
        return float(DataStructs.TanimotoSimilarity(fp1, fp2))
    except Exception:
        return 0.0


def get_murcko_scaffold(smiles: str) -> Optional[str]:
    """
    Extract Murcko scaffold from molecule.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Scaffold SMILES or None
    """
    mol = safe_mol_from_smiles(smiles)
    if mol is None:
        return None
    
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, True)
    except Exception:
        return None


def brics_decompose(smiles: str) -> List[str]:
    """
    Decompose molecule into BRICS fragments.
    
    Args:
        smiles: SMILES string
        
    Returns:
        List of fragment SMILES
    """
    mol = safe_mol_from_smiles(smiles)
    if mol is None:
        return []
    
    try:
        frags = list(BRICS.BRICSDecompose(mol))
        return [f for f in frags if f and len(f) > 1]
    except Exception:
        return []


def compute_bioavailability_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    """
    Compute bioavailability radar descriptors.
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        Dictionary of bioavailability descriptors
    """
    if mol is None:
        return {
            "Flexibility": 0.0,
            "Lipophilicity": 0.0,
            "Size": 0.0,
            "Polarity": 0.0,
            "Insolubility": 0.0,
            "Insaturation": 0.0
        }
    
    try:
        flex = Descriptors.NumRotatableBonds(mol)
        lip = Descriptors.MolLogP(mol)
        size = Descriptors.MolWt(mol)
        pol = Descriptors.TPSA(mol)
        insol = lip - pol / 100.0
        insat = Descriptors.FractionCSP3(mol)
        
        return {
            "Flexibility": float(flex),
            "Lipophilicity": float(lip),
            "Size": float(size),
            "Polarity": float(pol),
            "Insolubility": float(insol),
            "Insaturation": float(insat)
        }
    except Exception:
        return {
            "Flexibility": 0.0,
            "Lipophilicity": 0.0,
            "Size": 0.0,
            "Polarity": 0.0,
            "Insolubility": 0.0,
            "Insaturation": 0.0
        }


def classify_boiled_egg(logp: float, tpsa: float) -> str:
    """
    Classify molecule in BOILED-Egg plot.
    
    Args:
        logp: LogP value
        tpsa: TPSA value
        
    Returns:
        Classification string
    """
    in_bbb = (tpsa <= 90) and (-0.7 <= logp <= 6.0)
    in_gi = (tpsa <= 130) and (-0.7 <= logp <= 6.0)
    
    if in_bbb:
        return "BBB permeant (yellow yolk + GI)"
    elif in_gi:
        return "GI absorption (white egg only)"
    else:
        return "Outside BOILED-Egg (low GI/BBB probability)"


def calculate_lipinski_violations(mol: Chem.Mol) -> int:
    """
    Calculate number of Lipinski Rule of Five violations.
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        Number of violations (0-4)
    """
    if mol is None:
        return 4
    
    try:
        violations = 0
        
        mw = Descriptors.MolWt(mol)
        if mw > 500:
            violations += 1
        
        logp = Descriptors.MolLogP(mol)
        if logp > 5:
            violations += 1
        
        hbd = Descriptors.NumHDonors(mol)
        if hbd > 5:
            violations += 1
        
        hba = Descriptors.NumHAcceptors(mol)
        if hba > 10:
            violations += 1
        
        return violations
    except Exception:
        return 4
