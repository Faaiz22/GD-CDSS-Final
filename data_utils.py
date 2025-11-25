# gd_cdss/data_utils.py

"""
Data loading and processing utilities for GD-CDSS.
Handles robust loading of model artifacts and datasets with column normalization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple
import pickle

from config import *
from models import DualBranchNet


@st.cache_resource
def load_model_and_data():
    """
    Load trained model and all embeddings (cached).
    
    Returns:
        dict: Dictionary containing model, embeddings, and pair_df
    """
    try:
        # Load embeddings
        drug_emb = np.load(FEATURES_DRUG)
        prot_emb = np.load(FEATURES_PROTEIN)
        fusion_emb = np.load(FUSION_EMBEDDINGS)
        
        # Load pair labels/indices
        pair_df = pd.read_csv(PAIRWISE_INDEX_LABELS)
        
        # Normalize column names
        pair_df.columns = [c.strip() for c in pair_df.columns]
        
        # Load model
        model = DualBranchNet(drug_emb.shape[1], prot_emb.shape[1])
        model.load_state_dict(torch.load(MODEL, map_location="cpu"))
        model.eval()
        
        # Load scalers
        with open(SCALERS, "rb") as f:
            scalers = pickle.load(f)
        
        return {
            "model": model,
            "drug_emb": drug_emb,
            "prot_emb": prot_emb,
            "fusion_emb": fusion_emb,
            "pair_df": pair_df,
            "scalers": scalers
        }
    
    except FileNotFoundError as e:
        st.error(f"Missing required file: {e}")
        return None
    except Exception as e:
        st.error(f"Error loading model/data: {e}")
        return None


@st.cache_resource
def load_base_dataset():
    """
    Load base gene-drug dataset with SMILES and metadata.
    
    Returns:
        tuple: (DataFrame, smiles_column_name) or (None, None) on error
    """
    try:
        df = pd.read_csv(GENE_DRUG_DATASET)
        
        # Normalize column names
        df.columns = [c.strip() for c in df.columns]
        
        # Find SMILES column
        smiles_col = next(
            (c for c in df.columns if "smiles" in c.lower()),
            None
        )
        
        if smiles_col is None:
            st.warning("No SMILES column found in base dataset")
            return df, None
        
        # Drop rows with missing SMILES
        df = df.dropna(subset=[smiles_col])
        
        return df, smiles_col
    
    except FileNotFoundError:
        st.warning(f"Base dataset not found at {GENE_DRUG_DATASET}")
        return None, None
    except Exception as e:
        st.error(f"Error loading base dataset: {e}")
        return None, None


@st.cache_resource
def load_metadata():
    """
    Load gene and drug metadata for display names.
    
    Returns:
        dict: Dictionary with gene and drug metadata mappings
    """
    metadata = {
        "gene_symbol": {},
        "gene_name": {},
        "gene_alias": {},
        "drug_name": {},
        "drug_generic_name": {},
        "drug_alias": {}
    }
    
    # Load gene metadata
    if GENE_METADATA.exists():
        try:
            gene_df = pd.read_csv(GENE_METADATA).fillna("")
            gene_df.columns = [c.strip() for c in gene_df.columns]
            
            if "Gene_ID" in gene_df.columns:
                # Gene symbol
                symbol_col = next(
                    (c for c in gene_df.columns if "symbol" in c.lower()),
                    None
                )
                if symbol_col:
                    metadata["gene_symbol"] = dict(
                        zip(gene_df["Gene_ID"], gene_df[symbol_col])
                    )
                
                # Gene name
                name_col = next(
                    (c for c in gene_df.columns if "name" in c.lower() and "symbol" not in c.lower()),
                    None
                )
                if name_col:
                    metadata["gene_name"] = dict(
                        zip(gene_df["Gene_ID"], gene_df[name_col])
                    )
                
                # Gene aliases
                alias_col = next(
                    (c for c in gene_df.columns if "alias" in c.lower()),
                    None
                )
                if alias_col:
                    metadata["gene_alias"] = dict(
                        zip(gene_df["Gene_ID"], gene_df[alias_col])
                    )
        
        except Exception as e:
            st.warning(f"Error loading gene metadata: {e}")
    
    # Load drug metadata
    if DRUG_METADATA.exists():
        try:
            drug_df = pd.read_csv(DRUG_METADATA).fillna("")
            drug_df.columns = [c.strip() for c in drug_df.columns]
            
            if "Drug_ID" in drug_df.columns:
                # Drug name
                name_col = next(
                    (c for c in drug_df.columns if "name" in c.lower() and "generic" not in c.lower()),
                    None
                )
                if name_col:
                    metadata["drug_name"] = dict(
                        zip(drug_df["Drug_ID"], drug_df[name_col])
                    )
                
                # Drug generic name
                generic_col = next(
                    (c for c in drug_df.columns if "generic" in c.lower()),
                    None
                )
                if generic_col:
                    metadata["drug_generic_name"] = dict(
                        zip(drug_df["Drug_ID"], drug_df[generic_col])
                    )
                
                # Drug aliases
                alias_col = next(
                    (c for c in drug_df.columns if "alias" in c.lower()),
                    None
                )
                if alias_col:
                    metadata["drug_alias"] = dict(
                        zip(drug_df["Drug_ID"], drug_df[alias_col])
                    )
        
        except Exception as e:
            st.warning(f"Error loading drug metadata: {e}")
    
    return metadata


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize DataFrame column names (strip whitespace).
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with normalized column names
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def build_display_table(pair_df: pd.DataFrame,
                       base_df: pd.DataFrame,
                       metadata: Dict) -> pd.DataFrame:
    """
    Build enriched display table with metadata and SMILES.
    
    Args:
        pair_df: Gene-drug pair DataFrame
        base_df: Base dataset with SMILES
        metadata: Metadata dictionaries
        
    Returns:
        Enriched DataFrame for display
    """
    df = pair_df.copy()
    
    # Add gene metadata
    if "Gene_ID" in df.columns:
        df["Gene_Symbol"] = df["Gene_ID"].map(
            metadata.get("gene_symbol", {})
        ).fillna("")
        
        df["Gene_Name"] = df["Gene_ID"].map(
            metadata.get("gene_name", {})
        ).fillna("")
        
        df["Gene_Aliases"] = df["Gene_ID"].map(
            metadata.get("gene_alias", {})
        ).fillna("")
    
    # Add drug metadata
    if "Drug_ID" in df.columns:
        df["Drug_Name"] = df["Drug_ID"].map(
            metadata.get("drug_name", {})
        ).fillna("")
        
        df["Drug_Generic_Name"] = df["Drug_ID"].map(
            metadata.get("drug_generic_name", {})
        ).fillna("")
        
        df["Drug_Aliases"] = df["Drug_ID"].map(
            metadata.get("drug_alias", {})
        ).fillna("")
    
    # Add SMILES from base dataset
    if base_df is not None and "Drug_ID" in df.columns:
        # Find SMILES column in base_df
        smiles_col = next(
            (c for c in base_df.columns if "smiles" in c.lower()),
            None
        )
        
        if smiles_col:
            # Create Drug_ID -> SMILES mapping
            smiles_map = dict(
                zip(base_df["Drug_ID"], base_df[smiles_col])
            )
            df["Drug_SMILES"] = df["Drug_ID"].map(smiles_map).fillna("")
    
    return df


def get_drug_embedding_by_id(drug_id: str,
                             pair_df: pd.DataFrame,
                             drug_emb: np.ndarray) -> Optional[np.ndarray]:
    """
    Get drug embedding vector by Drug_ID.
    
    Args:
        drug_id: Drug identifier
        pair_df: Pair DataFrame with indices
        drug_emb: Drug embedding matrix
        
    Returns:
        Embedding vector or None if not found
    """
    indices = pair_df[pair_df["Drug_ID"] == drug_id].index.tolist()
    
    if not indices:
        return None
    
    # Use first occurrence
    return drug_emb[indices[0]]


def get_protein_embedding_by_id(gene_id: str,
                               pair_df: pd.DataFrame,
                               prot_emb: np.ndarray) -> Optional[np.ndarray]:
    """
    Get protein embedding vector by Gene_ID.
    
    Args:
        gene_id: Gene identifier
        pair_df: Pair DataFrame with indices
        prot_emb: Protein embedding matrix
        
    Returns:
        Embedding vector or None if not found
    """
    indices = pair_df[pair_df["Gene_ID"] == gene_id].index.tolist()
    
    if not indices:
        return None
    
    # Use mean of all occurrences for this gene
    return prot_emb[indices].mean(axis=0)


def get_smiles_by_drug_id(drug_id: str,
                         base_df: pd.DataFrame,
                         smiles_col: str) -> Optional[str]:
    """
    Get SMILES string by Drug_ID.
    
    Args:
        drug_id: Drug identifier
        base_df: Base dataset DataFrame
        smiles_col: Name of SMILES column
        
    Returns:
        SMILES string or None if not found
    """
    if base_df is None or smiles_col is None:
        return None
    
    matches = base_df[base_df["Drug_ID"] == drug_id]
    
    if matches.empty:
        return None
    
    return matches.iloc[0][smiles_col]


def save_predictions_csv(predictions: pd.DataFrame,
                        filename: str = "predictions.csv") -> str:
    """
    Convert predictions DataFrame to CSV string for download.
    
    Args:
        predictions: DataFrame with predictions
        filename: Filename for download
        
    Returns:
        CSV string
    """
    return predictions.to_csv(index=False)


def validate_artifacts() -> Tuple[bool, list]:
    """
    Validate that all required artifact files exist.
    
    Returns:
        tuple: (all_present: bool, missing_files: list)
    """
    required_files = [
        FEATURES_DRUG,
        FEATURES_PROTEIN,
        FUSION_EMBEDDINGS,
        SCALERS,
        MODEL,
        PAIRWISE_INDEX_LABELS,
        GENE_DRUG_DATASET
    ]
    
    missing = [f for f in required_files if not f.exists()]
    
    return len(missing) == 0, missing
