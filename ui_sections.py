# gd_cdss/ui_sections.py

"""
UI helper functions for Streamlit interface.
"""

import pandas as pd
from typing import Dict


def show_dataset_preview(df: pd.DataFrame, n_rows: int = 10):
    """
    Display a preview of the dataset with canonical columns.
    
    Args:
        df: DataFrame to display
        n_rows: Number of rows to show
    """
    display_cols = [
        "Drug_ID", "Drug_Generic_Name", "Drug_Name", "Drug_SMILES",
        "Gene_ID", "Gene_Symbol", "Gene_Name"
    ]
    
    # Filter to only existing columns
    available_cols = [c for c in display_cols if c in df.columns]
    
    return df[available_cols].head(n_rows)


def format_drug_option(row: pd.Series) -> str:
    """
    Format a drug row for selectbox display.
    
    Args:
        row: DataFrame row containing drug information
        
    Returns:
        Formatted string: "Drug_ID | GenericName (DrugName)"
    """
    drug_id = row.get("Drug_ID", "")
    generic = row.get("Drug_Generic_Name", "")
    name = row.get("Drug_Name", "")
    
    # Build label parts
    parts = [str(drug_id)]
    
    if generic and str(generic).strip():
        if name and str(name).strip() and name != generic:
            parts.append(f"{generic} ({name})")
        else:
            parts.append(generic)
    elif name and str(name).strip():
        parts.append(name)
    
    return " | ".join(parts)


def format_gene_option(row: pd.Series) -> str:
    """
    Format a gene row for selectbox display.
    
    Args:
        row: DataFrame row containing gene information
        
    Returns:
        Formatted string: "Gene_ID | GeneSymbol (GeneName)"
    """
    gene_id = row.get("Gene_ID", "")
    symbol = row.get("Gene_Symbol", "")
    name = row.get("Gene_Name", "")
    
    # Build label parts
    parts = [str(gene_id)]
    
    if symbol and str(symbol).strip():
        if name and str(name).strip():
            parts.append(f"{symbol} ({name})")
        else:
            parts.append(symbol)
    elif name and str(name).strip():
        parts.append(name)
    
    return " | ".join(parts)


def create_drug_lookup(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Create a lookup dictionary for drugs by formatted option string.
    
    Args:
        df: DataFrame containing drug information
        
    Returns:
        Dictionary mapping formatted option string to row data
    """
    lookup = {}
    for idx, row in df.iterrows():
        option_str = format_drug_option(row)
        lookup[option_str] = row
    return lookup


def create_gene_lookup(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Create a lookup dictionary for genes by formatted option string.
    
    Args:
        df: DataFrame containing gene information
        
    Returns:
        Dictionary mapping formatted option string to row data
    """
    lookup = {}
    for idx, row in df.iterrows():
        option_str = format_gene_option(row)
        lookup[option_str] = row
    return lookup


def format_smiles_display(smiles: str, max_length: int = 50) -> str:
    """
    Format SMILES string for display, truncating if too long.
    
    Args:
        smiles: SMILES string
        max_length: Maximum display length
        
    Returns:
        Formatted SMILES string
    """
    if not smiles or not isinstance(smiles, str):
        return ""
    
    smiles = str(smiles).strip()
    if len(smiles) <= max_length:
        return smiles
    else:
        return smiles[:max_length] + "..."


def get_association_label_text(label: int) -> str:
    """
    Convert association label to human-readable text.
    
    Args:
        label: 0 or 1 indicating association status
        
    Returns:
        Human-readable label
    """
    return "Associated" if label == 1 else "Not Associated"


def filter_dataframe_by_text(df: pd.DataFrame, 
                             search_text: str,
                             search_columns: list) -> pd.DataFrame:
    """
    Filter DataFrame rows by text search across specified columns.
    
    Args:
        df: DataFrame to filter
        search_text: Text to search for
        search_columns: List of column names to search in
        
    Returns:
        Filtered DataFrame
    """
    if not search_text or not search_text.strip():
        return df
    
    search_text = search_text.lower().strip()
    
    mask = pd.Series([False] * len(df), index=df.index)
    
    for col in search_columns:
        if col in df.columns:
            mask |= df[col].astype(str).str.lower().str.contains(search_text, na=False)
    
    return df[mask]
