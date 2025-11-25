# gd_cdss/visualization.py

"""
Visualization utilities for molecular properties and predictions.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Optional
from rdkit import Chem

from chem_utils import (
    safe_mol_from_smiles,
    compute_bioavailability_descriptors,
    classify_boiled_egg
)
from rdkit.Chem import Descriptors


def make_bioavailability_radar(desc_dict: Dict[str, float]) -> go.Figure:
    """
    Create bioavailability radar plot.
    
    Args:
        desc_dict: Dictionary of bioavailability descriptors
        
    Returns:
        Plotly Figure object
    """
    # Define normalization ranges
    ranges = {
        "Flexibility": (0, 9),
        "Lipophilicity": (-0.7, 5.0),
        "Size": (150, 500),
        "Polarity": (20, 130),
        "Insolubility": (-2.0, 6.0),
        "Insaturation": (0.25, 1.0),
    }
    
    categories = list(ranges.keys())
    
    # Normalize values to [0, 1]
    norm_vals = []
    for key in categories:
        val = desc_dict.get(key, 0)
        lo, hi = ranges[key]
        
        if hi > lo:
            x = (val - lo) / (hi - lo)
        else:
            x = 0.0
        
        x = max(0.0, min(1.0, x))
        norm_vals.append(x)
    
    # Close the radar plot
    categories_closed = categories + [categories[0]]
    norm_vals_closed = norm_vals + [norm_vals[0]]
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=norm_vals_closed,
        theta=categories_closed,
        fill='toself',
        name='Molecule',
        line=dict(color='rgb(31, 119, 180)')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        title="Bioavailability Radar (normalized descriptors)",
        height=500
    )
    
    return fig


def make_boiled_egg_plot(mol: Chem.Mol) -> tuple:
    """
    Create BOILED-Egg plot showing GI absorption and BBB permeability.
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        tuple: (Plotly Figure, classification string)
    """
    if mol is None:
        fig = go.Figure()
        fig.add_annotation(
            text="Invalid molecule",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig, "Invalid molecule"
    
    # Calculate properties
    tpsa = Descriptors.TPSA(mol)
    logp = Descriptors.MolLogP(mol)
    
    # Classify
    classification = classify_boiled_egg(logp, tpsa)
    
    # Create figure
    fig = go.Figure()
    
    # White egg (GI absorption zone)
    fig.add_shape(
        type="rect",
        x0=-0.7, x1=6.0,
        y0=0, y1=130,
        fillcolor="white",
        opacity=0.7,
        layer="below",
        line=dict(color="black", width=1)
    )
    
    # Yellow yolk (BBB permeant zone)
    fig.add_shape(
        type="rect",
        x0=-0.7, x1=6.0,
        y0=0, y1=90,
        fillcolor="gold",
        opacity=0.6,
        layer="below",
        line=dict(color="black", width=1)
    )
    
    # Legend traces
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(color="gold", size=10),
        name="BBB-permeant zone"
    ))
    
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(color="white", size=10, line=dict(color="black", width=1)),
        name="GI absorption zone"
    ))
    
    # Plot molecule
    fig.add_trace(go.Scatter(
        x=[logp],
        y=[tpsa],
        mode='markers+text',
        marker=dict(size=12, color='red', line=dict(color='black', width=1)),
        text=["Drug"],
        textposition="top center",
        name="Molecule"
    ))
    
    fig.update_layout(
        xaxis_title="LogP (MolLogP)",
        yaxis_title="TPSA (Ų)",
        title="BOILED-Egg Plot (GI vs BBB permeability)",
        xaxis=dict(range=[-2, 7]),
        yaxis=dict(range=[0, 200]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500
    )
    
    return fig, classification


def make_association_distribution_plot(df: pd.DataFrame,
                                      gene_id: Optional[str] = None) -> go.Figure:
    """
    Create pie chart showing association distribution.
    
    Args:
        df: DataFrame with Association_Label column
        gene_id: Optional gene ID for title
        
    Returns:
        Plotly Figure object
    """
    if "Association_Label" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No association labels available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig
    
    counts = df["Association_Label"].value_counts()
    
    fig = px.pie(
        values=counts.values,
        names=['No Association', 'Known Association'],
        title=f'Association Distribution{" for " + gene_id if gene_id else ""}',
        color_discrete_sequence=['#FF6B6B', '#4ECDC4']
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig


def make_prediction_bar_chart(predictions_df: pd.DataFrame,
                              label_col: str = "Drug_Name",
                              value_col: str = "Predicted_Probability",
                              top_k: int = 10) -> go.Figure:
    """
    Create bar chart of top predictions.
    
    Args:
        predictions_df: DataFrame with predictions
        label_col: Column to use for labels
        value_col: Column to use for values
        top_k: Number of top predictions to show
        
    Returns:
        Plotly Figure object
    """
    top_df = predictions_df.head(top_k).copy()
    
    fig = px.bar(
        top_df,
        x=label_col,
        y=value_col,
        color=value_col,
        color_continuous_scale='Blues',
        title=f'Top {top_k} Candidates by {value_col}'
    )
    
    fig.update_layout(
        xaxis_title=label_col,
        yaxis_title=value_col,
        xaxis={'categoryorder': 'total descending'},
        height=500
    )
    
    return fig


def make_scatter_matrix(df: pd.DataFrame,
                       dimensions: list,
                       color_col: Optional[str] = None) -> go.Figure:
    """
    Create scatter matrix for multi-dimensional data.
    
    Args:
        df: DataFrame with data
        dimensions: List of column names for dimensions
        color_col: Optional column for coloring
        
    Returns:
        Plotly Figure object
    """
    if color_col:
        fig = px.scatter_matrix(
            df,
            dimensions=dimensions,
            color=color_col,
            title="Multi-dimensional Scatter Matrix"
        )
    else:
        fig = px.scatter_matrix(
            df,
            dimensions=dimensions,
            title="Multi-dimensional Scatter Matrix"
        )
    
    fig.update_traces(diagonal_visible=False)
    
    return fig


def make_similarity_heatmap(similarity_matrix: np.ndarray,
                           labels: Optional[list] = None) -> go.Figure:
    """
    Create heatmap for similarity matrix.
    
    Args:
        similarity_matrix: Square matrix of similarities
        labels: Optional labels for axes
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=labels,
        y=labels,
        colorscale='Viridis',
        colorbar=dict(title="Similarity")
    ))
    
    fig.update_layout(
        title="Tanimoto Similarity Heatmap",
        xaxis_title="Molecule",
        yaxis_title="Molecule",
        height=600
    )
    
    return fig


def make_molecular_property_table(mol: Chem.Mol) -> pd.DataFrame:
    """
    Create table of molecular properties.
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        DataFrame with property names and values
    """
    if mol is None:
        return pd.DataFrame({
            'Property': ['Error'],
            'Value': ['Invalid molecule']
        })
    
    try:
        properties = {
            'Molecular Weight': f"{Descriptors.ExactMolWt(mol):.2f}",
            'LogP (MolLogP)': f"{Descriptors.MolLogP(mol):.2f}",
            'TPSA': f"{Descriptors.TPSA(mol):.2f}",
            'H-Bond Acceptors': str(Descriptors.NumHAcceptors(mol)),
            'H-Bond Donors': str(Descriptors.NumHDonors(mol)),
            'Rotatable Bonds': str(Descriptors.NumRotatableBonds(mol)),
            'Aromatic Rings': str(Descriptors.NumAromaticRings(mol)),
            'Heavy Atoms': str(mol.GetNumHeavyAtoms()),
            'Formal Charge': str(Chem.GetFormalCharge(mol))
        }
        
        return pd.DataFrame([
            {'Property': k, 'Value': v}
            for k, v in properties.items()
        ])
    
    except Exception:
        return pd.DataFrame({
            'Property': ['Error'],
            'Value': ['Could not compute properties']
        })


def make_confidence_gauge(probability: float) -> go.Figure:
    """
    Create gauge chart for prediction confidence.
    
    Args:
        probability: Prediction probability (0-1)
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Association Probability"},
        gauge={
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgray"},
                {'range': [0.3, 0.7], 'color': "gray"},
                {'range': [0.7, 1], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.5
            }
        }
    ))
    
    fig.update_layout(height=300)
    
    return fig
