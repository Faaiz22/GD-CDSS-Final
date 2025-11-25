# gd_cdss/app.py

"""
GD-CDSS: Gene-Drug Clinical Decision Support System
Streamlit Application for Gene-Drug Association Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Import local modules
from config import *
from models import DualBranchNet
from data_utils import (
    load_model_and_data,
    load_base_dataset,
    load_metadata,
    build_display_table
)
from ui_sections import (
    show_dataset_preview,
    format_drug_option,
    format_gene_option,
    create_drug_lookup,
    create_gene_lookup,
    filter_dataframe_by_text,
    get_association_label_text
)

# Page configuration
st.set_page_config(
    page_title="GD-CDSS: Gene-Drug Repurposing",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all_data():
    """Load all model and data artifacts (cached)."""
    try:
        # Load model and embeddings
        data = load_model_and_data()
        
        # Load base dataset
        base_df, smiles_col = load_base_dataset()
        
        # Load metadata
        metadata = load_metadata()
        
        return data, base_df, metadata
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None


def overview_page(data, base_df, metadata):
    """Overview page showing dataset summary."""
    st.header("📊 Overview")
    
    st.write("""
    **GD-CDSS** is a Gene-Drug Clinical Decision Support System that predicts associations 
    between genes and drugs using a dual-branch neural network trained on gene-drug interaction data.
    """)
    
    # System metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Gene-Drug Pairs", len(data["pair_df"]))
    
    with col2:
        n_genes = data["pair_df"]["Gene_ID"].nunique()
        st.metric("Unique Genes", n_genes)
    
    with col3:
        n_drugs = data["pair_df"]["Drug_ID"].nunique()
        st.metric("Unique Drugs", n_drugs)
    
    with col4:
        assoc_rate = data["pair_df"]["Association_Label"].mean()
        st.metric("Association Rate", f"{assoc_rate:.1%}")
    
    # Dataset preview
    st.subheader("📋 Dataset Preview")
    
    if base_df is not None and metadata is not None:
        display_df = build_display_table(
            data["pair_df"].head(100),
            base_df,
            metadata
        )
        preview = show_dataset_preview(display_df, n_rows=10)
        st.dataframe(preview, use_container_width=True)
    else:
        st.warning("Base dataset or metadata not available for preview.")


def explore_associations_page(data, base_df, metadata):
    """Page for exploring gene-drug associations."""
    st.header("🔍 Explore Associations")
    
    # Build full display table
    display_df = build_display_table(
        data["pair_df"],
        base_df,
        metadata
    )
    
    # Filters
    st.subheader("🔎 Search & Filter")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gene_search = st.text_input(
            "Search by Gene Symbol or Name:",
            placeholder="e.g., TP53, BRCA1"
        )
    
    with col2:
        drug_search = st.text_input(
            "Search by Drug Generic Name or Drug Name:",
            placeholder="e.g., Aspirin, Ibuprofen"
        )
    
    # Apply filters
    filtered_df = display_df.copy()
    
    if gene_search:
        filtered_df = filter_dataframe_by_text(
            filtered_df,
            gene_search,
            ["Gene_Symbol", "Gene_Name"]
        )
    
    if drug_search:
        filtered_df = filter_dataframe_by_text(
            filtered_df,
            drug_search,
            ["Drug_Generic_Name", "Drug_Name"]
        )
    
    # Display results
    st.subheader(f"📊 Results ({len(filtered_df)} associations)")
    
    # Add association label column for display
    if "Association_Label" in filtered_df.columns:
        filtered_df["Association_Status"] = filtered_df["Association_Label"].apply(
            get_association_label_text
        )
    
    # Select columns to display
    display_cols = [
        "Drug_ID", "Drug_Generic_Name", "Drug_Name", "Drug_SMILES",
        "Gene_ID", "Gene_Symbol", "Gene_Name", "Association_Status"
    ]
    display_cols = [c for c in display_cols if c in filtered_df.columns]
    
    st.dataframe(
        filtered_df[display_cols].head(100),
        use_container_width=True
    )
    
    # Statistics
    if len(filtered_df) > 0:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Filtered Pairs", len(filtered_df))
        
        with col2:
            if "Association_Label" in filtered_df.columns:
                assoc_count = filtered_df["Association_Label"].sum()
                st.metric("Associated Pairs", int(assoc_count))
        
        with col3:
            if "Association_Label" in filtered_df.columns:
                assoc_rate = filtered_df["Association_Label"].mean()
                st.metric("Association Rate", f"{assoc_rate:.1%}")


def single_pair_scoring_page(data, base_df, metadata):
    """Page for scoring individual gene-drug pairs."""
    st.header("🎯 Single Pair Scoring")
    
    st.write("Select a drug and gene to predict their association probability.")
    
    # Build full display table for lookups
    display_df = build_display_table(
        data["pair_df"],
        base_df,
        metadata
    )
    
    # Get unique drugs and genes
    unique_drugs = display_df.drop_duplicates(subset=["Drug_ID"])
    unique_genes = display_df.drop_duplicates(subset=["Gene_ID"])
    
    # Create selection options
    drug_options = [format_drug_option(row) for _, row in unique_drugs.iterrows()]
    gene_options = [format_gene_option(row) for _, row in unique_genes.iterrows()]
    
    # Create lookups
    drug_lookup = create_drug_lookup(unique_drugs)
    gene_lookup = create_gene_lookup(unique_genes)
    
    # Selection boxes
    col1, col2 = st.columns(2)
    
    with col1:
        selected_drug_option = st.selectbox(
            "Select Drug:",
            options=drug_options,
            help="Choose a drug from the list"
        )
    
    with col2:
        selected_gene_option = st.selectbox(
            "Select Gene:",
            options=gene_options,
            help="Choose a gene from the list"
        )
    
    # Predict button
    if st.button("🔮 Predict Association", type="primary"):
        # Get selected drug and gene data
        drug_row = drug_lookup[selected_drug_option]
        gene_row = gene_lookup[selected_gene_option]
        
        drug_id = drug_row["Drug_ID"]
        gene_id = gene_row["Gene_ID"]
        
        # Find indices in pair_df
        drug_indices = data["pair_df"][data["pair_df"]["Drug_ID"] == drug_id].index.tolist()
        gene_indices = data["pair_df"][data["pair_df"]["Gene_ID"] == gene_id].index.tolist()
        
        if not drug_indices or not gene_indices:
            st.error("Could not find embeddings for this drug-gene pair.")
        else:
            # Get embeddings (use first occurrence)
            drug_emb = data["drug_emb"][drug_indices[0]]
            prot_emb = data["prot_emb"][gene_indices[0]]
            
            # Predict
            prob = data["model"].predict_proba(drug_emb, prot_emb)
            
            # Display results
            st.success("✅ Prediction Complete!")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Association Probability", f"{prob:.3f}")
            
            with col2:
                confidence = "High" if prob > 0.7 or prob < 0.3 else "Moderate"
                st.metric("Confidence", confidence)
            
            with col3:
                prediction = "Associated" if prob > 0.5 else "Not Associated"
                st.metric("Prediction", prediction)
            
            # Interpretation
            st.subheader("📊 Interpretation")
            
            if prob > 0.7:
                st.success("✅ **Strong Association**: This drug-gene pair shows high predicted association. Consider for further investigation.")
            elif prob > 0.5:
                st.warning("⚠️ **Moderate Association**: This pair shows moderate association. Additional validation recommended.")
            else:
                st.info("ℹ️ **Low Association**: This pair shows low predicted association.")
            
            # Details table
            st.subheader("📋 Pair Details")
            
            details = pd.DataFrame([
                {"Property": "Drug ID", "Value": drug_id},
                {"Property": "Drug Generic Name", "Value": drug_row.get("Drug_Generic_Name", "")},
                {"Property": "Drug Name", "Value": drug_row.get("Drug_Name", "")},
                {"Property": "Drug SMILES", "Value": drug_row.get("Drug_SMILES", "")},
                {"Property": "Gene ID", "Value": gene_id},
                {"Property": "Gene Symbol", "Value": gene_row.get("Gene_Symbol", "")},
                {"Property": "Gene Name", "Value": gene_row.get("Gene_Name", "")},
            ])
            
            st.table(details)


def main():
    """Main application entry point."""
    
    # Header
    st.markdown(
        '<h1 class="main-header">💊 GD-CDSS: Gene-Drug Clinical Decision Support System</h1>',
        unsafe_allow_html=True
    )
    
    # Load data
    with st.spinner("Loading model and data..."):
        data, base_df, metadata = load_all_data()
    
    if data is None:
        st.error("❌ Failed to load required files. Please ensure all artifacts are present in the 'artifacts/' directory.")
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("📍 Navigation")
    page = st.sidebar.radio(
        "Select Page:",
        ["Overview", "Explore Associations", "Single Pair Scoring"]
    )
    
    # Route to appropriate page
    if page == "Overview":
        overview_page(data, base_df, metadata)
    elif page == "Explore Associations":
        explore_associations_page(data, base_df, metadata)
    elif page == "Single Pair Scoring":
        single_pair_scoring_page(data, base_df, metadata)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**GD-CDSS** v1.0\n\n"
        "Gene-Drug Clinical Decision Support System\n\n"
        "Powered by PyTorch & Streamlit"
    )


if __name__ == "__main__":
    main()
