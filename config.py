# gd_cdss/config.py

from pathlib import Path

# Directories
ARTIFACTS_DIR = Path("artifacts")
CACHE_DIR = Path("cache")

# Artifact file paths
FEATURES_DRUG = ARTIFACTS_DIR / "features_drug.npy"
FEATURES_PROTEIN = ARTIFACTS_DIR / "features_protein.npy"
FUSION_EMBEDDINGS = ARTIFACTS_DIR / "fusion_embeddings.npy"
SCALERS = ARTIFACTS_DIR / "scalers.pkl"
MODEL = ARTIFACTS_DIR / "model.pt"
PAIRWISE_INDEX_LABELS = ARTIFACTS_DIR / "pairwise_index_labels.csv"
GENE_DRUG_DATASET = ARTIFACTS_DIR / "gene_drug_dataset.csv"
DRUG_METADATA = ARTIFACTS_DIR / "drug_metadata.csv"
GENE_METADATA = ARTIFACTS_DIR / "gene_metadata.csv"

# Model parameters
RANDOM_SEED = 42
HIDDEN_SIZE = 128

# Feature parameters
FINGERPRINT_BITS = 2048
FINGERPRINT_PCA_DIM = 128
KMER = 3
KMER_PCA_DIM = 64
