
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import BRICS
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import pickle
import random
import time

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="GD-CDSS: Gene-Drug Repurposing System",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

ARTIFACTS_DIR = Path("artifacts")

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
class DualBranchNet(nn.Module):
    def __init__(self, d_in, p_in, hidden=128):
        super().__init__()
        self.drug_encoder = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden//2), nn.ReLU()
        )
        self.prot_encoder = nn.Sequential(
            nn.Linear(p_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden//2), nn.ReLU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1),
            nn.Sigmoid()
        )

    def forward(self, drug, prot):
        d = self.drug_encoder(drug)
        p = self.prot_encoder(prot)
        fused = torch.cat([d, p], dim=1)
        return self.fusion(fused).squeeze(1)

# -----------------------------------------------------------------------------
# LOADERS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model_and_data():
    try:
        drug_emb = np.load(ARTIFACTS_DIR / "features_drug.npy")
        prot_emb = np.load(ARTIFACTS_DIR / "features_protein.npy")

        model = DualBranchNet(drug_emb.shape[1], prot_emb.shape[1])
        model.load_state_dict(torch.load(ARTIFACTS_DIR / "model.pt", map_location="cpu"))
        model.eval()

        pair_df = pd.read_csv(ARTIFACTS_DIR / "pairwise_index_labels.csv")
        fusion_emb = np.load(ARTIFACTS_DIR / "fusion_embeddings.npy")

        with open(ARTIFACTS_DIR / "scalers.pkl", "rb") as f:
            scalers = pickle.load(f)

        return model, drug_emb, prot_emb, pair_df, fusion_emb, scalers
    except Exception as e:
        st.error(f"Error loading model or artifacts: {e}")
        return None, None, None, None, None, None


@st.cache_resource
def load_metadata():
    gene_map_symbol = {}
    gene_map_name = {}
    gene_map_alias = {}
    drug_map_name = {}
    drug_map_alias = {}

    gene_path = ARTIFACTS_DIR / "gene_metadata.csv"
    drug_path = ARTIFACTS_DIR / "drug_metadata.csv"

    if gene_path.exists():
        gdf = pd.read_csv(gene_path).fillna("")
        if "Gene_ID" in gdf.columns:
            gene_map_symbol = dict(zip(gdf["Gene_ID"], gdf.get("Gene_Symbol", "")))
            gene_map_name   = dict(zip(gdf["Gene_ID"], gdf.get("Gene_Name", "")))
            gene_map_alias  = dict(zip(gdf["Gene_ID"], gdf.get("Gene_Aliases", "")))

    if drug_path.exists():
        ddf = pd.read_csv(drug_path).fillna("")
        if "Drug_ID" in ddf.columns:
            drug_map_name  = dict(zip(ddf["Drug_ID"], ddf.get("Drug_Name", "")))
            drug_map_alias = dict(zip(ddf["Drug_ID"], ddf.get("Drug_Aliases", "")))

    return {
        "gene_symbol": gene_map_symbol,
        "gene_name": gene_map_name,
        "gene_alias": gene_map_alias,
        "drug_name": drug_map_name,
        "drug_alias": drug_map_alias,
    }

@st.cache_resource
def load_base_dataset():
    """
    Base gene–drug dataset with Drug_ID and SMILES.
    Change the filename if your artifact name differs.
    """
    path = ARTIFACTS_DIR / "gene_drug_dataset.csv"  # <-- update if needed
    if not path.exists():
        st.error(f"Base dataset not found at {path}. Put your training gene–drug CSV there.")
        return None, None, None

    df = pd.read_csv(path)
    smiles_col = next((c for c in df.columns if "smiles" in c.lower()), None)
    if smiles_col is None:
        st.error("No SMILES column found in base dataset.")
        return None, None, None
    if "Drug_ID" not in df.columns:
        st.error("Base dataset must contain a 'Drug_ID' column.")
        return None, None, None
    df = df.dropna(subset=[smiles_col])
    return df, smiles_col, df[smiles_col].unique().tolist()

@st.cache_resource
def build_fragment_and_scaffold_pools():
    df, smiles_col, all_smiles = load_base_dataset()
    if df is None:
        return None, None, None

    fragment_pool = set()
    for s in all_smiles:
        try:
            frags = BRICS.BRICSDecompose(s)
            for f in frags:
                if f and len(f) > 1:
                    fragment_pool.add(f)
        except Exception:
            continue
    fragment_pool = list(fragment_pool)

    scaffold_pool = set()
    for s in all_smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        try:
            scaff = MurckoScaffold.GetScaffoldForMol(m)
            sm = Chem.MolToSmiles(scaff, True)
            if sm and len(sm) > 1:
                scaffold_pool.add(sm)
        except Exception:
            continue
    scaffold_pool = list(scaffold_pool)

    return df, smiles_col, all_smiles, fragment_pool, scaffold_pool

# -----------------------------------------------------------------------------
# RDKit / descriptors / helpers
# -----------------------------------------------------------------------------
def safe_mol_from_smiles(smi):
    if not isinstance(smi, str) or not smi.strip():
        return None
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            m = Chem.MolFromSmiles(smi, sanitize=False)
            Chem.SanitizeMol(m)
        return m
    except Exception:
        try:
            m = Chem.MolFromSmiles(smi, sanitize=False)
            Chem.SanitizeMol(m)
            return m
        except Exception:
            return None

def smiles_is_valid(smi):
    return safe_mol_from_smiles(smi) is not None

def score_qed_smi(smi):
    m = safe_mol_from_smiles(smi)
    return float(QED.qed(m)) if m is not None else 0.0

def tanimoto(smi1, smi2, nBits=2048):
    m1, m2 = safe_mol_from_smiles(smi1), safe_mol_from_smiles(smi2)
    if m1 is None or m2 is None:
        return 0.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=nBits)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=nBits)
    from rdkit import DataStructs
    return float(DataStructs.TanimotoSimilarity(fp1, fp2))

def compute_drug_features(smiles, scalers):
    mol = safe_mol_from_smiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fp_arr = np.zeros((2048,), dtype=int)
    from rdkit import DataStructs
    DataStructs.ConvertToNumpyArray(fp, fp_arr)

    desc = {
        'MolLogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'RingCount': Descriptors.RingCount(mol),
        'MolWt': Descriptors.ExactMolWt(mol)
    }
    desc_arr = np.array(list(desc.values()), dtype=float)

    features = np.hstack([fp_arr[:128], desc_arr])
    features = scalers['drug'].transform(features.reshape(1, -1))
    return features[0]

def calculate_drug_likeness(smiles):
    mol = safe_mol_from_smiles(smiles)
    if mol is None:
        return 0.0, 10.0
    qed_score = QED.qed(mol)
    atoms = mol.GetNumHeavyAtoms()
    rings = len(Chem.GetSSSR(mol))
    logp = Descriptors.MolLogP(mol)
    sa = 5 + 0.05 * atoms + 0.1 * rings - 0.2 * logp
    sa = min(max(sa, 1.0), 10.0)
    return qed_score, sa

def predict_association(model, drug_features, prot_features):
    with torch.no_grad():
        drug_t = torch.tensor(drug_features, dtype=torch.float32).unsqueeze(0)
        prot_t = torch.tensor(prot_features, dtype=torch.float32).unsqueeze(0)
        prob = float(model(drug_t, prot_t).numpy()[0])
    return prob

# -----------------------------------------------------------------------------
# BIOAVAILABILITY + BOILED EGG
# -----------------------------------------------------------------------------
def compute_bioavailability_descriptors(mol):
    if mol is None:
        return None
    flex = Descriptors.NumRotatableBonds(mol)
    lip  = Descriptors.MolLogP(mol)
    size = Descriptors.MolWt(mol)
    pol  = Descriptors.TPSA(mol)
    insol = Descriptors.MolLogP(mol) - Descriptors.TPSA(mol) / 100.0
    insat = Descriptors.FractionCSP3(mol)
    return {
        "Flexibility": flex,
        "Lipophilicity": lip,
        "Size": size,
        "Polarity": pol,
        "Insolubility": insol,
        "Insaturation": insat,
    }

def normalize_for_radar(vals, ranges):
    norm = []
    for key, v in vals.items():
        lo, hi = ranges[key]
        x = (v - lo) / (hi - lo) if hi > lo else 0.0
        x = max(0.0, min(1.0, x))
        norm.append(x)
    return norm

def make_bioavailability_radar(desc_dict):
    ranges = {
        "Flexibility": (0, 9),
        "Lipophilicity": (-0.7, 5.0),
        "Size": (150, 500),
        "Polarity": (20, 130),
        "Insolubility": (-2.0, 6.0),
        "Insaturation": (0.25, 1.0),
    }
    categories = list(ranges.keys())
    norm_vals = normalize_for_radar(desc_dict, ranges)
    categories_closed = categories + [categories[0]]
    norm_vals_closed = norm_vals + [norm_vals[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=norm_vals_closed,
        theta=categories_closed,
        fill='toself',
        name='Molecule'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title="Bioavailability Radar (normalized descriptors)"
    )
    return fig

def classify_boiled_egg_point(logp, tpsa):
    in_bbb = (tpsa <= 90) and (-0.7 <= logp <= 6.0)
    in_gi  = (tpsa <= 130) and (-0.7 <= logp <= 6.0)
    if in_bbb:
        return "BBB permeant (yellow yolk + GI)"
    elif in_gi:
        return "GI absorption (white egg only)"
    else:
        return "Outside BOILED-Egg (low GI/BBB probability)"

def make_boiled_egg_plot(mol):
    if mol is None:
        return go.Figure(), "Invalid molecule"
    tpsa = Descriptors.TPSA(mol)
    logp = Descriptors.MolLogP(mol)
    classification = classify_boiled_egg_point(logp, tpsa)
    fig = go.Figure()
    fig.add_shape(
        type="rect",
        x0=-0.7, x1=6.0,
        y0=0, y1=130,
        fillcolor="white",
        opacity=0.7,
        layer="below",
        line=dict(color="black", width=1)
    )
    fig.add_shape(
        type="rect",
        x0=-0.7, x1=6.0,
        y0=0, y1=90,
        fillcolor="gold",
        opacity=0.6,
        layer="below",
        line=dict(color="black", width=1)
    )
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
        yaxis_title="TPSA (Å²)",
        title="BOILED-Egg Plot (GI vs BBB permeability)",
        xaxis=dict(range=[-2, 7]),
        yaxis=dict(range=[0, 200]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig, classification

# -----------------------------------------------------------------------------
# NAME / ALIAS HELPERS
# -----------------------------------------------------------------------------
def enrich_with_names(df, meta):
    df = df.copy()
    dm = meta["drug_name"]
    da = meta["drug_alias"]
    gs = meta["gene_symbol"]
    gn = meta["gene_name"]
    ga = meta["gene_alias"]
    if "Drug_ID" in df.columns:
        df["Drug_Name"]   = df["Drug_ID"].map(dm).fillna("")
        df["Drug_Aliases"] = df["Drug_ID"].map(da).fillna("")
    if "Gene_ID" in df.columns:
        df["Gene_Symbol"] = df["Gene_ID"].map(gs).fillna("")
        df["Gene_Name"]   = df["Gene_ID"].map(gn).fillna("")
        df["Gene_Aliases"] = df["Gene_ID"].map(ga).fillna("")
    return df

# -----------------------------------------------------------------------------
# RETRIEVAL + PERTURBATION + RANKING
# -----------------------------------------------------------------------------
def retrieval_candidates_for_gene_expanded(gene_id, pair_df, fusion, base_df, smiles_col,
                                           top_k=50, include_known=False):
    sub = pair_df[pair_df['Gene_ID'] == gene_id]
    if sub.empty:
        return pd.DataFrame()
    pos_idx = sub.index[sub['Association_Label'] == 1].tolist()
    if len(pos_idx) == 0:
        centroid = fusion[sub.index].mean(axis=0, keepdims=True)
    else:
        centroid = fusion[pos_idx].mean(axis=0, keepdims=True)
    sims = cosine_similarity(fusion, centroid).ravel()
    cand_df = pair_df.copy()
    cand_df['similarity'] = sims
    if not include_known:
        known_drugs = set(sub.loc[sub['Association_Label'] == 1, 'Drug_ID'].tolist())
        cand_df = cand_df[~cand_df['Drug_ID'].isin(known_drugs)]
    top = cand_df.sort_values('similarity', ascending=False).head(top_k).copy()

    def get_smiles_for(drugid):
        s = base_df.loc[base_df['Drug_ID'] == drugid, smiles_col]
        return s.iloc[0] if s.shape[0] > 0 else ""
    top['SMILES'] = top['Drug_ID'].map(get_smiles_for)
    top['SMILES_valid'] = top['SMILES'].map(smiles_is_valid)
    top['QED'] = top['SMILES'].map(score_qed_smi)
    return top.reset_index(drop=True)

def brics_recombine_from_pool(base_smi, pool, n_trials=200):
    results = set()
    try:
        _ = list(BRICS.BRICSDecompose(base_smi))
    except Exception:
        pass
    for _ in range(n_trials):
        try:
            k = random.randint(2, 4)
            chosen = random.sample(pool, min(k, len(pool)))
            mols = BRICS.BRICSBuild(chosen)
            for i, m in enumerate(mols):
                if i >= 3:
                    break
                if m is None:
                    continue
                s = Chem.MolToSmiles(m, True)
                if smiles_is_valid(s):
                    results.add(s)
        except Exception:
            continue
        if len(results) >= 300:
            break
    return list(results)

def murcko_graft_variants(base_smi, scaff_pool, max_variants=200):
    out = set()
    base_m = safe_mol_from_smiles(base_smi)
    if base_m is None:
        return []
    for scaff in random.sample(scaff_pool, min(len(scaff_pool), max_variants)):
        try:
            new_smi = scaff + "C"
            if smiles_is_valid(new_smi):
                out.add(new_smi)
        except Exception:
            continue
        if len(out) >= max_variants:
            break
    return list(out)

def conservative_atom_mutations(base_smi, n=200):
    out = set()
    m0 = safe_mol_from_smiles(base_smi)
    if m0 is None:
        return []
    atoms_idx = [a.GetIdx() for a in m0.GetAtoms()
                 if a.GetAtomicNum() in (6, 7, 8, 9, 15, 16, 17)]
    if not atoms_idx:
        return []
    for _ in range(n):
        try:
            m = Chem.RWMol(m0)
            action = random.choice(['add_methyl', 'remove_methyl', 'swap_hetero'])
            if action == 'add_methyl':
                idx = random.choice(atoms_idx)
                new_idx = m.AddAtom(Chem.Atom(6))
                m.AddBond(idx, new_idx, Chem.BondType.SINGLE)
            elif action == 'remove_methyl':
                candidates = [a.GetIdx() for a in m.GetAtoms()
                              if a.GetDegree() == 1 and a.GetAtomicNum() == 6]
                if candidates:
                    m.RemoveAtom(random.choice(candidates))
                else:
                    continue
            elif action == 'swap_hetero':
                heteros = [a for a in m.GetAtoms()
                           if a.GetAtomicNum() in (7, 8, 16, 15)]
                if not heteros:
                    continue
                a = random.choice(heteros)
                new_atomic = random.choice([7, 8, 16, 15])
                a.SetAtomicNum(new_atomic)
            new_m = m.GetMol()
            Chem.SanitizeMol(new_m)
            s = Chem.MolToSmiles(new_m, True)
            if smiles_is_valid(s):
                out.add(s)
        except Exception:
            continue
        if len(out) >= 500:
            break
    return list(out)

def predict_prob_via_nearest_embedding(candidate_smiles, gene_id,
                                       model, base_smiles, drug_emb, prot_emb, pair_df):
    best_idx = None
    best_sim = -1.0
    for i, s in enumerate(base_smiles):
        sim = tanimoto(candidate_smiles, s)
        if sim > best_sim:
            best_sim = sim
            best_idx = i
    if best_idx is None:
        return 0.0
    d_emb = drug_emb[best_idx]

    sub = pair_df[pair_df['Gene_ID'] == gene_id]
    pos_idx = sub.index[sub['Association_Label'] == 1].tolist()
    if len(pos_idx) == 0:
        prot_vec = prot_emb[sub.index[0]]
    else:
        prot_vec = prot_emb[pos_idx].mean(axis=0)

    with torch.no_grad():
        d_t = torch.tensor(d_emb, dtype=torch.float32).unsqueeze(0)
        p_t = torch.tensor(prot_vec, dtype=torch.float32).unsqueeze(0)
        prob = float(model(d_t, p_t).numpy().ravel()[0])
    return prob

def rank_candidates_multi(candidates, gene_id, model, base_smiles,
                          drug_emb, prot_emb, pair_df, top_k=20,
                          w_pred=0.6, w_qed=0.3, w_sim=0.1):
    scored = []
    top_retrieved = retrieval_candidates_for_gene_expanded(
        gene_id, pair_df, fusion_emb, base_df, smiles_col_base,
        top_k=20, include_known=True
    )
    for s in candidates:
        if not s or not isinstance(s, str):
            continue
        if not smiles_is_valid(s):
            continue
        q = score_qed_smi(s)
        pred = predict_prob_via_nearest_embedding(
            s, gene_id, model, base_smiles, drug_emb, prot_emb, pair_df
        )
        if top_retrieved.empty:
            sim = 0.0
        else:
            sims = [tanimoto(s, t) for t in top_retrieved['SMILES'].dropna().tolist() if t]
            sim = max(sims) if sims else 0.0
        scored.append((s, w_pred * pred + w_qed * q + w_sim * sim, pred, q, sim))
    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
    return scored_sorted[:top_k]

def robust_generate_for_gene(gene_id, model, pair_df, fusion,
                             base_df, smiles_col, base_smiles,
                             fragment_pool, scaffold_pool,
                             drug_emb, prot_emb,
                             retrieval_k=50, include_known=True,
                             perturb_budget=300, top_k=20):
    retrieved = retrieval_candidates_for_gene_expanded(
        gene_id, pair_df, fusion, base_df, smiles_col,
        top_k=retrieval_k, include_known=include_known
    )
    if retrieved.empty:
        return pd.DataFrame()

    base_smiles_seed = retrieved['SMILES'].dropna().unique().tolist()
    if len(base_smiles_seed) < 5:
        base_smiles_seed = base_smiles_seed + base_smiles
        base_smiles_seed = list(dict.fromkeys(base_smiles_seed))

    all_cands = set()
    for b in base_smiles_seed:
        tries = min(200, max(1, perturb_budget // max(1, len(base_smiles_seed))))
        all_cands.update(brics_recombine_from_pool(b, fragment_pool, n_trials=tries))
    for b in base_smiles_seed:
        all_cands.update(murcko_graft_variants(b, scaffold_pool, max_variants=100))
    for b in base_smiles_seed:
        all_cands.update(conservative_atom_mutations(
            b, n=max(1, perturb_budget // max(1, len(base_smiles_seed)))
        ))

    all_cands = {s for s in all_cands if s and smiles_is_valid(s)}
    if len(all_cands) == 0:
        return pd.DataFrame()

    ranked = rank_candidates_multi(
        list(all_cands), gene_id, model, base_smiles,
        drug_emb, prot_emb, pair_df, top_k=top_k
    )
    if not ranked:
        return pd.DataFrame()
    out = pd.DataFrame([{
        "SMILES": r[0],
        "score": r[1],
        "pred_prob": r[2],
        "QED": r[3],
        "tanimoto_top": r[4]
    } for r in ranked])
    return out

# SA score + PAINS
try:
    from rdkit.Chem.Scoring.Scoring import CalculateSAScore
    def calc_SA_score(mol):
        return CalculateSAScore(mol)
except Exception:
    def calc_SA_score(mol):
        atoms = mol.GetNumHeavyAtoms()
        rings = len(Chem.GetSSSR(mol))
        logp = Descriptors.MolLogP(mol)
        sa = 5 + 0.05 * atoms + 0.1 * rings - 0.2 * logp
        return float(min(max(sa, 1.0), 10.0))

params = FilterCatalogParams()
params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
pains_catalog = FilterCatalog(params)

def fails_pains(mol):
    return pains_catalog.HasMatch(mol)

def evaluate_candidates(candidate_df, top_k=15):
    results = []
    for _, row in candidate_df.iterrows():
        smi = row.get("SMILES") or row.get("smiles")
        m = safe_mol_from_smiles(smi)
        if m is None:
            continue
        qed = float(row.get("QED", QED.qed(m)))
        pred = float(row.get("pred_prob", row.get("pred", 0.0)))
        sa = calc_SA_score(m)
        pains_flag = fails_pains(m)
        sa_norm = (sa - 1) / 9 if not np.isnan(sa) else 0.5
        final_score = (0.5 * pred) + (0.25 * qed) + (0.25 * (1 - sa_norm))
        if pains_flag:
            final_score *= 0.5
        results.append({
            "SMILES": smi,
            "Pred_Assoc": round(pred, 3),
            "QED": round(qed, 3),
            "SA_Score": round(sa, 2) if not np.isnan(sa) else np.nan,
            "PAINS_Flag": pains_flag,
            "Final_Score": round(final_score, 3)
        })
    ranked = pd.DataFrame(results).sort_values("Final_Score", ascending=False).head(top_k).reset_index(drop=True)
    return ranked

# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------
def main():
    st.markdown(
        '<h1 class="main-header">💊 GD-CDSS: Gene-Drug Clinical Decision Support System</h1>',
        unsafe_allow_html=True
    )

    with st.spinner("Loading model and data..."):
        model, drug_emb, prot_emb, pair_df, fusion, scalers = load_model_and_data()
        meta = load_metadata()

    if model is None:
        st.error("Failed to load required files. Ensure all artifacts are present in 'artifacts/'.")
        return

    st.sidebar.title("🔬 Analysis Options")
    analysis_mode = st.sidebar.radio(
        "Select Mode:",
        [
            "Drug Repurposing",
            "New Drug Prediction",
            "Gene Analysis",
                    ]
    )

    # -------------------------------------------------------------------------
    # MODE 1: DRUG REPURPOSING
    # -------------------------------------------------------------------------
    if analysis_mode == "Drug Repurposing":
        st.header("🔄 Drug Repurposing Analysis")
        st.write("Find existing drugs that may be repurposed for a target gene.")

        available_genes = pair_df['Gene_ID'].unique().tolist()
        selected_gene = st.selectbox("Select Target Gene (ID):", available_genes)

        symbol = meta["gene_symbol"].get(selected_gene, "")
        gname  = meta["gene_name"].get(selected_gene, "")
        if symbol or gname:
            st.info(f"Selected gene: **{symbol}** — {gname}")

        top_k = st.slider("Number of candidates to retrieve:", 5, 50, 15)

        if st.button("🔍 Find Repurposing Candidates"):
            with st.spinner("Analyzing gene-drug associations..."):
                gene_indices = pair_df[pair_df['Gene_ID'] == selected_gene].index.tolist()
                target_prot_emb = prot_emb[gene_indices].mean(axis=0)

                target_fusion = fusion[gene_indices].mean(axis=0).reshape(1, -1)
                sims = cosine_similarity(fusion, target_fusion).ravel()

                known_drugs = set(
                    pair_df[
                        (pair_df['Gene_ID'] == selected_gene) &
                        (pair_df['Association_Label'] == 1)
                    ]['Drug_ID'].tolist()
                )

                candidates_df = pair_df.copy()
                candidates_df['similarity'] = sims
                candidates_df = candidates_df[~candidates_df['Drug_ID'].isin(known_drugs)]
                top_candidates = candidates_df.nlargest(top_k, 'similarity')

                predictions = []
                for idx in top_candidates.index:
                    drug_feat = drug_emb[idx]
                    prob = predict_association(model, drug_feat, target_prot_emb)
                    predictions.append(prob)

                top_candidates['Predicted_Probability'] = predictions
                top_candidates = top_candidates.sort_values('Predicted_Probability', ascending=False)

            st.success(f"Found {len(top_candidates)} repurposing candidates!")

            top_candidates = enrich_with_names(top_candidates, meta)

            col1, col2, col3 = st.columns(3)
            col1.metric("Target Gene (ID)", selected_gene)
            col2.metric("Candidates Found", len(top_candidates))
            col3.metric("Avg Probability", f"{top_candidates['Predicted_Probability'].mean():.3f}")

            st.subheader("📊 Top Candidates (with names)")
            display_df = top_candidates[
                [
                    'Drug_ID', 'Drug_Name', 'Drug_Aliases',
                    'Gene_ID', 'Gene_Symbol', 'Gene_Name',
                    'Predicted_Probability', 'similarity'
                ]
            ].head(top_k)
            st.dataframe(display_df, use_container_width=True)

            fig = px.bar(
                display_df.head(10),
                x='Drug_Name',
                y='Predicted_Probability',
                color='Predicted_Probability',
                color_continuous_scale='Blues',
                title='Top 10 Repurposing Candidates by Predicted Association'
            )
            st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # MODE 2: NEW DRUG PREDICTION
    # -------------------------------------------------------------------------
    elif analysis_mode == "New Drug Prediction":
        st.header("🆕 New Drug Candidate Prediction")
        st.write("Predict association probability and visualize bioavailability radar + BOILED-Egg.")

        input_method = st.radio("Input Method:", ["Enter SMILES", "Upload CSV"])

        if input_method == "Enter SMILES":
            smiles_input = st.text_input("Enter SMILES string:", placeholder="CCO")
            available_genes = pair_df['Gene_ID'].unique().tolist()
            selected_gene = st.selectbox("Select Target Gene (ID):", available_genes)

            symbol = meta["gene_symbol"].get(selected_gene, "")
            gname  = meta["gene_name"].get(selected_gene, "")
            if symbol or gname:
                st.info(f"Selected gene: **{symbol}** — {gname}")

            if st.button("🔮 Predict Association") and smiles_input:
                mol = safe_mol_from_smiles(smiles_input)
                if mol is None:
                    st.error("Invalid SMILES string. Please check your input.")
                else:
                    with st.spinner("Analyzing molecule..."):
                        drug_features = compute_drug_features(smiles_input, scalers)
                        gene_indices = pair_df[pair_df['Gene_ID'] == selected_gene].index.tolist()
                        target_prot_emb = prot_emb[gene_indices].mean(axis=0)
                        prob = predict_association(model, drug_features, target_prot_emb)
                        qed, sa = calculate_drug_likeness(smiles_input)
                        bio_desc = compute_bioavailability_descriptors(mol)
                        radar_fig = make_bioavailability_radar(bio_desc)
                        egg_fig, egg_class = make_boiled_egg_plot(mol)

                    st.success("Analysis Complete!")

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Association Probability", f"{prob:.3f}")
                    col2.metric("QED Score", f"{qed:.3f}")
                    col3.metric("SA Score", f"{sa:.2f}")
                    col4.metric("Drug-likeness", "High" if qed > 0.5 else "Moderate")

                    st.subheader("🧪 Molecular Properties")
                    mol_props = {
                        'Molecular Weight': Descriptors.ExactMolWt(mol),
                        'LogP (MolLogP)': Descriptors.MolLogP(mol),
                        'TPSA': Descriptors.TPSA(mol),
                        'H-Bond Acceptors': Descriptors.NumHAcceptors(mol),
                        'H-Bond Donors': Descriptors.NumHDonors(mol),
                        'Rotatable Bonds': Descriptors.NumRotatableBonds(mol)
                    }
                    props_df = pd.DataFrame(mol_props.items(), columns=['Property', 'Value'])
                    st.table(props_df)

                    st.subheader("📈 Bioavailability Visualization")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.plotly_chart(radar_fig, use_container_width=True)
                    with c2:
                        st.plotly_chart(egg_fig, use_container_width=True)
                        st.caption(f"BOILED-Egg classification: **{egg_class}**")

                    st.subheader("📝 Interpretation")
                    if prob > 0.7:
                        st.success("✅ Strong predicted association. This molecule shows high potential.")
                    elif prob > 0.5:
                        st.warning("⚠️ Moderate association. Further experimental validation recommended.")
                    else:
                        st.info("ℹ️ Low association probability. Consider structural modifications or alternative targets.")

        else:
            uploaded_file = st.file_uploader("Upload CSV with SMILES column", type=['csv'])
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                st.write("Preview:", df.head())
                smiles_col = st.selectbox("Select SMILES column:", df.columns)
                selected_gene = st.selectbox("Select Target Gene (ID):", pair_df['Gene_ID'].unique())
                symbol = meta["gene_symbol"].get(selected_gene, "")
                gname  = meta["gene_name"].get(selected_gene, "")
                if symbol or gname:
                    st.info(f"Selected gene: **{symbol}** — {gname}")

                if st.button("🔮 Batch Predict"):
                    with st.spinner("Processing batch predictions..."):
                        gene_indices = pair_df[pair_df['Gene_ID'] == selected_gene].index.tolist()
                        target_prot_emb = prot_emb[gene_indices].mean(axis=0)
                        predictions = []
                        for smi in df[smiles_col]:
                            try:
                                drug_features = compute_drug_features(smi, scalers)
                                prob = predict_association(model, drug_features, target_prot_emb)
                                qed, sa = calculate_drug_likeness(smi)
                                predictions.append({'SMILES': smi, 'Probability': prob, 'QED': qed, 'SA': sa})
                            except Exception:
                                predictions.append({'SMILES': smi, 'Probability': np.nan, 'QED': np.nan, 'SA': np.nan})
                        results_df = pd.DataFrame(predictions).sort_values('Probability', ascending=False)
                    st.success(f"Processed {len(results_df)} molecules!")
                    st.dataframe(results_df, use_container_width=True)
                    csv = results_df.to_csv(index=False)
                    st.download_button("📥 Download Results", csv, "predictions.csv", "text/csv")

    # -------------------------------------------------------------------------
    # MODE 3: GENE ANALYSIS
    # -------------------------------------------------------------------------
    elif analysis_mode == "Gene Analysis":
        st.header("🧬 Gene Association Analysis")
        st.write("Explore gene-drug association landscape.")

        selected_gene = st.selectbox("Select Gene (ID):", pair_df['Gene_ID'].unique())
        symbol = meta["gene_symbol"].get(selected_gene, "")
        gname  = meta["gene_name"].get(selected_gene, "")
        galias = meta["gene_alias"].get(selected_gene, "")
        if symbol or gname or galias:
            st.info(
                f"Gene ID: **{selected_gene}**  \n"
                f"Symbol: **{symbol}**  \n"
                f"Name: {gname}  \n"
                f"Aliases: {galias}"
            )

        if st.button("📊 Analyze Gene"):
            gene_data = pair_df[pair_df['Gene_ID'] == selected_gene]
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Associations", len(gene_data))
            col2.metric("Known Interactions", gene_data['Association_Label'].sum())
            col3.metric("Association Rate", f"{gene_data['Association_Label'].mean():.2%}")

            st.subheader("Association Distribution")
            counts = gene_data['Association_Label'].value_counts()
            fig = px.pie(
                values=counts.values,
                names=['No Association', 'Known Association'],
                title=f'Association Distribution for {symbol or selected_gene}'
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Top Associated Drugs (with names)")
            top_drugs = gene_data[gene_data['Association_Label'] == 1].copy()
            top_drugs = enrich_with_names(top_drugs, meta)
            st.dataframe(
                top_drugs[['Drug_ID', 'Drug_Name', 'Drug_Aliases', 'Gene_ID', 'Gene_Symbol']].head(10),
                use_container_width=True
            )


if __name__ == "__main__":
    main()
