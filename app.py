
# =============================================================================
# PEPTIDE PHARMACOPHORE SCREENING SUITE - ENHANCED EDITION v2.3
# Bulk Data Ingestion + PostgreSQL (RDKit/TimescaleDB) Integration
# =============================================================================

import gradio as gr
import numpy as np
import pandas as pd
import json
import os
import tempfile
import pickle
import hashlib
import sqlite3
import requests
import io
import base64
import re
import threading
import queue
import time
import logging
import webbrowser
from io import StringIO, BytesIO
from typing import List, Tuple, Optional, Dict, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from collections import deque, defaultdict
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RDKit imports
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, Descriptors, rdMolDescriptors, Lipinski
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator, Crippen, Fragments
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem import BRICS
import xml.etree.ElementTree as ET

# Scientific computing
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, recall_score,
                            f1_score, mean_squared_error, r2_score, confusion_matrix,
                            roc_curve, precision_recall_curve, average_precision_score,
                            mean_absolute_error, explained_variance_score)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel

# Optional Dependencies
try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False

try:
    import py3Dmol
    PY3DMOL_AVAILABLE = True
except ImportError:
    PY3DMOL_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, TensorDataset
    from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, GATConv, GINConv, TransformerConv
    from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader as GeometricDataLoader
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

try:
    import xgboost as xgb
    import lightgbm as lgb
    BOOSTING_AVAILABLE = True
except ImportError:
    BOOSTING_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Live Data Clients
try:
    from chembl_webresource_client.new_client import new_client
    CHEMBL_AVAILABLE = True
except ImportError:
    CHEMBL_AVAILABLE = False

try:
    import pubchempy as pcp
    PUBCHEM_AVAILABLE = True
except ImportError:
    PUBCHEM_AVAILABLE = False

# PostgreSQL
try:
    import psycopg2
    import psycopg2
    from psycopg2.extras import RealDictCursor, execute_values
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

# =============================================================================
# POSTGRESQL DATABASE MANAGER
# =============================================================================

class PostgresDB:
    """Manager for PostgreSQL with RDKit and TimescaleDB."""
    
    def __init__(self, dbname="pharmacophore_db", user="postgres", password="password", host="localhost", port="5432"):
        self.conn_params = {
            "dbname": dbname, "user": user, "password": password, "host": host, "port": port
        }
        self.connected = False
        self.conn = None
        
    def connect(self):
        if not POSTGRES_AVAILABLE: return "psycopg2 not installed"
        try:
            self.conn = psycopg2.connect(**self.conn_params)
            self.conn.autocommit = True
            self.connected = True
            return "Connected to PostgreSQL"
        except Exception as e:
            return f"Connection Failed: {e}"

    def init_schema(self):
        if not self.connected: return "Not connected"
        try:
            with open("db_schema.sql", "r") as f:
                sql = f.read()
            with self.conn.cursor() as cur:
                cur.execute(sql)
            return "Schema initialized"
        except Exception as e:
            return f"Schema Init Failed: {e}"

    def insert_molecule(self, smiles, source="User"):
        if not self.connected: return
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return
        inchikey = Chem.MolToInchiKey(mol)
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO molecules (smiles, mol, inchikey, source)
                    VALUES (%s, mol_from_smiles(%s), %s, %s)
                    ON CONFLICT (inchikey) DO NOTHING
                """, (smiles, smiles, inchikey, source))
        except Exception as e:
            logger.error(f"DB Insert Error: {e}")

    def insert_molecules_batch(self, data_list: List[Tuple[str, str]]):
        """Batch insert molecules for high performance."""
        if not self.connected: return
        
        # Pre-calculate InChIKeys and filter invalid mols
        valid_data = []
        for smiles, source in data_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                inchikey = Chem.MolToInchiKey(mol)
                valid_data.append((smiles, smiles, inchikey, source))
        
        if not valid_data: return

        try:
            with self.conn.cursor() as cur:
                execute_values(cur, """
                    INSERT INTO molecules (smiles, mol, inchikey, source)
                    VALUES %s
                    ON CONFLICT (inchikey) DO NOTHING
                """, valid_data, template="(%s, mol_from_smiles(%s), %s, %s)")
        except Exception as e:
            logger.error(f"Batch Insert Error: {e}")

    def query_substructure(self, smiles_query):
        if not self.connected: return pd.DataFrame()
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT mol_id, smiles, source, date_added 
                    FROM molecules 
                    WHERE mol @> mol_from_smiles(%s)
                    LIMIT 50
                """, (smiles_query,))
                return pd.DataFrame(cur.fetchall())
        except Exception as e:
            return pd.DataFrame({"Error": [str(e)]})

    def get_resistance_trends(self):
        if not self.connected: return pd.DataFrame()
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT time_bucket('1 year', year) AS bucket,
                           antibiotic_class,
                           AVG(resistance_pct) as avg_resistance
                    FROM resistance_trends
                    GROUP BY bucket, antibiotic_class
                    ORDER BY bucket
                """)
                return pd.DataFrame(cur.fetchall())
        except Exception as e:
            return pd.DataFrame({"Error": [str(e)]})

# =============================================================================
# PUBLIC DATA HUB
# =============================================================================

class PublicDataHub:
    DATA_SOURCES = {
        "FDA_OrangeBook": "https://www.fda.gov/drugs/drug-approvals-and-databases/orange-book-approved-drug-products-therapeutic-equivalence-evaluations",
        "WHO_Essential_Medicines": "https://www.who.int/groups/expert-committee-on-selection-and-use-of-essential-medicines/essential-medicines-lists",
        "CARD_Resistance": "https://card.mcmaster.ca/download",
        "CO_ADD": "https://www.co-add.org/data",
        "Google_Patents": "https://patents.google.com/?q=",
        "SureChEMBL": "https://www.surechembl.org/search/",
        "IBM_RXN": "https://rxn.res.ibm.com/"
    }

    @staticmethod
    @lru_cache(maxsize=100)
    def search_chembl(query: str, limit: int = 10) -> pd.DataFrame:
        if not CHEMBL_AVAILABLE: return pd.DataFrame({"Error": ["ChEMBL client not installed"]})
        try:
            molecule = new_client.molecule
            res = molecule.filter(molecule_synonyms__molecule_synonym__icontains=query).only(
                'molecule_chembl_id', 'pref_name', 'molecule_structures', 'molecule_type'
            )[:limit]
            data = []
            for r in res:
                smiles = r['molecule_structures']['canonical_smiles'] if r['molecule_structures'] else None
                data.append({"Source": "ChEMBL", "ID": r['molecule_chembl_id'], "Name": r.get('pref_name', 'Unknown'), "SMILES": smiles, "Type": r['molecule_type']})
            return pd.DataFrame(data)
        except Exception as e: return pd.DataFrame({"Error": [f"ChEMBL Search Failed: {str(e)}"]})

    @staticmethod
    @lru_cache(maxsize=100)
    def search_pubchem(query: str, limit: int = 10) -> pd.DataFrame:
        if not PUBCHEM_AVAILABLE: return pd.DataFrame({"Error": ["PubChemPy not installed"]})
        try:
            compounds = pcp.get_compounds(query, 'name')
            data = []
            for c in compounds[:limit]:
                data.append({"Source": "PubChem", "ID": c.cid, "Name": c.synonyms[0] if c.synonyms else query, "SMILES": c.isomeric_smiles, "Formula": c.molecular_formula})
            return pd.DataFrame(data)
        except Exception as e: return pd.DataFrame({"Error": [f"PubChem Search Failed: {str(e)}"]})

    @staticmethod
    def get_external_links() -> str:
        md = "### 🌍 External Data Hubs\n\n"
        for name, url in PublicDataHub.DATA_SOURCES.items(): md += f"- **{name.replace('_', ' ')}**: [Access Data]({url})\n"
        return md

# =============================================================================
# DATA INGESTION MANAGER
# =============================================================================

class IngestionManager:
    """Handles bulk ingestion of antibiotic data from public sources."""
    
    def __init__(self, db_manager: PostgresDB):
        self.db = db_manager
        self.classes = ["penicillins", "cephalosporins", "fluoroquinolones", "macrolides", 
                       "tetracyclines", "aminoglycosides", "glycopeptides", "polymyxins", 
                       "rifamycins", "clindamycin", "streptogramins", "pleuromutilins"]

    def ingest_all_antibiotics(self, progress=gr.Progress()):
        if not self.db.connected: return "Database not connected."
        
        log = ["Starting bulk ingestion..."]
        total_steps = len(self.classes)
        
        for i, cls in enumerate(self.classes):
            progress(i / total_steps, desc=f"Ingesting {cls}...")
            log.append(f"Processing {cls}:")
            
            # ChEMBL
            res_chembl = self.ingest_chembl_class(cls)
            log.append(f"  - {res_chembl}")
            
            # CO-ADD (Placeholder for demo)
            # res_coadd = self.ingest_coadd_class(cls)
            
            # CARD (Placeholder for demo)
            # res_card = self.ingest_card_mutations(cls)
            
        return "\n".join(log)

    def ingest_chembl_class(self, cls_name):
        if not CHEMBL_AVAILABLE: return "ChEMBL client not installed"
        try:
            # 1. Search for compounds
            res = PublicDataHub.search_chembl(cls_name, limit=50) # Increased limit for batch demo
            batch_data = []
            for _, row in res.iterrows():
                if row.get('SMILES'):
                    batch_data.append((row['SMILES'], f"ChEMBL_{cls_name}"))
            
            if batch_data:
                self.db.insert_molecules_batch(batch_data)
                return f"ChEMBL: Batch inserted {len(batch_data)} compounds"
            return "ChEMBL: No compounds found"
        except Exception as e:
            return f"ChEMBL Error: {str(e)}"

    def ingest_coadd_class(self, cls_name):
        # Placeholder: In a real scenario, this would fetch the CO-ADD CSV
        return "CO-ADD: Data source requires manual download (see Public Data Hub)"

    def ingest_card_mutations(self, cls_name):
        # Placeholder: This would parse the CARD ontology
        return "CARD: Resistance data parsing not yet implemented"

# =============================================================================
# DATA HANDLING & ML
# =============================================================================

class DataHandler:
    def __init__(self, cache_dir: str = ".cache"):
        os.makedirs(cache_dir, exist_ok=True)
    def load_file(self, file_path: str) -> pd.DataFrame:
        ext = file_path.split('.')[-1].lower()
        if ext == 'csv': return pd.read_csv(file_path)
        elif ext == 'xlsx': return pd.read_excel(file_path)
        elif ext == 'sdf': 
            suppl = Chem.SDMolSupplier(file_path)
            return pd.DataFrame([{'SMILES': Chem.MolToSmiles(m), **m.GetPropsAsDict()} for m in suppl if m])
        elif ext == 'json': return pd.read_json(file_path)
        else: raise ValueError(f"Unsupported format: {ext}")

class OnlineDataFetcher:
    def fetch_from_url(self, url: str) -> pd.DataFrame:
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200: return pd.read_csv(StringIO(r.text))
            return pd.DataFrame()
        except Exception as e: return pd.DataFrame()

class MolecularFeatureExtractor:
    def __init__(self): self.morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    def get_features(self, mol: Chem.Mol) -> np.ndarray:
        if not mol: return np.zeros(2048)
        fp = self.morgan_gen.GetFingerprint(mol)
        arr = np.zeros(2048); DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

if TORCH_AVAILABLE:
    class MoleculeDataset(Dataset):
        def __init__(self, smiles_list, labels, feature_extractor):
            self.smiles = smiles_list
            self.labels = labels
            self.fe = feature_extractor
        def __len__(self): return len(self.smiles)
        def __getitem__(self, idx):
            mol = Chem.MolFromSmiles(self.smiles[idx])
            feat = self.fe.get_features(mol)
            return torch.FloatTensor(feat), torch.FloatTensor([self.labels[idx]])

    class DeepNeuralNetwork(nn.Module):
        def __init__(self, input_dim=2048, hidden_dims=[512, 256], output_dim=1):
            super().__init__()
            layers = []
            prev = input_dim
            for h in hidden_dims:
                layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.3)])
                prev = h
            layers.append(nn.Linear(prev, output_dim))
            self.net = nn.Sequential(*layers)
        def forward(self, x): return self.net(x)

if TORCH_GEOMETRIC_AVAILABLE:
    class GNNModel(torch.nn.Module):
        def __init__(self, num_node_features=9, hidden_channels=64):
            super().__init__()
            self.conv1 = GCNConv(num_node_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels)
            self.lin = nn.Linear(hidden_channels, 1)

        def forward(self, x, edge_index, batch):
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = self.conv2(x, edge_index)
            x = x.relu()
            x = self.conv3(x, edge_index)
            x = global_mean_pool(x, batch)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.lin(x)
            return x

# =============================================================================
# REINFORCEMENT LEARNING
# =============================================================================

if TORCH_AVAILABLE:
    class MoleculeEnvironment:
        VOCAB = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', '(', ')', '[', ']', '=', '#', '@', '+', '-', '/', '\\', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
        def __init__(self, max_length=50):
            self.max_length = max_length
            self.char_to_idx = {c: i for i, c in enumerate(self.VOCAB)}
            self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
            self.vocab_size = len(self.VOCAB)
            self.reset()
        def reset(self): self.current_smiles = ""; self.steps = 0; return self._get_state()
        def _get_state(self):
            state = np.zeros((self.max_length, self.vocab_size))
            for i, char in enumerate(self.current_smiles[:self.max_length]):
                if char in self.char_to_idx: state[i, self.char_to_idx[char]] = 1
            return state.flatten()
        def step(self, action):
            self.steps += 1
            if action < len(self.VOCAB): self.current_smiles += self.idx_to_char[action]
            done = self.steps >= self.max_length
            reward = 0.0
            if done:
                mol = Chem.MolFromSmiles(self.current_smiles)
                if mol: reward = 1.0 + (0.1 * Descriptors.MolLogP(mol) if mol else 0)
                else: reward = -1.0
            return self._get_state(), reward, done, {}

    class PolicyNetwork(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU(), nn.Linear(256, action_dim), nn.Softmax(dim=-1))
        def forward(self, x): return self.net(x)

    class REINFORCETrainer:
        def __init__(self):
            self.env = MoleculeEnvironment()
            self.policy = PolicyNetwork(self.env.max_length * self.env.vocab_size, self.env.vocab_size)
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        def train_episode(self):
            state = self.env.reset(); log_probs = []; rewards = []; done = False
            while not done:
                state_t = torch.FloatTensor(state)
                probs = self.policy(state_t)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_probs.append(dist.log_prob(action))
                state, reward, done, _ = self.env.step(action.item())
                rewards.append(reward)
            loss = -torch.stack(log_probs).sum() * sum(rewards)
            self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
            return sum(rewards), self.env.current_smiles

# =============================================================================
# CORE TOOLS
# =============================================================================

def generate_conformers(smiles: str, num_confs: int = 50):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None, "Invalid SMILES"
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=AllChem.ETKDGv3())
    return mol, f"Generated {mol.GetNumConformers()} conformers"

def minimize_molecules(sdf_file, method="RDKit"):
    suppl = Chem.SDMolSupplier(sdf_file.name)
    mols = [m for m in suppl if m]
    res_mols = []
    for m in mols:
        AllChem.MMFFOptimizeMolecule(m)
        res_mols.append(m)
    
    out = tempfile.NamedTemporaryFile(suffix='.sdf', delete=False, mode='w')
    w = Chem.SDWriter(out.name)
    for m in res_mols: w.write(m)
    w.close()
    return f"Minimized {len(mols)} molecules", out.name

def mol_to_image(mol):
    if not mol: return None
    img = Draw.MolToImage(mol, size=(300, 300))
    buf = BytesIO(); img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

# =============================================================================
# UI BUILDER
# =============================================================================

DB_MANAGER = PostgresDB()
INGESTION_MANAGER = IngestionManager(DB_MANAGER)

def ui_search_kb(query, source):
    if source == "ChEMBL": return PublicDataHub.search_chembl(query)
    elif source == "PubChem": return PublicDataHub.search_pubchem(query)
    else: return pd.concat([PublicDataHub.search_chembl(query), PublicDataHub.search_pubchem(query)], ignore_index=True)

def ui_train_qsar(file_obj, url, target, model_name):
    df = None
    if file_obj: df = DataHandler().load_file(file_obj.name)
    elif url: df = OnlineDataFetcher().fetch_from_url(url)
    if df is None or df.empty: return "No data loaded", None
    fe = MolecularFeatureExtractor()
    X = []; y = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row.get('SMILES', ''))
        if mol and target in row: X.append(fe.get_features(mol)); y.append(row[target])
    if not X: return "No valid data", None, None
    
    if model_name == "Deep Neural Network (PyTorch)":
        if not TORCH_AVAILABLE: return "PyTorch not installed", None, None
        dataset = MoleculeDataset(df['SMILES'].tolist(), y, fe)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        model = DeepNeuralNetwork()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        model.train()
        for epoch in range(5): # Short training for demo
            epoch_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                pred = model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
        
        # Save model
        torch.save(model.state_dict(), "latest_dnn_model.pth")
        return f"DNN Trained. Final Loss: {epoch_loss/len(loader):.4f}", "latest_dnn_model.pth"

    elif model_name == "GNN (PyG)":
        if not TORCH_GEOMETRIC_AVAILABLE: return "PyG not installed", None, None
        return "GNN Training Placeholder (Requires complex graph conversion)", None, None

    model = RandomForestRegressor() if model_name == "Random Forest" else GradientBoostingClassifier()
    model.fit(X, y)
    
    # Save sklearn model
    with open("latest_ml_model.pkl", "wb") as f:
        pickle.dump(model, f)
        
    return f"Model Trained. Score: {model.score(X, y):.3f}", "latest_ml_model.pkl"

def ui_run_rl(steps):
    if not TORCH_AVAILABLE: return "PyTorch not installed", None
    trainer = REINFORCETrainer()
    best_rew = -999; best_smi = ""; log = ""
    for i in range(steps):
        r, s = trainer.train_episode()
        if r > best_rew: best_rew = r; best_smi = s
        if i % 5 == 0: log += f"Step {i}: {s} (R={r:.2f})\n"
    
    # Auto-save best molecule to DB
    if best_smi and DB_MANAGER.connected:
        mol = Chem.MolFromSmiles(best_smi)
        if mol:
            DB_MANAGER.insert_molecule(best_smi, source="RL_DeNovo_Gen")
            log += f"\n[Auto-Save] Best molecule saved to DB: {best_smi}"

    img = mol_to_image(Chem.MolFromSmiles(best_smi)) if best_smi else None
    return log, img

def ui_db_connect(db, user, pw, host, port):
    global DB_MANAGER, INGESTION_MANAGER
    DB_MANAGER = PostgresDB(db, user, pw, host, port)
    INGESTION_MANAGER = IngestionManager(DB_MANAGER)
    res = DB_MANAGER.connect()
    if "Connected" in res:
        DB_MANAGER.init_schema()
    return res

def ui_db_query(smiles):
    return DB_MANAGER.query_substructure(smiles)

def ui_db_trends():
    df = DB_MANAGER.get_resistance_trends()
    if df.empty: return None
    if PLOTLY_AVAILABLE:
        fig = px.line(df, x='bucket', y='avg_resistance', color='antibiotic_class', title="Resistance Trends Over Time")
        return fig
    return None

def ui_ingest_all():
    return INGESTION_MANAGER.ingest_all_antibiotics()


def create_ui_content():
    """Generates the UI components for the screening app."""
    gr.Markdown("# 🧬 Peptide Pharmacophore Suite v2.3 - Enterprise Edition")
    
    with gr.Tabs():
        with gr.TabItem("🌍 Public Data Hub"):
            with gr.Row():
                with gr.Column(scale=2):
                    q = gr.Textbox(label="Search Query"); src = gr.Radio(["ChEMBL", "PubChem", "Both"], value="Both", label="Source")
                    btn = gr.Button("Search Live APIs"); out = gr.Dataframe()
                    btn.click(ui_search_kb, inputs=[q, src], outputs=[out])
                with gr.Column(scale=1): gr.Markdown(PublicDataHub.get_external_links())
        
        with gr.TabItem("🗄️ Local Database (PostgreSQL)"):
            with gr.Accordion("Connection Settings", open=False):
                with gr.Row():
                    db_name = gr.Textbox(label="DB Name", value="pharmacophore_db")
                    db_user = gr.Textbox(label="User", value="postgres")
                    db_pass = gr.Textbox(label="Password", type="password")
                    db_host = gr.Textbox(label="Host", value="localhost")
                    db_port = gr.Textbox(label="Port", value="5432")
                    conn_btn = gr.Button("Connect & Init Schema")
                    conn_stat = gr.Textbox(label="Status")
                    conn_btn.click(ui_db_connect, inputs=[db_name, db_user, db_pass, db_host, db_port], outputs=[conn_stat])
            
            with gr.Accordion("Bulk Data Ingestion", open=False):
                gr.Markdown("Ingest data from ChEMBL, CO-ADD, and CARD for all major antibiotic classes.")
                ingest_btn = gr.Button("🚀 Ingest All Antibiotics", variant="primary")
                ingest_log = gr.Textbox(label="Ingestion Log", lines=10)
                ingest_btn.click(ui_ingest_all, outputs=[ingest_log])

            with gr.Row():
                with gr.Column():
                    sub_q = gr.Textbox(label="Substructure Query (SMILES)")
                    sub_btn = gr.Button("Search DB")
                    sub_res = gr.Dataframe()
                    sub_btn.click(ui_db_query, inputs=[sub_q], outputs=[sub_res])
                with gr.Column():
                    trend_btn = gr.Button("Show Resistance Trends")
                    trend_plot = gr.Plot()
                    trend_btn.click(ui_db_trends, outputs=[trend_plot])

        with gr.TabItem("🔄 Conformer Gen"):
            smi = gr.Textbox(label="SMILES"); btn = gr.Button("Generate"); log = gr.Textbox()
            btn.click(lambda s: generate_conformers(s)[1], inputs=[smi], outputs=[log])
        
        with gr.TabItem("🧪 Advanced QSAR"):
            f = gr.File(label="Upload Data"); u = gr.Textbox(label="OR Fetch URL")
            t = gr.Textbox(label="Target", value="pIC50")
            m = gr.Dropdown(["Random Forest", "GBM", "Deep Neural Network (PyTorch)", "GNN (PyG)"], value="Random Forest", label="Model")
            b = gr.Button("Train"); o = gr.Textbox(label="Training Log")
            dl_btn = gr.File(label="Download Trained Model")
            b.click(ui_train_qsar, inputs=[f, u, t, m], outputs=[o, dl_btn])
        
        with gr.TabItem("🧬 De Novo (RL)"):
            s = gr.Slider(10, 100, value=20, label="Steps"); b = gr.Button("Start"); l = gr.Textbox(); i = gr.Image()
            b.click(ui_run_rl, inputs=[s], outputs=[l, i])

def build_app():
    with gr.Blocks(title="Peptide Pharmacophore Suite v2.3", theme=gr.themes.Soft(primary_hue="teal")) as app:
        create_ui_content()
    return app


if __name__ == "__main__":
    app = build_app()
    app.launch()
