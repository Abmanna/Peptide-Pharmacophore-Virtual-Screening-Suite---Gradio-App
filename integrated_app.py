import gradio as gr
import sys
import os
import numpy as np

# Ensure the current directory is in the path so imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- Import Pharmacophore Screening App ---
try:
    from app import create_ui_content as create_screening_ui
except ImportError as e:
    print(f"Warning: Could not import Pharmacophore Screening App: {e}")
    create_screening_ui = lambda: gr.Markdown(f"## Error Loading Pharmacophore Screening App\n{e}")

# --- Import CombiChem Studio ---
try:
    from PyCombilab import PyCombLibGradioApp
except ImportError as e:
    print(f"Warning: Could not import PyCombilab: {e}")
    PyCombLibGradioApp = None

# --- Import Master Reactions ---
try:
    from ChemRXN import MasterReactions
    from rdkit import Chem
    from rdkit.Chem import Draw
    import base64
    from io import BytesIO
except ImportError as e:
    print(f"Warning: Could not import ChemRXN or RDKit: {e}")
    MasterReactions = None

# --- Import Peptide Optimizer ---
PEPTIDES_AVAILABLE = False
try:
    import tensorflow as tf
    from Peptides import PeptideScorer, PeptideOptimizer
    PEPTIDES_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. AI Peptide Design will be disabled.")
except Exception as e:
    print(f"Error loading Peptides module: {e}")


# --- Helper Functions ---

def create_reaction_interface():
    """Creates the UI for Master Reactions"""
    
    def run_rxn(reactant1, reactant2, rxn_name):
        if not MasterReactions:
            return None, "MasterReactions module not loaded."
        
        mol1 = Chem.MolFromSmiles(reactant1)
        if not mol1: return None, "Invalid Reactant 1 SMILES"
        
        mol2 = None
        if reactant2:
            mol2 = Chem.MolFromSmiles(reactant2)
        
        # Find the reaction object by name
        rxn = getattr(MasterReactions, rxn_name, None)
        if not rxn:
            return None, "Reaction not found"
            
        try:
            prod = MasterReactions.run(rxn, mol1, mol2)
            if not prod:
                return None, "Reaction produced no valid product"
            
            # Generate Image
            img = Draw.MolToImage(prod, size=(400, 300))
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return f"data:image/png;base64,{img_str}", Chem.MolToSmiles(prod)
            
        except Exception as e:
            return None, f"Error: {str(e)}"

    with gr.Row():
        with gr.Column():
            gr.Markdown("### ⚗️ Master Reaction Templates")
            r1 = gr.Textbox(label="Reactant 1 (SMILES)", placeholder="e.g. C(=O)O")
            r2 = gr.Textbox(label="Reactant 2 (SMILES) [Optional]", placeholder="e.g. N")
            
            # Get list of reactions dynamically
            rxn_choices = []
            if MasterReactions:
                for attr in dir(MasterReactions):
                    if not attr.startswith('_') and attr.isupper() and attr != "ALL":
                        rxn_choices.append(attr)
            
            rxn_menu = gr.Dropdown(choices=rxn_choices, label="Select Reaction", value="AMIDE_STANDARD" if "AMIDE_STANDARD" in rxn_choices else None)
            btn = gr.Button("Run Reaction", variant="primary")
        
        with gr.Column():
            out_img = gr.Image(label="Product Structure")
            out_smi = gr.Textbox(label="Product SMILES", interactive=False)
    
    btn.click(run_rxn, inputs=[r1, r2, rxn_menu], outputs=[out_img, out_smi])

def create_peptide_ui():
    if not PEPTIDES_AVAILABLE:
        gr.Markdown("## ⚠️ AI Peptide Design Unavailable\nTensorFlow is required for this feature.")
        return

    def run_opt(target, gens):
        optimizer = PeptideOptimizer(
            scoring_function=getattr(PeptideScorer, f"{target.lower()}_score", PeptideScorer.antimicrobial_score),
            population_size=20
        )
        # Fix: ensure proper string handling
        initial = [''.join(np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"), 15)) for _ in range(10)]
        results = optimizer.optimize(initial, generations=gens, target_length=15)
        
        out_text = ""
        for i, (seq, score) in enumerate(results[:5]):
            out_text += f"{i+1}. {seq} (Score: {score:.3f})\n"
        return out_text

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🧬 AI Peptide Design")
            t = gr.Dropdown(["Antimicrobial", "Cell_Penetrating", "Hemolytic_Risk"], label="Optimization Target", value="Antimicrobial")
            g = gr.Slider(1, 50, 10, label="Generations")
            b = gr.Button("Optimize", variant="primary")
        with gr.Column():
            o = gr.Textbox(label="Top Sequences", lines=10)
    
    # We won't wire the button effectively since logic is imported, but if we did:
    # b.click(run_opt, inputs=[t, g], outputs=[o]) 
    # BUT run_opt is defined above. We can wire it.
    b.click(run_opt, inputs=[t, g], outputs=[o])


# --- Main Application ---
with gr.Blocks(title="Unified Laboratory Suite", theme=gr.themes.Ocean()) as integrated_app:
    gr.Markdown("# 🔬 Unified Pharmacophore & CombiChem Laboratory")
    
    with gr.Tabs():
        with gr.TabItem("💊 Pharmacophore Screening"):
            create_screening_ui()
            
        with gr.TabItem("⚗️ CombiChem Studio"):
            if PyCombLibGradioApp:
                cc_app = PyCombLibGradioApp()
                cc_app.create_ui_content()
            else:
                gr.Markdown("PyCombilab module not loaded.")

        with gr.TabItem("🧪 Master Reactions"):
            create_reaction_interface()
            
        with gr.TabItem("🧬 AI Peptide Designer"):
            create_peptide_ui()

if __name__ == "__main__":
    integrated_app.launch(share=True)
