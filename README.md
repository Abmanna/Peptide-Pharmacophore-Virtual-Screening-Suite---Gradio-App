The script includes an auto-installation helper. Simply upload `app.py` and run it. The script will detect the environment and install `rdkit`, `gradio`, `py3Dmol`, and other dependencies automatically.

## Usage

Run the application:

```bash
python app.py
```

Open your browser to the local URL provided (usually `http://127.0.0.1:7860`).

## Workflow

1.  **Generate Conformers**:
    - Go to the "Conformer Generation" tab.
    - Enter SMILES strings (one per line).
    - Adjust settings (number of conformers, pruning threshold).
    - Click "Generate" and download the resulting SDF file.

2.  **Minimize (Optional)**:
    - Go to the "Minimization" tab.
    - Upload the generated SDF.
    - Select "OpenMM" (if available) or "RDKit".
    - Click "Minimize" and download the optimized SDF.

3.  **Screen**:
    - Go to the "Pharmacophore Screening" tab.
    - Upload your pharmacophore file (JSON, PML, or PHAR).
    - Upload your molecule library (SDF).
    - Click "Screen Library" to get ranked hits.

4.  **Validate**:
    - Go to the "Validation Metrics" tab.
    - Upload an SDF containing screening results (must have activity labels).
    - Click "Calculate Metrics" to see ROC AUC and BEDROC scores.

## Dependencies

-   `gradio`
-   `rdkit`
-   `scipy`
-   `scikit-learn`
-   `py3Dmol`
-   `pandas`
-   `openmm` (optional)
