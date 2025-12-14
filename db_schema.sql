-- Enable Extensions
CREATE EXTENSION IF EXISTS rdkit;
CREATE EXTENSION IF EXISTS timescaledb;

-- Molecules Table
CREATE TABLE IF NOT EXISTS molecules (
    mol_id SERIAL PRIMARY KEY,
    smiles TEXT,
    mol MOL,                    -- RDKit molecule type
    inchikey TEXT UNIQUE,
    source TEXT,
    date_added TIMESTAMPTZ DEFAULT NOW()
);

-- Bioactivities Table
CREATE TABLE IF NOT EXISTS bioactivities (
    activity_id SERIAL PRIMARY KEY,
    mol_id INT REFERENCES molecules(mol_id),
    target TEXT,
    organism TEXT,
    mic_um REAL,
    mic_mg_l REAL,
    pic50 REAL,
    class TEXT,                 -- penicillin, cephalosporin, etc.
    resistance_status TEXT,     -- VRE, MRSA, CRPA, etc.
    assay_type TEXT,
    source TEXT,
    pmid TEXT
);

-- Resistance Trends Table (Time-Series)
CREATE TABLE IF NOT EXISTS resistance_trends (
    year TIMESTAMPTZ NOT NULL,
    organism TEXT,
    antibiotic_class TEXT,
    resistance_pct REAL,
    region TEXT,
    source TEXT
);

-- Convert to Hypertable
-- Note: We use 'if not exists' logic or catch errors in application code if already converted
SELECT create_hypertable('resistance_trends', 'year', if_not_exists => TRUE);

-- Indexes
CREATE INDEX IF NOT EXISTS mol_idx ON molecules USING gist(mol);
CREATE INDEX IF NOT EXISTS smiles_idx ON molecules(smiles);
