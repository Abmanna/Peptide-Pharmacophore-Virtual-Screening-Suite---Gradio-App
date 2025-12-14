# Complete Enhanced Peptide WGAN-GP Code

```python
"""
Enhanced Peptide Generation using WGAN-GP with R-GCN
Includes: Property prediction, length control, cyclic peptides, and sequence validation
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from chemistry import InteractiveChemistryEngine

# ============================================================================
# CONFIGURATION
# ============================================================================

# Standard 20 amino acids + D-amino acids (optional)
AMINO_ACIDS = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
]

# Optional: D-amino acids (mirror forms)
D_AMINO_ACIDS = ['d' + aa.lower() for aa in AMINO_ACIDS]

# Physicochemical properties
AA_PROPERTIES = {
    'A': {'hydrophobic': 1, 'charge': 0, 'polar': 0, 'aromatic': 0, 'small': 1},
    'C': {'hydrophobic': 1, 'charge': 0, 'polar': 0, 'aromatic': 0, 'small': 1},
    'D': {'hydrophobic': 0, 'charge': -1, 'polar': 1, 'aromatic': 0, 'small': 0},
    'E': {'hydrophobic': 0, 'charge': -1, 'polar': 1, 'aromatic': 0, 'small': 0},
    'F': {'hydrophobic': 1, 'charge': 0, 'polar': 0, 'aromatic': 1, 'small': 0},
    'G': {'hydrophobic': 0, 'charge': 0, 'polar': 0, 'aromatic': 0, 'small': 1},
    'H': {'hydrophobic': 0, 'charge': 0.5, 'polar': 1, 'aromatic': 1, 'small': 0},
    'I': {'hydrophobic': 1, 'charge': 0, 'polar': 0, 'aromatic': 0, 'small': 0},
    'K': {'hydrophobic': 0, 'charge': 1, 'polar': 1, 'aromatic': 0, 'small': 0},
    'L': {'hydrophobic': 1, 'charge': 0, 'polar': 0, 'aromatic': 0, 'small': 0},
    'M': {'hydrophobic': 1, 'charge': 0, 'polar': 0, 'aromatic': 0, 'small': 0},
    'N': {'hydrophobic': 0, 'charge': 0, 'polar': 1, 'aromatic': 0, 'small': 0},
    'P': {'hydrophobic': 0, 'charge': 0, 'polar': 0, 'aromatic': 0, 'small': 1},
    'Q': {'hydrophobic': 0, 'charge': 0, 'polar': 1, 'aromatic': 0, 'small': 0},
    'R': {'hydrophobic': 0, 'charge': 1, 'polar': 1, 'aromatic': 0, 'small': 0},
    'S': {'hydrophobic': 0, 'charge': 0, 'polar': 1, 'aromatic': 0, 'small':
    'T': {'hydrophobic': 0, 'charge': 0, 'polar': 1, 'aromatic': 0, 'small': 0},
    'V': {'hydrophobic': 1, 'charge': 0, 'polar': 0, 'aromatic': 0, 'small': 0},
    'W': {'hydrophobic': 1, 'charge': 0, 'polar': 0, 'aromatic': 1, 'small': 0},
    'Y': {'hydrophobic': 1, 'charge': 0, 'polar': 1, 'aromatic': 1, 'small': 0},
}

# Peptide properties for conditional generation
PEPTIDE_PROPERTIES = [
    'antimicrobial',
    'cell_penetrating', 
    'antiviral',
    'antifungal',
    'anticancer',
    'hemolytic'
]

# Mappings
aa_to_idx = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
idx_to_aa = {idx: aa for idx, aa in enumerate(AMINO_ACIDS)}
prop_to_idx = {prop: idx for idx, prop in enumerate(PEPTIDE_PROPERTIES)}

# Bond types
BOND_TYPES = ['PEPTIDE', 'DISULFIDE', 'CYCLIC', 'NONE']
bond_to_idx = {bond: idx for idx, bond in enumerate(BOND_TYPES)}

# Dimensions
MAX_RESIDUES = 30
RESIDUE_DIM = len(AMINO_ACIDS) + 1  # +1 for padding
PROPERTY_DIM = 5  # Physicochemical properties
FEATURE_DIM = RESIDUE_DIM + PROPERTY_DIM
BOND_DIM = len(BOND_TYPES)
LATENT_DIM = 128
CONDITION_DIM = len(PEPTIDE_PROPERTIES)


# ============================================================================
# DATA CONVERSION
# ============================================================================

# Helper functions for graph conversion
def encode_amino_acids(sequence: str) -> np.ndarray:
    """Encode amino acids into one-hot vectors."""
    features = np.zeros((MAX_RESIDUES, RESIDUE_DIM), dtype=np.float32)
    for i, aa in enumerate(sequence):
        if aa in aa_to_idx:
            features[i, aa_to_idx[aa]] = 1.0
    return features

def encode_properties(sequence: str) -> np.ndarray:
    """Encode physicochemical properties of amino acids."""
    features = np.zeros((MAX_RESIDUES, PROPERTY_DIM), dtype=np.float32)
    for i, aa in enumerate(sequence):
        if aa in AA_PROPERTIES:
            props = AA_PROPERTIES[aa]
            features[i] = [
                props['hydrophobic'],
                (props['charge'] + 1) / 2,
                props['polar'],
                props['aromatic'],
                props['small']
            ]
    return features

def encode_bonds(sequence: str, is_cyclic: bool, disulfide_pairs: Optional[List[Tuple[int, int]]]) -> np.ndarray:
    """Encode bond information into adjacency matrices."""
    adjacency = np.zeros((BOND_DIM, MAX_RESIDUES, MAX_RESIDUES), dtype=np.float32)
    n = len(sequence)
    for i in range(n - 1):
        adjacency[bond_to_idx['PEPTIDE'], i, i + 1] = 1
        adjacency[bond_to_idx['PEPTIDE'], i + 1, i] = 1
    if is_cyclic and n > 2:
        adjacency[bond_to_idx['CYCLIC'], 0, n - 1] = 1
        adjacency[bond_to_idx['CYCLIC'], n - 1, 0] = 1
    if disulfide_pairs:
        for i, j in disulfide_pairs:
            adjacency[bond_to_idx['DISULFIDE'], i, j] = 1
            adjacency[bond_to_idx['DISULFIDE'], j, i] = 1
    bond_exists = np.sum(adjacency[:-1], axis=0)
    adjacency[-1] = 1 - np.clip(bond_exists, 0, 1)
    return adjacency

# Refactored sequence_to_graph function
def sequence_to_graph(sequence: str, is_cyclic: bool = False, disulfide_pairs: Optional[List[Tuple[int, int]]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert peptide sequence to graph representation.
    """
    if len(sequence) > MAX_RESIDUES:
        raise ValueError(f"Sequence length {len(sequence)} exceeds maximum {MAX_RESIDUES}")

    features = np.concatenate([
        encode_amino_acids(sequence),
        encode_properties(sequence)
    ], axis=1)

    adjacency = encode_bonds(sequence, is_cyclic, disulfide_pairs)

    return adjacency, features


def graph_to_sequence(
    adjacency: np.ndarray,
    features: np.ndarray,
    threshold: float = 0.5
) -> Dict:
    """
    Convert graph back to peptide sequence with structural info.
    
    Returns:
        dict with 'sequence', 'is_cyclic', 'disulfide_bonds'
    """
    aa_indices = np.argmax(features[:, :RESIDUE_DIM], axis=1)
    
    sequence = []
    for i, idx in enumerate(aa_indices):
        if idx == RESIDUE_DIM - 1 or features[i, RESIDUE_DIM-1] > threshold:
            break
        if idx < len(AMINO_ACIDS):
            sequence.append(idx_to_aa[idx])
    
    seq_str = ''.join(sequence)
    n = len(sequence)
    
    # Check if cyclic
    is_cyclic = False
    if n > 2:
        cyclic_bond = adjacency[bond_to_idx['CYCLIC'], 0, n-1]
        is_cyclic = cyclic_bond > threshold
    
    # Extract disulfide bonds
    disulfide_bonds = []
    disulfide_matrix = adjacency[bond_to_idx['DISULFIDE']]
    for i in range(n):
        for j in range(i+1, n):
            if disulfide_matrix[i, j] > threshold:
                disulfide_bonds.append((i, j))
    
    return {
        'sequence': seq_str,
        'is_cyclic': is_cyclic,
        'disulfide_bonds': disulfide_bonds,
        'length': n
    }


# ============================================================================
# PROPERTY PREDICTOR (Auxiliary Network)
# ============================================================================

class PropertyPredictor(keras.layers.Layer):
    """Predicts peptide properties from graph representation."""
    
    def __init__(self, num_properties, **kwargs):
        super().__init__(**kwargs)
        self.num_properties = num_properties
        
    def build(self, input_shape):
        self.dense1 = keras.layers.Dense(256, activation='relu')
        self.dropout1 = keras.layers.Dropout(0.3)
        self.dense2 = keras.layers.Dense(128, activation='relu')
        self.dropout2 = keras.layers.Dropout(0.3)
        self.output_layer = keras.layers.Dense(
            self.num_properties, 
            activation='sigmoid',
            name='property_predictions'
        )
        self.built = True
    
    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return self.output_layer(x)


# ============================================================================
# GRAPH CONVOLUTIONAL LAYER WITH ATTENTION
# ============================================================================

class RelationalGraphConvLayer(keras.layers.Layer):
    """R-GCN layer with multi-head attention mechanism."""
    
    def __init__(self, units=128, num_heads=4, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.activation = keras.activations.get(activation)
    
    def build(self, input_shape):
        bond_dim = input_shape[0][1]
        feature_dim = input_shape[1][2]
        
        # Separate weights for each bond type
        self.kernels = []
        for _ in range(bond_dim):
            self.kernels.append(
                self.add_weight(
                    shape=(feature_dim, self.units),
                    initializer='glorot_uniform',
                    trainable=True
                )
            )
        
        # Multi-head attention
        self.attention_weights = []
        for _ in range(self.num_heads):
            self.attention_weights.append(
                self.add_weight(
                    shape=(self.units, 1),
                    initializer='glorot_uniform',
                    trainable=True
                )
            )
        
        self.combine_heads = keras.layers.Dense(self.units)
        self.built = True
    
    def call(self, inputs, training=False):
        adjacency, features = inputs
        batch_size = tf.shape(features)[0]
        
        # Process each bond type
        outputs = []
        for i, kernel in enumerate(self.kernels):
            # Aggregate neighbors
            adj_slice = adjacency[:, i, :, :]  # (batch, nodes, nodes)
            aggregated = tf.matmul(adj_slice, features)  # (batch, nodes, features)
            transformed = tf.matmul(aggregated, kernel)  # (batch, nodes, units)
            outputs.append(transformed)
        
        # Stack bond-type outputs
        x = tf.stack(outputs, axis=1)  # (batch, bond_types, nodes, units)
        
        # Multi-head attention over bond types
        attention_outputs = []
        for att_w in self.attention_weights:
            scores = tf.matmul(x, att_w)  # (batch, bond_types, nodes, 1)
            weights = tf.nn.softmax(scores, axis=1)  # Softmax over bond types
            attended = tf.reduce_sum(x * weights, axis=1)  # (batch, nodes, units)
            attention_outputs.append(attended)
        
        # Combine attention heads
        x = tf.concat(attention_outputs, axis=-1)
        x = self.combine_heads(x)
        
        return self.activation(x)


# ============================================================================
# ENHANCED GENERATOR WITH LENGTH CONTROL
# ============================================================================

class ConditionalPeptideGenerator(keras.Model):
    """
    Generator with property conditioning and length control.
    """
    
    def __init__(
        self,
        dense_units=[256, 512, 1024],
        dropout_rate=0.2,
        latent_dim=LATENT_DIM,
        condition_dim=CONDITION_DIM,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        
        # Condition encoder
        self.condition_encoder = keras.Sequential([
            keras.layers.Dense(64, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(128, activation='relu'),
        ], name='condition_encoder')
        
        # Length encoder (embedded length signal)
        self.length_encoder = keras.Sequential([
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
        ], name='length_encoder')
        
        # Main dense layers
        self.dense_layers = []
        for units in dense_units:
            self.dense_layers.append(keras.layers.Dense(units, activation='tanh'))
            self.dense_layers.append(keras.layers.Dropout(dropout_rate))
        
        # Adjacency head
        self.adj_dense = keras.layers.Dense(512, activation='relu')
        self.adj_output = keras.layers.Dense(
            BOND_DIM * MAX_RESIDUES * MAX_RESIDUES
        )
        
        # Feature head  
        self.feat_dense = keras.layers.Dense(512, activation='relu')
        self.feat_output = keras.layers.Dense(
            MAX_RESIDUES * FEATURE_DIM
        )
    
    def call(self, inputs, training=False):
        z, conditions, target_length = inputs
        
        # Encode conditions
        cond_emb = self.condition_encoder(conditions, training=training)
        
        # Encode target length (normalized to [0, 1])
        length_norm = target_length / MAX_RESIDUES
        length_emb = self.length_encoder(length_norm, training=training)
        
        # Combine all inputs
        x = tf.concat([z, cond_emb, length_emb], axis=-1)
        
        # Process through dense layers
        for layer in self.dense_layers:
            x = layer(x, training=training)
        
        # Generate adjacency matrix
        adj = self.adj_dense(x, training=training)
        adj = self.adj_output(adj)
        adj = tf.reshape(adj, (-1, BOND_DIM, MAX_RESIDUES, MAX_RESIDUES))
        # Symmetrize
        adj = (adj + tf.transpose(adj, (0, 1, 3, 2))) / 2
        adj = tf.nn.softmax(adj, axis=1)
        
        # Generate features
        feat = self.feat_dense(x, training=training)
        feat = self.feat_output(feat)
        feat = tf.reshape(feat, (-1, MAX_RESIDUES, FEATURE_DIM))
        # Softmax over amino acid types
        feat_aa = tf.nn.softmax(feat[:, :, :RESIDUE_DIM], axis=-1)
        feat_props = tf.nn.sigmoid(feat[:, :, RESIDUE_DIM:])
        feat = tf.concat([feat_aa, feat_props], axis=-1)
        
        return adj, feat


# ============================================================================
# ENHANCED DISCRIMINATOR WITH PROPERTY PREDICTION
# ============================================================================

class EnhancedPeptideDiscriminator(keras.Model):
    """
    Discriminator that also predicts peptide properties.
    """
    
    def __init__(
        self,
        gconv_units=[128, 256, 256, 128],
        dense_units=[512, 256],
        dropout_rate=0.2,
        predict_properties=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.predict_properties = predict_properties
        
        # Graph convolution layers
        self.gconv_layers = []
        for units in gconv_units:
            self.gconv_layers.append(
                RelationalGraphConvLayer(units, num_heads=4)
            )
            self.gconv_layers.append(keras.layers.Dropout(dropout_rate))
        
        # Global pooling
        self.pool_mean = keras.layers.GlobalAveragePooling1D()
        self.pool_max = keras.layers.GlobalMaxPooling1D()
        
        # Dense layers for main task
        self.dense_layers = []
        for units in dense_units:
            self.dense_layers.append(keras.layers.Dense(units, activation='relu'))
            self.dense_layers.append(keras.layers.Dropout(dropout_rate))
        
        # Real/fake output
        self.realfake_output = keras.layers.Dense(1, dtype='float32', name='realfake')
        
        # Property prediction head
        if self.predict_properties:
            self.property_predictor = PropertyPredictor(
                num_properties=CONDITION_DIM,
                name='property_predictor'
            )
    
    def call(self, inputs, training=False):
        adjacency, features = inputs
        
        # Graph convolutions
        x = features
        for layer in self.gconv_layers:
            if isinstance(layer, RelationalGraphConvLayer):
                x = layer([adjacency, x], training=training)
            else:
                x = layer(x, training=training)
        
        # Global pooling
        x_mean = self.pool_mean(x)
        x_max = self.pool_max(x)
        x_pooled = tf.concat([x_mean, x_max], axis=-1)
        
        # Dense processing
        x_dense = x_pooled
        for layer in self.dense_layers:
            x_dense = layer(x_dense, training=training)
        
        # Real/fake score
        realfake_score = self.realfake_output(x_dense)
        
        # Property predictions
        if self.predict_properties:
            property_preds = self.property_predictor(x_pooled, training=training)
            return realfake_score, property_preds
        
        return realfake_score


# ============================================================================
# ENHANCED WGAN WITH MULTIPLE OBJECTIVES
# ============================================================================

class EnhancedPeptideWGAN(keras.Model):
    """
    WGAN-GP with property prediction and sequence validation.
    """
    
    def __init__(
        self,
        generator,
        discriminator,
        discriminator_steps=5,
        generator_steps=1,
        gp_weight=10.0,
        property_weight=1.0,
        sequence_weight=0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.discriminator_steps = discriminator_steps
        self.generator_steps = generator_steps
        self.gp_weight = gp_weight
        self.property_weight = property_weight
        self.sequence_weight = sequence_weight
        self.latent_dim = self.generator.latent_dim
    
    def compile(
        self,
        optimizer_generator,
        optimizer_discriminator,
        **kwargs
    ):
        super().compile(**kwargs)
        self.optimizer_generator = optimizer_generator
        self.optimizer_discriminator = optimizer_discriminator
        
        # Metrics
        self.metric_gen = keras.metrics.Mean(name='loss_gen')
        self.metric_disc = keras.metrics.Mean(name='loss_disc')
        self.metric_prop = keras.metrics.Mean(name='loss_prop')
        self.metric_seq = keras.metrics.Mean(name='loss_seq')
    
    def train_step(self, data):
        if isinstance(data, tuple):
            graph_real, properties_real, lengths_real = data
        else:
            graph_real = data
            properties_real = None
            lengths_real = None
        
        batch_size = tf.shape(graph_real[0])[0]
        
        # Train discriminator
        for _ in range(self.discriminator_steps):
            z = tf.random.normal((batch_size, self.latent_dim))
            
            # Random conditions for generation
            if properties_real is not None:
                conditions = properties_real
                lengths = lengths_real
            else:
                conditions = tf.random.uniform((batch_size, CONDITION_DIM))
                lengths = tf.random.uniform(
                    (batch_size, 1),
                    minval=5,
                    maxval=MAX_RESIDUES
                )
            
            with tf.GradientTape() as tape:
                # Generate fake samples
                graph_fake = self.generator(
                    [z, conditions, lengths],
                    training=True
                )
                
                # Discriminator predictions
                disc_real = self.discriminator(graph_real, training=True)
                disc_fake = self.discriminator(graph_fake, training=True)
                
                # Wasserstein loss
                if isinstance(disc_real, tuple):
                    real_score, real_props = disc_real
                    fake_score, fake_props = disc_fake
                else:
                    real_score = disc_real
                    fake_score = disc_fake
                    real_props = None
                
                loss_disc = tf.reduce_mean(fake_score) - tf.reduce_mean(real_score)
                
                # Gradient penalty
                loss_gp = self._gradient_penalty(graph_real, graph_fake)
                loss_disc = loss_disc + self.gp_weight * loss_gp
                
                # Property prediction loss (if available)
                if real_props is not None and properties_real is not None:
                    loss_prop = tf.reduce_mean(
                        keras.losses.binary_crossentropy(
                            properties_real,
                            real_props
                        )
                    )
                    loss_disc = loss_disc + self.property_weight * loss_prop
                    self.metric_prop.update_state(loss_prop)
            
            # Update discriminator
            grads = tape.gradient(loss_disc, self.discriminator.trainable_weights)
            self.optimizer_discriminator.apply_gradients(
                zip(grads, self.discriminator.trainable_weights)
            )
            self.metric_disc.update_state(loss_disc)
        
        # Train generator
        for _ in range(self.generator_steps):
            z = tf.random.normal((batch_size, self.latent_dim))
            
            if properties_real is not None:
                conditions = properties_real
                lengths = lengths_real
            else:
                conditions = tf.random.uniform((batch_size, CONDITION_DIM))
                lengths = tf.random.uniform(
                    (batch_size, 1),
                    minval=5,
                    maxval=MAX_RESIDUES
                )
            
            with tf.GradientTape() as tape:
                graph_fake = self.generator(
                    [z, conditions, lengths],
                    training=True
                )
                
                disc_fake = self.discriminator(graph_fake, training=True)
                
                if isinstance(disc_fake, tuple):
                    fake_score, fake_props = disc_fake
                else:
                    fake_score = disc_fake
                    fake_props = None
                
                # Generator loss (want high discriminator score)
                loss_gen = -tf.reduce_mean(fake_score)
                
                # Sequence validity loss
                loss_seq = self._sequence_validity_loss(graph_fake, lengths)
                loss_gen = loss_gen + self.sequence_weight * loss_seq
                self.metric_seq.update_state(loss_seq)
                
                # Property matching loss
                if fake_props is not None:
                    loss_prop_gen = tf.reduce_mean(
                        keras.losses.binary_crossentropy(conditions, fake_props)
                    )
                    loss_gen = loss_gen + self.property_weight * loss_prop_gen
            
            # Update generator
            grads = tape.gradient(loss_gen, self.generator.trainable_weights)
            self.optimizer_generator.apply_gradients(
                zip(grads, self.generator.trainable_weights)
            )
            self.metric_gen.update_state(loss_gen)
        
        return {m.name: m.result() for m in self.metrics}
    
    def _gradient_penalty(self, graph_real, graph_fake):
        """Compute gradient penalty for WGAN-GP."""
        adj_real, feat_real = graph_real
        adj_fake, feat_fake = graph_fake
        
        batch_size = tf.shape(adj_real)[0]
        alpha = tf.random.uniform([batch_size, 1, 1, 1])
        adj_interp = adj_real * alpha + adj_fake * (1 - alpha)
        
        alpha = tf.reshape(alpha, [batch_size, 1, 1])
        feat_interp = feat_real * alpha + feat_fake * (1 - alpha)
        
        with tf.GradientTape() as tape:
            tape.watch(adj_interp)
            tape.watch(feat_interp)
            disc_interp = self.discriminator([adj_interp, feat_interp], training=True)
            if isinstance(disc_interp, tuple):
                disc_interp = disc_interp[0]
        
        grads = tape.gradient(disc_interp, [adj_interp, feat_interp])
        grad_adj = tf.reshape(grads[0], [batch_size, -1])
        grad_feat = tf.reshape(grads[1], [batch_size, -1])
        grad_combined = tf.concat([grad_adj, grad_feat], axis=1)
        
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad_combined), axis=1) + 1e-8)
        gp = tf.reduce_mean(tf.square(grad_norm - 1.0))
        
        return gp
    
    def _sequence_validity_loss(self, graph_fake, target_lengths):
        """
        Penalize invalid peptide sequences.
        Encourages proper length and valid amino acid distributions.
        """
        adj, feat = graph_fake
        batch_size = tf.shape(feat)[0]
        
        # Check peptide bond connectivity
        peptide_bonds = adj[:, bond_to_idx['PEPTIDE'], :, :]
        
        # Encourage sequential connectivity
        diagonal_upper = tf.linalg.diag_part(peptide_bonds[:, :-1, 1:])
        connectivity_loss = tf.reduce_mean(1.0 - diagonal_upper)
        
        # Encourage correct length
        aa_probs = feat[:, :, :RESIDUE_DIM]
        aa_confidence = tf.reduce_max(aa_probs, axis=-1)
        
        length_mask = tf.sequence_mask(
            tf.cast(tf.squeeze(target_lengths), tf.int32),
            MAX_RESIDUES,
            dtype=tf.float32
        )
        
        length_loss = tf.reduce_mean(
            tf.abs(aa_confidence - length_mask)
        )
        
        return connectivity_loss + length_loss
    
    @property
    def metrics(self):
        return [
            self.metric_gen,
            self.metric_disc,
            self.metric_prop,
            self.metric_seq
        ]


# ============================================================================
# PEPTIDE DATASET LOADER
# ============================================================================

class PeptideDataset:
    """
    Dataset handler for peptide sequences with properties.
    Refactored to include a generic load_peptides function.
    """
    
    def __init__(self):
        self.peptides = []
    
    def add_peptide(
        self,
        sequence: str,
        properties: List[str] = None,
        is_cyclic: bool = False,
        disulfide_pairs: List[Tuple[int, int]] = None
    ):
        self.peptides.append({
            "sequence": sequence,
            "properties": properties or [],
            "is_cyclic": is_cyclic,
            "disulfide_pairs": disulfide_pairs or []
        })
    
    def load_peptides(self, peptide_data: List[Tuple[str, List[str], Optional[bool], Optional[List[Tuple[int, int]]]]]):
        """
        Generic function to load peptides into the dataset.
        Args:
            peptide_data: List of tuples containing sequence, properties, is_cyclic, and disulfide_pairs.
        """
        for seq, props, is_cyclic, ss_bonds in peptide_data:
            self.add_peptide(seq, props, is_cyclic or False, ss_bonds or [])
    
    def load_antimicrobial_peptides(self):
        """Load predefined antimicrobial peptides."""
        antimicrobial_data = [
            ("FLGFLGFLG", ["antimicrobial"], False, None),
            ("RWKFGGFKWR", ["antimicrobial", "cell_penetrating"], False, None),
            ("KLKLLKKLLKK", ["antimicrobial"], False, None),
        ]
        self.load_peptides(antimicrobial_data)
    
    def load_cell_penetrating_peptides(self):
        """Load predefined cell-penetrating peptides."""
        cpp_data = [
            ("KKKKKKKKK", ["cell_penetrating"], False, None),
            ("VRLRIRVAVIRA", ["cell_penetrating"], False, None),
        ]
        self.load_peptides(cpp_data)
    
    def load_cyclic_peptides(self):
        """Load predefined cyclic peptides."""
        cyclic_data = [
            ("CYCLIC1", ["cyclic"], True, [(0, 5)]),
            ("CYCLIC2", ["cyclic"], True, [(1, 4)]),
        ]
        self.load_peptides(cyclic_data)
    
    def to_tensors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert dataset to training tensors."""
        adjacencies = []
        features = []
        
        for i, pep in enumerate(self.peptides):
            seq = pep['sequence']
            adj, feat = sequence_to_graph(
                seq,
                is_cyclic=pep.get('is_cyclic', False),
                disulfide_pairs=pep.get('disulfide_pairs') if pep.get('disulfide_pairs') else None
            )
            adjacencies.append(adj)
            features.append(feat)
        
        adjacency_tensor = np.array(adjacencies, dtype=np.float32)
        feature_tensor = np.array(features, dtype=np.float32)
        property_tensor = np.array([p['properties'] for p in self.peptides], dtype=np.float32)
        length_tensor = np.array([[len(p['sequence'])] for p in self.peptides], dtype=np.float32)
        
        return adjacency_tensor, feature_tensor, property_tensor, length_tensor
    
    def create_tf_dataset(self, batch_size: int = 32, shuffle: bool = True):
        """Create a TensorFlow dataset for training."""
        adj, feat, props, lengths = self.to_tensors()
        
        dataset = tf.data.Dataset.from_tensor_slices((
            (adj, feat),
            props,
            lengths
        ))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.peptides))
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def __len__(self):
        return len(self.peptides)


# ============================================================================
# SAMPLING AND VALIDATION
# ============================================================================

class PeptideSampler:
    """
    Sample and validate peptides from the trained generator.
    Refactored to include a helper function for common sampling logic.
    """

    def __init__(self, generator: ConditionalPeptideGenerator):
        self.generator = generator

    def _generate_samples(self, num_samples: int, target_properties: List[str] = None, target_length: int = None):
        """
        Helper function to generate peptide samples.
        """
        samples = []
        for _ in range(num_samples):
            z = tf.random.normal([1, LATENT_DIM])
            conditions = tf.zeros([1, CONDITION_DIM])
            if target_properties:
                for prop in target_properties:
                    conditions[0, prop_to_idx[prop]] = 1.0
            length = tf.constant([[target_length or MAX_RESIDUES]], dtype=tf.float32)
            generated = self.generator([z, conditions, length], training=False)
            samples.append(generated)
        return samples

    def sample(self, num_samples: int, target_properties: List[str] = None, target_length: int = None) -> List[Dict]:
        """
        Generate peptide samples without filtering.
        """
        return self._generate_samples(num_samples, target_properties, target_length)

    def sample_with_filtering(
        self,
        num_samples: int,
        target_properties: List[str] = None,
        target_length: int = None,
        min_length: int = 5,
        max_length: int = MAX_RESIDUES,
        require_valid: bool = True,
        max_attempts: int = 10
    ) -> List[Dict]:
        """
        Generate peptide samples with filtering criteria.
        """
        filtered_samples = []
        attempts = 0
        while len(filtered_samples) < num_samples and attempts < max_attempts:
            samples = self._generate_samples(num_samples, target_properties, target_length)
            for sample in samples:
                sequence = sample['sequence']
                if min_length <= len(sequence) <= max_length:
                    if not require_valid or self._is_valid(sequence):
                        filtered_samples.append(sample)
                if len(filtered_samples) >= num_samples:
                    break
            attempts += 1
        return filtered_samples

    def _is_valid(self, sequence: str) -> bool:
        """
        Validate peptide sequence (placeholder for actual validation logic).
        """
        return True


# ============================================================================
# PEPTIDE ANALYSIS UTILITIES
# ============================================================================

class PeptideAnalyzer:
    """
    Analyze generated peptides for various properties.
    """
    
    @staticmethod
    def compute_composition(sequence: str) -> Dict[str, float]:
        """Compute amino acid composition."""
        n = len(sequence)
        if n == 0:
            return {}
        
        composition = {}
        for aa in AMINO_ACIDS:
            count = sequence.count(aa)
            composition[aa] = count / n
        
        return composition
    
    @staticmethod
    def compute_charge(sequence: str, pH: float = 7.0) -> float:
        """Compute net charge at given pH."""
        # pKa values
        pKa = {'K': 10.5, 'R': 12.5, 'H': 6.0, 'D': 3.9, 'E': 4.1}
        
        charge = 0.0
        for aa in sequence:
            if aa in ['K', 'R', 'H']:
                charge += 1.0 / (1.0 + 10**(pH - pKa.get(aa, 7.0)))
            elif aa in ['D', 'E']:
                charge -= 1.0 / (1.0 + 10**(pKa.get(aa, 7.0) - pH))
        
        # N-terminus
        charge += 1.0 / (1.0 + 10**(pH - 9.0))
        # C-terminus
        charge -= 1.0 / (1.0 + 10**(2.0 - pH))
        
        return charge
    
    @staticmethod
    def compute_hydrophobicity(sequence: str) -> float:
        """Compute average hydrophobicity (Kyte-Doolittle scale)."""
        kd_scale = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
        
        if not sequence:
            return 0.0
        
        total = sum(kd_scale.get(aa, 0) for aa in sequence)
        return total / len(sequence)
    
    @staticmethod
    def compute_molecular_weight(sequence: str) -> float:
        """Compute molecular weight in Daltons."""
        mw = {
            'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2,
            'Q': 146.2, 'E': 147.1, 'G': 75.1, 'H': 155.2, 'I': 131.2,
            'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2, 'P': 115.1,
            'S': 105.1, 'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1
        }
        
        total = sum(mw.get(aa, 0) for aa in sequence)
        # Subtract water for peptide bonds
        total -= 18.0 * (len(sequence) - 1)
        
        return total
    
    @staticmethod
    def analyze_peptide(sequence: str) -> Dict:
        """Comprehensive peptide analysis."""
        return {
            'sequence': sequence,
            'length': len(sequence),
            'composition': PeptideAnalyzer.compute_composition(sequence),
            'charge_ph7': PeptideAnalyzer.compute_charge(sequence, 7.0),
            'hydrophobicity': PeptideAnalyzer.compute_hydrophobicity(sequence),
            'molecular_weight': PeptideAnalyzer.compute_molecular_weight(sequence),
            'cysteine_count': sequence.count('C'),
            'proline_count': sequence.count('P'),
            'positive_residues': sum(1 for aa in sequence if aa in 'KRH'),
            'negative_residues': sum(1 for aa in sequence if aa in 'DE'),
            'aromatic_residues': sum(1 for aa in sequence if aa in 'FWY'),
        }
    
    @staticmethod
    def batch_analyze(peptides: List[Dict]) -> List[Dict]:
        """Analyze a batch of peptides."""
        results = []
        for pep in peptides:
            if pep['sequence']:
                analysis = PeptideAnalyzer.analyze_peptide(pep['sequence'])
                analysis['is_cyclic'] = pep.get('is_cyclic', False)
                analysis['disulfide_bonds'] = pep.get('disulfide_bonds', [])
                results.append(analysis)
        return results


# ============================================================================
# TRAINING CALLBACKS
# ============================================================================

class PeptideGenerationCallback(keras.callbacks.Callback):
    """
    Callback to sample and display peptides during training.
    """
    
    def __init__(
        self,
        generator,
        sample_interval: int = 10,
        num_samples: int = 5,
        target_properties: List[str] = None
    ):
        super().__init__()
        self.generator = generator
        self.sample_interval = sample_interval
        self.num_samples = num_samples
        self.target_properties = target_properties
        self.sampler = None
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.sample_interval == 0:
            if self.sampler is None:
                self.sampler = PeptideSampler(self.generator)
            
            print(f"\n--- Epoch {epoch + 1}: Generated Peptides ---")
            samples = self.sampler.sample_with_filtering(
                num_samples=self.num_samples,
                target_properties=self.target_properties,
                min_length=5
            )
            
            for i, sample in enumerate(samples):
                seq = sample['sequence']
                if seq:
                    charge = PeptideAnalyzer.compute_charge(seq)
                    hydro = PeptideAnalyzer.compute_hydrophobicity(seq)
                    cyclic_str = " [CYCLIC]" if sample['is_cyclic'] else ""
                    print(f"  {i+1}. {seq} (len={len(seq)}, charge={charge:.1f}, H={hydro:.2f}){cyclic_str}")
            print()


class ModelCheckpointCallback(keras.callbacks.Callback):
    """
    Save model checkpoints during training.
    """
    
    def __init__(self, save_path: str, save_interval: int = 50):
        super().__init__()
        self.save_path = save_path
        self.save_interval = save_interval
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_interval == 0:
            self.model.generator.save_weights(
                f"{self.save_path}/generator_epoch_{epoch+1}.h5"
            )
            self.model.discriminator.save_weights(
                f"{self.save_path}/discriminator_epoch_{epoch+1}.h5"
            )
            print(f"Saved checkpoint at epoch {epoch + 1}")


# ============================================================================
# VISUALIZATION
# ============================================================================

# Helper functions for graph visualization
def create_graph(adjacency: np.ndarray, features: np.ndarray) -> nx.Graph:
    """Create a NetworkX graph from adjacency and feature matrices."""
    info = graph_to_sequence(adjacency, features)
    sequence = info['sequence']
    n = len(sequence)

    G = nx.Graph()
    for i, aa in enumerate(sequence):
        G.add_node(i, label=aa)

    for i in range(n):
        for j in range(i + 1, n):
            for bond_type, bond_idx in bond_to_idx.items():
                if adjacency[bond_idx, i, j] > 0.5:
                    G.add_edge(i, j, bond_type=bond_type)
    return G

def style_graph(G: nx.Graph) -> Tuple[Dict, Dict]:
    """Apply styles to the graph nodes and edges."""
    node_colors = ['lightblue' for _ in G.nodes]
    edge_colors = []
    edge_styles = []

    for u, v, data in G.edges(data=True):
        bond_type = data.get('bond_type', 'NONE')
        if bond_type == 'PEPTIDE':
            edge_colors.append('black')
            edge_styles.append('solid')
        elif bond_type == 'DISULFIDE':
            edge_colors.append('gold')
            edge_styles.append('dashed')
        elif bond_type == 'CYCLIC':
            edge_colors.append('red')
            edge_styles.append('dotted')
        else:
            edge_colors.append('gray')
            edge_styles.append('solid')

    return node_colors, {'edge_color': edge_colors, 'style': edge_styles}

# Refactored visualize_peptide_graph function
def visualize_peptide_graph(adjacency: np.ndarray, features: np.ndarray):
    """Visualize peptide as a graph (requires matplotlib and networkx)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for visualization.")

    G = create_graph(adjacency, features)
    node_colors, edge_styles = style_graph(G)

    pos = nx.spring_layout(G, seed=42)
    labels = nx.get_node_attributes(G, 'label')

    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw(
        G, pos, ax=ax, with_labels=True, labels=labels,
        node_color=node_colors, node_size=700, font_size=10,
        **edge_styles
    )
    plt.tight_layout()
    plt.show()


# ============================================================================
# EXAMPLE USAGE - COMPLETE TRAINING PIPELINE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Enhanced Peptide WGAN-GP with R-GCN")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # 1. Create and load dataset
    # -------------------------------------------------------------------------
    print("\n[1] Loading peptide dataset...")
    
    dataset = PeptideDataset()
    
    # Load example peptides from different categories
    dataset.load_antimicrobial_peptides()
    dataset.load_cell_penetrating_peptides()
    dataset.load_cyclic_peptides()
    
    # Add more custom peptides if desired
    custom_peptides = [
        ("KKKKKKKKK", ["cell_penetrating"]),  # Poly-K
        ("AAAAAAAA", []),  # Poly-A
        ("FLGFLGFLG", ["antimicrobial"]),  # Repetitive
        ("RWKFGGFKWR", ["antimicrobial", "cell_penetrating"]),
        ("KLKLLKKLLKK", ["antimicrobial"]),
        ("GRFKRFRKKFKKLFKKLS", ["antimicrobial"]),
        ("GIGKHVGKALKGLKGLLKGLGES", ["antimicrobial"]),
        ("VRLRIRVAVIRA", ["cell_penetrating"]),
    ]
    
    for seq, props in custom_peptides:
        dataset.add_peptide(seq, props)
    
    print(f"   Total peptides in dataset: {len(dataset)}")
    
    # Convert to tensors
    adj_tensor, feat_tensor, prop_tensor, len_tensor = dataset.to_tensors()
    print(f"   Adjacency tensor shape: {adj_tensor.shape}")
    print(f"   Feature tensor shape: {feat_tensor.shape}")
    print(f"   Property tensor shape: {prop_tensor.shape}")
    print(f"   Length tensor shape: {len_tensor.shape}")
    
    # -------------------------------------------------------------------------
    # 2. Build models
    # -------------------------------------------------------------------------
    print("\n[2] Building models...")
    
    generator = ConditionalPeptideGenerator(
        dense_units=[256, 512, 1024],
        dropout_rate=0.2,
        latent_dim=LATENT_DIM,
        condition_dim=CONDITION_DIM
    )
    
    discriminator = EnhancedPeptideDiscriminator(
        gconv_units=[128, 256, 256, 128],
        dense_units=[512, 256],
        dropout_rate=0.2,
        predict_properties=True
    )
    
    # Build models by calling with dummy input
    dummy_z = tf.zeros((1, LATENT_DIM))
    dummy_cond = tf.zeros((1, CONDITION_DIM))
    dummy_len = tf.zeros((1, 1))
    dummy_adj = tf.zeros((1, BOND_DIM, MAX_RESIDUES, MAX_RESIDUES))
    dummy_feat = tf.zeros((1, MAX_RESIDUES, FEATURE_DIM))
    
    _ = generator([dummy_z, dummy_cond, dummy_len])
    _ = discriminator([dummy_adj, dummy_feat])
    
    print(f"   Generator parameters: {generator.count_params():,}")
    print(f"   Discriminator parameters: {discriminator.count_params():,}")
    
    # -------------------------------------------------------------------------
    # 3. Create WGAN
    # -------------------------------------------------------------------------
    print("\n[3] Creating WGAN-GP...")
    
    wgan = EnhancedPeptideWGAN(
        generator=generator,
        discriminator=discriminator,
        discriminator_steps=5,
        generator_steps=1,
        gp_weight=10.0,
        property_weight=1.0,
        sequence_weight=0.5
    )
    
    wgan.compile(
        optimizer_generator=keras.optimizers.Adam(
            learning_rate=1e-4, beta_1=0.5, beta_2=0.9
        ),
        optimizer_discriminator=keras.optimizers.Adam(
            learning_rate=1e-4, beta_1=0.5, beta_2=0.9
        )
    )
    
    # -------------------------------------------------------------------------
    # 4. Setup callbacks
    # -------------------------------------------------------------------------
    print("\n[4] Setting up callbacks...")
    
    callbacks = [
        PeptideGenerationCallback(
            generator=generator,
            sample_interval=10,
            num_samples=5,
            target_properties=['antimicrobial']
        ),
        keras.callbacks.EarlyStopping(
            monitor='loss_gen',
            patience=50,
            restore_best_weights=True
        ),
    ]
    
    # -------------------------------------------------------------------------
    # 5. Create TensorFlow dataset
    # -------------------------------------------------------------------------
    print("\n[5] Preparing training data...")
    
    # Create training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((
        (adj_tensor, feat_tensor),
        prop_tensor,
        len_tensor
    ))
    
    BATCH_SIZE = 16
    train_dataset = train_dataset.shuffle(buffer_size=len(dataset))
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Number of batches: {len(dataset) // BATCH_SIZE}")
    
    # -------------------------------------------------------------------------
    # 6. Train the model
    # -------------------------------------------------------------------------
    print("\n[6] Training model...")
    print("-" * 70)
    
    EPOCHS = 100  # Increase for better results
    
    history = wgan.fit(
        train_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # -------------------------------------------------------------------------
    # 7. Sample and analyze generated peptides
    # -------------------------------------------------------------------------
    print("\n[7] Generating and analyzing peptides...")
    print("-" * 70)
    
    sampler = PeptideSampler(generator)
    
    # Generate antimicrobial peptides
    print("\nAntimicrobial Peptides:")
    amp_peptides = sampler.sample_with_filtering(
        num_samples=10,
        target_properties=['antimicrobial'],
        target_length=15,
        min_length=8,
        max_length=25
    )
    
    for i, pep in enumerate(amp_peptides):
        if pep['sequence']:
            analysis = PeptideAnalyzer.analyze_peptide(pep['sequence'])
            print(f"  {i+1}. {analysis['sequence']}")
            print(f"      Length: {analysis['length']}, Charge: {analysis['charge_ph7']:.1f}, "
                  f"Hydrophobicity: {analysis['hydrophobicity']:.2f}")
            print(f"      +Residues: {analysis['positive_residues']}, "
                  f"-Residues: {analysis['negative_residues']}")
    
    # Generate cell-penetrating peptides
    print("\nCell-Penetrating Peptides:")
    cpp_peptides = sampler.sample_with_filtering(
        num_samples=10,
        target_properties=['cell_penetrating'],
        target_length=12,
        min_length=8,
        max_length=20
    )
    
    for i, pep in enumerate(cpp_peptides):
        if pep['sequence']:
            analysis = PeptideAnalyzer.analyze_peptide(pep['sequence'])
            print(f"  {i+1}. {analysis['sequence']}")
            print(f"      Length: {analysis['length']}, Charge: {analysis['charge_ph7']:.1f}, "
                  f"MW: {analysis['molecular_weight']:.1f} Da")
    
    # -------------------------------------------------------------------------
    # 8. Diversity analysis
    # -------------------------------------------------------------------------
    print("\n[8] Diversity Analysis...")
    print("-" * 70)
    
    all_samples = sampler.sample_with_filtering(
        num_samples=100,
        min_length=5
    )
    
    valid_sequences = [s['sequence'] for s in all_samples if s['sequence']]
    unique_sequences = set(valid_sequences)
    
    print(f"  Valid peptides: {len(valid_sequences)}/100")
    print(f"  Unique peptides: {len(unique_sequences)}")
    print(f"  Uniqueness rate: {len(unique_sequences)/max(len(valid_sequences), 1)*100:.1f}%")
    
    # Length distribution
    lengths = [len(s) for s in valid_sequences]
    if lengths:
        print(f"  Average length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
        print(f"  Length range: {min(lengths)} - {max(lengths)}")
    
    # Charge distribution
    charges = [PeptideAnalyzer.compute_charge(s) for s in valid_sequences]
    if charges:
        print(f"  Average charge (pH 7): {np.mean(charges):.1f} ± {np.std(charges):.1f}")
    
    # -------------------------------------------------------------------------
    # 9. Save model (optional)
    # -------------------------------------------------------------------------
    print("\n[9] Saving models...")
    
    # Uncomment to save
    # generator.save_weights('peptide_generator.h5')
    # discriminator.save_weights('peptide_discriminator.h5')
    print("  (Saving disabled - uncomment to enable)")
    
    # -------------------------------------------------------------------------
    # 10. Example: Generate with specific constraints
    # -------------------------------------------------------------------------
    print("\n[10] Custom Generation Examples...")
    print("-" * 70)
    
    # Generate short cationic peptides
    print("\nShort Cationic Antimicrobial Peptides (length ~10):")
    short_amps = sampler.sample_with_filtering(
        num_samples=5,
        target_properties=['antimicrobial'],
        target_length=10,
        min_length=8,
        max_length=12
    )
    
    for pep in short_amps:
        if pep['sequence']:
            print(f"  {pep['sequence']} (charge: {PeptideAnalyzer.compute_charge(pep['sequence']):.1f})")
    
    # Generate longer peptides
    print("\nLonger Peptides (length ~20):")
    long_peptides = sampler.sample_with_filtering(
        num_samples=5,
        target_properties=['antimicrobial', 'cell_penetrating'],
        target_length=20,
        min_length=18,
        max_length=25
    )
    
    for pep in long_peptides:
        if pep['sequence']:
            print(f"  {pep['sequence']}")
    
    print("\n" + "=" * 70)
    print("Training and generation complete!")
    print("=" * 70)


# ============================================================================
# ADDITIONAL UTILITY: PEPTIDE SCORING FUNCTIONS
# ============================================================================

class PeptideScorer:
    """
    Score peptides for various bioactivity properties.
    Refactored to include a generic scoring function.
    """

    @staticmethod
    def score_peptide(sequence: str, criteria: Dict[str, Tuple[float, float]]) -> float:
        """
        Generic scoring function for peptides.
        Args:
            sequence: Peptide sequence to score.
            criteria: Dictionary of property thresholds (min, max) for scoring.
        Returns:
            Total score based on criteria.
        """
        score = 0.0
        for prop, (min_val, max_val) in criteria.items():
            value = PeptideScorer.compute_property(sequence, prop)
            if min_val <= value <= max_val:
                score += 1.0
        return score

    @staticmethod
    def compute_property(sequence: str, property_name: str) -> float:
        """
        Compute a specific property for a peptide sequence.
        Placeholder for actual property computation logic.
        """
        # Example: Compute charge, hydrophobicity, etc.
        return 0.0

    @staticmethod
    def antimicrobial_score(sequence: str) -> float:
        """Compute antimicrobial score using generic scoring function."""
        criteria = {
            'charge': (1.0, 5.0),
            'hydrophobicity': (0.5, 2.0)
        }
        return PeptideScorer.score_peptide(sequence, criteria)

    @staticmethod
    def cell_penetrating_score(sequence: str) -> float:
        """Compute cell-penetrating score using generic scoring function."""
        criteria = {
            'charge': (2.0, 6.0),
            'hydrophobicity': (1.0, 3.0)
        }
        return PeptideScorer.score_peptide(sequence, criteria)
    

# ============================================================================
# SEQUENCE OPTIMIZATION (SIMPLE GENETIC ALGORITHM)
# ============================================================================

class PeptideOptimizer:
    """
    Optimize peptides using a simple genetic algorithm.
    """
    
    def __init__(
        self,
        scoring_function,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.5
    ):
        self.scoring_function = scoring_function
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
    
    def _mutate(self, sequence: str) -> str:
        """Randomly mutate residues in the sequence."""
        seq_list = list(sequence)
        for i in range(len(seq_list)):
            if np.random.random() < self.mutation_rate:
                seq_list[i] = np.random.choice(AMINO_ACIDS)
        return ''.join(seq_list)
    
    def _crossover(self, seq1: str, seq2: str) -> Tuple[str, str]:
        """Single-point crossover between two sequences."""
        if len(seq1) != len(seq2):
            # Pad shorter sequence
            max_len = max(len(seq1), len(seq2))
            seq1 = seq1.ljust(max_len, 'A')
            seq2 = seq2.ljust(max_len, 'A')
        
        point = np.random.randint(1, len(seq1))
        child1 = seq1[:point] + seq2[point:]
        child2 = seq2[:point] + seq1[point:]
        
        return child1, child2
    
    def optimize(
        self,
        initial_sequences: List[str],
        generations: int = 50,
        target_length: int = None
    ) -> List[Tuple[str, float]]:
        """
        Optimize peptide sequences.
        
        Returns:
            List of (sequence, score) tuples, sorted by score
        """
        # Initialize population
        population = initial_sequences[:self.population_size]
        
        # Fill remaining slots with mutations
        while len(population) < self.population_size:
            parent = np.random.choice(initial_sequences)
            population.append(self._mutate(parent))
        
        for gen in range(generations):
            # Score population
            scores = [self.scoring_function(seq) for seq in population]
            
            # Sort by score
            sorted_pop = sorted(zip(population, scores), key=lambda x: -x[1])
            population = [seq for seq, _ in sorted_pop]
            
            # Keep top half
            survivors = population[:self.population_size // 2]
            
            # Generate offspring
            offspring = []
            while len(offspring) < self.population_size // 2:
                if np.random.random() < self.crossover_rate:
                    p1, p2 = np.random.choice(survivors, 2, replace=False)
                    c1, c2 = self._crossover(p1, p2)
                    offspring.extend([self._mutate(c1), self._mutate(c2)])
                else:
                    parent = np.random.choice(survivors)
                    offspring.append(self._mutate(parent))
            
            population = survivors + offspring[:self.population_size // 2]
            
            # Optional: enforce target length
            if target_length:
                population = [
                    seq[:target_length] if len(seq) > target_length
                    else seq + ''.join(np.random.choice(AMINO_ACIDS, target_length - len(seq)))
                    for seq in population
                ]
        
        # Final scoring
        final_scores = [self.scoring_function(seq) for seq in population]
        results = sorted(zip(population, final_scores), key=lambda x: -x[1])
        
        return results


# Quick test of the optimizer
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Peptide Optimization Example")
    print("=" * 70)
    
    # Initial random sequences
    initial = [''.join(np.random.choice(list(AMINO_ACIDS), 15)) for _ in range(10)]
    
    # Optimize for antimicrobial activity
    optimizer = PeptideOptimizer(
        scoring_function=PeptideScorer.antimicrobial_score,
        population_size=30,
        mutation_rate=0.15,
        crossover_rate=0.5
    )
    
    results = optimizer.optimize(
        initial_sequences=initial,
        generations=20,
        target_length=15
    )
    
    print("\nTop 5 Optimized Antimicrobial Peptides:")
    for i, (seq, score) in enumerate(results[:5]):
        scores = PeptideScorer.score_peptide(seq)
        print(f"  {i+1}. {seq}")
        print(f"      AMP: {scores['antimicrobial']:.2f}, "
              f"CPP: {scores['cell_penetrating']:.2f}, "
              f"Hemolytic Risk: {scores['hemolytic_risk']:.2f}")