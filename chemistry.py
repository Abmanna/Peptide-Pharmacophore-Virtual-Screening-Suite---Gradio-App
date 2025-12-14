# chemistry.py
from rdkit import Chem
from rdkit.Chem import AllChem
import itertools

class InteractiveChemistryEngine:
    """Real-time chemistry engine with optimization capabilities"""

    def __init__(self):
        self.reactions = self._load_reaction_templates()
        self.fragments = self._load_fragment_library()
        self.current_library = []
        self.optimization_history = []

    def _load_reaction_templates(self):
        """Load optimized reaction templates"""
        return {
            'amide': AllChem.ReactionFromSmarts('[C:1](=[O:2])-[OH:3].[N:4]-[*:5]>>[C:1](=[O:2])-[N:4]-[*:5]'),
            'ester': AllChem.ReactionFromSmarts('[C:1](=[O:2])-[OH:3].[O:4]-[*:5]>>[C:1](=[O:2])-[O:4]-[*:5]'),
            'click': AllChem.ReactionFromSmarts('[N-:1]=[N+]=[N:2].[C:3]#[C:4]>>[N:1]1-N=N-[C:2]-[C:3]-[C:4]-1'),
            'glycosidic': AllChem.ReactionFromSmarts('[C:1]1-[OH:2]-[C:3].[OH:4]-[c:5]>>[C:1]1-[O:4]-[c:5]'),
        }

    def _load_fragment_library(self):
        """Load building block library"""
        return {
            'pharmacophores': [
                ('Vancomycin Core', 'c1c(cc(c(c1O)C2c3c(cc(c(c3)O)O)c4c(c2)c(c(c(c4O)O)NC(=O)C(N)Cc5ccc(O)cc5)C(=O)N)O'),
                ('Quinolone', 'O=C1c2ccccc2N(C)C(=O)N1'),
                ('Macrolide Core', 'CC1CC(=O)OC(C)C(O)C(C)C(=O)C(C)CC(O)C(C)CC(=O)O1'),
            ],
            'amino_acids': [
                ('Lysine', 'NCCCC[C@H](N)C(=O)O'),
                ('Arginine', 'NC(=N)NCCC[C@H](N)C(=O)O'),
                ('Aspartic Acid', 'OC(=O)C[C@H](N)C(=O)O'),
                ('Phenylalanine', 'N[C@@H](Cc1ccccc1)C(=O)O'),
                ('Proline', 'N1[C@@H](CCC1)C(=O)O'),
            ],
            'sugars': [
                ('Glucose', 'OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O'),
                ('Vancosamine', 'C[C@H]1O[C@@H](O)[C@H](N(C)C)[C@@H](O)[C@@H]1O'),
                ('Galactose', 'OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@H]1O'),
            ],
            'linkers': [
                ('PEG3', 'C(COCCOCCO)'),
                ('Alkyl C6', 'CCCCCC'),
                ('Aromatic', 'c1ccc(cc1)'),
                ('Triazole', 'c1cnn[nH]1'),
            ]
        }

    def generate_peptide_library(self, template_name, params):
        """Generate combinatorial peptide library using specified reaction template."""
        if template_name not in self.reactions:
            raise ValueError(f"Reaction template '{template_name}' not found.")

        reaction = self.reactions[template_name]
        fragments = self.fragments['amino_acids']
        library = []

        for frag1, frag2 in itertools.product(fragments, repeat=2):
            mol1 = Chem.MolFromSmiles(frag1[1])
            mol2 = Chem.MolFromSmiles(frag2[1])
            products = reaction.RunReactants((mol1, mol2))

            for product in products:
                try:
                    Chem.SanitizeMol(product[0])
                    library.append(Chem.MolToSmiles(product[0]))
                except Chem.rdchem.KekulizeException:
                    continue

        self.current_library = library
        return library