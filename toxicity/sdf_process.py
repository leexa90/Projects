import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import AllChem


my_sdf_file = 'nr-ahr.sdf'
suppl = Chem.SDMolSupplier(my_sdf_file)

for mol in suppl:
    print(mol.GetNumAtoms())
    Chem.GetAdjacencyMatrix(mol)
    Chem.MolFromMolBlock(mol)
    Chem.MolToSmiles(mol)
    die

die
frame = PandasTools.LoadSDF(my_sdf_file,
                            smilesName='SMILES',
                            molColName='Molecule',
                            includeFingerprints=True)

frame.to_csv('test.csv')
