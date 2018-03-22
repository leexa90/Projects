import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import numpy as np

train = pd.read_csv('../bioactive_PDB_cpp.csv').iloc[::]

smiles = []
for i in range(len(train)):
    try:
        mol = Chem.MolFromFASTA(train.iloc[i]['seq'].lower())
        smiles += [Chem.MolToSmiles(mol),]
    except :
        smiles += [np.nan,]
        print (train.iloc[i])
train['SMILES'] = smiles
train = train[train['size'] <=20]

train = train[~train['SMILES'].isnull()].reset_index(drop=True)
train['ID'] = train['seq']
del train['seq']

train.to_csv('peptide_permability.csv',index=0)
result = {}
counter = 0
dictt = { 1.0 : 0 , 1.5 : 1 , 2.0:2}
all = []
all2 = []
for iii in range(len(train)):
        if iii%100==0:
            print (iii)
        counter += 1
        smile = train.iloc[iii]['SMILES']
        try:
            #print (counter)
            mol = Chem.MolFromSmiles(smile)
            mol = Chem.AddHs(mol)
            #AllChem.Compute2DCoords(mol)
            #AllChem.EmbedMolecule(mol,AllChem.ETKDG())
            #open('temp_%s.sdf'%(counter-1),'w').write(Chem.MolToMolBlock(mol));plt.show()
            #Chem.MolFromMolBlock(mol)
            adj = Chem.GetAdjacencyMatrix(mol)*1.0
            dist =  Chem.GetDistanceMatrix(mol)* (Chem.GetDistanceMatrix(mol)<255)
            bond_order = np.zeros(shape = adj.shape + (3,))
            atoms = list(map(lambda x:x.GetAtomicNum(),list(mol.GetAtoms())))
            all += [len(atoms),]
            all2 += atoms
            for i in range(len(adj)):
                for j in range(len(adj)):
                    if adj[i,j] == 1:
                        bond_type = mol.GetBondBetweenAtoms(i,j).GetBondTypeAsDouble()
                        if bond_type in dictt:
                            index = dictt[bond_type]
                            bond_order[i,j,index] = 1
                            bond_order[j,i,index] =  bond_order[i,j,index]
            temp = np.concatenate([np.stack([adj,dist],-1),bond_order],-1)
            result[train.iloc[iii]['ID']] = [temp.astype(np.uint8),atoms]
        except :
            print ('errror',counter,smile);
    
##    [ i.GetBondTypeAsDouble() for i  in mol.GetBonds()]
##    Chem.MolToSmiles(mol)
##    Descriptors.TPSA(mol)
##    Descriptors.MolLogP(mol)
##    AllChem.CalcNumLipinskiHBA(mol)
##    AllChem.CalcNumLipinskiHBD(mol)
##    i.IsInRingSize(2)

np.save('peptide_permability.npy',result)
