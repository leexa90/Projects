import pandas as pd
import numpy as np
data1 = pd.read_csv('nr-ar.smiles',sep='\t',engine='python',header=None)
data2 = pd.read_csv('nr-ahr.smiles',sep='\t',engine='python',header=None)
data3 = pd.read_csv('nr-ar-lbd.smiles',sep='\t',engine='python',header=None)
data4 = pd.read_csv('nr-er.smiles',sep='\t',engine='python',header=None)
data5 = pd.read_csv('nr-er-lbd.smiles',sep='\t',engine='python',header=None)
data6 = pd.read_csv('nr-aromatase.smiles',sep='\t',engine='python',header=None)
data7 = pd.read_csv('nr-ppar-gamma.smiles',sep='\t',engine='python',header=None)

data = data1
counter = 1
for i in [data2,data3,data4,data5,data6,data7]:
    i['2_'+str(counter)] = i[2];counter += 1
    print i.groupby(2)[2].apply(len)
    del i[2]
    data = pd.merge(data,i,how='outer',on=[1,0])


def got_null(x):
    result = 0
    for i in x:
        if i >= 0:
            result +=1
    return result

data['got'] = map(got_null, data[[2, '2_1', '2_2', '2_3', '2_4', '2_5', '2_6']].values)
import matplotlib.pyplot as plt
x=plt.hist(data['got'],6)
data = data.fillna(-999)
from rdkit.Chem import AllChem
from rdkit import Chem
for i in range(len(data2)):
    if i%100 == 0:
        print i
    m1 = Chem.MolToSmiles(Chem.MolFromSmiles(data2.iloc[i][0]))
    m2 = Chem.MolFromSmiles(m1)
    AllChem.Compute2DCoords(m2)
    #open('./nr-ahr/data1_%s.sdf'%i,'w').write(Chem.MolToMolBlock(m2) )
    AllChem.EmbedMolecule(m2,AllChem.ETKDG()) #add 2D coordinates by embedding the molecule
    m3 = Chem.AddHs(m2)
    AllChem.EmbedMolecule(m3,AllChem.ETKDG()) #3d cord
    m3 = Chem.RemoveHs(m3) #remove H
    open('./nr-ahr/data2_%s.sdf'%i,'w').write(Chem.MolToMolBlock(m3) )
import openbabel



import pybel
for mymol in pybel.readfile("sdf", "nr-ahr.sdf"):
    print mymol
    die
