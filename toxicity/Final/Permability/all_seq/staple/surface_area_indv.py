import pandas as pd
pd.options.display.max_rows = 20
import numpy as np
import sklearn
import xgboost
import scipy
from scipy import histogram, digitize, stats, mean, std
all  = []
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt
for line in open('../bond_lengths.csv','r'):
    all += [line.split(),]
bond_L= pd.DataFrame(all[1:],columns=all[0])
all = []
for line in open('../VDW.csv','r'):
    all += [line.split(),]
VDW= pd.DataFrame(all[1:],columns=all[0])

test = pd.read_csv('../stapled_peptide_permability_features.csv')
for feats in ['SMR_VSA','PEOE_VSA','SlogP_VSA']:
    feat_all = [ x for x in test.keys() if feats in x]
    for feat in feat_all:
        test[feat+'_norm']=test[feat]/np.sum(test[feat_all],1)
def get_surface_area(smile):
    print (smile[-25:])
    mol0 = Chem.MolFromSmiles(smile)
    mol = Chem.AddHs(mol0)
    AllChem.Compute2DCoords(mol)
    adj = (Chem.GetDistanceMatrix(mol)==1)*1
    adj2 = (Chem.GetDistanceMatrix(mol)==2)*1
    molMMFF = AllChem.MMFFGetMoleculeProperties(mol)
    # Chem.MolSurf._LabuteHelper(mol) indiv contribution of surface area
    atoms = list(
                            map(lambda x: molMMFF.GetMMFFAtomType(x),
                                range(len(mol.GetAtoms()))
                                 )
                            )
    AllChem.ComputeGasteigerCharges(mol)
    charges = np.array([float(mol.GetAtomWithIdx(x).GetProp('_GasteigerCharge')) for x in range(len(atoms))])
    surf= np.array(Chem.MolSurf._LabuteHelper(mol))
    return (charges,surf[1:],atoms)
test['charge_surf'] = test['SMILES'].apply( get_surface_area)
bins = [-999,-0.3 , -0.25, -0.2 , -0.15, -0.1 , -0.05,  0.  ,  0.05,  0.1 ,
        0.15,  0.2 ,  0.25,  0.3 ,999]
if True:
    charge_csv =pd.read_csv('../stapled_peptide_geisteiger_charge.csv')
    for atom in [35, 71, 15, 11, 66, 39, 23, 34, 64, 16, 21, 63,  2, 29, 41, 57, 32,
            6, 56, 36,  7, 10,  3, 28, 37,  1,  5]:
        bins = histogram(charge_csv[charge_csv['atoms']==atom]['charge'].values,20)[1]
        for i in range(len(bins)-1):
            def func(x,atom=atom):
                sum_ =  np.sum(x[1][(x[0]  >= bins[i] )&(x[0]  <= bins[i+1] ) &(np.array(x[2])==atom)])
                if sum_ >= 0:
                    return sum_
                else :
                    return 0.0
            def func_norm(x,atom=atom):
                sum_ =  np.sum(x[1][(x[0]  >= bins[i] )&(x[0]  <= bins[i+1] ) &(np.array(x[2])==atom)])
                if sum_ >= 0:
                    return sum_/np.sum(x[1])
                else :
                    return 0.0
            test['charge_%s_%s_%s'%(atom,bins[i],bins[i+1])] = test['charge_surf'].apply(func)
            test['charge_norm_%s_%s_%s'%(atom,bins[i],bins[i+1])] = test['charge_surf'].apply(func_norm)
            if np.mean(test['charge_%s_%s_%s'%(atom,bins[i],bins[i+1])])==0:
                del test['charge_%s_%s_%s'%(atom,bins[i],bins[i+1])]
                print ('charge_%s_%s_%s'%(atom,bins[i],bins[i+1]))

if True:
    results = []
    for i in test.keys()[10:]:#(0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3):
        try:
            X,Y1,Y2 = [],[],[]
            from sklearn import linear_model,preprocessing ,decomposition
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import Lasso
            alpha =0.00#alpha#10**alpha
            clf = linear_model.Lasso(alpha=alpha)
            temptrain  = StandardScaler().fit(test[[i,]]).transform(test[[i,]])
            preds = clf.fit(temptrain,np.log10(test['10'])).predict(temptrain)
            se = np.sum((preds-np.log10(test['10']))**2)/np.sum((temptrain-np.mean(temptrain))**2)
            se = 2*(se/215)**.5
            if  clf.coef_[0]-se >0 or clf.coef_[0]+se <0 :
                results += [(i,np.corrcoef(preds,np.log10(test['10']))[0,1],
                             clf.coef_[0],se,[clf.coef_[0]-se,clf.coef_[0]+se]),]
        except : print (i)
    print (sorted(results,key =  lambda x : x[1]))
#test[''] = 
plt.plot(test['charge_norm_7_-0.27398123963107096_-0.27251541490535375'],np.log10(test['permability']),'ro');plt.show()
