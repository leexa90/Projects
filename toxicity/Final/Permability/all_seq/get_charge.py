import pandas as pd
train = pd.read_csv('peptide_permability.csv')
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import  numpy as np
import scipy
from MI_code import mutual_information
for i in range(1,15):
    print ("train['PEOE_VSA%s'] = list(map( lambda smile : Descriptors.PEOE_VSA%s(smile),train['SMILES'].values))" %(i,i))
for i in range(1,11):
    print ("train['SMR_VSA%s'] = list(map( lambda smile : Descriptors.SMR_VSA%s(smile),train['SMILES'].values))" %(i,i))
for i in range(1,12):
    print ("train['SlogP_VSA%s'] = list(map( lambda smile : Descriptors.SlogP_VSA%s(smile),train['SMILES'].values))" %(i,i))
for i in range(1,11):
    print ("train['VSA_EState%s'] = list(map( lambda smile : Descriptors.VSA_EState%s(smile),train['SMILES'].values))" %(i,i))
	
def make_info(x,num=None):
    dictt = { 1.0 : 0 , 1.5 : 1 , 2.0:2}
    mol0 = Chem.MolFromSmiles(x)
    mol = Chem.AddHs(mol0)
    AllChem.Compute2DCoords(mol)
    molMMFF = AllChem.MMFFGetMoleculeProperties(mol)
    atoms = list(
                            map(lambda x: molMMFF.GetMMFFAtomType(x),
                                range(len(mol.GetAtoms()))
                                 )
                            )
    AllChem.ComputeGasteigerCharges(mol)
    charges = [float(mol.GetAtomWithIdx(x).GetProp('_GasteigerCharge')) for x in range(len(atoms))]
    return [atoms,charges]

train['all']= train['SMILES'].apply(lambda x : make_info(x))
die
train['atoms']= train['all'].apply(lambda x : x[2])
#train['bond']= train['all'].apply(lambda x : x[1][:,:,-3:])
if True:
    train['charges'] = train['all'].apply(lambda x : x[-1])
    train['charge_skew'] = train['charges'].apply(scipy.stats.skew)
    train['charge_kurtosis'] = train['charges'].apply(scipy.stats.kurtosis)
    train['charge_std']= train['charges'].apply(np.std)
    ##for i in range(0,3):
    ##    train['bond_%s'%i]= train['bond'].apply(lambda x : np.sum(x[:,:,i]))
    ##    train['bond_sum_%s'%i]= train['bond'].apply(lambda x : 1.0*np.sum(x[:,:,i])/np.sum(x))
    train['SMILES_temp'] = train['SMILES']
    train['SMILES'] = train['all'].apply(lambda x : x[0])
    for i in set(np.concatenate(train['atoms'],0)):
        train['atom_%s'%i] = train['atoms'].apply(lambda x : np.sum((np.array(x) ==i)))
        train['atom_sum_%s'%i] = train['atoms'].apply(lambda x : np.mean((np.array(x) ==i)))
       
    train['TPSA'] = list(map( lambda smile : Descriptors.TPSA(smile),
                              train['SMILES'].values))
    train['TPSA_norm'] = train['TPSA']/train['atoms'].apply(len) 
    train['LabuteASA'] = list(map( lambda smile : Descriptors.LabuteASA(smile),
                              train['SMILES'].values))
    train['LabuteASA_norm'] = train['LabuteASA']/train['atoms'].apply(len) 
    train['PEOE_VSA1'] = list(map( lambda smile : Descriptors.PEOE_VSA1(smile),train['SMILES'].values))
    train['PEOE_VSA2'] = list(map( lambda smile : Descriptors.PEOE_VSA2(smile),train['SMILES'].values))
    train['PEOE_VSA3'] = list(map( lambda smile : Descriptors.PEOE_VSA3(smile),train['SMILES'].values))
    train['PEOE_VSA4'] = list(map( lambda smile : Descriptors.PEOE_VSA4(smile),train['SMILES'].values))
    train['PEOE_VSA5'] = list(map( lambda smile : Descriptors.PEOE_VSA5(smile),train['SMILES'].values))
    train['PEOE_VSA6'] = list(map( lambda smile : Descriptors.PEOE_VSA6(smile),train['SMILES'].values))
    train['PEOE_VSA7'] = list(map( lambda smile : Descriptors.PEOE_VSA7(smile),train['SMILES'].values))
    train['PEOE_VSA8'] = list(map( lambda smile : Descriptors.PEOE_VSA8(smile),train['SMILES'].values))
    train['PEOE_VSA9'] = list(map( lambda smile : Descriptors.PEOE_VSA9(smile),train['SMILES'].values))
    train['PEOE_VSA10'] = list(map( lambda smile : Descriptors.PEOE_VSA10(smile),train['SMILES'].values))
    train['PEOE_VSA11'] = list(map( lambda smile : Descriptors.PEOE_VSA11(smile),train['SMILES'].values))
    train['PEOE_VSA12'] = list(map( lambda smile : Descriptors.PEOE_VSA12(smile),train['SMILES'].values))
    train['PEOE_VSA13'] = list(map( lambda smile : Descriptors.PEOE_VSA13(smile),train['SMILES'].values))
    train['PEOE_VSA14'] = list(map( lambda smile : Descriptors.PEOE_VSA14(smile),train['SMILES'].values))
    train['SMR_VSA1'] = list(map( lambda smile : Descriptors.SMR_VSA1(smile),train['SMILES'].values))
    train['SMR_VSA2'] = list(map( lambda smile : Descriptors.SMR_VSA2(smile),train['SMILES'].values))
    train['SMR_VSA3'] = list(map( lambda smile : Descriptors.SMR_VSA3(smile),train['SMILES'].values))
    train['SMR_VSA4'] = list(map( lambda smile : Descriptors.SMR_VSA4(smile),train['SMILES'].values))
    train['SMR_VSA5'] = list(map( lambda smile : Descriptors.SMR_VSA5(smile),train['SMILES'].values))
    train['SMR_VSA6'] = list(map( lambda smile : Descriptors.SMR_VSA6(smile),train['SMILES'].values))
    train['SMR_VSA7'] = list(map( lambda smile : Descriptors.SMR_VSA7(smile),train['SMILES'].values))
    train['SMR_VSA8'] = list(map( lambda smile : Descriptors.SMR_VSA8(smile),train['SMILES'].values))
    train['SMR_VSA9'] = list(map( lambda smile : Descriptors.SMR_VSA9(smile),train['SMILES'].values))
    train['SMR_VSA10'] = list(map( lambda smile : Descriptors.SMR_VSA10(smile),train['SMILES'].values))
    train['SlogP_VSA1'] = list(map( lambda smile : Descriptors.SlogP_VSA1(smile),train['SMILES'].values))
    train['SlogP_VSA2'] = list(map( lambda smile : Descriptors.SlogP_VSA2(smile),train['SMILES'].values))
    train['SlogP_VSA3'] = list(map( lambda smile : Descriptors.SlogP_VSA3(smile),train['SMILES'].values))
    train['SlogP_VSA4'] = list(map( lambda smile : Descriptors.SlogP_VSA4(smile),train['SMILES'].values))
    train['SlogP_VSA5'] = list(map( lambda smile : Descriptors.SlogP_VSA5(smile),train['SMILES'].values))
    train['SlogP_VSA6'] = list(map( lambda smile : Descriptors.SlogP_VSA6(smile),train['SMILES'].values))
    train['SlogP_VSA7'] = list(map( lambda smile : Descriptors.SlogP_VSA7(smile),train['SMILES'].values))
    train['SlogP_VSA8'] = list(map( lambda smile : Descriptors.SlogP_VSA8(smile),train['SMILES'].values))
    train['SlogP_VSA9'] = list(map( lambda smile : Descriptors.SlogP_VSA9(smile),train['SMILES'].values))
    train['SlogP_VSA10'] = list(map( lambda smile : Descriptors.SlogP_VSA10(smile),train['SMILES'].values))
    train['SlogP_VSA11'] = list(map( lambda smile : Descriptors.SlogP_VSA11(smile),train['SMILES'].values))
    train['VSA_EState1'] = list(map( lambda smile : Descriptors.VSA_EState1(smile),train['SMILES'].values))
    train['VSA_EState2'] = list(map( lambda smile : Descriptors.VSA_EState2(smile),train['SMILES'].values))
    train['VSA_EState3'] = list(map( lambda smile : Descriptors.VSA_EState3(smile),train['SMILES'].values))
    train['VSA_EState4'] = list(map( lambda smile : Descriptors.VSA_EState4(smile),train['SMILES'].values))
    train['VSA_EState5'] = list(map( lambda smile : Descriptors.VSA_EState5(smile),train['SMILES'].values))
    train['VSA_EState6'] = list(map( lambda smile : Descriptors.VSA_EState6(smile),train['SMILES'].values))
    train['VSA_EState7'] = list(map( lambda smile : Descriptors.VSA_EState7(smile),train['SMILES'].values))
    train['VSA_EState8'] = list(map( lambda smile : Descriptors.VSA_EState8(smile),train['SMILES'].values))
    train['VSA_EState9'] = list(map( lambda smile : Descriptors.VSA_EState9(smile),train['SMILES'].values))
    train['VSA_EState10'] = list(map( lambda smile : Descriptors.VSA_EState10(smile),train['SMILES'].values))
    train['HBA'] = list(map( lambda smile : Chem.rdMolDescriptors.CalcNumHBA(smile),train['SMILES'].values))
    train['HBD'] = list(map( lambda smile : Chem.rdMolDescriptors.CalcNumHBD(smile),train['SMILES'].values))
    train['Rotatable_num'] = list(map( lambda smile : Descriptors.NumRotatableBonds(smile),train['SMILES'].values))
    train['HBA_norm'] = train['HBA']/train['atoms'].apply(len) 
    train['HBD_norm'] = train['HBD']/train['atoms'].apply(len)
    train['SMILES'] = train['SMILES_temp']
    train['atom_len'] = train['atoms'].apply(len)
    train['atom_size'] = train['atoms'].apply(sum)
    del train['atoms']
    #del train['bond']
    del train['SMILES_temp']
    del train['charges']
    train['permability'] = (train['source'] == 2)*1
    train.to_csv('peptide_permability_features.csv',index=0)
if True:
    from scipy import histogram, digitize, stats, mean, std
    import matplotlib.pyplot as plt
    def mutual_information(x,y):
        #discritize to 100 bins
    #   binx = histogram(x,100,density=True)
    #   biny = histogram(y,100,density=True)
        temp = plt.hist2d(x,y,20)
        plt.close()
        prob,x_bin,y_bin = temp[0],temp[1],temp[2]
        prob = temp[0]/np.sum(temp[0])
        result = 0.0
        for i in range(100):
            sumI = np.sum(prob[i,:])
            if sumI != 0:
                for j in range(20):
                    sumJ = np.sum(prob[:,j])
                    if sumJ != 0 and prob[i][j] != 0:
                        result += prob[i][j]*np.log(prob[i][j]/(sumI*sumJ))
        return result

    corr = train.corr()['permability']
    import matplotlib.pyplot as plt
    import math
    from matplotlib.colors import LogNorm
    for i in train.keys():
        try:
            print (i,mutual_information(train['permability'],train[i]),corr[i])
            plt.title([i,mutual_information(train['permability'],train[i]),corr[i]])
            
            temp50,temp25,temp75,y =[],[],[],[]
            
            _,bins = histogram(train[i],20)
            bins = [-1000,] + list(bins) +[ 1000,]
            for j in range(1,22):
                 if math.isnan(np.median(train[(train[i] < bins[j+1])  & (train[i] >= bins[j])]['permability'])) is False:
                     try :
                         A = [np.median(train[(train[i] < bins[j+1])  & (train[i] >= bins[j])]['permability']),]
                         B = [A[0]-np.percentile(train[(train[i] < bins[j+1])  & (train[i] >= bins[j])]['permability'],25),]
                         C = [np.percentile(train[(train[i] < bins[j+1])  & (train[i] >= bins[j])]['permability'],75)-A[0],]
                         temp50 += A
                         temp25 += B
                         temp75 += C
                         y += [bins[j]+0.5*(bins[5]-bins[4]),]
                     except IndexError : None
        
            x=plt.hist2d(train[i],train['permability'],20,
                        cmap ='viridis',norm=LogNorm())
            plt.colorbar(x[-1])
            plt.errorbar(y,temp50,yerr=np.vstack([temp25,temp75]),
                         barsabove ='above',marker='o',color='orange',
                         capthick = 2,linewidth=2)
            plt.xlabel(i)
            plt.ylabel('log Permability, cm per sec')
            plt.savefig('%s.png'%i,dpi=300)
            plt.close()
        except :print(i,'failed')

def get_more_plot(x,y):
    None
