from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import pandas as pd
#[C@H]
dictt = {'AC':'CC(=O)',
         'A':'N[C@H](C)C(=O)',
         'C':'N[C@H](CS)C(=O)',
         'D':'N[C@H](CC(=O)O)C(=O)',
         'E':'N[C@H](CCC(=O)O)C(=O)',
         'F':'N[C@H](Cc1ccccc1)C(=O)',
         'G':'NCC(=O)',
         'H':'N[C@H](Cc1c[nH]cn1)C(=O)',
         'I':'N[C@H]([C@@H](C)CC)C(=O)',#NC([C@@H](CC)C)C(=O)
         'K':'N[C@H](CCCCN)C(=O)',
         'L':'N[C@H](CC(C)C)C(=O)',
         'M':'N[C@H](CCSC)C(=O)',
         'N':'N[C@H](CC(N)=O)C(=O)',
         'NL':'N[C@@H](CC(N)=O)C(=O)',
         'P':'N1CCCC1C(=O)',
         'Q':'N[C@H](CCC(N)=O)C(=O)',
         'R':'N[C@H](CCCNC(=N)N)C(=O)',
         'S':'N[C@H](CO)C(=O)',
         'T':'N[C@H]([C@H](O)C)C(=O)', #NC([C@H](O)C)C(=O)
         'V':'N[C@H](C(C))C(=O)',
         'W':'N[C@H](Cc1c[nH]c2ccccc12)C(=O)',
         'Y':'N[C@H](Cc1ccc(O)cc1)C(=O)',
         'S5':'N[C@](CCCC=2)(C)C(=O)',
         'S8':'N[C@](CCCCCCC=2)(C)C(=O)',
         'R5':'N[C@@](CCCC=2)(C)C(=O)',
         'R8':'N[C@@](CCCCCCC=2)(C)C(=O)',
         'S53':'N[C@](CCCC=3)(C)C(=O)',
         'S83':'N[C@](CCCCCCC=3)(C)C(=O)',
         'R53':'N[C@@](CCCC=3)(C)C(=O)',
         'R83':'N[C@@](CCCCCCC=3)(C)C(=O)',
         'B8':'NC(CCCC=2)(CCCC=3)C(=O)', #???
         'B5':'NC(CCCC=2)(CCCC=3)C(=O)',#'S8':'N[C@](CCCCCCC=3)(C)C(=O)' for S8
         'BAla':'NCCC(=O)',
         'PEG1':'NCCOCCOCC(=O)',
         'PEG2':'NCCOCOCCOCCC(=O)',
         'PEG5':'NCCOCCOCCOCCOCCOCCOCCC(=O)',
         'EEA':'NCCOCCOCCNC(=O)CCC(=O)',
         'pff':'N[C@@H](Cc1c(F)c(F)c(F)c(F)c1(F))C(=O)',
         'FITC':'C(NC1=CC=C2C(=C1)C4(OC2=O)C3=C(C=C(C=C3)O)OC5=C4C=CC(=C5)O)(=S)'}
         


         
data  = pd.read_csv('dataset_stapled.csv',header=None)
         
         
def process_string(str):
    result = []
    special_resi = False
    for i in str:
        if i =='(':
            return result
        elif special_resi and i !='-': #inside special residue
            temp += i
        elif i =='-':
            if special_resi : #closing of special residue
                result += [temp,]
                special_resi = False
                temp  = ''
            else: #starting of special residue
                special_resi = True 
                temp = ''
        else: #normal residue
            result += [i,]
    return result

def save_smile(i,counter ):
    smile = process_string(i)
    smile_str = ''
    second_ii = False
    for ii in smile[1:]:
        if second_ii==0.5 and ii in ['B5','B8']:
            second_ii = True
        elif second_ii is True and ii in ['S8','S5','R8','R5']:
            second_ii = False
            ii = ii +'3'
        smile_str += dictt[ii]
        if second_ii is  False and ii in ['S8','S5','R8','R5']:
            second_ii = 0.5
    smile_str += 'O'
    try:
        mol0 = Chem.MolFromSmiles(smile_str)
        mol = Chem.AddHs(mol0)
        AllChem.Compute2DCoords(mol)
        #AllChem.EmbedMolecule(mol,AllChem.ETKDG())
        open('temp_%s.sdf'%(counter),'w').write(Chem.MolToMolBlock(mol))
        return smile_str
    except :
        print (counter,smile_str)
        return 'failed' 


data['list'] = data[1].apply(process_string)
print (data[1].apply(process_string))
counter = 0
smile = []
for i in data[1]:
    smile += [save_smile(i,counter = counter),]
    counter += 1
data['SMILES'] = smile
data.to_csv('../stapled_nofitc_200.csv',index=0)
