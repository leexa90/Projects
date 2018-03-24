from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
#[C@H]
dictt = {'A':'N[C@H](C)C(=O)',
         'C':'N[C@H](CS)C(=O)',
         'D':'N[C@H](CC(=O)O)C(=O)',
         'E':'N[C@H](CCC(=O)O)C(=O)',
         'F':'N[C@H](Cc1ccccc1)C(=O)',
         'G':'NCC(=O)',
         'H':'N[C@H](Cc1c[nH]cn1)C(=O)',
         'I':'N[C@H](C(C)CC)C(=O)',#NC([C@@H](CC)C)C(=O)
         'K':'N[C@H](CCCCN)C(=O)',
         'L':'N[C@H](CC(C)C)C(=O)',
         'M':'N[C@H](CCSC)C(=O)',
         'N':'N[C@H](CC(N)=O)C(=O)',
         'NL':'N[C@@H](CC(N)=O)C(=O)'
         'P':'N1CCCC1C(=O)',
         'Q':'N[C@H](CCC(N)=O)C(=O)',
         'R':'N[C@H](CCCNC(=N)N)C(=O)',
         'S':'N[C@H](CO)C(=O)',
         'T':'N[C@H](C(O)C)C(=O)', #NC([C@H](O)C)C(=O)
         'V':'N[C@H](C(C))C(=O)',
         'W':'N[C@H](Cc1c[nH]c2ccccc12)C(=O)',
         'Y':'N[C@H](Cc1ccc(O)cc1)C(=O)',
         'S5':'N[C@](CCCC=2)(C)C(=O)',
         'S8':'N[C@](CCCCCCC=2)(C)C(=O)',
         'R5':'N[C@@](CCCC=2)(C)C(=O)',
         'R8':'N[C@@](CCCCCCC=2)(C)C(=O)',
         'B5':'NC(CCCC=2)(CCCC=3)C(=O)',#'S8':'N[C@](CCCCCCC=3)(C)C(=O)' for S8
         'BetaA':'NCCC(=O)',
         'PEG1':'NCCOCCOCC(=O)',
         'PEG2':'NCCOCOCCOCCC(=O)',
         'PEG5':'NCCOCCOCCOCCOCCOCCOCCC(=O)',
         'EEA':'NCCOCCOCCNC(=O)CCC(=O)',
         'PPF':'N[C@@H](Cc1(F)c(F)c(F)c(F)c(F)c1(F))C(=O)',
         'FITC':'C(NC1=CC=C2C(=C1)C4(OC2=O)C3=C(C=C(C=C3)O)OC5=C4C=CC(=C5)O)(=S)'}
         
         
         
         
         
