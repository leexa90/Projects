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
import openbabel
import pybel
for mymol in pybel.readfile("sdf", "nr-ahr.sdf"):
    print mymol
    die
