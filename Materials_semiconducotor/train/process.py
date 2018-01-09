import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
# get individial atoms'
def L1_norm(vec):
    return np.sum(map(np.absolute,vec))
def get_dist(atom_type,temp):
    if True:
        cor1 = np.zeros((0,3))
        cor2 = np.zeros((0,3))
        for i in range(len(temp)):
            if temp[4].iloc[i] == atom_type:
                cor1 = np.append(cor1, np.stack([temp[[1,2,3]].values[i],]*len(temp)),0)
                cor2 = np.append(cor2,temp[[1,2,3]].values,0) 
        dist = np.sum((cor1 - cor2)**2,1)**.5
        dist = dist[dist>0.01]
        dist = dist[dist <= 7]
        return list(dist)
data = pd.DataFrame(None,columns=['atom','X','Y','Z','type'])
data = pd.DataFrame(None,columns=range(6))
data[5] = data[5].astype(np.int32)
files = [x[0] for x in  os.walk('.') if '/' in x[0] ]
dist = {'Al':[],'In':[],'Ga':[],'O':[]}
for i in sorted(files,key = lambda x : int(x[2:]))[:]:
    if i[-1] == '0':
        print i
    temp= pd.read_csv(i+'/geometry.xyz',skiprows=3,sep=' ',header=None)
    temp = temp.iloc[3:]
    #temp['L1'] = map(L1_norm,temp[[1,2,3]].values)
    #temp = temp.sort_values('L1').reset_index(drop=True)
    for atom_type in ['Al','In','Ga','O']:
        dist[atom_type] += get_dist(atom_type,temp)
    #temp[5] = int(i[2:])
    #data = data.append(temp)
#data.to_csv('train_cord.csv',index=0)
''' https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)
13 	Al 	aluminium 	125 	118 	184  	118 	111 	143
49 	In 	indium_ 	155 	156 	193 	144 	146 	167
31 	Ga 	gallium 	130 	136 	187 	126 	121 	135
8 	O 	oxygen_ 	60 	48 	152 	73 	53
'''
bond_data = [['Al',1.25,1.18,1.84,1.18,1.11,1.43],
            ['In',1.55,1.56,1.93,1.44,1.46,1.67],
            ['Ga',1.30,1.36,1.87,1.26,1.21,1.35],
            ['O',.60,.48,1.52,.73,.53,np.nan]]
bond_data = pd.DataFrame(bond_data, columns=['atom','empirical',
                                             'calcul','VDW','Cov1',
                                             'Cov3','metallic'])

def L1_norm(vec):
    return np.sum(map(np.absolute,vec))
temp = temp.iloc[3:]
temp['L1'] = map(L1_norm,temp[[1,2,3]].values)
temp = temp.sort_values('L1').reset_index(drop=True)
cor1 = np.zeros((0,3))
cor2 = np.zeros((0,3))
for i in range(len(temp)):
    cor1 = np.append(cor1, np.stack([temp[[1,2,3]].values[i],]*len(temp)),0)
    cor2 = np.append(cor2,temp[[1,2,3]].values,0) 
dist = np.sum((cor1 - cor2)**2,1)**.5
dist_mat = np.reshape(dist,(len(temp),len(temp)))

# get individial atoms'
def L1_norm(vec):
    return np.sum(map(np.absolute,vec))
def get_dist(atom_type,temp=temp):
    if True:
        cor1 = np.zeros((0,3))
        cor2 = np.zeros((0,3))
        for i in range(len(temp)):
            if temp[4].iloc[i] == atom_type:
                cor1 = np.append(cor1, np.stack([temp[[1,2,3]].values[i],]*len(temp)),0)
                cor2 = np.append(cor2,temp[[1,2,3]].values,0) 
        dist = np.sum((cor1 - cor2)**2,1)**.5
        dist = dist[dist>0.01]
        dist = dist[dist <= 10]
        return dist

    
