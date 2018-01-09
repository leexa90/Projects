import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
# get individial atoms'
def L1_norm(vec):
    return np.sum(map(np.absolute,vec))
def get_dist(atom_typeA,atom_typeB,temp):
    if True:
        temp1 = temp[temp[4]==atom_typeA]
        temp2 = temp[temp[4]==atom_typeB]
        if len(temp1) ==0 or  len(temp2) ==0:
            return []
        cor1 = np.zeros((0,3))
        cor2 = np.zeros((0,3))
        for i in range(len(temp1)):
            cor1 = np.append(cor1, np.stack([temp1[[1,2,3]].values[i],]*len(temp2)),0)
            cor2 = np.append(cor2,temp2[[1,2,3]].values,0) 
        dist = np.sum((cor1 - cor2)**2,1)**.5
        dist = dist[dist>0.01]
        dist = dist[dist <= 6.5]
        return list(dist)
train =pd.read_csv('../train.csv')
for target in ['formation_energy_ev_natom','bandgap_energy_ev']:
    train[target] = (train[target] - np.mean(train[target]))/np.std(train[target])

data = pd.DataFrame(None,columns=['atom','X','Y','Z','type'])
data = pd.DataFrame(None,columns=range(6))
data[5] = data[5].astype(np.int32)
files = [x[0] for x in  os.walk('.') if '/' in x[0] ]
dist = {('Al', 'Ga'): [], ('Al', 'In'): [], ('Al', 'Al'): [],
        ('Ga', 'Ga'): [], ('Al', 'O'): [], ('In', 'Ga'): [],
        ('In', 'In'): [], ('Ga', 'O'): [], ('In', 'O'): [],
        ('O', 'O'): []}
'''
distance view and chosen by guesstimation. GMM model could not handle it 
'''
dist_values = {('Al', 'Ga'): [(2.6,4),(4,5.1),(5.1,6.0),(6.0,6.4)],
        ('Al', 'In'): [(2.6,4),(4,4.4),(4.4,5.3),(5.3,6.5)],
        ('Al', 'Al'): [(2.6,4),(4,4.4),(4.4,5.3),(5.3,6)],
        ('Ga', 'Ga'): [(2.6,4),(4,4.4),(4.4,5.2),(5.2,6)],
        ('Al', 'O') : [(1.5,2.5),(2.9,3.2),(3.2,4.1),(4.1,5),(5,6)],
        ('In', 'Ga'): [(2.6,4),(4,4.4),(4.4,5.2),(5.2,6)],
        ('In', 'In'): [(2.6,3.5),(3.4,5.2),(4.2,4.8),(4.8,5.4),(5.4,6.5)],
        ('Ga', 'O'):  [(1.5,2.6),(2.8,3.1),(3.1,4.2),(4.2,5.1),(5.1,6)],
        ('In', 'O'):  [(1.5,2.6),(2.9,3.2),(3.2,4.3),(4.3,5.5)],
        ('O', 'O'):   [(2.4,3.6),(3.6,4.7),(4.7,6.2)]}
features = [('Al', 'Ga', 0), ('Al', 'Ga', 1), ('Al', 'Ga', 2), ('Al', 'Ga', 3),
            ('In', 'O', 0), ('In', 'O', 1), ('In', 'O', 2), ('In', 'O', 3),
            ('Al', 'In', 0), ('Al', 'In', 1), ('Al', 'In', 2), ('Al', 'In', 3),
            ('Al', 'Al', 0), ('Al', 'Al', 1), ('Al', 'Al', 2), ('Al', 'Al', 3),
            ('Ga', 'Ga', 0), ('Ga', 'Ga', 1), ('Ga', 'Ga', 2), ('Ga', 'Ga', 3),
            ('Al', 'O', 0), ('Al', 'O', 1), ('Al', 'O', 2), ('Al', 'O', 3),('Al', 'O', 4),
            ('In', 'Ga', 0), ('In', 'Ga', 1), ('In', 'Ga', 2),('In', 'Ga', 3),
            ('O', 'O', 0), ('O', 'O', 1), ('O', 'O', 2),
            ('In', 'In', 0), ('In', 'In', 1), ('In', 'In', 2), ('In', 'In', 3),('In', 'In', 4),
            ('Ga', 'O', 0), ('Ga', 'O', 1), ('Ga', 'O', 2),('Ga', 'O', 3), ('Ga', 'O', 4)]

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn import datasets
from fastcluster import linkage
def seriation(Z,N,cur_index):
    '''
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z
            
        seriation computes the order implied by a hierarchical tree (dendrogram)
    '''
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index-N,0])
        right = int(Z[cur_index-N,1])
        return (seriation(Z,N,left) + seriation(Z,N,right))
    
def compute_serial_matrix(dist_mat,method="ward"):
    '''
        input:
            - dist_mat is a distance matrix
            - method = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)
        
        compute_serial_matrix transforms a distance matrix into 
        a sorted distance matrix according to the order implied 
        by the hierarchical tree (dendrogram)
    '''
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method,preserve_input=True)
    res_order = seriation(res_linkage, N, N + N-2)
    seriated_dist = np.zeros((N,N))
    a,b = np.triu_indices(N,k=1)
    seriated_dist[a,b] = dist_mat[ [res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b,a] = seriated_dist[a,b]
    
    return seriated_dist, res_order, res_linkage


data_features = pd.DataFrame(None,columns=[5,]+features)
data_features[5] = data_features[5].astype(np.int32)
pictures = {}
for i in sorted(files,key = lambda x : int(x[2:]))[::1]:
    if i[-1] == '0':
        print i
    temp= pd.read_csv(i+'/geometry.xyz',skiprows=3,sep=' ',header=None)
    temp = temp.iloc[3:]
    temp_features = []
    for j in dist:
        None#dist[j] += get_dist(j[0],j[1],temp)
    for k in features:
        temp_dist = np.array(get_dist(k[0],k[1],temp))
        thres = dist_values[k[:2]][k[-1]]
        temp_features += [len(temp_dist[(temp_dist >= thres[0]) & (temp_dist < thres[1]) ]),]
    temp_features = pd.DataFrame([temp_features],columns=features)
    temp_features[5] = int(i[2:])
    data_features=data_features.append(temp_features)
    temp[5] = int(i[2:])
    data = data.append(temp)
    # distance matrix
    if True:
        #make all distance start from 0
        cor1 = np.zeros((0,3))
        cor2 = np.zeros((0,3))
        for ii in range(len(temp)):
            cor1 = np.append(cor1, np.stack([temp[[1,2,3]].values[ii],]*len(temp)),0)
            cor2 = np.append(cor2,temp[[1,2,3]].values,0) 
        dist = np.sum((cor1 - cor2)**2,1)**.5
        dist_mat = np.reshape(dist ,(len(temp),len(temp)))
        methods = ["ward","single","average","complete"]
        f, ax = plt.subplots(1,4,figsize=(20,5))
        counter = 0
        ax[counter].set_title(train[['formation_energy_ev_natom' , 'bandgap_energy_ev']].iloc[int(i[2:])-1])
        pictures[int(i[2:])] = []
        for method in methods:
            ordered_dist_mat, res_order, res_linkage = compute_serial_matrix(dist_mat,method);
            ax[counter].imshow(ordered_dist_mat)
            counter += 1
            pictures[int(i[2:])] += [ordered_dist_mat]
        plt.savefig('../picture/%s.png'%(int(i[2:])),dpi=200) 
        plt.close()
data.to_csv('train_cord.csv',index=0)
data_features.to_csv('train_data_features.csv',index=0)
np.save('../pictures.npy',pictures)
counter = 0
if True:
    f, ax = plt.subplots(1,10,figsize=(50,5));k=0
    from sklearn.mixture import GaussianMixture
    def gauss(mean,var,x):
            return np.exp(-(mean-x)**2/(2*var**2))/(22.507*var)
    for i in range(len(dist)):
        ax[i].hist(dist[dist.keys()[i]],bins=np.linspace(1,7,121),normed=True)
        ax[i].set_title(dist.keys()[i])
        z= GaussianMixture(5,means_init =[[2], [3], [3.5], [4.5], [5]]).fit([[x] for x in dist[dist.keys()[i]] if (x >= 0.1 and x <6)])
        for ii in range(0,5):
            ax[i].plot(np.linspace(1,6,120),
                      gauss(z.means_[ii,0],z.covariances_[ii,0,0],np.linspace(1,7,120)))
            
    plt.savefig('../dist_train.png',dpi=300)
die


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

    
