import numpy as np
import matplotlib.pyplot
import pandas as pd
import matplotlib.pyplot as plt
data = pd.DataFrame()
atom2Id = np.load('atom2Id.npy').item()
Id2atom = {atom2Id[x]: x for x in atom2Id.keys()}
bonds2 = np.load('bonds2.npy.zip')['bonds2.npy']
embed = np.load('final_embeddings_28.npy')
dictt = {}
for i,j in bonds2:
    try:
        dictt[i] += 1
    except :
        dictt[i] = 1
    try:
        dictt[j] += 1
    except :
        dictt[j] = 1            
            
commonwords=sorted(dictt.keys(), key = lambda x : dictt[x],reverse=True)
data = pd.DataFrame([[atom2Id[x], x] for x in atom2Id.keys()])
data[3] = data[0].map(dictt)
data = data.sort_values(3).reset_index(drop=True)

def main(num):
    data_common = data[data[3] > num].reset_index(drop=True)

    impt = []
    for i in range(8):
        data_common['pc_'+str(i)]= 0
        impt += ['pc_'+str(i)]

    from sklearn.decomposition import PCA
    pca = PCA(n_components=8)
    z = pca.fit_transform(embed[data_common[0].values,:])

    # see of atoms make sense
    data_common['atom'] = data_common[1].apply(lambda x : x[0][0])
    data_common = data_common.set_value(data_common.index,
                                        impt,
                                        z)

    if True:
        fig, ax = plt.subplots()
        for i in sorted(pd.unique(data_common['atom'])):
            temp = data_common[data_common['atom']==i]
            for j, txt in enumerate(temp[1]):
                ax.annotate(txt[0], (temp['pc_1'].iloc[j],temp['pc_2'].iloc[j]),
                            size=5)
            ax.plot(temp['pc_1'],temp['pc_2'],'o',label=i)
        plt.legend()
        plt.show()
    if True:
        fig, ax = plt.subplots()
        for i in sorted(pd.unique(data_common['atom'])):
            temp = data_common[data_common['atom']==i]
            for j, txt in enumerate(temp[1]):
                ax.annotate(txt[0], (temp['pc_1'].iloc[j],temp['pc_3'].iloc[j]),
                            size=5)
            ax.plot(temp['pc_1'],temp['pc_3'],'o',label=i)
        plt.legend()
        plt.show()
