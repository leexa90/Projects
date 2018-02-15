import numpy as np
import pandas as pd
dictt_train = np.load('dictt_train.npy').item()
dictt_chal = np.load('dictt_chal.npy').item()
dictt_test = np.load('dictt_test.npy').item()
dictt = {}
print len(dictt_train)
print len(dictt_chal)
print len(dictt_test)
def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z
dictt = merge_two_dicts(merge_two_dicts(dictt_train,dictt_chal),dictt_test)
data = pd.DataFrame()
data['ID'] = dictt_train.keys() + dictt_chal.keys() + dictt_test.keys()
data['embedding'] = data['ID'].map(dictt) 
X = np.stack(data['embedding'].values)
from sklearn.decomposition import PCA

pca = PCA(n_components=32)

Y = pca.fit(X[:2525]).transform(X)
print np.sum(pca.explained_variance_ratio_)
import matplotlib.pyplot as plt
for i in range(32):
        j=i
    #for j in range(i+1,10):
        plt.hist([Y[2525:,i],Y[:2525,i]],bins=20,label=['Train','NPL'])
        plt.legend()
        #plt.plot(Y[:2525,i],Y[:2525,j],'ro')
        #plt.plot(Y[2525:,i],Y[2525:,j],'bo',alpha=0.7)
        plt.title('%s_%s'%(i,j))
        plt.savefig('%s_%s.png'%(i,j))
        plt.clf()
#        plt.show()
data['dist'] = np.sum((X-np.mean(X,0))**2,1)
values = data['dist'].values
