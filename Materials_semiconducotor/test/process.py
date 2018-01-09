import numpy as np
import pandas as pd
import os
data = pd.DataFrame(None,columns=['atom','X','Y','Z','type'])
data = pd.DataFrame(None,columns=range(6))
data[5] = data[5].astype(np.int32)
files = [x[0] for x in  os.walk('.') if '/' in x[0] ]
for i in sorted(files,key = lambda x : int(x[2:])):
    if i[-1] == '0':
        print i
    temp= pd.read_csv(i+'/geometry.xyz',skiprows=3,sep=' ',header=None)
    temp[5] = int(i[2:])
    data = data.append(temp)
data.to_csv('train_cord.csv',index=0)
