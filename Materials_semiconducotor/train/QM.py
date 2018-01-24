import pandas as pd
import numpy as np
data = pd.read_csv('./1/geometry2.xyz',sep = ' ',header=None,skiprows=2,)
X_min,X_max = np.min(data[1]),np.max(data[1])
Y_min,Y_max = np.min(data[2]),np.max(data[2])
Z_min,Z_max = np.min(data[3]),np.max(data[3])
data[1]= data[1] + (1.4-X_min)
data[2]= data[2] + (1.4-Y_min)
data[3]= data[3] + (1.4-Z_min)

X_min,X_max = np.min(data[1]),np.max(data[1])
Y_min,Y_max = np.min(data[2]),np.max(data[2])
Z_min,Z_max = np.min(data[3]),np.max(data[3])

a=-X_min+X_max+2.8
b=-Y_min+Y_max+2.8
c=-Z_min+Z_max+2.8
data[1]=np.round(data[1]*10,1)
data[2]=np.round(data[2]*10,1)
data[3]=np.round(data[3]*10,1)
data[1]=(data[1]).astype(np.int32)
data[2]=(data[2]).astype(np.int32)
data[3]=(data[3]).astype(np.int32)
a = int(a/0.1)
b = int(b/0.1)
c = int(c/0.1)
dictt = {'Al' : 0.5235,'Ga' : 0.9982,'In' :2.2258, 'O' : 11.4927 }
dictt2 = {'Al' : 25,'Ga' : 38.4,'In' :65.61, 'O' : 14**2 }
data['out'] = data[0].map(dictt)
grid = np.zeros((a,b,c))
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure()

    
