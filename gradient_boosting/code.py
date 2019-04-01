'''
coding gradient boosting algorithm
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.tree import DecisionTreeRegressor
np.random.seed(1)
X_ori = np.expand_dims(np.linspace(-10,10,1001),-1)
y_ori = np.sin(X_ori[:,0])#+(X_ori[:,1]**2)**.5


fig = plt.figure()
ax = plt.axes(projection='3d')
plt.close()

#ax.scatter(X_ori[:,0],X_ori[:,1],y_ori)
#plt.show()



def OLS_solver(X,y):
    return np.linalg.solve(X,y)
y = np.zeros(len(y_ori))+np.mean(y_ori)
X = np.copy(X_ori)
ensemble = np.zeros(len(y_ori)) + y
y = y_ori - ensemble 

print np.mean((y_ori-ensemble)**2)
for i in range(100):
    regressor = DecisionTreeRegressor(random_state=0,
                                      max_depth=2).fit(X,y)
    #print B
    ym = regressor.predict(X)
    y =y-0.01*ym
    ensemble += ym
    if i%10 ==0:
        plt.plot(X[:,0],y_ori,'bo')
        plt.plot(X[:,0],ensemble,'r')
        plt.show()
    y = y_ori - ensemble
    print np.mean((y_ori-ensemble)**2)
    
