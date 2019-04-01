'''
coding gradient boosting algorithm, implementation for undnerstanding.
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
                                      max_depth=1).fit(X,y)
    #print B
    ym = regressor.predict(X)
    y =y-0.01*ym
    ensemble += ym
    if i%1 ==0:
        plt.plot(X[:,0],y_ori,'bo',label='data')
        plt.plot(X[:,0],ensemble,'r',label='boosted predictions')
        plt.legend(loc='upper right')
        num = '{0:03d}'.format(i)
        plt.title('Boosted  descision trees num %s'%num)
        plt.xlabel('X var')
        plt.ylabel('prediction')
        plt.savefig('{0:03d}'.format(i)+'png')
        plt.close()
    y = y_ori - ensemble
    print np.mean((y_ori-ensemble)**2)
#ffmpeg -framerate 1  -pattern_type glob -i '*.png' -filter:v tblend video.mp4
#convert -loop 0 *png out.gif
