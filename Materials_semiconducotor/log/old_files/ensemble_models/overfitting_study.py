import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
train  = pd.read_csv('train.csv')

test = pd.read_csv('test.csv')
target1 = 'formation_energy_ev_natom'
target2 = 'bandgap_energy_ev'

train[target1] = np.log1p(train[target1])
train[target2] = np.log1p(train[target2])

import os
for i in [x for x in os.listdir('.') if 'model_train' in x and np.float(x.split('_')[3][:-4]) < 0.0514]:
    name = i[11:]
    temp = pd.read_csv(i).sort_values('id').reset_index(drop=True)
    train['predict1'+name] = temp[temp.keys()[1]]
    train['predict2'+name] = temp[temp.keys()[2]]
scores0 = []
scores1 = []
for i in range(0,1000):
    temp = train.sample(600,random_state=i,replace=True)
    temp1 = train.sample(600,random_state=i,replace=False)
    temp2 = train.sample(100,random_state=i,replace=False)
    col1 = [x for x in  temp.keys() if 'predict1' in x]
    col2 = [x for x in  temp.keys() if 'predict2' in x]
    for i in np.random.choice(range(len(col1)),5):
        predict1 = temp1[col1[i]]
        predict2 = temp1[col2[i]]
        a,b = np.mean((predict1-temp1[target1])**2)**.5, np.mean((predict2-temp1[target2])**2)**.5 #private lb
        predict1 = temp2[col1[i]]
        predict2 = temp2[col2[i]]
        c,d = np.mean((predict1-temp2[target1])**2)**.5, np.mean((predict2-temp2[target2])**2)**.5 #public lb
        scores0 += [(a+b)/2,]
        scores1 += [(c+d)/2,]
plt.plot(scores0,scores1,'ro',markersize=0.4)
regr = linear_model.LinearRegression()
regr =regr.fit(np.array([[x] for x in scores0]),np.array(scores1))
plt.xlabel('Private Leader Board score (500samples)')
plt.ylabel('Public Leader Board score (100samples)')

plt.title('Regression line y= %sX + %s'%(regr.coef_[0],regr.intercept_))
regr = regr.predict(np.array([[x] for x in scores0]))
plt.plot(scores0,regr,'b');plt.savefig('Private_Vs_public.png',dpi=300)
plt.show()

scores0 = []
scores1 = []
for i in range(0,1000):
    temp = train.sample(2900,random_state=i,replace=True)
    temp1 = train.sample(2400,random_state=i,replace=False)
    temp2 = train.sample(500,random_state=i,replace=False)
    col1 = [x for x in  temp.keys() if 'predict1' in x]
    col2 = [x for x in  temp.keys() if 'predict2' in x]
    for i in np.random.choice(range(len(col1)),5):
        predict1 = temp1[col1[i]]
        predict2 = temp1[col2[i]]
        a,b = np.mean((predict1-temp1[target1])**2)**.5, np.mean((predict2-temp1[target2])**2)**.5 #private lb
        predict1 = temp2[col1[i]]
        predict2 = temp2[col2[i]]
        c,d = np.mean((predict1-temp2[target1])**2)**.5, np.mean((predict2-temp2[target2])**2)**.5 #public lb
        scores0 += [(a+b)/2,]
        scores1 += [(c+d)/2,]
plt.plot(np.random.normal(0,0.00003,5000)+np.array(scores0),scores1,'go',markersize=0.8)
regr = linear_model.LinearRegression()
regr =regr.fit(np.array([[x] for x in scores0]),np.array(scores1))
plt.xlabel('Cross validation score (1900samples)')
plt.ylabel('Private Leader Board score (500samples)')
plt.title('Regression line y= %sX + %s'%(regr.coef_[0],regr.intercept_))
regr = regr.predict(np.array([[x] for x in scores0]))
plt.plot(scores0,regr,'b');plt.savefig('Private_Vs_CV.png',dpi=300)
plt.show()
