import sys
import pandas as pd
import os
import numpy as np
np.random.seed(0)
data = [x for x in os.listdir('.') if 'model_train' in x][:]
train = pd.read_csv(data[0])
test = pd.read_csv('model_test'+data[0][11:])
for i in range(1,len(data)):
    temptrain = pd.read_csv(data[i])
    temptest = pd.read_csv('model_test'+data[i][11:])
    train = pd.merge(train,temptrain,on='id')
    test = pd.merge(test,temptest,on='id')

target1 = 'formation_energy_ev_natom_x'
target2 = 'bandgap_energy_ev_x'
train['bias'] = 1
old = np.array([0,1.0,1.0,1.0])/3
old_score = 0.05
counter =0.0
for i in range(0,100):
    new = old + np.random.normal(0,0.1,(4))
    train['ensemble1'] = np.matmul(train[['bias',]+[x for x in train.keys() if 'predict1' in x]].values,new)
    train['ensemble2'] = np.matmul(train[['bias',]+[x for x in train.keys() if 'predict2' in x]].values,new)
    a = np.mean((train['ensemble1'] - np.log1p(train[target1]))**2)**.5
    b = np.mean((train['ensemble2'] - np.log1p(train[target2]))**2)**.5
    #print a,b,(a+b)/2
    new_score = (a+b)/2
    if np.exp(-(old_score-new_score)/0.03) < np.random.uniform(0,1):
        old_score = new_score
        old = new
        counter += 1
        print old_score, counter/(i+1),old
train['ensemble1'] = np.mean(train[[x for x in train.keys() if 'predict1' in x]].values,1)
train['ensemble2'] = np.mean(train[[x for x in train.keys() if 'predict2' in x]].values,1)
a = np.mean((train['ensemble1'] - np.log1p(train[target1]))**2)**.5
b = np.mean((train['ensemble2'] - np.log1p(train[target2]))**2)**.5
print a,b,(a+b)/2
new_score = (a+b)/2        
train['bias'] = 1
A = train[['bias',]+[x for x in train.keys() if 'predict1' in x]].values
A_At_inv = np.linalg.inv(np.matmul(A.T,A))
Ab = np.matmul(A.T,train[target1].values)
print np.matmul(A_At_inv,Ab)

train['ensemble1b'] = np.matmul(A,np.matmul(A_At_inv,Ab))

A = train[['bias',]+[x for x in train.keys() if 'predict2' in x]].values
A_At_inv = np.linalg.inv(np.matmul(A.T,A))
Ab = np.matmul(A.T,train[target2].values)
print np.matmul(A_At_inv,Ab)
train['ensemble2b'] = np.matmul(A,np.matmul(A_At_inv,Ab))
a = np.mean((train['ensemble1b'] - np.log1p(train[target1]))**2)**.5
b = np.mean((train['ensemble2b'] - np.log1p(train[target2]))**2)**.5
print a,b,(a+b)/2
