import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
train = np.load('train_pictures.npy').item()
test = np.load('test_pictures.npy').item()
dictt = {'Al' : 1, 'O' : 2,'Ga' :3, 'In' : 4}

dictt2 = {('Al', 'Al'): 0, ('Al', 'Ga'): 1, ('Al', 'In'): 2,  ('Al', 'O'): 3,
          ('Ga', 'Ga'): 4, ('Ga', 'In'): 5, ('Ga', 'O'): 6, 
          ('In', 'In'): 7, ('In', 'O'): 8,
          ('O', 'O'): 9}
train_result = {}
for i in train.keys():
    temp_temp_pic, temp_resi = [],[]
    for l in range(0,4):
        temp_pic = 1/(train[i][l][0]+0.001)*(train[i][l][0] != 0)
        temp_atoms = train[i][l][1][:,3]
        temp_atoms2 = map(lambda x : dictt[x], temp_atoms)
        resi = np.zeros((len(temp_atoms),len(temp_atoms),10))
        for j in range(len(temp_atoms2)):
            for k in range(j,len(temp_atoms2)):
                res_num = dictt2[tuple(sorted((temp_atoms[j],temp_atoms[k])))]
                resi[j,k,res_num] = 1
                resi[k,k,res_num] = 1
        temp_temp_pic +=  [temp_pic,]
        temp_resi += [resi,]
        train_result[i] = [np.stack(temp_temp_pic),\
                           np.stack(temp_resi)]
    del train[i] 

test_result = {}
for i in test.keys():
    temp_temp_pic, temp_resi = [],[]
    for l in range(0,4):
        temp_pic = 1/(test[i][l][0]+0.001)*(test[i][l][0] != 0)
        temp_atoms = test[i][l][1][:,3]
        temp_atoms2 = map(lambda x : dictt[x], temp_atoms)
        resi = np.zeros((len(temp_atoms),len(temp_atoms),10))
        for j in range(len(temp_atoms2)):
            for k in range(j,len(temp_atoms2)):
                res_num = dictt2[tuple(sorted((temp_atoms[j],temp_atoms[k])))]
                resi[j,k,res_num] = 1
                resi[k,k,res_num] = 1
        temp_temp_pic +=  [temp_pic,]
        temp_resi += [resi,]
        test_result[i] = [np.stack(temp_temp_pic),\
                           np.stack(temp_resi)]
    del test[i] 
input = tf.placeholder(tf.float32,[1,None,None,3],
                                             name='input')
if True:
    np.save('train_pic.npy',train_result)
    np.save('test_pic.npy',test_result)
