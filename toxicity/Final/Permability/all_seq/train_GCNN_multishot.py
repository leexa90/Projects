import tensorflow as tf
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import sys,os

import os
tot_m, used_m, free_m = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
print used_m
print free_m
print tot_m
target1 = 'permable1'
target2 = 'permable2'
train = pd.read_csv('peptide_permability.csv').iloc[::1]
train = train[(train['size'] >=5) & (train['size'] <=15)].reset_index(drop=True)
train.head()
print len(train)
train[target1]  = (train['source'] == 2)*1
def fn1(str):
    #if structure contains over .334 R or K return 0 or else
    # return number of distinct amino acids the seq has
    # use to weed out low-complexity sequences and hightly positively charged sequences
    num=0.0
    for i in str:
        if i.upper() == 'R' or i.upper() == 'K':
            num += 1
    if num/len(str) > .334: #should be tweaked
        return 0
    return len(set(str))
train = train[train['ID'].apply(fn1)>= 4]

train_caco = pd.read_csv('permability.csv')
train_caco = train_caco[train_caco['set'] != 'val']
train_caco[target2] = train_caco['permability']
train = train.append(train_caco).reset_index(drop=True)

train_bond = np.load('peptide_permability.npy.zip')['peptide_permability.npy'].item()
train_bond.update( np.load('permability.npy.zip')['permability.npy'].item())
train['array']= train['ID'].map(train_bond)
train['P_matrix'] = train['ID'].apply(lambda x : train_bond[x][0][:,:,1:])
train['A_matrix'] = train['ID'].apply(lambda x : train_bond[x][1])
train['adj_1'] = train['ID'].apply(lambda x : train_bond[x][0][:,:,0])
adjs_list = ['adj_1',]
for i in range(2,6):
    adjs_list += ['adj_%s'%i,]
    train['adj_%s'%i] = map(lambda x : 1*(np.matmul(x[0],x[1])>=1), train[['adj_1','adj_%s'%(i-1)]].values)
    tot_m, used_m, free_m = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
    print used_m
    print free_m
    print tot_m
train['Adj_matrix'] = map(lambda x : np.stack(x,-1), train[adjs_list].values)
target1 = 'permable1'
target2 = 'permable2'
train[target2] = train['permability']
train['predict1'] = 0
train['predict2'] = 0
def atom2Vec(x): #one hot encode atom
    result = []
    for i in x:
        temp= [0,]*11
        if i  == 1: temp[0] = 1
        elif i == 6: temp[1] = 1
        elif i == 7: temp[2] = 1
        elif i == 8: temp[3] = 1
        elif i == 9: temp[4] = 1
        elif i == 15: temp[5] = 1
        elif i == 16: temp[6] = 1
        elif i == 17: temp[7] = 1
        elif i == 35: temp[8] = 1
        elif i == 53: temp[9] = 1
        else : temp[10] = 1
        result += [temp,]
    return np.stack(result,0)
train['A_matrix'] = train['A_matrix'].apply(atom2Vec)
train.head()
train.head()
test = train[train['set'] == 'Te']
train = train[train['set'] != 'Te']
####### TENSORFLOW ########
Adj_matrix = tf.placeholder(tf.float32, [None,None,None,5]) # adjacency matrix to 5th order
A_matrix = tf.placeholder(tf.float32, [None,None,11]) #3 atoms-wise features
P_matrix = tf.placeholder(tf.float32, [None,None,None,4]) # distance, Pair-wise features 
num_atoms =  tf.placeholder(tf.int32,None)
training= tf.placeholder(tf.bool,None)
### WEAVE MODULES ###
epsilon = 1e-7
def layer_norm2(x,name=None):
    #problematic with zero padding. (zeros gets normalized differently
    #for diferent paddings sizes)
    
    # Will be followed by RELU
    if len(x.get_shape()) == 4:
        mean,var = tf.nn.moments(x,[1,2],keep_dims=True)

    elif len(x.get_shape()) ==3:
        mean,var = tf.nn.moments(x,[1],keep_dims=True)

    elif len(x.get_shape()) ==2:
        mean,var = tf.nn.moments(x,[1],keep_dims=True)
    scale = tf.Variable(tf.ones(mean.shape[-1]))
    beta = tf.Variable(tf.zeros(mean.shape[-1]))
    final = tf.nn.batch_normalization(x,mean,var,beta,scale,epsilon)
    return final

#def layer_norm2(x,name=None,training=training): # this is batch norm per layer
#    final = tf.layers.batch_normalization(x,training=training)
#    return final

def P_P (P_matrix,name,activation=tf.nn.relu,depth=50):
    temp =  tf.layers.conv2d(P_matrix,depth,(1,1),padding='same',
                               activation=None,name=name+'P_P')
    temp = layer_norm2(temp,name+'norm')
    if activation == tf.nn.relu :
       temp = tf.nn.relu(temp) 
    return temp

def A_A (A_matrix,name,activation=tf.nn.relu,depth=50):
    temp = tf.layers.conv1d(A_matrix,depth,1,padding='same',
                               activation=activation,name=name+'A_A')
    temp = layer_norm2(temp,name+'norm')
    if activation == tf.nn.relu :
       temp = tf.nn.relu(temp) 
    return temp

def P_A(P_matrix,name,Adj_matrix=Adj_matrix,activation=tf.nn.relu,depth=50):
    cnn = tf.layers.conv2d(P_matrix,depth,(1,1),padding='same',
                               activation=None,name=name+'1_P_A')
##    temp = []
##    for i in range(Adj_matrix.shape[-1]): #SHOULD TRY TO STACK THE IMAGES FIRST
##        cnn = tf.multiply(Adj_matrix[:,:,:,i:i+1],cnn,name=name+'2_P_A') #broadcasting in effect
##        temp += [cnn,]
    temp = cnn#tf.concat(temp,-1)
    temp = layer_norm2(temp,name+'norm')
    if activation == tf.nn.relu :
       temp = tf.nn.relu(temp) 
    return tf.reduce_sum(temp,(-2),name=name+'P_A')

def A_P(A_matrix,num_atoms,name,activation=tf.nn.relu,depth=50):
    result = []
    i =0
    x=A_matrix[:,:,:]
    x1 = tf.expand_dims(x,1)
    x2 = tf.expand_dims(x,2)
    x1b= tf.tile(x1,(1,num_atoms,1,1))
    x2b= tf.tile(x2,(1,1,num_atoms,1))
    x3 = tf.concat([x1b,x2b],-1)
    temp = tf.layers.conv2d(x3,depth,(1,1),padding='same',
                            activation=None,name=name + '%sA_P' %i)
    temp = layer_norm2(temp,name+'norm')
    if activation == tf.nn.relu :
       temp = tf.nn.relu(temp) 
    return temp
### WEAVE FUNCTION ###
def weave(A_matrix,P_matrix,
          num_atoms,name,
          Adj_matrix=Adj_matrix):
    # this module has relu activation function and also 50 depth
    name = name+'_weave'
    #A_matrix = layer_norm2(A_matrix , name+'A')
    #P_matrix = layer_norm2(P_matrix , name+'A')
    P0 = P_P (P_matrix,name+'_P0_')
    P1 = A_P(A_matrix,num_atoms,name+'_P1_')
    A0 = A_A (A_matrix,name+'_A0_')
    A1 = P_A(P_matrix,name+'_A1_',Adj_matrix=Adj_matrix)
    P2 = tf.concat([P0,P1],-1,name+'_P2_')
    A2 = tf.concat([A0,A1],-1,name+'_A2_')
    P3 = P_P(P2,name+'_P3_')
    A3 = A_A(A2,name+'_A3_')
    return A3,P3

def weave_final(A_matrix,P_matrix,
                num_atoms,name,
                Adj_matrix=Adj_matrix,):
    # this module has no activation function and also 128 depth
    name = name+'_weave'
    #A_matrix = layer_norm2(A_matrix , name+'A')
    #P_matrix = layer_norm2(P_matrix , name+'A')
    activation = tf.nn.relu
    P0 = P_P (P_matrix,name+'_P0_',activation=activation,depth=128)
    P1 = A_P(A_matrix,num_atoms,name+'_P1_',activation=activation,depth=128)
    A0 = A_A (A_matrix,name+'_A0_',activation=activation,depth=128)
    A1 = P_A(P_matrix,name+'_A1_',Adj_matrix=Adj_matrix,activation=activation,depth=128)
    P2 = tf.concat([P0,P1],-1,name+'_P2_')
    A2 = tf.concat([A0,A1],-1,name+'_A2_')
    activation = None #final convolution
    P3 = P_P(P2,name+'_P3_',activation=activation,depth=128)
    A3 = A_A(A2,name+'_A3_',activation=activation,depth=128)
    return A3,P3
### WEAVE FUNCTION END###

def gaussian_histogram( x):
    gaussian_memberships = [(-1.645, 0.283), (-1.080, 0.170), (-0.739, 0.134),
                            (-0.468, 0.118), (-0.228, 0.114), (0., 0.114),
                            (0.228, 0.114), (0.468, 0.118), (0.739, 0.134),
                            (1.080, 0.170), (1.645, 0.283)]
    dist = [
        tf.contrib.distributions.Normal(p[0], p[1])
        for p in gaussian_memberships
    ]
    dist_max = [dist[i].prob(gaussian_memberships[i][0]) for i in range(11)]
    outputs1 = [dist[i].prob(x) / dist_max[i] for i in range(11)]
    outputs = tf.stack(outputs1, axis=2)
    outputs = outputs / (0.00001+tf.reduce_sum(outputs, axis=2, keep_dims=True))
    return outputs,dist_max,outputs1


### BUILDING MODEL ###
# two weave modules #
A0,P0 = weave(A_matrix,P_matrix,num_atoms,'weave1')
A1,P1 = weave_final(A0,P0,num_atoms,'weavef') #paper includes seperate, final convolution,i just rused the last layer
#molecular level featurization
#meanA, varA = tf.nn.moments(A1, axes=[1])
#meanP, varP = tf.nn.moments(P1, axes=[1,2])
#Dense layer
dense0 = A1
'''
Quote from Paper
Gaussian features
'''
dense1,dist_max,outputs1 = gaussian_histogram(dense0)
meanA, varA = tf.nn.moments(dense1, axes=[1]) #sum over all points
shapeNumFeat,shapeHisBins = map(int,meanA.shape[-2:]) #gotta convert to int
meanA2,varA2= tf.reshape(meanA,(-1,shapeNumFeat*shapeHisBins)),tf.reshape(varA,(-1,shapeNumFeat*shapeHisBins))
dense2r= layer_norm2(meanA2)
dense2 = tf.nn.relu(dense2r)
dense3r= layer_norm2(tf.layers.dense(dense2,256,None))
dense3 = tf.nn.relu(dense3r)
dense4r= layer_norm2(tf.layers.dense(dense3,64,None)) #layer_nrom contains RELU
dense4 = tf.nn.relu(dense4r)
dense5b= tf.layers.dense(dense4,1,None) #caco
dense5a= tf.layers.dense(dense4,1,None) #classification
dense5 = tf.nn.sigmoid(dense4) #classification
# Evaluation
answer = tf.placeholder(tf.float32,[None,2],name='answer') #(classification,caco2)
cost_mask = tf.placeholder(tf.float32,[None,2],name='cost_mask') 
#cost = tf.reduce_mean(tf.squared_difference(dense5,answer )) #linear reg
costA =tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = dense5a, labels = answer[:,0:1] )* cost_mask[:,0:1])
costB =tf.reduce_mean(tf.squared_difference(dense5b,answer[:,1:]  )* cost_mask[:,1:])
cost = costA+costB
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
learning_rate = tf.Variable(0,dtype=tf.float32,name='learning_rate')

with tf.control_dependencies(update_ops):
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.5).minimize(cost)
def R2(truth,pred):
    truth = np.array(truth)
    pred = np.array(pred)
    all_var = truth - np.mean(truth)
    all_var = np.sum(all_var**2)
    expl_var = truth - pred
    expl_var = np.sum(expl_var**2)
    return (1.0-expl_var/all_var)

### MODEL END ###
train[target1] = train[target1].fillna(-999)
train[target2] = train[target2].fillna(-999)
init = tf.global_variables_initializer();sess = tf.Session();sess.run(init)
def print_param():
    total_parameters = 0
    print (''' ### PARAMETERS ### ''')
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        print (variable.name,variable_parameters,variable)
        total_parameters += variable_parameters
    print (' ### model parameters',total_parameters,'###')
import sklearn
print 'Imbalanced',int((1-np.mean(train[target1]))/np.mean(train[target1]))
for fold in  range(0,5):
    init = tf.global_variables_initializer();sess = tf.Session();sess.run(init)
    all_id = range(len(train))
    #random.shuffle(all_id)
    train_id = [x for x in range(len(all_id)) if x%5 != fold]
    val_id =  [x for x in range(len(all_id)) if x%5== fold]
    for epoch in range(22):
        shuffle = []
        for i in train_id:
            if train.iloc[i][target1] == 1:
                shuffle += [i,]*8#int((1-np.mean(train[target1]))/np.mean(train[target1]))
            elif train.iloc[i][target1] == -999:
                shuffle += [i,]*10#int((1-np.mean(train[target1]))/np.mean(train[target1]))
            else:
                shuffle += [i,]
        print (epoch),len(shuffle),
        random.shuffle(shuffle)
        train_cost = []
        val_cost = []
        test_cost = []
        counter = 0
        def zero_padding(x,mode = 'constant'):
            return np.array([x[0].astype(np.float32)])
##            max_len = max(map(len,x))
##            for i in range(len(x)):
##                offset = max_len - len(x[i])
##                temp = len(x[i].shape)-1
##                tup = (offset//2,(offset+1)//2)
##                
##                x[i] = np.pad(x[i],(tup,)*temp+((0,0),),
##                              mode = mode)
##            return np.stack(x,0)
            
        if True: #not training
            for i in  range(0,len(shuffle),1):
                sample = shuffle[i:i+1]
                counter += 1
                lr = 1+np.cos(1.0*counter*3.142/len(shuffle))
                lr = lr/3000#3
                x1 = zero_padding(train.iloc[sample]['Adj_matrix'].values)
                x2 = zero_padding(train.iloc[sample]['A_matrix'].values)
                x3 = zero_padding(train.iloc[sample]['P_matrix'].values)
                x4 = np.max(train.iloc[sample]['P_matrix'].map(len))
                y = train.iloc[sample][[target1,target2]].values
                y2 = 1*(y != -999)
                _, c = sess.run([optimizer, cost],
                                feed_dict={Adj_matrix: x1,
                                           A_matrix: x2,
                                           P_matrix: x3,
                                           num_atoms: x4,
                                           answer: y,
                                           learning_rate : lr,
                                           cost_mask : y2,
                                           training : True})
                #die
        if True:
            train_pred = []
            train_gd = []
            for sample in train_id:
                x1,x2,x3 = train.iloc[sample][['Adj_matrix','A_matrix','P_matrix']]
                x4 = len(x1)
                y = train.iloc[sample][[target1,target2]].values
                y2 = 1*(y != -999)
                outa,outb, c = sess.run([dense5a,dense5b,costB], feed_dict={Adj_matrix: [x1],
                                                              A_matrix: [x2],
                                                              P_matrix: [x3],
                                                              num_atoms: x4,
                                                              answer: [y],
                                                              learning_rate : 0,
                                                             cost_mask : [y2],
                                                             training : False})
                train_cost += [c,]
                train_pred += [(outa[0][0],outb[0][0]),]
                train_gd += [y,]
            gd = np.array([x[0] for x in train_gd])[train.iloc[train_id][target2]==-999]
            pred = np.array([x[0] for x in train_pred])[train.iloc[train_id][target2]==-999]
            gd2 = np.array([x[0] for x in train_gd])[train.iloc[train_id][target2]!=-999]
            pred2 = np.array([x[0] for x in train_pred])[train.iloc[train_id][target2]!=-999]
            print (len(train_cost),np.mean(train_cost),sklearn.metrics.roc_auc_score(gd,pred),R2(gd,pred))
            val_pred = []
            val_gd = []
            for sample in val_id:
                x1,x2,x3 = train.iloc[sample][['Adj_matrix','A_matrix','P_matrix']]
                x4 = len(x1)
                y = train.iloc[sample][[target1,target2]].values
                y2 = 1*(y != -999)
                outa,outb, c = sess.run([dense5a,dense5b, costB], feed_dict={Adj_matrix: [x1],
                                                              A_matrix: [x2],
                                                              P_matrix: [x3],
                                                              num_atoms: x4,
                                                              answer: [y],
                                                              learning_rate : 0,
                                                             cost_mask : [y2],
                                                             training : False})
                val_cost += [c,]
                val_pred += [(outa[0][0],outb[0][0]),]
                val_gd += [y,]
            gd = np.array([x[0] for x in val_gd])[train.iloc[val_id][target2]==-999]
            pred = np.array([x[0] for x in val_pred])[train.iloc[val_id][target2]==-999]
            gd2 = np.array([x[0] for x in val_gd])[train.iloc[val_id][target2]!=-999]
            pred2 = np.array([x[0] for x in val_pred])[train.iloc[val_id][target2]!=-999]
            print (len(val_cost),np.mean(val_cost),sklearn.metrics.roc_auc_score(gd,pred),R2(gd2,pred2))
            test_pred = []
            test_gd = []
            for sample in range(len(test)):
                x1,x2,x3 = test.iloc[sample][['Adj_matrix','A_matrix','P_matrix']]
                x4 = len(x1)
                y = test.iloc[sample][[target1,target2]].values
                y2 = 1*(y != -999)
                outa,outb, c = sess.run([dense5a,dense5b, costB], feed_dict={Adj_matrix: [x1],
                                                              A_matrix: [x2],
                                                              P_matrix: [x3],
                                                              num_atoms: x4,
                                                              answer: [y],
                                                              learning_rate : 0,
                                                             cost_mask : [y2],
                                                             training : False})
                test_cost += [c,]
                test_pred += [(outa[0][0],outb[0][0]),]
                test_gd += [y,]
            gd = np.array([x[1] for x in test_gd])
            pred = np.array([x[1] for x in test_pred])
            print (len(test_cost),np.mean(test_cost),R2(gd,pred))
##            test_pred = []
##            test_gd = []
##            test_cost = []
##            for sample in range(len(test)):
##                x1,x2,x3 = test.iloc[sample][['Adj_matrix','A_matrix','P_matrix']]
##                x4,y = len(x1),test.iloc[sample][target1]
##                out, c = sess.run([dense5, cost], feed_dict={Adj_matrix: [x1],
##                                                              A_matrix: [x2],
##                                                              P_matrix: [x3],
##                                                              num_atoms: x4,
##                                                              answer: [[y]],
##                                                              learning_rate : 0 ,
##                                                             training : False})
##                test_cost += [c,]
##                test_pred += [out[0][0],]
##                test_gd += [y,]
##            print (len(test_cost),np.mean(test_cost),sklearn.metrics.roc_auc_score(test_gd,test_pred))
##            chal_pred = []
##            chal_gd = []
##            chal_cost = []
##            for sample in range(len(chal)):
##                x1,x2,x3 = chal.iloc[sample][['Adj_matrix','A_matrix','P_matrix']]
##                x4,y = len(x1),chal.iloc[sample][target1]
##                out, c = sess.run([dense5, cost], feed_dict={Adj_matrix: [x1],
##                                                              A_matrix: [x2],
##                                                              P_matrix: [x3],
##                                                              num_atoms: x4,
##                                                              answer: [[y]],
##                                                              learning_rate : 0,
##                                                             training : False})
##                chal_cost += [c,]
##                chal_pred += [out[0][0],]
##                chal_gd += [y,]
##            print (len(chal_cost),np.mean(chal_cost),sklearn.metrics.roc_auc_score(chal_gd,chal_pred))
            if epoch >= 17:
                if 'predict1_%s'%epoch not in train.keys():
                    train['predict1_%s'%epoch] = 0
                    train['predict2_%s'%epoch] = 0
                train = train.set_value(val_id,
                                                        ['predict1_%s'%epoch,'predict2_%s'%epoch],
                                                         np.array(val_pred))
##                test = test.set_value(range(len(test)),
##                                                      'predict1',
##                                                      test['predict1'] + np.array(test_pred)/5)
##                chal = chal.set_value(range(len(chal)),
##                                                      'predict1',
##                                                      chal['predict1'] + np.array(chal_pred)/5)
##                print ('ENSEMBLE',len(test_cost),sklearn.metrics.roc_auc_score(test[target1],test['predict1']))
##                print ('ENSEMBLE',len(chal_cost),sklearn.metrics.roc_auc_score(chal[target1],chal['predict1']))
##                print '\n'
##
                train.to_csv('train_CNN.csv',index=0)
                test.to_csv('test_CNN.csv', index=0)
##                save_path = saver.save(sess,'./model_final_reweigh_loss_%s_%s.ckpt' %(fold,epoch))


