import tensorflow as tf
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
'''
graph CNN
https://arxiv.org/abs/1603.00856
Molecular Graph Convolutions: Moving Beyond Fingerprints
implementation with some changes

# Generally code was simplified instead of using tensordot (in deepchem/nn/graph_layers.py) ,
# simplifed to using 1D and 2D CNN with stride 1 (becomes linear combination). 50 lines reduced to 1 line

1. Batch_norm to layer_norm
2. Removal of gaussian features. Just use mean and var

model inputs
Adj_matrix = tf.placeholder(tf.float32, [1,None,None,5]) # adjacency matrix to 5th order
A_matrix = tf.placeholder(tf.float32, [1,None,3]) #3 atoms-wise features
P_matrix = tf.placeholder(tf.float32, [1,None,None,1]) # distance, Pair-wise features 
num_atoms =  tf.placeholder(tf.int32,None)
answer = tf.placeholder(tf.float32,[1,],name='answer')


train_bond contains
1. pairwise features (adjacent matrix,distance_matrix,bond-order matrix)
2. atoms
3. properties
'''

train_bond = np.load('NPL_nr-ahr.npy').item()
test_bond = np.load('test_nr-ahr.npy.zip')['test_nr-ahr.npy'].item()
chal_bond = np.load('challenge_nr-ahr.npy.zip')['challenge_nr-ahr.npy'].item()
train = pd.read_csv('../nr-ahr.smiles',sep='\t',engine='python',header=None)
train = pd.DataFrame(None)
train['ID'] = sorted(train_bond.keys())
train['NR.AhR'] = 0
chal = pd.read_csv('../tox21_compoundData.csv')
test = pd.read_csv('../tox21_compoundData.csv')
chal['array'] = chal['ID'].map(chal_bond)
train['array']= train['ID'].map(train_bond)
test['array']= test['ID'].map(test_bond)
def got_array(x):
    if type(x) == list:return 1
    else:return 0
train = train[train['array'].apply(got_array)==1] # remove those without features
test = test[test['array'].apply(got_array)==1]
chal = chal[chal['array'].apply(got_array)==1]

train['P_matrix'] = train['ID'].apply(lambda x : train_bond[x][0][:,:,1:])
chal['P_matrix'] = chal['ID'].apply(lambda x : chal_bond[x][0][:,:,1:])
test['P_matrix'] =   test['ID'].apply(lambda x : test_bond[x][0][:,:,1:])
train['A_matrix'] = train['ID'].apply(lambda x : train_bond[x][1])
chal['A_matrix'] = chal['ID'].apply(lambda x : chal_bond[x][1])
test['A_matrix'] = test['ID'].apply(lambda x : test_bond[x][1])
train['adj_1'] = train['ID'].apply(lambda x : train_bond[x][0][:,:,0])
chal['adj_1'] = chal['ID'].apply(lambda x : chal_bond[x][0][:,:,0])
test['adj_1'] = test['ID'].apply(lambda x : test_bond[x][0][:,:,0])
adjs_list = ['adj_1',]
for i in range(2,6):
    adjs_list += ['adj_%s'%i,]
    train['adj_%s'%i] = map(lambda x : 1*(np.matmul(x[0],x[1])>=1), train[['adj_1','adj_%s'%(i-1)]].values)
    chal['adj_%s'%i] = map(lambda x : 1*(np.matmul(x[0],x[1])>=1), chal[['adj_1','adj_%s'%(i-1)]].values)
    test['adj_%s'%i] = map(lambda x : 1*(np.matmul(x[0],x[1])>=1), test[['adj_1','adj_%s'%(i-1)]].values)
train['Adj_matrix'] = map(lambda x : np.stack(x,-1), train[adjs_list].values)
chal['Adj_matrix'] = map(lambda x : np.stack(x,-1), chal[adjs_list].values)
test['Adj_matrix'] = map(lambda x : np.stack(x,-1), test[adjs_list].values)
target1 = 'formation_energy_ev_natom'
train[target1] = train['NR.AhR']
chal[target1] = chal['NR.AhR']
test[target1] = test['NR.AhR']
train = train[~train[target1].isnull()]
chal = chal[~chal[target1].isnull()]
test = test[~test[target1].isnull()]
train= train.sort_values(target1).reset_index(drop=True)
train['predict1'] = 0
test['predict1'] = 0
chal['predict1'] = 0
print train.head()
print test.head()
def atom2Vec(x):
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
chal['A_matrix'] = chal['A_matrix'].apply(atom2Vec)
test['A_matrix'] = test['A_matrix'].apply(atom2Vec)
####### TENSORFLOW ########
Adj_matrix = tf.placeholder(tf.float32, [1,None,None,5]) # adjacency matrix to 5th order
A_matrix = tf.placeholder(tf.float32, [1,None,11]) #3 atoms-wise features
P_matrix = tf.placeholder(tf.float32, [1,None,None,4]) # distance, Pair-wise features 
num_atoms =  tf.placeholder(tf.int32,None)
### WEAVE MODULES ###
epsilon = 1e-7
def layer_norm2(x,name=None):
    #layer normalization because batchsize == 1
    if len(x.get_shape()) == 4:
        mean,var = tf.nn.moments(x,[0,1,2],keep_dims=False)
        scale = tf.Variable(tf.ones([x.shape[-1]]))
        beta = tf.Variable(tf.zeros([x.shape[-1]]))
        final = tf.nn.batch_normalization(x,mean,var,beta,scale,epsilon)
    elif len(x.get_shape()) ==3:
        mean,var = tf.nn.moments(x,[0,1],keep_dims=False)
        scale = tf.Variable(tf.ones([x.shape[-1]]))
        beta = tf.Variable(tf.zeros([x.shape[-1]]))
        final = tf.nn.batch_normalization(x,mean,var,beta,scale,epsilon)
    elif len(x.get_shape()) ==2:
        mean,var = tf.nn.moments(x,[1],keep_dims=False)
        scale = tf.Variable(tf.ones([x.shape[0]]))
        beta = tf.Variable(tf.zeros([x.shape[0]]))
        final = tf.nn.batch_normalization(x,mean,var,beta,scale,epsilon)
    return final

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
    temp = []
    for i in range(Adj_matrix.shape[-1]):
        cnn = tf.multiply(Adj_matrix[:,:,:,i:i+1],cnn,name=name+'2_P_A') #broadcasting in effect
        temp += [cnn,]
    temp = tf.concat(temp,-1)
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
dense4r= layer_norm2(tf.layers.dense(dense3,32,None)) #layer_nrom contains RELU
dense4 = tf.nn.relu(dense4r)
dense5a= tf.layers.dense(dense4,1,None)
dense5 = tf.nn.sigmoid(dense5a)
# Evaluation
answer = tf.placeholder(tf.float32,[1,1],name='answer')
#cost = tf.reduce_mean(tf.squared_difference(dense5,answer )) #linear reg
cost =tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = dense5a, labels = answer ))
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
learning_rate = tf.Variable(0,dtype=tf.float32,name='learning_rate')

with tf.control_dependencies(update_ops):
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.5).minimize(cost)


### MODEL END ###

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
'''
Adj_matrix = tf.placeholder(tf.float32, [1,None,None,5]) # adjacency matrix to 5th order
A_matrix = tf.placeholder(tf.float32, [1,None,3]) #3 atoms-wise features
P_matrix = tf.placeholder(tf.float32, [1,None,None,1]) # distance, Pair-wise features 
num_atoms =  tf.placeholder(tf.int32,None)
answer = tf.placeholder(tf.float32,[1,],name='answer')
'''
np.random.seed(0)
print_param()
import os
saver = tf.train.Saver()
# Initializing the variables
init = tf.global_variables_initializer();sess = tf.Session();sess.run(init)
saved_files = [xxx[:-5] for xxx in os.listdir('.') if (xxx.startswith('model') and xxx.endswith('.ckpt.meta'))]
next_epoch = 0
if len(saved_files) >= 1:
    last_file = sorted(saved_files,key=lambda x : int(x.split('_')[-2]))[-1]
    next_epoch = int(last_file.split('_')[-2])
    print ('starting from :' ,last_file)
    saver.restore(sess,'./'+last_file)
import sklearn
for fold in  range(0,10):
    all_id = range(len(train))
    #random.shuffle(all_id)
    train_id = [x for x in range(len(all_id)) if x%10 != fold]
    val_id =  [x for x in range(len(all_id)) if x%10 == fold]
    for epoch in range(1):
        print (epoch),
        shuffle = []
        for i in train_id:
            if train.iloc[i][2] == 1:
                shuffle += [i,]*7
            else:
                shuffle += [i,]
        random.shuffle(shuffle)
        train_cost = []
        val_cost = []
        counter = 0
        if False: #not training
            for sample in  shuffle:
                counter += 1
                lr = 1+np.cos(1.0*counter*3.142/len(shuffle))
                lr = lr/600
                x1,x2,x3 = train.iloc[sample][['Adj_matrix','A_matrix','P_matrix']]
                x4,y = len(x1),train.iloc[sample][target1]
                _, c = sess.run([optimizer, cost], feed_dict={Adj_matrix: [x1],
                                                              A_matrix: [x2],
                                                              P_matrix: [x3],
                                                              num_atoms: x4,
                                                              answer: [[y]],
                                                              learning_rate : lr })
        if True:
            dictt_train ={}
            dictt_chal ={}
            dictt_test ={}
            train_pred = []
            train_gd = []
            for sample in train_id:
                x1,x2,x3 = train.iloc[sample][['Adj_matrix','A_matrix','P_matrix']]
                x4,y = len(x1),train.iloc[sample][target1]
                out, c = sess.run([dense4, cost], feed_dict={Adj_matrix: [x1],
                                                              A_matrix: [x2],
                                                              P_matrix: [x3],
                                                              num_atoms: x4,
                                                              answer: [[y]],
                                                              learning_rate : 0 })
                dictt_train[train.iloc[sample]['ID']] = out[0]
                train_cost += [c,]
                train_pred += [out[0][0],]
                train_gd += [y,]
            val_pred = []
            val_gd = []
            for sample in val_id:
                x1,x2,x3 = train.iloc[sample][['Adj_matrix','A_matrix','P_matrix']]
                x4,y = len(x1),train.iloc[sample][target1]
                out, c = sess.run([dense4, cost], feed_dict={Adj_matrix: [x1],
                                                              A_matrix: [x2],
                                                              P_matrix: [x3],
                                                              num_atoms: x4,
                                                              answer: [[y]],
                                                              learning_rate : 0 })
                dictt_train[train.iloc[sample]['ID']] = out[0]
                val_cost += [c,]
                val_pred += [out[0][0],]
                val_gd += [y,]
            test_pred = []
            test_gd = []
            test_cost = []
            for sample in range(len(test)):
                x1,x2,x3 = test.iloc[sample][['Adj_matrix','A_matrix','P_matrix']]
                x4,y = len(x1),test.iloc[sample][target1]
                out, c = sess.run([dense4, cost], feed_dict={Adj_matrix: [x1],
                                                              A_matrix: [x2],
                                                              P_matrix: [x3],
                                                              num_atoms: x4,
                                                              answer: [[y]],
                                                              learning_rate : 0 })
                dictt_test[test.iloc[sample]['ID']] = out[0]
                test_cost += [c,]
                test_pred += [out[0][0],]
                test_gd += [y,]
            chal_pred = []
            chal_gd = []
            chal_cost = []
            for sample in range(len(chal)):
                x1,x2,x3 = chal.iloc[sample][['Adj_matrix','A_matrix','P_matrix']]
                x4,y = len(x1),chal.iloc[sample][target1]
                out, c = sess.run([dense4, cost], feed_dict={Adj_matrix: [x1],
                                                              A_matrix: [x2],
                                                              P_matrix: [x3],
                                                              num_atoms: x4,
                                                              answer: [[y]],
                                                              learning_rate : 0 })
                dictt_chal[chal.iloc[sample]['ID']] = out[0]
                chal_cost += [c,]
                chal_pred += [out[0][0],]
                chal_gd += [y,]
            np.save('dictt_train.npy',dictt_train)
            np.save('dictt_chal.npy',dictt_chal)
            np.save('dictt_test.npy',dictt_test)






die
####t_histo_rows = [
####        tf.histogram_fixed_width(
####            tf.gather([-2.5,-1.5,-.5,.5,1.5,2.5], [row]),
####            vals, nbins)
####        for row in range(11)]
####
####t_histo = tf.pack(t_histo_rows, axis=0)
a = np.array([[0,1,2,3],[0,0,0,1],[1,0,0,0],[1,1,0,0],[1,1,10,10],[1,1,1,1]])
b = np.stack([a,2*a],-1)
c = np.concatenate([b,b*3],0)
temp = tf.placeholder(tf.float32, [None,6,4])
temp2 = layer_norm2(temp,'test')
temp3 = gaussian_histogram(temp2)
init = tf.global_variables_initializer();sess = tf.Session();sess.run(init)
'''
array([[[-1.4142135 ,  0.7071067 , -0.04657465,  0.1428572 ],
        [-1.4142135 , -1.4142135 , -0.60547036, -0.42857143],
        [ 0.7071067 , -1.4142135 , -0.60547036, -0.71428573],
        [ 0.7071067 ,  0.7071067 , -0.60547036, -0.71428573],
        [ 0.7071067 ,  0.7071067 ,  2.189008  ,  2.142857  ],
        [ 0.7071067 ,  0.7071067 , -0.3260225 , -0.42857143]]],
'''
