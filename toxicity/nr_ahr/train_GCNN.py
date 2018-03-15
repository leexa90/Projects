import tensorflow as tf
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
'''
batch norml is SLOWSLOW
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

train_bond = np.load('nr-ahr.npy.zip')['nr-ahr.npy'].item()
test_bond = train_bond
train = pd.read_csv('../nr-ahr.smiles',sep='\t',engine='python',header=None)
test = train.iloc[:10]

train['P_matrix'] = train['id'].apply(lambda x : np.stack([train_bond[x][0]],-1))
test['P_matrix'] = test['id'].apply(lambda x : np.stack([test_bond[x][0]],-1))
train['A_matrix'] = train['id'].apply(lambda x : train_bond[x][1])
test['A_matrix'] = test['id'].apply(lambda x : test_bond[x][1])
train['adj_1'] = train['id'].apply(lambda x : 1*(train_bond[x][0] < 2.7))
test['adj_1'] = test['id'].apply(lambda x : 1*(test_bond[x][0] < 2.7))
for i in range(2,6):
    train['adj_%s'%i] = map(lambda x : 1*(np.matmul(x[0],x[1])>=1), train[['adj_1','adj_%s'%(i-1)]].values)
    test['adj_%s'%i] = map(lambda x : 1*(np.matmul(x[0],x[1])>=1), test[['adj_1','adj_%s'%(i-1)]].values)
train['Adj_matrix'] = map(lambda x : np.stack(x,-1), train[[x for x in train.keys() if 'adj_' in x]].values)
test['Adj_matrix'] = map(lambda x : np.stack(x,-1), test[[x for x in train.keys() if 'adj_' in x]].values)
target1 = 'formation_energy_ev_natom'
target2 = 'bandgap_energy_ev'

train[target1] = np.log(1+train[target1])
train[target2] = np.log(1+train[target2])
Adj_matrix = tf.placeholder(tf.float32, [1,None,None,5]) # adjacency matrix to 5th order
A_matrix = tf.placeholder(tf.float32, [1,None,4]) #3 atoms-wise features
P_matrix = tf.placeholder(tf.float32, [1,None,None,1]) # distance, Pair-wise features 
num_atoms =  tf.placeholder(tf.int32,None)
test[target1] = 0
### WEAVE MODULES ###

def layer_norm2(x,name=None,training=training):
    # this is just batch norm, not layer norm
    final = tf.layers.batch_normalization(x,training=training)
    return final

def layer_norm2(x,name=None):
    epsilon = 1e-7
    #this is real layer normalization
    #problematic with zero padding. (std is different
    #for diferent paddings sizes)
    # Will be followed by RELU
    if len(x.get_shape()) == 4:
        mean,var = tf.nn.moments(x,[1,2],keep_dims=True)

    elif len(x.get_shape()) ==3:
        mean,var = tf.nn.moments(x,[1],keep_dims=True)

    scale = tf.Variable(tf.ones(mean.shape[-1]))
    beta = tf.Variable(tf.zeros(mean.shape[-1]))
    final = tf.nn.batch_normalization(x,mean,var,beta,scale,epsilon)
    return final



def P_P (P_matrix,name,activation=tf.nn.relu,depth=50):
    return tf.layers.conv2d(P_matrix,depth,(1,1),padding='same',
                               activation=activation,name=name+'P_P')

def A_A (A_matrix,name,activation=tf.nn.relu,depth=50):
    return tf.layers.conv1d(A_matrix,depth,1,padding='same',
                               activation=activation,name=name+'A_A')

def P_A(P_matrix,name,Adj_matrix=Adj_matrix,activation=tf.nn.relu,depth=50):
    cnn = tf.layers.conv2d(P_matrix,depth,(1,1),padding='same',
                               activation=activation,name=name+'1_P_A')
    temp = []
    for i in range(Adj_matrix.shape[-1]): #SHOULD TRY TO STACK THE IMAGES FIRST
        cnn = tf.multiply(Adj_matrix[:,:,:,i:i+1],cnn,name=name+'2_P_A') #broadcasting in effect
        temp += [cnn,]
    cnn = tf.concat(temp,-1)
    return tf.reduce_sum(cnn,(-2),name=name+'P_A')

def A_P(A_matrix,num_atoms,name,activation=tf.nn.relu,depth=50):
    result = []
    i =0
    x=A_matrix[:,:,:]
    x1 = tf.expand_dims(x,1)
    x2 = tf.expand_dims(x,2)
    x1b= tf.tile(x1,(1,num_atoms,1,1))
    x2b= tf.tile(x2,(1,1,num_atoms,1))
    x3 = tf.concat([x1b,x2b],-1)
    result = tf.layers.conv2d(x3,depth,(1,1),padding='same',
                            activation=activation,name=name + '%sA_P' %i)
    return result
### WEAVE MODULES END###

### WEAVE FUNCTION ###
def weave(A_matrix,P_matrix,
          num_atoms,name,
          Adj_matrix=Adj_matrix):
    # this module has relu activation function and also 50 depth
    name = name+'_weave'
    A_matrix = layer_norm2(A_matrix , name+'A')
    P_matrix = layer_norm2(P_matrix , name+'A')
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
    A_matrix = layer_norm2(A_matrix , name+'A')
    P_matrix = layer_norm2(P_matrix , name+'A')
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
dense2 = tf.concat([meanA2],-1)
dense3 = tf.layers.dense(dense2,256,tf.nn.relu)
dense4 = tf.layers.dense(dense3,32,tf.nn.relu)
dense5 = tf.layers.dense(dense4,1,None)
# Evaluation
answer = tf.placeholder(tf.float32,[1,1],name='answer')
cost = tf.reduce_mean(tf.squared_difference(dense5,answer ))
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
learning_rate = tf.Variable(0,dtype=tf.float32,name='learning_rate')

with tf.control_dependencies(update_ops):
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.5).minimize(cost)

die
### MODEL END ###

init = tf.global_variables_initializer();sess = tf.Session();sess.run(init)
def print_param():
    total_parameters = 0
    print ''' ### PARAMETERS ### '''
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        print variable.name,variable_parameters,variable
        total_parameters += variable_parameters
    print' ### model parameters',total_parameters,'###'
'''
Adj_matrix = tf.placeholder(tf.float32, [1,None,None,5]) # adjacency matrix to 5th order
A_matrix = tf.placeholder(tf.float32, [1,None,3]) #3 atoms-wise features
P_matrix = tf.placeholder(tf.float32, [1,None,None,1]) # distance, Pair-wise features 
num_atoms =  tf.placeholder(tf.int32,None)
answer = tf.placeholder(tf.float32,[1,],name='answer')
'''
np.random.seed(0)
print_param()
for fold in  range(0,10):
    init = tf.global_variables_initializer();sess = tf.Session();sess.run(init)
    all_id = range(len(train))
    #random.shuffle(all_id)
    train_id = [x for x in range(len(all_id)) if x%10 != fold]
    val_id =  [x for x in range(len(all_id)) if x%10 == fold]
    for epoch in range(100):
        print (epoch),
        shuffle = train_id
        random.shuffle(shuffle)
        train_cost = []
        val_cost = []
        counter = 0
        for sample in  shuffle:
            counter += 1
            lr = 1+np.cos(1.0*counter*3.142/len(shuffle))
            lr = lr/300
            x1,x2,x3 = train.iloc[sample][['Adj_matrix','A_matrix','P_matrix']]
            x4,y = len(x1),train.iloc[sample][target1]
            _, c = sess.run([optimizer, cost], feed_dict={Adj_matrix: [x1],
                                                          A_matrix: [x2],
                                                          P_matrix: [x3],
                                                          num_atoms: x4,
                                                          answer: [[y]],
                                                          learning_rate : lr })
            #print sess.run([extra_feat,extra_feat1], feed_dict={D_input: X,answer:[y], learning_rate : lr, extra_feat:z })
        for sample in train_id:
            x1,x2,x3 = train.iloc[sample][['Adj_matrix','A_matrix','P_matrix']]
            x4,y = len(x1),train.iloc[sample][target1]
            out, c = sess.run([dense5, cost], feed_dict={Adj_matrix: [x1],
                                                          A_matrix: [x2],
                                                          P_matrix: [x3],
                                                          num_atoms: x4,
                                                          answer: [[y]],
                                                          learning_rate : 0 })
            train_cost += [c,]
        print (np.mean(train_cost)),
        val_answer = []
        for sample in val_id:
            x1,x2,x3 = train.iloc[sample][['Adj_matrix','A_matrix','P_matrix']]
            x4,y = len(x1),train.iloc[sample][target1]
            out, c = sess.run([dense5, cost], feed_dict={Adj_matrix: [x1],
                                                          A_matrix: [x2],
                                                          P_matrix: [x3],
                                                          num_atoms: x4,
                                                          answer: [[y]],
                                                          learning_rate : 0 })
            val_cost += [c,]
            val_answer += [out[0][0],]
        test_answer = []
        for sample in range(len(test)):
            x1,x2,x3 = train.iloc[sample][['Adj_matrix','A_matrix','P_matrix']]
            x4,y = len(x1),train.iloc[sample][target1]
            out, c = sess.run([dense5, cost], feed_dict={Adj_matrix: [x1],
                                                          A_matrix: [x2],
                                                          P_matrix: [x3],
                                                          num_atoms: x4,
                                                          answer: [[y]],
                                                          learning_rate : 0 })
            test_answer += [out[0][0],]
        print (np.mean(val_cost))
        if epoch >= 80:
            train = train.set_value(val_id,
                                                    'predict1',
                                                    train['predict1'].iloc[val_id] + np.array(val_answer))
            test = test.set_value(range(len(test)),
                                                  'predict1',
                                                  test['predict1'] + np.array(test_answer)/10)




            train.to_csv('train_CNN.csv',index=0)
            test.to_csv('test_CNN.csv', index=0)








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
