import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
train = np.load('train_pic.npy.zip')['train_pic'].item()
test = np.load('test_pic.npy.zip')['test_pic'].item()
X_train = []
X_val = []
X_test = []

def make_array(train):
    X_train = []
    for i in train:
        tempA = train[i][1][:2]*np.stack([train[i][0][:2],]*10,-1)
        tempB = np.stack([train[i][0][:2]],-1)
        temp = np.concatenate([tempA,tempB],-1)
        final = []
        for i in temp:
            final +=[i,]
        final = np.array([np.concatenate(final,-1)])
        del train[i]
        if i%2 ==0 :
            X_train += [final,]
        else:
            X_test += [final,]


def one_layer(D_input,output,name):
    layer1a = tf.layers.conv2d(D_input,output,(1,1),padding='same',
                               activation=None,name=name+'a') #linear comb
    layer1b = tf.layers.conv2d(layer1a,output//2,(5,5),padding='same',
                               activation=tf.nn.relu,name=name+'b')
    layer1c = tf.layers.conv2d(layer1a,output//2,(3,3),padding='same',
                               activation=tf.nn.relu,name=name+'c')
    layer1 = tf.concat([layer1b,layer1c],-1,name=name+'concat')
    layer1_pool = tf.layers.average_pooling2d(layer1,(2,2),strides =(2,2),
                                              name=name+'pool')
    return layer1_pool
D_input = tf.placeholder(tf.float32,[1,None,None,22],name='input')
answer = tf.placeholder(tf.float32,[1,],name='answer')
with tf.name_scope('layers') as scope:
    layer1 = one_layer(D_input,32,'layer1')
    layer2 = one_layer(layer1,64,'layer2')
    layer3 = one_layer(layer2,128,'layer3')
with tf.name_scope('out') as scope:
    global_average_pool = tf.reduce_mean(layer3, [1,2])
    dense1 = tf.layers.dense(global_average_pool,64,tf.nn.relu)
    dense2 = tf.layers.dense(global_average_pool,32,None)
    dense3 = tf.layers.dense(global_average_pool,1,None)
cost = tf.reduce_mean(tf.squared_difference(dense3,answer ))
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
learning_rate = tf.Variable(0,dtype=tf.float32,name='learning_rate')

with tf.control_dependencies(update_ops):
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.5).minimize(cost)

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
init = tf.global_variables_initializer();sess = tf.Session();sess.run(init)
