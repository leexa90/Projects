import numpy as np
import pandas as pd
import tensorflow as tf
#import matplotlib.pyplot as plt
import random
random.seed(0)
train = np.load('../train_pic.npy.zip')['train_pic'].item()
test = np.load('../test_pic.npy.zip')['test_pic'].item()
target1 = 'formation_energy_ev_natom'
target2 = 'bandgap_energy_ev'
train_y = np.log1p(pd.read_csv('../train.csv')[target1])
train_predict = pd.read_csv('../train_CNN_all.csv')
test_predict = pd.read_csv('../test_CNN_all.csv')
train_predict['predict1'] = 0
test_predict['predict1'] = 0
cols = [x for x in  train_predict.keys() if 'array' not in x and \
        'predict' not in x and x[0] != x and 'CNN' not in x and \
        x not in ['id',target1,target2,'predict1','predict2'] ]
X_test = {}
X_train = {}
num = 0
import sys
if len(sys.argv) ==3:
    num = int(sys.argv[1])
    if sys.argv[2] == str(2):
        train_y = np.log1p(pd.read_csv('../train.csv')[target2])
    else:
        train_y = np.log1p(pd.read_csv('../train.csv')[target1])
for i in train.keys():
    tempA = train[i][1][:]*np.stack([train[i][0][:],]*10,-1)
    tempB = np.stack([train[i][0][:]],-1)
    temp = np.concatenate([tempA,tempB],-1)
    final = []
    y = train_y[i-1]
    feat = train_predict[cols].iloc[i-1:i].values
    for ii in temp:
        final +=[ii,]
    final = [np.array([np.concatenate(final,-1)]),y,feat]
    del train[i]
    X_train[i-1] = final#id vs pandas id offset 1

for i in test.keys():
    tempA = test[i][1][:]*np.stack([test[i][0][:],]*10,-1)
    tempB = np.stack([test[i][0][:]],-1)
    temp = np.concatenate([tempA,tempB],-1)
    final = []
    feat = test_predict[cols].iloc[i-1:i].values
    for ii in temp:
        final +=[ii,]
    final =[np.array([np.concatenate(final,-1)]),-999,feat]
    del test[i]
    X_test[i-1] = final #id vs pandas id offset 1



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
D_input = tf.placeholder(tf.float32,[1,None,None,44],name='input')
answer = tf.placeholder(tf.float32,[1,],name='answer')
extra_feat = tf.placeholder(tf.float32,[1,len(cols)],name='extra_features')
with tf.name_scope('layers') as scope:
    layer1 = one_layer(D_input,32,'layer1')
    layer2 = one_layer(layer1,64,'layer2')
    layer3 = one_layer(layer2,128,'layer3')
with tf.name_scope('out') as scope:
    global_average_pool = tf.concat([tf.reduce_mean(layer3, [1,2]),extra_feat],1)
    dense0 = tf.layers.dense(global_average_pool,64,tf.nn.relu)
    dense1 = tf.layers.dense(dense0,128,tf.nn.relu)
    dense2 = tf.layers.dense(dense1,64,tf.nn.relu)
    dense3 = tf.layers.dense(dense2,32,None)
    dense4 = tf.layers.dense(dense3,1,None)
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
range
for fold in  range(0,10):
    init = tf.global_variables_initializer();sess = tf.Session();sess.run(init)
    all_id = range(len(X_train))
    #random.shuffle(all_id)
    train_id = [all_id[x] for x in range(len(all_id)) if x%10 != fold]
    val_id =  [all_id[x] for x in range(len(all_id)) if x%10 == fold]
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
            lr = lr/100
            X,y,z = X_train[sample]
            _, c = sess.run([optimizer, cost], feed_dict={D_input: X,answer:[y], learning_rate : lr, extra_feat:z })
        for i in train_id:
            X,y,z = X_train[i]
            out, c = sess.run([dense3, cost], feed_dict={D_input: X,answer:[y], learning_rate : lr,  extra_feat:z})
            train_cost += [c,]
        print (np.mean(train_cost)),
        val_answer = []
        for i in val_id:
            X,y,z = X_train[i]
            out, c = sess.run([dense3, cost], feed_dict={D_input: X,answer:[y], learning_rate : lr, extra_feat:z })
            val_cost += [c,]
            val_answer += [out[0][0],]
        test_answer = []
        for i in range(len(X_test)):
            X,y,z = X_test[i]
            out = sess.run(dense3, feed_dict={D_input: X,answer:[y], learning_rate : 0 , extra_feat:z})
            test_answer += [out[0][0],]
        print (np.mean(val_cost))
        if epoch >= 80:
            train_predict = train_predict.set_value(val_id,
                                                    'predict1',
                                                    train_predict['predict1'].iloc[val_id] + np.array(val_answer))
            test_predict = test_predict.set_value(range(len(test_predict)),
                                                  'predict1',
                                                  test_predict['predict1'] + np.array(test_answer)/10)




            train_predict.to_csv('train_CNN.csv',index=0)
            test_predict.to_csv('test_CNN.csv', index=0)


