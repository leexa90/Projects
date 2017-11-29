# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import  numpy as np
import tensorflow as tf
import pandas as pd
print '### TENSORFLOW VERSION %s ###'%tf.__version__
float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
dictt = {'A': 0, 'C': 1, 'E': 2, 'D': 3, 'G': 4,
         'F': 5, 'I': 6, 'H': 7, 'K': 8, 'M': 9,
         'L': 10, 'N': 11, 'Q': 12, 'P': 13, 'S': 14,
         'R': 15, 'T': 16, 'W': 17, 'V': 18, 'Y': 19, 'X' :20  }
import xgboost 
#Interface Scale
#ΔGwif (kcal/mol) 	Octanol Scale
#ΔGwoct (kcal/mol) 	Octanol − Interface
#Scale

dictt_hydropathy  = {
'I' : [ 	-0.31, 	-1.12, 	-0.81 ,0 ],
'L' : [ 	-0.56,	-1.25, 	-0.69 ,0],
'F' : [ 	-1.13, 	-1.71,	-0.58 ,0],
'V' : [ 	0.07 ,	-0.46, 	-0.53 ,0],
'M' : [ 	-0.23, 	-0.67,	-0.44 ,0],
'P' : [ 	0.45 ,	0.14 ,	-0.31 ,0],
'W' : [ 	-1.85, 	-2.09, 	-0.24 ,0],
'T' : [ 	0.14 , 	0.25 , 	0.11 ,0],
'Q' : [ 	0.58 ,	0.77 ,	0.19 ,0],
'C' : [ 	-0.24,	-0.02, 	0.22 ,0],
'Y' : [ 	-0.94, 	-0.71, 	0.23 ,0],
'A' : [ 	0.17 ,	0.50, 	0.33 ,0],
'S' : [ 	0.13 ,	0.46, 	0.33 ,0],
'N' : [ 	0.42 ,	0.85, 	0.43 ,0],
'R' : [ 	0.81 ,	1.81, 	1.00 ,1],
'G' : [ 	0.01 ,	1.15, 	1.14 ,0],
'H' : [ 	0.96 ,	2.33, 	1.37 ,0],
'E' : [ 	2.02 ,	3.63, 	1.61 ,-1],
'K' : [ 	0.99 ,	2.80, 	1.81 ,1],
'D' : [ 	1.23 ,	3.64, 	2.41 ,-1],
'Z' : [ 1.3,  2.2,  0.9, -0.5], #E or Q (mass spec data cannot differentiate)
'B' : [ 0.825,  2.245,  1.42 , -0.5  ], #D or N (mass spec data cannot differentiate)
'X' : [0.0,0.0,0.0,0.0]} 
data = pd.read_csv('Small_public.csv').dropna()

def fn1(str):
    num=0.0
    for i in str:
        if i.upper() == 'R' or i.upper() == 'K':
            num += 1
    if num/len(str) > .334: #should be tweaked
        return 0
    return len(set(str))
def fn2(str): # find if sequence has weird bonds
    for i in list(set(str)):
        if i.upper() == '-':
            return 2
    for i in list(set(str)):
        if i.upper() not in dictt or i.upper()=='X':
            return 1    
    return 0
#notable non-cannonical residues , U - selnocysteine
data['type_aa'] = data['seq'].apply(fn1)
data = data[data['type_aa'] >= 4]
print 'removing these number of non-cannonical peptides' ,len(data[data['seq'].apply(fn2) == 1])
data = data[data['seq'].apply(fn2) == 0] # remove peptides with weird chemical bonds , and non-cannonical res (mostly negavtives)
data['len'] = data.seq.apply(len)
data = data[data['len'] <=30]
data = data.sort_values(by = ['len','source']).reset_index(drop=True)
# get frequencies of amino acids
all = np.concatenate([data.seq])
result = {}
for i in all:
	for j in i :
		if j not in result:
			result[j] =1
		else:
			result[j] += 1
print 'different aa frequencies',result
total_aa = np.sum([result[x] for x in result])
for i in result:
	result[i] = np.round(result[i]*1.0 /total_aa,3)
print 'different aa frequencies normalized',result
######### Start of initilziing data ###
X = []
counter = 0
## creates num of residue...
features = []
for i in dictt:
    def fn(str):
        result = 0.0
        for j in str:
            if j.upper() == i:
                result += 1
        return result
    if  i != 'X':
        data['num_'+str(i)] = data['seq'].apply(fn)
        data['per_'+str(i)] = data['num_'+str(i)]/data['len']
        features += ['per_'+str(i),'num_'+str(i)]

for idx in range(len(data)):
    i = data.iloc[idx]['seq'].upper()
    ii = i
    i = i + 'X'*(30-len(i))
    temp = np.zeros((len(i),5))
    for j in  range(len(i)):
        res = dictt[i[j]]
        temp[j][0] = res
        #temp[j][-1] = len(i)*0.01
        temp[j][-4:] = dictt_hydropathy[i[j]]
    temp = temp.T
    alternative = [len(i),np.sum(temp[-1,:]),np.sum(temp[-2,:]),np.sum(temp[-3,:])]
    alternative += list(data.iloc[idx][features]) #add ratio for aa
    per = data.iloc[idx]['source'] == 2 
    X += [[temp[0],temp[1:],alternative,ii,per*1],]
    #print '> %s\n%s' %(i,i)
    counter += 1
data['X'] = X
print '''### DATA PREPROCESSING ENDS HERE ###\n### DATA STORED IN VARIBLE X ###'''
print '''MODEL STARTS HERE'''
batch_size = None
epsilon = 1e-3
def batch_normalization(x,name='batchnorm',feature_norm = False):
    if feature_norm : 
        x = tf.contrib.layers.layer_norm (x,trainable=False)
    else:
        x = tf.contrib.layers.layer_norm (x,trainable=False)
    return x

with tf.name_scope('inputs') as scope:
    Inp0 = tf.placeholder(tf.int32,[batch_size,None],name='sequence_factors1')
    Inp1 = tf.placeholder(tf.float32,[batch_size,4,None],name='sequence_factors2')
    labels = tf.placeholder(tf.float32 , [batch_size,1],name='labels')
    dropout = tf.placeholder(tf.float32,name='dropout')
    Inp2 = tf.placeholder(tf.float32, [batch_size,44] ,name = 'globa_seq_info')
    sequence_length = tf.placeholder(tf.int32,shape=(batch_size),name='sequence_lenght')

with tf.name_scope('embedding') as scope:
    aa_embeddings = tf.get_variable('aa_embeddings',[21, 5])
    embedded_word_ids = tf.gather(aa_embeddings,range(0,21))
    embed0 = tf.nn.embedding_lookup(aa_embeddings,Inp0,name='lookup')
    embed1 = tf.transpose(embed0,(0,2,1))
    unstack0 = tf.unstack(Inp1,axis=-2,name='unstack0')
    unstack1 = tf.unstack(embed1 , axis=-2,name='unstack1')
    layer0 = tf.stack(unstack0+unstack1,axis=1)


from tensorflow.contrib import rnn
with tf.name_scope('RNN') as scope:
    rnn_cell  = rnn.BasicLSTMCell(370,activation= tf.nn.relu)
    output_,state_= tf.nn.bidirectional_dynamic_rnn(rnn_cell,rnn_cell,
                                                    tf.transpose(layer0,(0,2,1)),
                                                    dtype=tf.float32,parallel_iterations=32,
                                                    sequence_length=sequence_length)
    last_output = tf.concat([state_[0][1],state_[1][1]],-1,name='last_output')
with tf.name_scope('dense') as scope:
    gbmp_extra = tf.concat([Inp2,last_output],axis = 1, name ='gbmp_extra')
    dense1 = tf.layers.dense(gbmp_extra,256,activation = tf.nn.relu , name = 'dense1' )
    layer6_DO = tf.layers.dropout(dense1,rate=dropout,name='Drop6',training=True)
    dense2 = tf.layers.dense(layer6_DO,128,activation = tf.nn.relu , name = 'dense2' )
    dense3 = tf.layers.dense(dense2,64,activation = None , name = 'dense3' )
    dense4 = tf.layers.dense(dense3,1 , name = 'dense4' )

with tf.name_scope('loss') as scope:
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                      logits = dense4,
                                                      name='loss')
with tf.name_scope('output') as scope:
    out_softmax = tf.nn.sigmoid(dense4)
#learning_rate = tf.Variable(0,dtype= np.float32)
mean_loss = tf.reduce_mean(loss)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
learning_rate = tf.Variable(0,dtype=tf.float32,name='learning_rate')

with tf.control_dependencies(update_ops):
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.5).minimize(mean_loss)
with tf.name_scope('accuracy') as scope:
    predict_boo = tf.greater(out_softmax,0.5)
    predict = tf.cast(predict_boo, np.float32)
    acc = tf.reduce_mean(tf.cast(tf.equal(labels,predict),tf.float32),name='accuracy')
import sklearn.metrics,random
saver = tf.train.Saver()
# Initializing the variables
init = tf.global_variables_initializer();sess = tf.Session();sess.run(init)
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

training_epochs  = 100

import sklearn.metrics,random

RESULT = {}
import sys
if len(sys.argv)==2:
    test = int(sys.argv[1])
else:
    test = 2
print "TEST SET IS  FOLD",test
all_data =[]
test_emsemble= []
def get_data_from_X(X,y,i): #get tensor inputs from X and y
    Inp0_ = np.array([X[i][0]])
    Inp1_ = np.array([X[i][1]])
    Inp2_ = np.array([X[i][2]])
    labels_ = np.array([[y[i],]])
    length_ = np.array([len(X[i][3])])
    return Inp0_,Inp1_,Inp2_,labels_,length_
def get_batch_from_X(Xy): #get batch tensor inputs from X and y
    Inp0_ = np.array([x[0][0] for x in Xy])
    Inp1_ = np.array([x[0][1] for x in Xy])
    Inp2_ = np.array([x[0][2] for x in Xy])
    labels_ = np.array([[x[1] for x in Xy],])
    length_ = np.array([len(x[0][3]) for x in Xy])
    return Inp0_,Inp1_,Inp2_,labels_.T,length_
folds= 5
# ensure no leakage of train to test, was getting 95 AUC
X_test = []
y_test = []
print 'train_val_test size:' , len(X)
for i in range(test,len(X),folds):
    if i%5 == test:
        x = X[i]
        X_test += [(x[0],x[1],x[2],x[3]),]
        y_test += [x[-1],]
        X[i] = 'test'
X = [x for x in X if x != 'test']
print 'train+val  size :', len(X)
saver = tf.train.Saver( max_to_keep=5000)
repeat = 0
for repeat in range(0,1): #perform 5 repeats
    for CV in range(folds-1): #for each repeat, do 4 fold CV. (test set is kept constant throughtout)# 
        init = tf.global_variables_initializer();sess = tf.Session();sess.run(init)
        RESULT[CV] = []
        X_train = []
        y_train = []
        X_train_weighted = []
        y_train_weighted = []
        X_val = []
        y_val = []    
        for i in range(len(X)):
            x = X[i]
            if i%4 == CV:
                X_val += [(x[0],x[1],x[2],x[3]),]
                y_val += [x[-1],]
            else:
                if x[-1] == 1:
                    mul = 18 #ratio of pos to neg is 1:18.4, 
                else: mul = 1
                X_train += [(x[0],x[1],x[2],x[3]),]
                y_train += [x[-1],]
                X_train_weighted += [(x[0],x[1],x[2],x[3]),]*mul
                y_train_weighted += [x[-1],]*mul
        print 'size of different sets Train/val/test:',len(X_train_weighted),len(X_val),len(X_test)

        best_roc_val = {}     # stores val ROC for trainnig epochs per CV+repeat run
        for epoch in range(training_epochs):#training_epochs):
            if epoch%50 ==0 :
                init = tf.global_variables_initializer();sess = tf.Session();sess.run(init)
            logit_train = []
            cost_train = []
            random.seed(epoch)
            sorted_data = zip(X_train_weighted,y_train_weighted)
            shuffle = sorted_data #shuffle index 
            random.shuffle(shuffle)
            counter = 0
            for i in range(0,len(shuffle),32): #training with bagging
                lr = 0.012*np.abs(np.cos(0.5*3.142*counter/len(shuffle[::32])))
                counter += 1
                Inp0_,Inp1_,Inp2_,labels_,length_ = get_batch_from_X(sorted_data[i:i+32])       
                _, c = sess.run([optimizer, acc], feed_dict={Inp0: Inp0_,Inp2: Inp2_,
                                                                   Inp1: Inp1_,sequence_length:length_,
                                                                   labels: labels_,
                                                                   dropout : 0.0,learning_rate : lr}) #sgd
            logit_train = []
            cost_train = []
            lr = 0
            sorted_data = zip(X_train,y_train)#sorted(zip(X_train,y_train),key = lambda x : x[0][-1][0])
            y_temp = []
            Inp0_,Inp1_,Inp2_,labels_,length_ = get_batch_from_X(sorted_data)
            c, out = sess.run ([acc, out_softmax],
                               feed_dict={Inp0: Inp0_,Inp1: Inp1_,Inp2: Inp2_,sequence_length : length_,
                                labels: labels_,dropout : 0,learning_rate : 0})
            cost_train += [c,]*len(Inp0_)
            logit_train += list(out[:,0])
            y_temp += list(labels_[:,0])
            roc_train   = sklearn.metrics.roc_auc_score(y_temp,logit_train)
                    
            logit_val = []
            cost_val = []
            sorted_data = zip(X_val,y_val)#sorted(zip(X_val,y_val),key = lambda x : x[0][-1][0])
            y_temp = []
            Inp0_,Inp1_,Inp2_,labels_,length_ = get_batch_from_X(sorted_data)
            c, out = sess.run ([acc, out_softmax],
                               feed_dict={Inp0: Inp0_,Inp1: Inp1_,Inp2: Inp2_,sequence_length : length_,
                                labels: labels_,dropout : 0,learning_rate : 0})
            cost_val += [c,]*len(Inp0_)
            logit_val += list(out[:,0])
            y_temp += list(labels_[:,0])
            roc_val   = sklearn.metrics.roc_auc_score(y_temp,logit_val)

            logit_test = []
            cost_test = []

            sorted_data = zip(X_test,y_test)
            y_temp = []
            Inp0_,Inp1_,Inp2_,labels_,length_ = get_batch_from_X(sorted_data)
            c, out = sess.run ([acc, out_softmax],
                               feed_dict={Inp0: Inp0_,Inp1: Inp1_,Inp2: Inp2_,sequence_length : length_,
                                labels: labels_,dropout : 0,learning_rate : lr})
            cost_test += [c,]*len(Inp0_)
            logit_test += list(out[:,0])
            y_temp += list(labels_[:,0])
            roc_test   = sklearn.metrics.roc_auc_score(y_temp,logit_test)
                    
            print np.mean(cost_train),np.mean(cost_val)
            print roc_train,roc_val,roc_test
            ### code below saves top 3 models of each iteration
            best_roc_val[epoch] = [roc_train,roc_val,roc_test,logit_test]
            all_data += [[roc_train,roc_val,roc_test],]
            if roc_val <= 0.5: # if val worst then chance, reinitalize tarining
                init = tf.global_variables_initializer();sess = tf.Session();sess.run(init)
            best_logit_test = sorted([best_roc_val[ep] for ep in best_roc_val], key = lambda x :x[1])[-3:]
            if len(best_logit_test) >=3 and best_logit_test[0][1] <= roc_val:
                model_name = './CNN_layer2_test.pyLSTM1_%s_%s_%s_%s_%s_%s.ckpt' %(test,CV,repeat,str(roc_train)[:5],str(roc_val)[:5],str(roc_test)[:5])
                #saver.save(sess,model_name),
                print 'SAVED\n'
        for j in best_logit_test:
            test_emsemble += [j[3],]
        print [x[1:3] for x in best_logit_test]
        print sklearn.metrics.roc_auc_score(y_test,np.mean(np.array(test_emsemble),0))

	
        
data['pred'] = 0
data.set_value(range(test,len(data),5),'pred',np.mean(np.array(test_emsemble),0))
data[['seq','source','pred']].iloc[range(test,len(data),5)].to_csv('model5_%s.csv'%test,index=0)
        
