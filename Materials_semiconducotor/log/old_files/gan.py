import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import tensorflow as tf
import keras as keras
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
# calculate the volume of the structure

train  = pd.read_csv('train_v2_nan.csv')
test = pd.read_csv('test_v2_nan.csv')
target1 = 'formation_energy_ev_natom'
target2 = 'bandgap_energy_ev'
'''
convert spacegroup to one-hot-encoded
'''
all_dummy = pd.get_dummies(pd.concat([train['spacegroup'],test['spacegroup']]))

train[map(str,all_dummy.keys())] = all_dummy.iloc[:len(train)]
test[map(str,all_dummy.keys())] = all_dummy.iloc[len(train):]
cols = [x for x in train.keys() if (x not in ['id',target1,target2,'soacegroup'] and 'array' not in x and 'predict' not in x)]

for i in cols:
    train[i] = train[i].fillna(np.mean(train[i]))
    test[i] = test[i].fillna(np.mean(train[i]))
# 157 features
cols = ["('Al', 'Al')_115_125", "('Al', 'Al')_125_135", "('Al', 'Al')_85_95", "('Al', 'Al')_95_105",
        "('Al', 'Al')_A_mean", "('Al', 'Al')_A_std", "('Al', 'Al', 0)", "('Al', 'Al', 2)",
        "('Al', 'Al', 3)", "('Al', 'Ga')_115_125", "('Al', 'Ga')_125_135", "('Al', 'Ga')_85_95",
        "('Al', 'Ga')_95_105", "('Al', 'Ga')_A_mean", "('Al', 'Ga')_A_std", "('Al', 'Ga', 0)",
        "('Al', 'Ga', 1)", "('Al', 'Ga', 2)", "('Al', 'Ga', 3)", "('Al', 'In')_105_115",
        "('Al', 'In')_115_125", "('Al', 'In')_125_135", "('Al', 'In')_85_95", "('Al', 'In')_95_105",
        "('Al', 'In')_A_mean", "('Al', 'In')_A_std", "('Al', 'In', 0)", "('Al', 'In', 2)",
        "('Al', 'In', 3)", "('Al', 'O')_1.8_1.9", "('Al', 'O')_1.9_2.0", "('Al', 'O')_2.0_2.1",
        "('Al', 'O')_2.1_2.2", "('Al', 'O')_2.2_2.3", "('Al', 'O', 0)", "('Al', 'O', 1)",
        "('Al', 'O', 2)", "('Al', 'O', 3)", "('Al', 'O', 4)", "('Ga', 'Ga')_115_125", "('Ga', 'Ga')_125_135",
        "('Ga', 'Ga')_85_95", "('Ga', 'Ga')_95_105", "('Ga', 'Ga')_A_mean", "('Ga', 'Ga')_A_std",
        "('Ga', 'Ga', 0)", "('Ga', 'In')_105_115", "('Ga', 'In')_115_125", "('Ga', 'In')_125_135",
        "('Ga', 'In')_85_95", "('Ga', 'In')_95_105", "('Ga', 'In')_A_mean", "('Ga', 'In')_A_std",
        "('Ga', 'O')_1.8_1.9", "('Ga', 'O')_1.9_2.0", "('Ga', 'O')_2.0_2.1", "('Ga', 'O')_2.1_2.2",
        "('Ga', 'O', 0)", "('Ga', 'O', 2)", "('Ga', 'O', 3)", "('Ga', 'O', 4)", "('In', 'Ga', 0)",
        "('In', 'Ga', 2)", "('In', 'Ga', 3)", "('In', 'In')_105_115", "('In', 'In')_115_125",
        "('In', 'In')_125_135", "('In', 'In')_85_95", "('In', 'In')_95_105", "('In', 'In')_A_mean",
        "('In', 'In')_A_std", "('In', 'In', 0)", "('In', 'In', 1)", "('In', 'In', 3)", "('In', 'In', 4)",
        "('In', 'O')_1.7_1.8", "('In', 'O')_1.8_1.9", "('In', 'O')_1.9_2.0", "('In', 'O')_2.0_2.1",
        "('In', 'O')_2.1_2.2", "('In', 'O')_2.2_2.3", "('In', 'O', 0)", "('In', 'O', 2)", "('In', 'O', 3)",
        "('O', 'O')_105_115", "('O', 'O')_155_165", "('O', 'O')_175_185", "('O', 'O')_75_85",
        "('O', 'O')_85_95", "('O', 'O')_95_105", "('O', 'O')_A_mean", "('O', 'O')_A_std", "('O', 'O', 1)",
        "('O', 'O', 2)", "Bond_('Al', 'O')_mean", "Bond_('Al', 'O')_std", "Bond_('Ga', 'O')_mean",
        "Bond_('Ga', 'O')_std", "Bond_('In', 'O')_mean", "Bond_('In', 'O')_std", 'IonChar_mean', 'IonChar_std',
        "N('Al', 'Al', 0)", "N('Al', 'Al', 2)", "N('Al', 'Al', 3)", "N('Al', 'Ga', 0)", "N('Al', 'Ga', 1)",
        "N('Al', 'Ga', 2)", "N('Al', 'Ga', 3)", "N('Al', 'In', 0)", "N('Al', 'In', 2)", "N('Al', 'In', 3)",
        "N('Al', 'O', 0)", "N('Al', 'O', 1)", "N('Al', 'O', 2)", "N('Al', 'O', 3)", "N('Al', 'O', 4)",
        "N('Ga', 'Ga', 0)", "N('Ga', 'Ga', 2)", "N('Ga', 'Ga', 3)", "N('Ga', 'O', 0)", "N('Ga', 'O', 2)",
        "N('Ga', 'O', 3)", "N('Ga', 'O', 4)", "N('In', 'Ga', 0)", "N('In', 'Ga', 2)", "N('In', 'Ga', 3)",
        "N('In', 'In', 0)", "N('In', 'In', 1)", "N('In', 'In', 2)", "N('In', 'In', 3)", "N('In', 'In', 4)",
        "N('In', 'O', 0)", "N('In', 'O', 1)", "N('In', 'O', 2)", "N('In', 'O', 3)", "N('O', 'O', 0)",
        "N('O', 'O', 1)", "N('O', 'O', 2)", 'Norm2', 'Norm3', 'Norm5', 'lattice_angle_alpha_degree',
        'lattice_angle_beta_degree', 'lattice_angle_gamma_degree', 'lattice_vector_1_ang', 'lattice_vector_2_ang',
        'lattice_vector_3_ang', 'percent_atom_al', 'percent_atom_ga', 'percent_atom_in', 'period_mean', 'period_std',
        'spacegroup', 'vol'][::3]
train_ori = train.copy(deep = True)
cols_ori = list(np.copy(cols))
seeds = [1,5516,643,5235,2352,12,5674,19239,41241,1231,151,34,1235,2663,75765,2314][:1]
print seeds,len(seeds)
bad = []
train['pred1'] = 0
train['pred2'] = 0
test['pred1'] = 0
test['pred2'] = 0
num = 2
for seed in seeds:
    train = train.sample(2400,random_state=seed+1).reset_index(drop=True)
    for i in range(0,num):
        test_id = [x for x in range(0,2400) if x%num == i] 
        train_id = [x for x in range(0,2400) if x%num != i] 
        scaler = StandardScaler()
        regr = linear_model.LinearRegression()
        scaler = scaler.fit(train.iloc[train_id][cols_ori].values,train[target1].values)
        train['z1'] = regr.fit(scaler.transform(train.iloc[train_id][cols_ori].values),
                               train.iloc[train_id][target1].values).predict(scaler.transform(train[cols_ori].values))
        train['z2'] = regr.fit(scaler.transform(train.iloc[train_id][cols_ori].values),
                               train.iloc[train_id][target2].values).predict(scaler.transform(train[cols_ori].values))
        test['z1'] = regr.fit(scaler.transform(train.iloc[train_id][cols_ori].values),
                              train.iloc[train_id][target1].values).predict(scaler.transform(test[cols_ori].values))
        test['z2'] = regr.fit(scaler.transform(train.iloc[train_id][cols_ori].values),
                              train.iloc[train_id][target2].values).predict(scaler.transform(test[cols_ori].values))
        if 'z1' not in cols:
            cols += ['z1','z2']
        X_train, y_train1,y_train2 = train.iloc[train_id][cols], train.iloc[train_id][target1],\
                                     train.iloc[train_id][target2]
        X_test, y_test1,y_test2 = train.iloc[test_id][cols], train.iloc[test_id][target1],\
                                  train.iloc[test_id][target2]
        #generative network
        from keras import backend as K
        K.clear_session()

        def generator_func(generator_input):
            layer0 = tf.layers.batch_normalization(generator_input,trainable=False,training=False)
            layer1 = tf.layers.dense(layer0,train[cols].values.shape[1],activation=tf.nn.relu, use_bias=True)
            layer1 = tf.layers.batch_normalization(layer1,trainable=True,training=True)
            layer1 = tf.layers.dropout(layer1,rate=0.2)
            layer2 = tf.layers.dense(layer1,train[cols].values.shape[1],activation=tf.nn.relu, use_bias=True)
            layer2 = tf.layers.dropout(layer2,rate=0.2)
            layer2 = tf.layers.batch_normalization(layer2,trainable=True,training=True)
            layer3 = tf.layers.dense(layer2,train[cols].values.shape[1],activation=tf.nn.relu, use_bias=True)
            layer3 = tf.layers.dropout(layer3,rate=0.2)
            return tf.layers.dense(layer3,train[cols].values.shape[1],activation=None, use_bias=True)
        def discriminator_func(discriminator_input):
            Dlayer0 = tf.layers.batch_normalization(discriminator_input,trainable=False,training=False)
            Dlayer1 = tf.layers.dense(Dlayer0,train[cols].values.shape[1],activation=tf.nn.relu, use_bias=True)
            Dlayer1 = tf.layers.batch_normalization(Dlayer1,trainable=True,training=True)
            Dlayer1 = tf.layers.dropout(Dlayer1,rate=0.2)
            Dlayer2 = tf.layers.dense(Dlayer1,train[cols].values.shape[1],activation=tf.nn.relu, use_bias=True)
            Dlayer2 = tf.layers.dropout(Dlayer2,rate=0.2)
            Dlayer2 = tf.layers.batch_normalization(Dlayer2,trainable=True,training=True)
            Dlayer3 = tf.layers.dense(Dlayer2,train[cols].values.shape[1],activation=tf.nn.relu, use_bias=True)
            Dlayer3 = tf.layers.dropout(Dlayer3,rate=0.2)
            return tf.sigmoid(tf.layers.dense(Dlayer3,1,activation=None, use_bias=True))
        with tf.variable_scope('G'):
            generator_input = tf.placeholder(tf.float32,[None,train[cols].values.shape[1]],
                                             name='generator_input')
            generator = generator_func(generator_input)
        with tf.variable_scope('D') as scope:
            #disriminator #real are 1, generated = 0
            discriminator_input = tf.placeholder(tf.float32,[None,train[cols].values.shape[1]],
                                                     name='discrminator_input')
            discriminator_real  = discriminator_func(discriminator_input)
        with tf.variable_scope('D',reuse=True) as scope:
            discriminator_input_fake = tf.placeholder(tf.float32,[None,train[cols].values.shape[1]],
                                                     name='discrminator_input_fake')
            discriminator_fake = discriminator_func(discriminator_input_fake)
            
                                        
        

        loss_d = tf.reduce_mean(-tf.log(discriminator_real)+\
                                -tf.log(1-discriminator_fake))
        loss_g = tf.reduce_mean(-tf.log(discriminator_fake))
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        lr_d = tf.Variable(0,dtype = tf.float32,trainable =False,
                                             name='lr_d')
        lr_g = tf.Variable(0,dtype = tf.float32,trainable =False,
                                             name='lr_g')
        with tf.control_dependencies(update_ops):
            d_optimizer = tf.train.MomentumOptimizer(learning_rate=lr_d,momentum=0.5).minimize(loss_d)
            g_optimizer = tf.train.MomentumOptimizer(learning_rate=lr_g,momentum=0.5).minimize(loss_g)
            dg_optimizer = tf.train.MomentumOptimizer(learning_rate=lr_g+lr_d,momentum=0.5).minimize(loss_g+loss_d)
        init = tf.global_variables_initializer();sess = tf.Session();sess.run(init)
        for epoch in range(0,1000):
            avg_g,avg_d = 0,0
            prob_g_avg,prob_d_avg = [] , []
            train = train.sample(2400,random_state=seed+1).reset_index(drop=True)
            for num in range(0,64*(len(train)//64),64):
                REAL = train[cols].iloc[num:num+64].values
                NOISE = np.random.uniform(0,1,(64,len(cols))) #noise vector
                FAKE  = sess.run(generator, feed_dict= { generator_input:NOISE, 
                                                 discriminator_input : REAL,lr_d : 0.00001,lr_g : 0.00001})
                for i in range(0,3):
                    FAKE,_ = sess.run([generator,g_optimizer],
                                    feed_dict= { generator_input:NOISE, discriminator_input_fake:FAKE,
                                                 discriminator_input : REAL,lr_d : 0.0001,lr_g : 0.00003})
                __,c_g,c_d,prob_g,prob_d = sess.run([d_optimizer,loss_d,loss_g,discriminator_fake,discriminator_real],
                                        feed_dict= { generator_input:NOISE,discriminator_input_fake:FAKE, discriminator_input : REAL,lr_d : 0.0001,lr_g : 0.00001})
                avg_g += c_g #real
                avg_d += c_d #fake
                prob_g_avg += list(prob_g)
                prob_d_avg += list(prob_d)
            print np.mean(avg_g),np.mean(avg_d),np.mean(prob_g_avg),np.mean(prob_d_avg)
