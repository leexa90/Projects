import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
# calculate the volume of the structure

'''
added volume
added bond angles
added dihe 

'''
train  = pd.read_csv('train.csv')
train2 = pd.read_csv('train_data_features2.csv')
test = pd.read_csv('test.csv')
test2 = pd.read_csv('test_data_features2.csv')

train2['id'] = train2['5']

def get_vol(a, b, c, alpha, beta, gamma):
    """
    Args:
        a (float) - lattice vector 1
        b (float) - lattice vector 2
        c (float) - lattice vector 3
        alpha (float) - lattice angle 1 [radians]
        beta (float) - lattice angle 2 [radians]
        gamma (float) - lattice angle 3 [radians]
    Returns:
        volume (float) of the parallelepiped unit cell
    """
    return a*b*c*np.sqrt(1 + 2*np.cos(alpha)*np.cos(beta)*np.cos(gamma)
                           - np.cos(alpha)**2
                           - np.cos(beta)**2
                           - np.cos(gamma)**2)

# convert lattice angles from degrees to radians for volume calculation
lattice_angles = ['lattice_angle_alpha_degree', 'lattice_angle_beta_degree',
                  'lattice_angle_gamma_degree']
for lang in lattice_angles:
    train[lang+'_r'] = np.pi * train[lang] / 180
    test[lang+'_r'] = np.pi * test[lang] / 180
    
    
# compute the cell volumes 
train['vol'] = get_vol(train['lattice_vector_1_ang'],
                        train['lattice_vector_2_ang'],
                        train['lattice_vector_3_ang'],
                        train['lattice_angle_alpha_degree_r'],
                        train['lattice_angle_beta_degree_r'],
                        train['lattice_angle_gamma_degree_r'])
test['vol'] = get_vol(test['lattice_vector_1_ang'],
                        test['lattice_vector_2_ang'],
                        test['lattice_vector_3_ang'],
                        test['lattice_angle_alpha_degree_r'],
                        test['lattice_angle_beta_degree_r'],
                        test['lattice_angle_gamma_degree_r'])
for lang in lattice_angles:
    del train[lang+'_r']
    del test[lang+'_r']
del train2['5']
test2['id'] = test2['5']
del test2['5']
train = pd.merge(train,train2,on='id')
test = pd.merge(test,test2,on='id')
extra_feature = [x for x in train.keys() if '(' in x and 'Bond' not in x]
for i in extra_feature:
    train['N'+i] = train[i]/train['vol']
    test['N'+i] = test[i]/test['vol']
# bond angles
train_angle = np.load('train_bond_angles.npy').item()
test_angle = np.load('test_bond_angles.npy').item()
def more_less_than(x,a,b):
    return np.mean((x > a) * (x < b))
for i in [('Al', 'Ga'), ('Al', 'In'), ('Ga', 'In'),
          ('Al', 'Al'), ('Ga', 'Ga'), ('In', 'In'),
          ('O', 'O')]:
    temp = []
    all_angle = []
    all_list = []
    train[str(i)+'_A_mean'] = 0
    train[str(i)+'_A_std'] = 0
    train['array_'+str(i)] = 0
    for j in train_angle.keys():
        temp += [[np.mean(train_angle[j][i]),np.std(train_angle[j][i])],];
        all_angle += train_angle[j][i]
        all_list += [np.array(train_angle[j][i]),]
    all_angle = [ x for x in all_angle if x  <= 500 and x >=-500];
    plt.hist(all_angle,bins = 100);plt.savefig(str(i)+'.png');plt.close()
    train = train.set_value(train.index,[str(i)+'_A_mean',str(i)+'_A_std'],temp)
    train = train.set_value(train.index,'array_'+str(i),all_list)
    def percentile(x,per):
        if x != []:
            return np.percentile(x,per)
    train[str(i)+'_A_75'] = train['array_'+str(i)].map(lambda x : percentile(x,75))
    train[str(i)+'_A_50'] = train['array_'+str(i)].map(lambda x : percentile(x,50))
    train[str(i)+'_A_25'] = train['array_'+str(i)].map(lambda x : percentile(x,25))
    
    temp = []
    all_list = []
    test[str(i)+'_A_mean'] = 0
    test[str(i)+'_A_std'] = 0
    test['array_'+str(i)] = 0
    for j in test_angle.keys():
        temp += [[np.mean(test_angle[j][i]),np.std(test_angle[j][i])],]
        all_list += [np.array(test_angle[j][i]),]
    test = test.set_value(test.index,[str(i)+'_A_mean',str(i)+'_A_std'],temp)    
    test = test.set_value(test.index,'array_'+str(i),all_list)
    test[str(i)+'_A_75'] = test['array_'+str(i)].map(lambda x : percentile(x,75))
    test[str(i)+'_A_50'] = test['array_'+str(i)].map(lambda x : percentile(x,50))
    test[str(i)+'_A_25'] = test['array_'+str(i)].map(lambda x : percentile(x,25))
    for ii in range(75,180,10):
        train[str(i)+'_%s_%s'%(ii,ii+10)] = train['array_'+str(i)].apply(lambda x : more_less_than(x,ii,ii+10))
        test[str(i)+'_%s_%s' %(ii,ii+10)] = test['array_'+str(i)].apply(lambda x : more_less_than(x,ii,ii+10))
        if np.mean(train[str(i)+'_%s_%s'%(ii,ii+10)]) == 0:
            del train[str(i)+'_%s_%s'%(ii,ii+10)]
            del test[str(i)+'_%s_%s'%(ii,ii+10)]
            print str(i)+'_%s_%s'%(ii,ii+10)

train_bond = np.load('train_bond_distribution.npy').item()
test_bond = np.load('test_bond_distribution.npy').item()
for i in [('Al', 'O'),('Ga', 'O'),('In', 'O')]:
    temp = []
    train['array_'+str(i)] = 0
    for j in train_bond.keys():
        temp += [np.array(train_bond[j][i]),]
    train = train.set_value(train.index, 'array_'+str(i),temp)
    def percentile(x,per):
        if x != []:
            return np.percentile(x,per)
    train[str(i)+'_B_75'] = train['array_'+str(i)].map(lambda x : percentile(x,75))
    train[str(i)+'_B_50'] = train['array_'+str(i)].map(lambda x : percentile(x,50))
    train[str(i)+'_B_25'] = train['array_'+str(i)].map(lambda x : percentile(x,25))
    temp = []
    test['array_'+str(i)] = 0
    for j in test_bond.keys():
        temp += [np.array(test_bond[j][i]),]
    test = test.set_value(test.index, 'array_'+str(i),temp)
    test[str(i)+'_B_75'] = test['array_'+str(i)].map(lambda x : percentile(x,75))
    test[str(i)+'_B_50'] = test['array_'+str(i)].map(lambda x : percentile(x,50))
    test[str(i)+'_B_25'] = test['array_'+str(i)].map(lambda x : percentile(x,25))
    for ii in np.linspace(1.5,2.7,13):
        train[str(i)+'_%s_%s'%(ii,ii+0.1)] = train['array_'+str(i)].apply(lambda x : more_less_than(x,ii,ii+0.1))
        test[str(i)+'_%s_%s'%(ii,ii+0.1)] = test['array_'+str(i)].apply(lambda x : more_less_than(x,ii,ii+0.1))
        if np.mean(train[str(i)+'_%s_%s'%(ii,ii+0.1)] ) == 0:
            del train[str(i)+'_%s_%s'%(ii,ii+0.1)]
            del test[str(i)+'_%s_%s'%(ii,ii+0.1)]
            print str(i)+'_%s_%s'%(ii,ii+0.1)

### get dihedrals ###
train_dihe = np.load('train_dihe.npy').item()
test_dihe = np.load('test_dihe.npy').item()

train['array_dihe'] = train['id'].map(train_dihe)
train['dihe_mean'] =  train['array_dihe'].map(lambda x : np.array(x)[np.array(x) >= -0.001]).map(lambda x : np.mean(x))
train['dihe_std'] =  train['array_dihe'].map(lambda x : np.array(x)[np.array(x) >= -0.001]).map(lambda x : np.std(x))
train['dihe_nan'] =  train['array_dihe'].map(lambda x : 1.0*(len(x)-len(np.array(x)[np.array(x) >= -0.001]))/len(x))
train['dihe_25'] =  train['array_dihe'].map(lambda x : np.array(x)[np.array(x) >= -0.001]).map(lambda x : np.percentile(x,25))
train['dihe_50'] =  train['array_dihe'].map(lambda x : np.array(x)[np.array(x) >= -0.001]).map(lambda x : np.percentile(x,50))
train['dihe_75'] =  train['array_dihe'].map(lambda x : np.array(x)[np.array(x) >= -0.001]).map(lambda x : np.percentile(x,75))
for ii in np.linspace(0,180,10)[:-1]:
    diff= np.linspace(0,180,10)[1]-np.linspace(0,180,10)[0]
    train['dihe_%s_%s'%(ii,ii+diff)] = train['array_dihe'].apply(lambda x : more_less_than(x,ii,ii+diff))
test['array_dihe'] = test['id'].map(train_dihe)
test['dihe_mean'] = test['array_dihe'].map(lambda x : np.array(x)[np.array(x) >= -0.001]).map(lambda x : np.mean(x))
test['dihe_std'] = test['array_dihe'].map(lambda x : np.array(x)[np.array(x) >= -0.001]).map(lambda x : np.std(x))
test['dihe_nan'] = test['array_dihe'].map(lambda x : 1.0*(len(x)-len(np.array(x)[np.array(x) >= -0.001]))/len(x))
test['dihe_25'] = test['array_dihe'].map(lambda x : np.array(x)[np.array(x) >= -0.001]).map(lambda x : np.percentile(x,25))
test['dihe_50'] = test['array_dihe'].map(lambda x : np.array(x)[np.array(x) >= -0.001]).map(lambda x : np.percentile(x,50))
test['dihe_75'] = test['array_dihe'].map(lambda x : np.array(x)[np.array(x) >= -0.001]).map(lambda x : np.percentile(x,75))

for ii in np.linspace(0,180,10)[:-1]:
    diff= np.linspace(0,180,10)[1]-np.linspace(0,180,10)[0]
    test['dihe_%s_%s'%(ii,ii+diff)] = test['array_dihe'].apply(lambda x : more_less_than(x,ii,ii+diff))
target1 = 'formation_energy_ev_natom'
target2 = 'bandgap_energy_ev'
cols = [x for x in train.keys() if (x not in ['id',target1,target2] and 'array' not in x)]
train[target1] = np.log1p(train[target1])
train[target2] = np.log1p(train[target2])
train['predict1'] = 0
test[target1] = 0
train['predict2'] = 0
test[target2] = 0
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
dictt_EN = {'O' : 3.44, 'In': 1.78, 'Al' : 1.61, 'Ga' : 1.81}
dictt_cols1 = pd.DataFrame(cols + ['z1','z2'],columns=[1])
dictt_cols2 = pd.DataFrame(cols + ['z1','z2'],columns=[1])

train_ori = train.copy(deep = True)
test_ori = test.copy(deep = True)


cols_ori = list(np.copy(cols))
seeds = [1,5516,643,5235,2352,12,5674,19239,41241,1231,151,34,1235,2664,75764,2314][:3]
print seeds,len(seeds)
comps=30
from sklearn.metrics import normalized_mutual_info_score
dictt =[]
for ii in range(len(cols)):      
    i = cols[ii]
    a= normalized_mutual_info_score(train[i],train[target1])
    b = normalized_mutual_info_score(train[i],train[target2])
    dictt += [[a,b,i],]
#print sorted(dictt,key = lambda x : x[0]+x[1])
for ii in range(30):
    for jj in range(ii+1,30):
        i = [x[2] for x in  sorted(dictt,key = lambda x : x[0]+x[1])[-30:]][ii]
        j = [x[2] for x in  sorted(dictt,key = lambda x : x[0]+x[1])[-30:]][jj]
        a1= normalized_mutual_info_score(train[i],train[target1])
        b1 = normalized_mutual_info_score(train[j],train[target1])
        a2= normalized_mutual_info_score(train[i],train[target2])
        b2 = normalized_mutual_info_score(train[j],train[target2]) 
        c1 = normalized_mutual_info_score(train[i]/train[j],train[target1])
        d1 = normalized_mutual_info_score(train[j]/train[i],train[target1])
        e1 = normalized_mutual_info_score(train[i]*train[j],train[target1])
        f1 = normalized_mutual_info_score(train[i]-train[j],train[target1])
        c2 = normalized_mutual_info_score(train[i]/train[j],train[target2])
        d2 = normalized_mutual_info_score(train[j]/train[i],train[target2])
        e2 = normalized_mutual_info_score(train[i]*train[j],train[target2])
        f2 = normalized_mutual_info_score(train[i]-train[j],train[target2])
        if max([c1,d1,e1,f1]) > max([a1,b1]):
            if max([a1,b1])/max([c1,d1,e1,f1]) < 0.9985:
                print i,j ,max([a1,b1])/max([c1,d1,e1,f1])
        if max([c2,d2,e2,f2]) > max([a2,b2]):
            if max([a2,b2])/max([c2,d2,e2,f2]) < 0.9985:
                print i,j ,max([a2,b2])/max([c2,d2,e2,f2])
        
die
for seed in seeds:
    train = train.sample(2400,random_state=seed).reset_index(drop=True)
    for i in range(0,2):

        test_id = [x for x in range(0,2400) if x%2 == i] 
        train_id = [x for x in range(0,2400) if x%2 != i] 

        X_train, y_train1,y_train2 = train.iloc[train_id][cols], train.iloc[train_id][target1],\
                                     train.iloc[train_id][target2]
        X_test, y_test1,y_test2 = train.iloc[test_id][cols], train.iloc[test_id][target1],\
                                  train.iloc[test_id][target2]
        

'''
251                      dihe_20.0_40.0   130.012179
121                   ('Al', 'Ga')_A_50   130.798634
61                          IonChar_std   132.210442
196                    ('O', 'O')_85_95   137.731206
244                           dihe_mean   139.551943
210  ('Al', 'O')_1.9000000000000001_2.0   145.527317
204                    ('Al', 'O')_B_75   146.294590
10           lattice_angle_gamma_degree   148.565780
194                     ('O', 'O')_A_25   157.148507
157                   ('Al', 'Al')_A_50   164.906867
245                            dihe_std   170.060184
232                    ('In', 'O')_B_25   171.594448
8            lattice_angle_alpha_degree   172.645551
11                                  vol   176.503351
250                       dihe_0.0_20.0   204.677845
249                             dihe_75   250.919187
259                                  z1   338.325087
66                          period_mean   357.524081
67                           period_std   377.370168
260                                  z2  1693.477356
                                      1          avg
134                   ('Al', 'In')_A_25   132.217341
139                ('Al', 'In')_115_125   132.758748
206                    ('Al', 'O')_B_25   133.874317
58                Bond_('In', 'O')_mean   136.235346
190                   ('O', 'O')_A_mean   139.878442
256                    dihe_120.0_140.0   141.173582
250                       dihe_0.0_20.0   142.106365
9             lattice_angle_beta_degree   145.640964
8            lattice_angle_alpha_degree   159.974783
84                     N('Al', 'In', 0)   164.036258
59                 Bond_('In', 'O')_std   168.165314
14                      ('Al', 'Ga', 2)   168.993687
55                 Bond_('Al', 'O')_std   172.226444
244                           dihe_mean   175.073378
78                     N('Al', 'Ga', 2)   177.603239
180                   ('In', 'In')_A_75   178.861538
245                            dihe_std   184.683608
249                             dihe_75   281.730499
210  ('Al', 'O')_1.9000000000000001_2.0   305.131766
259                                  z1  1285.689419
'''
