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
train['CNN1']= 0
test['CNN1']= 0
train['CNN2']= 0
test['CNN2']= 0
for i in ['11/','12/','13/','14/','11B/','12B/','13B/','14B/','15A/'][:4]:
    #train[i] = pd.read_csv(i+'train_CNN.csv')['predict1']
    #test[i] = pd.read_csv(i+'test_CNN.csv')['predict1']
    train['CNN1'] =  train['CNN1']+pd.read_csv(i+'train_CNN.csv')['predict1']
    test['CNN1'] = test['CNN1']+pd.read_csv(i+'test_CNN.csv')['predict1']
for i in ['21/','22/','23/','24/','21B/','22B/','23B/'][:3]:
    #train[i] = pd.read_csv(i+'train_CNN.csv')['predict1']
    #test[i] = pd.read_csv(i+'test_CNN.csv')['predict1']
    train['CNN2'] = train['CNN2']+pd.read_csv(i+'train_CNN.csv')['predict1']
    test['CNN2'] = test['CNN2']+pd.read_csv(i+'test_CNN.csv')['predict1']

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
##    del train[i]
##    del test[i]
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
    #plt.hist(all_angle,bins = 100);plt.savefig(str(i)+'.png');plt.close()
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

##### get dihedrals ###
train_dihe = np.load('train_dihe.npy').item()
test_dihe = np.load('test_dihe.npy').item()

train['array_dihe'] = train['id'].map(train_dihe)
train['dihe_mean'] =  train['array_dihe'].map(lambda x : np.array(x)[np.array(x) >= -0.001]).map(lambda x : np.mean(x))
train['dihe_std'] =  train['array_dihe'].map(lambda x : np.array(x)[np.array(x) >= -0.001]).map(lambda x : np.std(x))
train['dihe_nan'] =  train['array_dihe'].map(lambda x : 1.0*(len(x)-len(np.array(x)[np.array(x) >= -0.001]))/len(x))
train['dihe_25'] =  train['array_dihe'].map(lambda x : np.array(x)[np.array(x) >= -0.001]).map(lambda x : np.percentile(x,25))
train['dihe_50'] =  train['array_dihe'].map(lambda x : np.array(x)[np.array(x) >= -0.001]).map(lambda x : np.percentile(x,50))
train['dihe_75'] =  train['array_dihe'].map(lambda x : np.array(x)[np.array(x) >= -0.001]).map(lambda x : np.percentile(x,75))

test['array_dihe'] = test['id'].map(train_dihe)
test['dihe_mean'] = test['array_dihe'].map(lambda x : np.array(x)[np.array(x) >= -0.001]).map(lambda x : np.mean(x))
test['dihe_std'] = test['array_dihe'].map(lambda x : np.array(x)[np.array(x) >= -0.001]).map(lambda x : np.std(x))
test['dihe_nan'] = test['array_dihe'].map(lambda x : 1.0*(len(x)-len(np.array(x)[np.array(x) >= -0.001]))/len(x))
test['dihe_25'] = test['array_dihe'].map(lambda x : np.array(x)[np.array(x) >= -0.001]).map(lambda x : np.percentile(x,25))
test['dihe_50'] = test['array_dihe'].map(lambda x : np.array(x)[np.array(x) >= -0.001]).map(lambda x : np.percentile(x,50))
test['dihe_75'] = test['array_dihe'].map(lambda x : np.array(x)[np.array(x) >= -0.001]).map(lambda x : np.percentile(x,75))

for ii in np.linspace(0,180,91)[:-1]:
    diff= np.linspace(0,180,91)[1]-np.linspace(0,180,91)[0]
    train['dihe_%s_%s'%(ii,ii+diff)] = train['array_dihe'].apply(lambda x : more_less_than(x,ii,ii+diff))
    test['dihe_%s_%s'%(ii,ii+diff)] = test['array_dihe'].apply(lambda x : more_less_than(x,ii,ii+diff))
##### get energy ###
train_E = np.load('train_energy.npy').item()
test_E = np.load('test_energy.npy').item()

train['array_energy'] = train['id'].map(train_E)
train['lj_E'] = train['array_energy'].map(lambda x : x[1])
train['M_E'] = train['array_energy'].map(lambda x : x[4])
train['force1_mean'] = train['array_energy'].map(lambda x : np.mean(np.sum(x[3]**2,1),0))
train['force2_mean'] = train['array_energy'].map(lambda x : np.mean(np.sum(x[-1]**2,1),0))
train['force1_std'] = train['array_energy'].map(lambda x : np.std(np.sum(x[3]**2,1),0))
train['force2_std'] = train['array_energy'].map(lambda x : np.std(np.sum(x[-1]**2,1),0))
train['force1_25'] = train['array_energy'].map(lambda x : np.percentile(np.sum(x[3]**2,1),25))
train['force2_25'] = train['array_energy'].map(lambda x : np.percentile(np.sum(x[-1]**2,1),25))
train['force1_50'] = train['array_energy'].map(lambda x : np.percentile(np.sum(x[3]**2,1),50))
train['force2_50'] = train['array_energy'].map(lambda x : np.percentile(np.sum(x[-1]**2,1),50))
train['force1_75'] = train['array_energy'].map(lambda x : np.percentile(np.sum(x[3]**2,1),75))
train['force2_75'] = train['array_energy'].map(lambda x : np.percentile(np.sum(x[-1]**2,1),75))

test['array_energy'] = test['id'].map(test_E)
test['lj_E'] = test['array_energy'].map(lambda x : x[1])
test['M_E'] = test['array_energy'].map(lambda x : x[4])
test['force1_mean'] = test['array_energy'].map(lambda x : np.mean(np.sum(x[3]**2,1),0))
test['force2_mean'] = test['array_energy'].map(lambda x : np.mean(np.sum(x[-1]**2,1),0))
test['force1_std'] = test['array_energy'].map(lambda x : np.std(np.sum(x[3]**2,1),0))
test['force2_std'] = test['array_energy'].map(lambda x : np.std(np.sum(x[-1]**2,1),0))
test['force1_25'] = test['array_energy'].map(lambda x : np.percentile(np.sum(x[3]**2,1),25))
test['force2_25'] = test['array_energy'].map(lambda x : np.percentile(np.sum(x[-1]**2,1),25))
test['force1_50'] = test['array_energy'].map(lambda x : np.percentile(np.sum(x[3]**2,1),50))
test['force2_50'] = test['array_energy'].map(lambda x : np.percentile(np.sum(x[-1]**2,1),50))
test['force1_75'] = test['array_energy'].map(lambda x : np.percentile(np.sum(x[3]**2,1),75))
test['force2_75'] = test['array_energy'].map(lambda x : np.percentile(np.sum(x[-1]**2,1),75))


# CNN features based on distance matrix
train['CNN1']= 0
test['CNN1']= 0
train['CNN2']= 0
test['CNN2']= 0
for i in ['11/','12/','13/','14/',]:
    #train[i] = pd.read_csv(i+'train_CNN.csv')['predict1']
    #test[i] = pd.read_csv(i+'test_CNN.csv')['predict1']
    train['CNN1'] =  train['CNN1']+pd.read_csv(i+'train_CNN.csv')['predict1']
    test['CNN1'] = test['CNN1']+pd.read_csv(i+'test_CNN.csv')['predict1']
for i in ['22/','23/','24/']:
    #train[i] = pd.read_csv(i+'train_CNN.csv')['predict1']
    #test[i] = pd.read_csv(i+'test_CNN.csv')['predict1']
    train['CNN2'] = train['CNN2']+pd.read_csv(i+'train_CNN.csv')['predict1']
    test['CNN2'] = test['CNN2']+pd.read_csv(i+'test_CNN.csv')['predict1']

## eleemntal properties, https://www.kaggle.com/cbartel/random-forest-using-elemental-properties/data
dictt_ = {}
dictt_['EN'] = {'O' : 3.44, 'In': 1.78, 'Al' : 1.61, 'Ga' : 1.81}
dictt_['EA'] = {'O' : -0.22563, 'In': -0.3125, 'Al' : -0.2563, 'Ga' : -0.1081}
dictt_['HOMO'] = {'O' : -2.74, 'In': -2.784, 'Al' : -2.697, 'Ga' : -2.732}
dictt_['LUMO'] = {'O' : 0.397, 'In': 0.695, 'Al' : 0.368, 'Ga' : 0.13}
dictt_['IP'] = {'O' :-5.71187, 'In': -5.5374, 'Al' : -5.78, 'Ga' : -5.8182}
dictt_['MASS'] = {'O' :16, 'In': 115, 'Al' : 27, 'Ga' :70}
dictt_['RD'] = {'O' :2.4033, 'In': 1.94, 'Al' : 3.11, 'Ga' :2.16}
dictt_['RP'] = {'O' :1.406667, 'In': 1.39, 'Al' : 1.5, 'Ga' :1.33}
dictt_['RS'] = {'O' :1.07, 'In': 1.09, 'Al' : 1.13, 'Ga' :0.99}
dictt_['VOL'] = {'Al' : 0.5235,'Ga' : 0.9982,'In' :2.2258, 'O' : 11.4927 }
train_ele = np.load('train_resi.npy').item()
test_ele = np.load('test_resi.npy').item()
train['array_ele'] = train['id'].map(train_ele)
test['array_ele'] = test['id'].map(test_ele)
def get_all_ele(x,d):
    return [d['Al'],]*x[0] + [d['Ga'],]*x[1] + [d['In'],]*x[2] + [d['O'],]*x[3]

for element in dictt_.keys():
    temp1 = train['array_ele'].map(lambda x : get_all_ele(x,dictt_[element]))
    train[element+'_mean'] = map(np.mean,temp1)
    train[element+'_std'] = map(np.std,temp1)
    train[element+'_25'] = map(lambda x : np.percentile(x,25),temp1)
    train[element+'_50'] = map(lambda x : np.percentile(x,50),temp1)
    train[element+'_75'] = map(lambda x : np.percentile(x,75),temp1)
    temp2 = test['array_ele'].map(lambda x : get_all_ele(x,dictt_[element]))
    test[element+'_mean'] = map(np.mean,temp2)
    test[element+'_std'] = map(np.std,temp2)
    test[element+'_25'] = map(lambda x : np.percentile(x,25),temp2)
    test[element+'_50'] = map(lambda x : np.percentile(x,50),temp2)
    test[element+'_75'] = map(lambda x : np.percentile(x,75),temp2)
    if element == 'VOL':
        train[element+'_sum'] = 1-map(np.sum,temp1)/train['vol']
        test[element+'_sum'] = 1-map(np.sum,temp2)/test['vol']

train['N_number_of_total_atoms']= train['number_of_total_atoms']/train['vol']
test['N_number_of_total_atoms']= test['number_of_total_atoms']/test['vol']
for i in range(0,4):
    res = ['Al', 'Ga', 'In', 'O']
    train['N_num'+res[i]] = train['array_ele'].map(lambda x : x[i])
    train['N_num'+res[i]] = train['N_num'+res[i]] /train['vol']
    test['N_num'+res[i]] = test['array_ele'].map(lambda x : x[i])
    test['N_num'+res[i]] = test['N_num'+res[i]] /test['vol']
for i in train.keys():
    if 'array'  in i or  np.std(train[i]) <= 0.0001:
        print i#
        del train[i],test[i]
corr = train.corr()
for ii in range(len(corr)):
    for jj in range(ii+1,len(corr)):
        i =corr.keys()[ii]
        j =corr.keys()[jj]
        if i!=j and corr[i][j]**2 >= 0.99**2 :
            try:
                print i ,j ,corr[i][j]
                del train[j]
                del test[j]
            except KeyError:
                None
## finished ellemental properties
target1 = 'formation_energy_ev_natom'
target2 = 'bandgap_energy_ev'


train[target1] = np.log1p(train[target1])
train[target2] = np.log1p(train[target2])
train['predict1'] = 0
test[target1] = 0
train['predict2'] = 0
test[target2] = 0
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
##train['predict1_0.498'] = pd.read_csv('model_train_removeCorr_0.0498.csv').\
##                          sort_values('id').reset_index(drop=True)['predict1_0.0498']
##test['predict1_0.498'] = pd.read_csv('model_test_removeCorr_0.0498.csv').\
##                         sort_values('id').reset_index(drop=True)['predict1_0.0498']
##train['predict2_0.498'] = pd.read_csv('model_train_removeCorr_0.0498.csv').\
##                          sort_values('id').reset_index(drop=True)['predict2_0.0498']
##test['predict2_0.498'] = pd.read_csv('model_test_removeCorr_0.0498.csv').\
##                         sort_values('id').reset_index(drop=True)['predict2_0.0498']

cols = ["N('O', 'O', 2)", "('In', 'In')_125_135", "('Al', 'O')_2.2_2.3000000000000003", "('Al', 'Al')_85_95", "N('In', 'Ga', 0)", 'period_mean', "N('Al', 'Al', 0)", "('Al', 'Ga')_115_125", 'dihe_mean', "('In', 'In')_A_50", "N('In', 'In', 0)", "('O', 'O')_85_95", "('Ga', 'In')_95_105", "('Al', 'In')_125_135", "('Al', 'Al')_A_50", "('Al', 'O')_B_25", "Bond_('Ga', 'O')_std", "('Al', 'In')_A_50", "('Al', 'Ga')_A_50", "('In', 'In')_A_std", "Bond_('In', 'O')_mean", "N('Al', 'O', 2)", "('Ga', 'Ga')_95_105", "('Ga', 'Ga')_A_75", "N('In', 'In', 1)", "('Ga', 'Ga')_115_125", "('Al', 'O')_1.8_1.9000000000000001", "('O', 'O')_75_85", "('Al', 'Ga')_85_95", "('O', 'O')_A_mean", 'spacegroup', "('Ga', 'O')_B_75", "('In', 'In')_95_105", 'dihe_nan', "N('Al', 'Ga', 1)", "('In', 'In')_A_mean", "('Al', 'In')_95_105", "N('O', 'O', 1)", 'IonChar_mean', "('Ga', 'O')_2.1_2.2", "N('In', 'In', 4)", "N('Ga', 'Ga', 2)", 'lattice_angle_beta_degree', "N('Al', 'In', 0)", 'IonChar_std', "('In', 'In')_A_75", "('Al', 'In')_105_115", "('Al', 'Al')_A_75", "('Ga', 'In')_115_125", "('O', 'O')_155_165", "('Ga', 'O')_B_50", 'dihe_std', "N('Al', 'O', 0)", "('Ga', 'In')_85_95", "('Ga', 'In')_A_mean", "N('Al', 'Ga', 0)", 'dihe_25', "N('In', 'O', 0)", "N('Ga', 'O', 4)", "('Al', 'Ga')_95_105", "Bond_('Al', 'O')_mean", 'z1', 'z2', "('Ga', 'O')_1.8_1.9000000000000001", 'period_std', "('Ga', 'Ga')_A_mean", "N('Ga', 'Ga', 3)", "('O', 'O')_175_185", "Bond_('In', 'O')_std", 'lattice_vector_3_ang', "('Al', 'O')_B_75", "('Al', 'Ga')_A_mean", "('Al', 'Ga')_125_135", "N('Al', 'Ga', 3)", "('O', 'O')_A_75", "('Al', 'In')_85_95", "('Al', 'Ga')_A_25", "('Ga', 'In')_A_25", "('Al', 'In')_115_125", 'lattice_angle_alpha_degree', "N('In', 'O', 1)", "N('Ga', 'O', 3)", "N('Al', 'O', 1)", "('Ga', 'Ga')_125_135", "N('Ga', 'Ga', 0)", "('Ga', 'Ga')_85_95", "('Ga', 'Ga')_A_25", 'dihe_50', "('Al', 'In')_A_mean", "N('Al', 'In', 2)", "('Ga', 'O')_1.9000000000000001_2.0", "Bond_('Ga', 'O')_mean", "('Ga', 'O')_B_25", "('In', 'O')_2.0_2.1", "('Al', 'In')_A_75", 'percent_atom_in', 'percent_atom_ga', "('In', 'In')_115_125", "('In', 'O')_1.9000000000000001_2.0", "('In', 'O')_2.1_2.2", "('Ga', 'In')_A_50", "N('Al', 'O', 3)", "N('Ga', 'O', 2)", "('O', 'O')_95_105", "('O', 'O')_105_115", "N('In', 'O', 2)", "('Al', 'Al')_125_135", "('In', 'In')_A_25", "N('Al', 'In', 3)", "('O', 'O')_A_25", "N('In', 'Ga', 3)", "('Al', 'Al')_A_25", "('Ga', 'Ga')_A_std", "Bond_('Al', 'O')_std", "('Al', 'O')_B_50", 'Norm3', 'Norm2', 'Norm7', 'Norm5', "('Al', 'O')_2.0_2.1", "('Ga', 'In')_A_std", "('O', 'O')_A_50", "N('O', 'O', 0)", 'vol', "('Al', 'O')_2.1_2.2", "('Al', 'In')_A_std", "N('Ga', 'O', 1)", "('In', 'O')_2.2_2.3000000000000003", "('O', 'O')_A_std", 'lattice_angle_gamma_degree', "N('Al', 'Ga', 2)", "N('In', 'O', 3)", 'dihe_75', "('In', 'O')_1.8_1.9000000000000001", "('In', 'In')_85_95", "N('Al', 'Al', 2)", "('Al', 'O')_1.9000000000000001_2.0", "N('In', 'In', 2)", "N('In', 'Ga', 2)", "('In', 'O')_B_75", "('Al', 'Al')_95_105", 'lattice_vector_2_ang', "('In', 'O')_B_25", "('Al', 'In')_A_25", "('Al', 'Al')_A_mean", "('Al', 'Ga')_A_75", "('In', 'In')_105_115", "('Ga', 'Ga')_A_50", "('Ga', 'O')_2.2_2.3000000000000003", 'lattice_vector_1_ang', "('Ga', 'In')_A_75", "('Al', 'Al')_A_std", "N('Al', 'O', 4)", "N('Ga', 'O', 0)", "N('Al', 'Al', 3)", "('In', 'O')_B_50", "N('In', 'In', 3)", 'percent_atom_al', "('Al', 'Al')_115_125", "('Ga', 'O')_2.0_2.1", "('Al', 'Ga')_A_std"]
cols = [x for x in cols if x not in ['z1','z2']]
cols = [x for x in train.keys() if x not in ['id',target1,target2,'predict1','predict2'] and 'array' not in x]
print cols
dictt_cols1 = pd.DataFrame(cols + ['z1','z2'],columns=[1])
dictt_cols2 = pd.DataFrame(cols + ['z1','z2'],columns=[1])
test[target1]=0
test[target2]=0
train_ori = train.copy(deep = True)
test_ori = test.copy(deep = True)
def maximum_likelihood_feature(train,test,feature):
    def discrete(z,trn,tst): #original variables are not modified
        change_z = np.copy(z) #only training, percetiles based on here
        change_trn = np.copy(trn) #test set
        change_tst = np.copy(tst) #entire training set (include validation)
        final = 100
        mul = 100.0/final
        final_l = []
        for i in range(0,final):
            lower = np.percentile(z[~np.isnan(z)],mul*i)
            upper = np.percentile(z[~np.isnan(z)],mul*(1+i))
            change_z[(z >= lower) & (z < upper)] = i
            change_trn[(trn >= lower) & (trn < upper)] = i
            change_tst[(tst >= lower) & (tst < upper)] = i
            final_l += [lower,]
        lower = np.percentile(z[~np.isnan(z)],mul*(final-1))
        final_l += [np.max(z[~np.isnan(z)]),]
        change_z[ (z > lower)] = final
        change_trn[ (trn > lower)] = final
        change_tst[ (tst > lower)] = final
        return change_z,change_trn,change_tst,final_l
    train_copy = train.copy()[['id',feature,target1,target2]]
    return discrete(train.copy().iloc[train_id][feature].values,
                     train.copy()[feature].values,test.copy()[feature].values)
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import mutual_info_score
from MI_continous import mutual_information
if True:
    result = {}
    mutual_information = mutual_info_score
    train = train_ori.copy()
    train_id = range(2400)
    for i in cols+[target1,target2]:
        if len(pd.unique(train[i])) >=101:
            x,y,z,final = maximum_likelihood_feature(train,test,i)
            train[i] = y
    for i in cols:
            x,y = mutual_information(train[target1],train[i]),mutual_information(train[target2],train[i])
            if np.mean(train[i].isnull()) <=0.03:
                    print i,x,y
                    f, ax = plt.subplots(2,1,figsize=(5,10));
                    ax[0].plot(train[i],train[target1],'bo')
                    ax[1].plot(train[i],train[target2],'ro')
                    ax[0].set_title(x)
                    ax[1].set_title(y)
                    plt.savefig('z'+i+'.png')
                    plt.close()
                    result[i] = [x,y]
best=  sorted(result,key = lambda x : result[x][0],reverse = True)[:10]
a=train.iloc[train_id].groupby(best[0])[target1].apply(np.mean)
b=train.iloc[train_id].groupby(best[1])[target1].apply(np.mean)
c=train.iloc[train_id].groupby(best[2])[target1].apply(np.mean)
d=train.iloc[train_id].groupby(best[7])[target1].apply(np.mean)
e=train.iloc[train_id].groupby(best[8])[target1].apply(np.mean)
train['1'] = train[best[0]].map(a.to_dict()) + train[best[1]].map(b.to_dict())+train[best[2]].map(c.to_dict()) + train[best[7]].map(d.to_dict())+ train[best[8]].map(e.to_dict())
 
c=pd.merge(a,b)
