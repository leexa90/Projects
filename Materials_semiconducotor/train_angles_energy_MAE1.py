import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
# calculate the volume of the structure

#https://stackoverflow.com/questions/45006341/xgboost-how-to-use-mae-as-objective-function
# calculate with approximations to MAE objective function
def huber_approx_obj(preds, dtrain):
    d = preds - dtrain.get_label()   #remove .get_labels() for sklearn
    h = 1  #h is delta in the graphic
    scale = 1 + (d / h) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt
    hess = 1 / scale / scale_sqrt
    return grad, hess
def fair_obj(preds, dtrain):
    """y = c * abs(x) - c * np.log(abs(abs(x) + c))"""
    x = preds - dtrain.get_label() 
    c = 1
    den = abs(x) + c
    grad = c*x / den
    hess = c*c / den ** 2
    return grad, hess
def log_cosh_obj(preds, dtrain):
    x = preds - dtrain.get_label() 
    grad = np.tanh(x)
    hess = 1 / np.cosh(x)**2
    return grad, hess
def log_rmse(y_pred,dtrain):
    answer = dtrain.get_label()
    highest_mcc = -999
    answer = np.array(answer)
    prediction_prob = np.array(y_pred)
    loss = np.mean((np.log1p(y_pred) - np.log1p(answer))**2)
    return 'mae error',loss**.5

obj = huber_approx_obj
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
##train['CNN1']= 0
##test['CNN1']= 0
##train['CNN2']= 0
##test['CNN2']= 0
##for i in ['11/','12/','13/','14/','11B/','12B/','13B/','14B/','15A/'][:]:
##    #train[i] = pd.read_csv(i+'train_CNN.csv')['predict1']
##    #test[i] = pd.read_csv(i+'test_CNN.csv')['predict1']
##    train['CNN1'] =  train['CNN1']+pd.read_csv(i+'train_CNN.csv')['predict1']
##    test['CNN1'] = test['CNN1']+pd.read_csv(i+'test_CNN.csv')['predict1']
##for i in ['21/','22/','23/','24/','21B/','22B/','23B/','24B/'][:]:
##    #train[i] = pd.read_csv(i+'train_CNN.csv')['predict1']
##    #test[i] = pd.read_csv(i+'test_CNN.csv')['predict1']
##    train['CNN2'] = train['CNN2']+pd.read_csv(i+'train_CNN.csv')['predict1']
##    test['CNN2'] = test['CNN2']+pd.read_csv(i+'test_CNN.csv')['predict1']

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

test['array_dihe'] = test['id'].map(test_dihe)
test['dihe_mean'] = test['array_dihe'].map(lambda x : np.array(x)[np.array(x) >= -0.001]).map(lambda x : np.mean(x))
test['dihe_std'] = test['array_dihe'].map(lambda x : np.array(x)[np.array(x) >= -0.001]).map(lambda x : np.std(x))
test['dihe_nan'] = test['array_dihe'].map(lambda x : 1.0*(len(x)-len(np.array(x)[np.array(x) >= -0.001]))/len(x))
test['dihe_25'] = test['array_dihe'].map(lambda x : np.array(x)[np.array(x) >= -0.001]).map(lambda x : np.percentile(x,25))
test['dihe_50'] = test['array_dihe'].map(lambda x : np.array(x)[np.array(x) >= -0.001]).map(lambda x : np.percentile(x,50))
test['dihe_75'] = test['array_dihe'].map(lambda x : np.array(x)[np.array(x) >= -0.001]).map(lambda x : np.percentile(x,75))

for ii in np.linspace(0,180,91)[:-1]: #improved accuracy by 0.013
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
if True:
    a = pd.read_csv('model_train_removeCorr_0.04958.csv').sort_values('id').reset_index(drop=True)['predict1_0.04958']
    b = pd.read_csv('model_train_removeCorrLess_0.04969.csv').sort_values('id').reset_index(drop=True)['predict1_0.04969']
    train['predict1_0.495'] = (a + b)/2
    a = pd.read_csv('model_train_removeCorr_0.04958.csv').sort_values('id').reset_index(drop=True)['predict2_0.04958']
    b = pd.read_csv('model_train_removeCorrLess_0.04969.csv').sort_values('id').reset_index(drop=True)['predict2_0.04969']
    train['predict2_0.495'] = (a + b)/2
    a = pd.read_csv('model_test_removeCorr_0.04958.csv').sort_values('id').reset_index(drop=True)['predict1_0.04958']
    b = pd.read_csv('model_test_removeCorrLess_0.04969.csv').sort_values('id').reset_index(drop=True)['predict1_0.04969']
    test['predict1_0.495'] = (a + b)/2
    a = pd.read_csv('model_test_removeCorr_0.04958.csv').sort_values('id').reset_index(drop=True)['predict2_0.04958']
    b = pd.read_csv('model_test_removeCorrLess_0.04969.csv').sort_values('id').reset_index(drop=True)['predict2_0.04969']
    test['predict2_0.495'] = (a + b)/2
    aa,bb = np.mean((train['predict1_0.495']-train[target1])**2)**.5, np.mean((train['predict2_0.495']-train[target2])**2)**.5

target1 = 'formation_energy_ev_natom'
target2 = 'bandgap_energy_ev'

cols = [x for x in train.keys() if (x not in ['id',target1,target2] and 'array' not in x)]
# mae , comment below
##train[target1] = np.log1p(train[target1])
##train[target2] = np.log1p(train[target2])
train['predict1'] = 0
test[target1] = 0
train['predict2'] = 0
test[target2] = 0
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

cols = ["N('O', 'O', 2)", "('In', 'In')_125_135", "('Al', 'O')_2.2_2.3000000000000003", "('Al', 'Al')_85_95", "N('In', 'Ga', 0)", 'period_mean', "N('Al', 'Al', 0)", "('Al', 'Ga')_115_125", 'dihe_mean', "('In', 'In')_A_50", "N('In', 'In', 0)", "('O', 'O')_85_95", "('Ga', 'In')_95_105", "('Al', 'In')_125_135", "('Al', 'Al')_A_50", "('Al', 'O')_B_25", "Bond_('Ga', 'O')_std", "('Al', 'In')_A_50", "('Al', 'Ga')_A_50", "('In', 'In')_A_std", "Bond_('In', 'O')_mean", "N('Al', 'O', 2)", "('Ga', 'Ga')_95_105", "('Ga', 'Ga')_A_75", "N('In', 'In', 1)", "('Ga', 'Ga')_115_125", "('Al', 'O')_1.8_1.9000000000000001", "('O', 'O')_75_85", "('Al', 'Ga')_85_95", "('O', 'O')_A_mean", 'spacegroup', "('Ga', 'O')_B_75", "('In', 'In')_95_105", 'dihe_nan', "N('Al', 'Ga', 1)", "('In', 'In')_A_mean", "('Al', 'In')_95_105", "N('O', 'O', 1)", 'IonChar_mean', "('Ga', 'O')_2.1_2.2", "N('In', 'In', 4)", "N('Ga', 'Ga', 2)", 'lattice_angle_beta_degree', "N('Al', 'In', 0)", 'IonChar_std', "('In', 'In')_A_75", "('Al', 'In')_105_115", "('Al', 'Al')_A_75", "('Ga', 'In')_115_125", "('O', 'O')_155_165", "('Ga', 'O')_B_50", 'dihe_std', "N('Al', 'O', 0)", "('Ga', 'In')_85_95", "('Ga', 'In')_A_mean", "N('Al', 'Ga', 0)", 'dihe_25', "N('In', 'O', 0)", "N('Ga', 'O', 4)", "('Al', 'Ga')_95_105", "Bond_('Al', 'O')_mean", 'z1', 'z2', "('Ga', 'O')_1.8_1.9000000000000001", 'period_std', "('Ga', 'Ga')_A_mean", "N('Ga', 'Ga', 3)", "('O', 'O')_175_185", "Bond_('In', 'O')_std", 'lattice_vector_3_ang', "('Al', 'O')_B_75", "('Al', 'Ga')_A_mean", "('Al', 'Ga')_125_135", "N('Al', 'Ga', 3)", "('O', 'O')_A_75", "('Al', 'In')_85_95", "('Al', 'Ga')_A_25", "('Ga', 'In')_A_25", "('Al', 'In')_115_125", 'lattice_angle_alpha_degree', "N('In', 'O', 1)", "N('Ga', 'O', 3)", "N('Al', 'O', 1)", "('Ga', 'Ga')_125_135", "N('Ga', 'Ga', 0)", "('Ga', 'Ga')_85_95", "('Ga', 'Ga')_A_25", 'dihe_50', "('Al', 'In')_A_mean", "N('Al', 'In', 2)", "('Ga', 'O')_1.9000000000000001_2.0", "Bond_('Ga', 'O')_mean", "('Ga', 'O')_B_25", "('In', 'O')_2.0_2.1", "('Al', 'In')_A_75", 'percent_atom_in', 'percent_atom_ga', "('In', 'In')_115_125", "('In', 'O')_1.9000000000000001_2.0", "('In', 'O')_2.1_2.2", "('Ga', 'In')_A_50", "N('Al', 'O', 3)", "N('Ga', 'O', 2)", "('O', 'O')_95_105", "('O', 'O')_105_115", "N('In', 'O', 2)", "('Al', 'Al')_125_135", "('In', 'In')_A_25", "N('Al', 'In', 3)", "('O', 'O')_A_25", "N('In', 'Ga', 3)", "('Al', 'Al')_A_25", "('Ga', 'Ga')_A_std", "Bond_('Al', 'O')_std", "('Al', 'O')_B_50", 'Norm3', 'Norm2', 'Norm7', 'Norm5', "('Al', 'O')_2.0_2.1", "('Ga', 'In')_A_std", "('O', 'O')_A_50", "N('O', 'O', 0)", 'vol', "('Al', 'O')_2.1_2.2", "('Al', 'In')_A_std", "N('Ga', 'O', 1)", "('In', 'O')_2.2_2.3000000000000003", "('O', 'O')_A_std", 'lattice_angle_gamma_degree', "N('Al', 'Ga', 2)", "N('In', 'O', 3)", 'dihe_75', "('In', 'O')_1.8_1.9000000000000001", "('In', 'In')_85_95", "N('Al', 'Al', 2)", "('Al', 'O')_1.9000000000000001_2.0", "N('In', 'In', 2)", "N('In', 'Ga', 2)", "('In', 'O')_B_75", "('Al', 'Al')_95_105", 'lattice_vector_2_ang', "('In', 'O')_B_25", "('Al', 'In')_A_25", "('Al', 'Al')_A_mean", "('Al', 'Ga')_A_75", "('In', 'In')_105_115", "('Ga', 'Ga')_A_50", "('Ga', 'O')_2.2_2.3000000000000003", 'lattice_vector_1_ang', "('Ga', 'In')_A_75", "('Al', 'Al')_A_std", "N('Al', 'O', 4)", "N('Ga', 'O', 0)", "N('Al', 'Al', 3)", "('In', 'O')_B_50", "N('In', 'In', 3)", 'percent_atom_al', "('Al', 'Al')_115_125", "('Ga', 'O')_2.0_2.1", "('Al', 'Ga')_A_std"]
cols = [x for x in cols if x not in ['z1','z2']]
cols = [x for x in train.keys() if x not in ['id',target1,target2,'predict1','predict2'] and 'array' not in x]
print cols
dictt_cols1 = pd.DataFrame(cols + ['z1','z2'],columns=[1])
dictt_cols2 = pd.DataFrame(cols + ['z1','z2'],columns=[1])
train_ori = train.copy(deep = True)
test_ori = test.copy(deep = True)
for i in cols:
    train[i] = train[i].fillna(0)
    test[i] = test[i].fillna(0)
train = train.fillna(0)
test = test.fillna(0)
ori_cols = list(np.copy(cols))
seeds = [1,5516,643,5235,2352,12,5674,19239,41241,1231,151,34,1235,2664,
         75764,2314,1111,2222,3333,4444][:]
print seeds,len(seeds)
comps=1
for seed in seeds:
    train = train.sample(2400,random_state=seed).reset_index(drop=True)
    for i in range(0,10):
##    train = train.copy(deep = True)
##    train['outlier'] = 1
##    train = train.set_value(train[(train[target1] <= 0.40) & (train[target2] <= 1.81) & (train[target2] >= 0.1) ].index,
##                            'outlier',0)
##    train = train.sort_values('outlier').reset_index(drop= True)
##    del train['outlier']
        test_id = [x for x in range(0,2400) if x%10 == i] 
        train_id = [x for x in range(0,2400) if x%10 != i] 
        scaler = StandardScaler()
        regr = linear_model.LinearRegression()
        scaler = scaler.fit(pd.concat([train,test])[ori_cols].values)
        train['z1'] = regr.fit(scaler.transform(train.iloc[train_id][ori_cols].values),
                               train.iloc[train_id][target1].values).predict(scaler.transform(train[ori_cols].values))
        train['z2'] = regr.fit(scaler.transform(train.iloc[train_id][ori_cols].values),
                               train.iloc[train_id][target2].values).predict(scaler.transform(train[ori_cols].values))
        test['z1'] = regr.fit(scaler.transform(train.iloc[train_id][ori_cols].values),
                              train.iloc[train_id][target1].values).predict(scaler.transform(test[ori_cols].values))
        test['z2'] = regr.fit(scaler.transform(train.iloc[train_id][ori_cols].values),
                              train.iloc[train_id][target2].values).predict(scaler.transform(test[ori_cols].values))
        xgtest = xgb.DMatrix(test[cols].values,missing=np.NAN,feature_names=cols)
        from sklearn.decomposition import PCA
        regr = PCA(n_components=comps)
        names = []
        for i in range(comps):
            names += ['zz'+str(i),]
        for name in names :
            train[name] = 0
            test[name] = 0
        temp = regr.fit(scaler.transform(train.iloc[train_id][ori_cols].values)).transform(scaler.transform(train[ori_cols].values))
        train = train.set_value(train.index,names,temp);
        temp = regr.fit(scaler.transform(train.iloc[train_id][ori_cols].values)).transform(scaler.transform(test[ori_cols].values))
        test = test.set_value(test.index,names,temp)

        if 'z1' not in cols:
            cols += ['z1','z2'] #+names
        X_train, y_train1,y_train2 = train.iloc[train_id][cols], train.iloc[train_id][target1],\
                                     train.iloc[train_id][target2]
        X_test, y_test1,y_test2 = train.iloc[test_id][cols], train.iloc[test_id][target1],\
                                  train.iloc[test_id][target2]
        

        params = {}
        params["objective"] = 'reg:linear' 
        params["eta"] = 0.05
        params["min_child_weight"] = 30
        params["subsample"] = 0.4
        params["colsample_bytree"] = 0.20 # many features here
        params["scale_pos_weight"] = 1
        params["silent"] = 0
        params["max_depth"] = 8
        params['seed']=seed
        #params['maximize'] =True
        params['eval_metric'] =  'rmse'
        if seed %2 == 0:
            None#params["eta"] = params["eta"]/3
        plst = list(params.items())
        xgtrain = xgb.DMatrix(X_train[cols].values, label=y_train1.values,missing=np.NAN,feature_names=cols)
        xgval = xgb.DMatrix(X_test[cols].values, label=y_test1.values,missing=np.NAN,feature_names=cols)
        watchlist  = [ (xgtrain,'train'),(xgval,'test')]
        model1_a = {}
        model1=xgb.train(plst,xgtrain,5000,watchlist,early_stopping_rounds=200,
                         evals_result=model1_a,maximize=False,verbose_eval=1000,
                         feval =log_rmse,obj =obj)
        train = train.set_value(test_id,'predict1',train.iloc[test_id]['predict1']+model1.predict(xgval)/len(seeds))
        test = test.set_value(test.index, target1, test[target1]+model1.predict(xgtest)/(10*len(seeds)))
        dictt_cols1[len(dictt_cols1.keys())+1] = dictt_cols1[1].map(model1.get_fscore())
        xgtrain = xgb.DMatrix(X_train[cols].values, label=y_train2.values,missing=np.NAN,feature_names=cols)
        xgval = xgb.DMatrix(X_test[cols].values, label=y_test2.values,missing=np.NAN,feature_names=cols)
        watchlist  = [ (xgtrain,'train'),(xgval,'test')]
        model1_a = {}
        params["eta"] = 0.015
        params["max_depth"] = 8
        if seed %2 == 0:
            None#params["eta"] = params["eta"]/3
        plst = list(params.items())
        model2=xgb.train(plst,xgtrain,6500,watchlist,early_stopping_rounds=200,
                         evals_result=model1_a,maximize=False,verbose_eval=1000,
                         feval =log_rmse, obj=obj)
        train = train.set_value(test_id,'predict2',train.iloc[test_id]['predict2']+model2.predict(xgval)/len(seeds))
        test = test.set_value(test.index, target2, test[target2]+model2.predict(xgtest)/(10*len(seeds)))
        dictt_cols2[len(dictt_cols2.keys())+1] = dictt_cols2[1].map(model2.get_fscore())
dictt_cols1 = dictt_cols1.fillna(0)
dictt_cols2 = dictt_cols2.fillna(0)
for i in range(2,1+len(dictt_cols1.keys())):
    dictt_cols1[i] = 100*dictt_cols1[i] / np.sum(dictt_cols1[i])
    dictt_cols2[i] = 100*dictt_cols2[i] / np.sum(dictt_cols2[i])
#print np.sum(dictt_cols1[range(2,len(dictt_cols1.keys()))].values,1)
#print np.sum(dictt_cols2[range(2,len(dictt_cols1.keys()))].values,1)
dictt_cols1['avg'] =  np.sum(dictt_cols1[range(2,len(dictt_cols1.keys()))].values,1)
dictt_cols2['avg'] =  np.sum(dictt_cols2[range(2,len(dictt_cols1.keys()))].values,1)
if True:
    print dictt_cols2[[1,'avg']].sort_values('avg').iloc[:50]
    print dictt_cols1[[1,'avg']].sort_values('avg').iloc[:50]
if True:
    print dictt_cols2[[1,'avg']].sort_values('avg').iloc[-50:]
    print dictt_cols1[[1,'avg']].sort_values('avg').iloc[-50:]
print list(set(list(dictt_cols1[[1,'avg']].sort_values('avg').iloc[-160:][1])+list(dictt_cols2[[1,'avg']].sort_values('avg').iloc[-160:][1])))
train[target1] = np.log1p(train[target1])
train[target2] = np.log1p(train[target2])
train['predict1'] = np.log1p(train['predict1'])
train['predict2'] = np.log1p(train['predict2'])
test[target1] = np.log1p(test[target1])
test[target1] = np.log1p(test[target1])
a,b = np.mean((train['predict1']-train[target1])**2)**.5, np.mean((train['predict2']-train[target2])**2)**.5
print a
print b
print (a+b)/2
train[target1] = np.exp(train[target1])-1
train[target2] = np.exp(train[target2])-1

permutation_seed = 'MAE1'
if True:
    train.to_csv('train_%s_%s.csv'%(permutation_seed,np.round((a+b)/2,5)),index=0)
    test.to_csv('test_%s_%s.csv'%(permutation_seed,np.round((a+b)/2,5)),index=0)
    np.save('cols_%s_%s.npy'%(permutation_seed,np.round((a+b)/2,5)),cols)
if True:
    name1 = 'predict1_%s'%np.round((a+b)/2,5)
    name2 = 'predict2_%s'%np.round((a+b)/2,5)
    test[name1] = test[target1]
    test[name2] = test[target2]
    train[name1] = train['predict1']
    train[name2] = train['predict2']
    train[['id',name1,name2,target1,target2]].to_csv('model_train_%s_%s.csv'%(permutation_seed,np.round((a+b)/2,5)),index=0)
    test[['id',name1,name2]].to_csv('model_test_%s_%s.csv'%(permutation_seed,np.round((a+b)/2,5)),index=0)
if True: #write to file
    test[target1] = np.exp(test[target1])-1
    test[target2] = np.exp(test[target2])-1
    test[['id',target1,target2]].to_csv('submit_test_%s_%s.csv'%(permutation_seed,np.round((a+b)/2,5)),index=0)
list(set(list(dictt_cols1[[1,'avg']].sort_values('avg').iloc[-70:][1]) + list(dictt_cols2[[1,'avg']].sort_values('avg').iloc[-70:][1])))
import matplotlib.pyplot as plt
##for i in cols:
##	f, ax = plt.subplots(1,2,figsize=(10,5));
##	ax[0].plot(train[i],train[target1],'ro')
##	ax[1].plot(train[i],train[target2],'ro')
##	ax[0].set_title(i+'_'+target1)
##	ax[1].set_title(i+'_'+target2)
##	ax[0].set_xlabel(train[[i,target1]].corr()[i])
##	ax[1].set_xlabel(train[[i,target2]].corr()[i])
##	plt.savefig(i+'.png',dpi=300,bbox_inches='tight');plt.close();
'''
RMSE obj 
0.024040672264456046
0.07733700943946409
0.05068884085196007

Fair objective
0.024094492163879683
0.07741023540876714
0.05075236378632341

# log_cosh_obj *20
0.02375701222456912
0.0762436476695883
0.05000032994707871
'''
