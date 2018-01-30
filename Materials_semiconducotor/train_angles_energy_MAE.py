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

obj = None#huber_approx_obj
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
for i in ['11/','12/','13/','14/','11B/','12B/','13B/','14B/','15A/'][:]:
    #train[i] = pd.read_csv(i+'train_CNN.csv')['predict1']
    #test[i] = pd.read_csv(i+'test_CNN.csv')['predict1']
    train['CNN1'] =  train['CNN1']+pd.read_csv(i+'train_CNN.csv')['predict1']
    test['CNN1'] = test['CNN1']+pd.read_csv(i+'test_CNN.csv')['predict1']
for i in ['21/','22/','23/','24/','21B/','22B/','23B/','24B/'][:]:
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


# CNN features based on distance matrix
train['CNN1']= 0
test['CNN1']= 0
train['CNN2']= 0
test['CNN2']= 0
for i in ['11/','12/','13/','14/','11B/','12B/','13B/','14B/','15A/'][:]:
    #train[i] = pd.read_csv(i+'train_CNN.csv')['predict1']
    #test[i] = pd.read_csv(i+'test_CNN.csv')['predict1']
    train['CNN1'] =  train['CNN1']+pd.read_csv(i+'train_CNN.csv')['predict1']
    test['CNN1'] = test['CNN1']+pd.read_csv(i+'test_CNN.csv')['predict1']
for i in ['21/','22/','23/','24/','21B/','22B/','23B/'][:]:
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

permutation_seed = 'RMSE'
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
#RMSE *20
0.0237838236201
0.0762322149592
0.0500080192897

#huber 1 *20
0.0237777744767
0.0760913266231
0.0499345505499

# fair *20
0.0236724974596
0.0763541846614
0.0500133410605


# log_cosh_obj *20
0.02375701222456912
0.0762436476695883
0.05000032994707871

204  ('Al', 'O')_1.8_1.9000000000000001  106.515709
57                 Bond_('Al', 'O')_std  107.673123
146                ('Ga', 'In')_115_125  108.792827
141                   ('Ga', 'In')_A_25  109.067776
176                   ('In', 'In')_A_50  109.831599
187                     ('O', 'O')_A_75  111.298338
91                      N('Al', 'O', 0)  111.548220
129                   ('Al', 'In')_A_25  112.415982
116                   ('Al', 'Ga')_A_50  112.558600
337                          force1_std  112.853015
345                             RP_mean  116.351656
79                     N('Al', 'In', 0)  116.888654
152                   ('Al', 'Al')_A_50  117.034291
60                Bond_('In', 'O')_mean  117.855520
343                           force2_75  118.131769
366                            LUMO_std  118.230310
246                        dihe_2.0_4.0  121.449709
342                           force1_75  122.030901
331                    dihe_176.0_178.0  122.619481
64                          period_mean  125.855844
356                              IP_std  126.174361
310                    dihe_134.0_136.0  134.212470
101                      N('O', 'O', 1)  137.286423
353                            VOL_mean  137.537669
95                      N('Al', 'O', 4)  137.586152
340                           force1_50  139.542029
128                   ('Al', 'In')_A_50  139.562198
362                             RD_mean  139.963667
10           lattice_angle_gamma_degree  142.271195
62                         IonChar_mean  143.625536
65                           period_std  145.061991
7                  lattice_vector_3_ang  147.205563
185                   ('O', 'O')_A_mean  148.874762
134                ('Al', 'In')_115_125  154.740344
83                     N('Al', 'Al', 0)  160.915838
243                             dihe_50  161.387891
265                      dihe_42.0_44.0  175.358697
289                      dihe_90.0_92.0  176.713707
311                    dihe_136.0_138.0  184.916804
302                    dihe_118.0_120.0  186.322952
189                     ('O', 'O')_A_25  188.126284
355                             VOL_sum  194.675774
288                      dihe_88.0_90.0  202.230743
244                             dihe_75  209.484396
240                            dihe_std  222.323572
8            lattice_angle_alpha_degree  242.753997
11                                 CNN1  267.326352
12                                 CNN2  312.083576
370                                  z1  365.375649
371                                  z2  526.888961
                                      1         avg
59                 Bond_('Ga', 'O')_std  107.503469
12                                 CNN2  108.764691
265                      dihe_42.0_44.0  109.682996
79                     N('Al', 'In', 0)  109.805354
200                    ('Al', 'O')_B_50  109.906015
311                    dihe_136.0_138.0  110.592448
219                 ('Ga', 'O')_2.0_2.1  112.478860
334                                 M_E  113.051886
206                 ('Al', 'O')_2.0_2.1  114.241137
333                                lj_E  114.647368
56                Bond_('Al', 'O')_mean  115.466724
125                 ('Al', 'In')_A_mean  116.341577
9             lattice_angle_beta_degree  116.922494
112                     N('Ga', 'O', 4)  117.778442
152                   ('Al', 'Al')_A_50  118.122299
289                      dihe_90.0_92.0  118.241121
368                            HOMO_std  118.663327
245                        dihe_0.0_2.0  119.225640
61                 Bond_('In', 'O')_std  119.876888
129                   ('Al', 'In')_A_25  120.407142
175                   ('In', 'In')_A_75  121.481929
201                    ('Al', 'O')_B_25  123.592157
128                   ('Al', 'In')_A_50  124.330870
57                 Bond_('Al', 'O')_std  124.845360
232  ('In', 'O')_1.9000000000000001_2.0  125.360339
182                ('In', 'In')_115_125  125.435801
185                   ('O', 'O')_A_mean  127.727285
177                   ('In', 'In')_A_25  127.959354
7                  lattice_vector_3_ang  128.073572
199                    ('Al', 'O')_B_75  128.416068
214                    ('Ga', 'O')_B_25  129.325781
10           lattice_angle_gamma_degree  130.296102
239                           dihe_mean  130.905380
288                      dihe_88.0_90.0  130.944158
102                      N('O', 'O', 2)  135.677124
173                 ('In', 'In')_A_mean  142.161139
176                   ('In', 'In')_A_50  145.731809
243                             dihe_50  149.710162
189                     ('O', 'O')_A_25  153.418897
134                ('Al', 'In')_115_125  156.968351
371                                  z2  171.186817
242                             dihe_25  172.732519
205  ('Al', 'O')_1.9000000000000001_2.0  178.262479
8            lattice_angle_alpha_degree  189.378496
100                      N('O', 'O', 0)  194.503671
101                      N('O', 'O', 1)  205.979184
244                             dihe_75  213.602266
240                            dihe_std  251.656940
11                                 CNN1  412.552995
370                                  z1  612.488680
if True:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats

    sns.set(style='white')
    iris = sns.load_dataset('iris')

    def corrfunc(x, y, **kws):
      r, p = stats.pearsonr(x, y)
      p_stars = ''
      if p <= 0.05:
        p_stars = '*'
      if p <= 0.01:
        p_stars = '**'
      if p <= 0.001:
        p_stars = '***'
      ax = plt.gca()
      ax.annotate('r = {:.2f} '.format(r) + p_stars,
                  xy=(0.05, 0.9), xycoords=ax.transAxes)

    def annotate_colname(x, **kws):
      ax = plt.gca()
      ax.annotate(x.name, xy=(0.05, 0.9), xycoords=ax.transAxes,
                  fontweight='bold')

    def cor_matrix(df):
      sns.regplot.func_defaults=(None, None, None, 'ci', True, True, 95, 1000, None, 1, False, False, False, False, None, None, False, True, None, None, None, None, 'o', None, None, None)
      g = sns.PairGrid(df, palette=['red'])
      # Use normal regplot as `lowess=True` doesn't provide CIs.
      g.map_upper(sns.regplot, scatter_kws={'s':1},line_kws={'lw':1})
      g.map_diag(sns.distplot)
      g.map_diag(annotate_colname)
      #g.map_lower(sns.kdeplot, cmap='Blues_d')
      #g.map_lower(corrfunc)
      # Remove axis labels, as they're in the diagonals.
      for ax in g.axes.flatten():
        ax.set_ylabel('')
        ax.set_xlabel('')
      return g
cor_matrix(train[['dihe_75','dihe_std','lattice_angle_alpha_degree',target1,target2]])
a=train.groupby(['dihe_75','dihe_std'])[target1].apply(np.mean)
b=train.groupby(['dihe_75','dihe_std'])[target1].apply(len)
x=plt.plot(range(len(a.keys())),a.values,'ro');x=plt.plot(range(len(b.keys())),b.values/20,'bo');plt.show()


'''
def plot(a,b):
    old_num = 0
    for i in range(len(a)-1):
        if a.keys().labels[0][i] != a.keys().labels[0][i+1]  :
            plt.plot(range(len(a.keys()))[old_num:i+1],a.values[old_num:i+1],'o')
            plt.plot(range(len(a.keys()))[old_num:i+1],b[old_num:i+1],'o')
            old_num = i+1
    plt.show()
def plot20(train,var1,var2,target1=target1,target2=target2):
    from matplotlib.colors import LogNorm
    f, ax = plt.subplots(8,5,figsize=(15,24));
    for i in range(0,20):
        temp = train[train[var1]==i]
        ax[i//5,i%5].hist2d(temp[var2].values,temp[target1].values,10,norm=LogNorm())
        ax[i//5,i%5].set_title(str(i)+'__'+str(np.corrcoef(temp[var2].values,temp[target1].values)[0,1])[:5]+'__'+str(len(temp)))
        ax[i//5,i%5].set_xticks(temp.groupby(var2)[target1].apply(np.mean).keys())
        ax[i//5,i%5].set_xticklabels( map(lambda x : str(np.round(x,1)),temp.groupby(var2)[target1].apply(np.mean).values))
        ax[i//5+4,i%5].hist2d(temp[var2].values,temp[target2].values,10,norm=LogNorm())
        ax[i//5+4,i%5].set_title(str(i)+'__'+str(np.corrcoef(temp[var2].values,temp[target2].values)[0,1])[:5]+'__'+str(len(temp)))
        ax[i//5+4,i%5].set_xticks(temp.groupby(var2)[target2].apply(np.mean).keys())
        ax[i//5+4,i%5].set_xticklabels(map(lambda x : str(np.round(x,1)),temp.groupby(var2)[target2].apply(np.mean).values))
    plt.savefig( var1+'_'+var2+'_.png',bbox_inches='tight');plt.close()


def plot20B(train_old,var1,var2,target1=target1,target2=target2):
    # function discretise features to 20 bins and then plots them together
    def discrete(z):
        change_z = np.copy(z)
        final = 20
        mul = 100.0/final
        for i in range(0,final):
            lower = np.percentile(z,mul*i)
            upper = np.percentile(z,mul*(1+i))
            change_z[(z >= lower) & (z < upper)] = i
        lower = np.percentile(z,mul*(final-1))
        change_z[ (z > lower)] = final-1
        return change_z
    train = train_old.copy()
    train[var1] = discrete(train[var1])
    train[var2] = discrete(train[var2])
    f, ax = plt.subplots(8,5,figsize=(15,24));
    for i in range(0,20):
        temp = train[train[var1]==i]
        a = temp.groupby(var2)[target1].apply(np.mean)
        b = temp.groupby(var2)[target1].apply(np.std)
        ax[i//5,i%5].set_xlim([0,20])
        ax[i//5+4,i%5].set_xlim([0,20])
        ax[i//5,i%5].set_ylim([0,0.65])
        ax[i//5+4,i%5].set_ylim([0,5.3])
        ax[i//5,i%5].plot(temp[var2].values,temp[target1].values,'o',markersize=1.5)
        ax[i//5,i%5].plot(a.keys(),a.values,'orange')
        ax[i//5,i%5].errorbar(a.keys(),a.values,b.values,fmt='o', ecolor='g')
        ax[i//5,i%5].set_title(str(i)+'__'+str(np.corrcoef(temp[var2].values,temp[target1].values)[0,1])[:5]+'__'+str(len(temp)))
        ax[i//5,i%5].set_xticks(a.keys())
        ax[i//5,i%5].set_xticklabels( temp.groupby(var2)[target1].apply(lambda x : np.round(np.log10(len(x)),1)).values,fontsize=4)
        a = temp.groupby(var2)[target2].apply(np.mean)
        b = temp.groupby(var2)[target2].apply(np.std)
        ax[i//5+4,i%5].plot(temp[var2].values,temp[target2].values,'o',markersize=1.5)
        ax[i//5+4,i%5].plot(a.keys(),a.values,'orange')
        ax[i//5+4,i%5].errorbar(a.keys(),a.values,b.values,fmt='o', ecolor='g')
        ax[i//5+4,i%5].set_title(str(i)+'__'+str(np.corrcoef(temp[var2].values,temp[target2].values)[0,1])[:5]+'__'+str(len(temp)))
        ax[i//5+4,i%5].set_xticks(a.keys())
        ax[i//5+4,i%5].set_xticklabels( temp.groupby(var2)[target2].apply(lambda x : np.round(np.log10(len(x)),1)).values,fontsize=4)
    if sorted((var1,var2))[0] == var1:
        plt.savefig( str(sorted((var1,var2)))+'_1_.png',bbox_inches='tight',dpi=150);plt.close()
    else:
        plt.savefig( str(sorted((var1,var2)))+'_2_.png',bbox_inches='tight',dpi=150);plt.close()

a=train.groupby(['dihe_75','dihe_std'])[target1].apply(np.mean);b=train.groupby(['dihe_75','dihe_std'])[target2].apply(np.mean);plot(a,b.values)

def discrete(z):
    change_z = np.copy(z)
    final = 20
    mul = 100.0/final
    for i in range(0,final):
        lower = np.percentile(z,mul*i)
        upper = np.percentile(z,mul*(1+i))
        change_z[(z >= lower) & (z < upper)] = i
    lower = np.percentile(z,mul*(final-1))
    change_z[ (z > lower)] = final-1
    return change_z

train_ori = train.copy()
train = train_ori.copy()
for i in cols:
    if len(pd.unique(train[i]))>=100:
        z = train[i].copy(deep=True).values
        z1=discrete(z)
        train[i] = z1
for i in range(len(impt)):
	for j in range(i+1,len(impt)):
		plot20B(train,impt[i],impt[j])
impt = ['dihe_std','dihe_75',"N('O', 'O', 1)","N('O', 'O', 0)",
        "lattice_angle_alpha_degree","dihe_25","dihe_88.0_90.0",
        "N('O', 'O', 2)","lattice_vector_3_ang","('O', 'O')_A_25'",
        'dihe_mean','dihe_50','period_std','period_mean',"N('Al', 'O', 4)",
        "('O', 'O')_A_mean"]

def maximum_likelihood_feature(train,test,feature):
    def discrete(z,trn,tst): #original variables are not modified
        change_z = np.copy(z) #only training, percetiles based on here
        change_trn = np.copy(trn) #test set
        change_tst = np.copy(tst) #entire training set (include validation)
        final = 100
        mul = 100.0/final
        final_l = []
        for i in range(0,final):
            lower = np.percentile(z,mul*i)
            upper = np.percentile(z,mul*(1+i))
            change_z[(z >= lower) & (z < upper)] = i
            change_trn[(trn >= lower) & (trn < upper)] = i
            change_tst[(tst >= lower) & (tst < upper)] = i
            final_l += [lower,]
        lower = np.percentile(z,mul*(final-1))
        final_l += [np.max(z),]
        change_z[ (z > lower)] = final
        change_trn[ (trn > lower)] = final
        change_tst[ (tst > lower)] = final
        return change_z,change_trn,change_tst,final_l
    train_copy = train.copy()[['id',feature,target1,target2]]
    return discrete(train.copy().iloc[train_id][feature].values,
                     train.copy()[feature].values,test.copy()[feature].values)
        
        
