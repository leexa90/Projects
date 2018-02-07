import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
# calculate the volume of the structure


'''
RF 20 times
0.024381132559098906
0.07570733725112831
150 3 90 0.05004423490511361


# No CNN, no eswald, predict1_0.495 for model2
[2314, 1111, 2222, 3333, 4444]
0.023777557694703744
0.07322642574011586
0.0485019917174098
[5555, 6666, 7777, 8888, 9999]
0.024048941798210904
0.07365744823160975
0.048853195014910325
[111, 222, 333, 444, 555]
0.02399012900705931
0.07369018753514377
0.04884015827110154


BENCHMARK
if True:
  A=[0.024425624291,0.0243964088121,0.0243356065627,0.0242540405675,0.0243152888378]
  B=[0.0751856146305,0.0745420428233,0.0748317206887,0.0750704768338,0.0742665348147]
  print np.mean(A)-np.std(A),np.mean(A),np.mean(A)+np.std(A)
  print np.mean(B)-np.std(B),np.mean(B),np.mean(B)+np.std(B)
0.024284752285550458 0.024345393814219997 0.024406035342889536
0.07444109381994804 0.07477927795820001 0.07511746209645198

0.024425624291
0.0751856146305
0.0498056194608

0.0243964088121
0.0745420428233
0.0494692258177

0.0243356065627
0.0748317206887
0.0495836636257

0.0242540405675
0.0750704768338
0.0496622587006

0.0243152888378
0.0742665348147
0.0492909118263

## removed features good results
predict2_0.495 confirmed - 0.0735
[1, 5516, 643, 5235, 2352]
0.0244532380861
0.0740942551534
0.0492737466198

#no predict_495
[151, 34, 1235, 2664, 75764]
0.0239114185401
0.0765263106143
0.0502188645772


#no predict_495 ,no CNN, no eswald
[151, 34, 1235, 2664, 75764]
0.0236123879521
0.0764022549346
0.0500073214434


'''
name = 'benchmark'
#dictt_cols1_old = pd.DataFrame(np.load('./log/dictt_cols1.npy'))
#dictt_cols2_old = pd.DataFrame(np.load('./log/dictt_cols2.npy'))
#removed_features = list(set(list(dictt_cols1_old[0].iloc[-10:])+list(dictt_cols2_old[0].iloc[-10:])))[-3:]
#sprint removed_features
log = open('./log/'+ name+'.txt','w')
removed_feature = ''
if True:#for removed_feature in removed_features:
    log.write(removed_feature+'\n')
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
    if False:
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
    print '###doing bond angles features###'
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

    print '###doing bond features###'
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
    print '###doing dihe features###'
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

    for ii in np.linspace(0,180,91)[:-1]:
        diff= np.linspace(0,180,91)[1]-np.linspace(0,180,91)[0]
        train['dihe_%s_%s'%(ii,ii+diff)] = train['array_dihe'].apply(lambda x : more_less_than(x,ii,ii+diff))
        test['dihe_%s_%s'%(ii,ii+diff)] = test['array_dihe'].apply(lambda x : more_less_than(x,ii,ii+diff))
    ##### get energy ###
    print '###doing VDW energy features###'
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
    print '###doing elemental features###'
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
    if True:
        print '###doing predict495 features###'
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

    '''0.023510690358920693
    0.07351348695632946
    0.048512088657625074'''
    if False:
        print '###doing predict485 features###'
        a = pd.read_csv('model_train_L2_0.04851.csv').sort_values('id').reset_index(drop=True)['predict1_0.04851']
        b = pd.read_csv('model_train_L2_0.04875.csv').sort_values('id').reset_index(drop=True)['predict1_0.04875']
        train['predict1_0.485'] = (a + b)/2
        a = pd.read_csv('model_train_L2_0.04851.csv').sort_values('id').reset_index(drop=True)['predict2_0.04851']
        b = pd.read_csv('model_train_L2_0.04875.csv').sort_values('id').reset_index(drop=True)['predict2_0.04875']
        train['predict2_0.485'] = (a + b)/2
        a = pd.read_csv('model_test_L2_0.04851.csv').sort_values('id').reset_index(drop=True)['predict1_0.04851']
        b = pd.read_csv('model_test_L2_0.04875.csv').sort_values('id').reset_index(drop=True)['predict1_0.04875']
        test['predict1_0.485'] = (a + b)/2
        a = pd.read_csv('model_test_L2_0.04851.csv').sort_values('id').reset_index(drop=True)['predict2_0.04851']
        b = pd.read_csv('model_test_L2_0.04875.csv').sort_values('id').reset_index(drop=True)['predict2_0.04875']
        test['predict2_0.485'] = (a + b)/2
    if False:
        print '###doing eswald features###'
        train_E = np.load('train_ewald_sum_data.npy.zip')['train_ewald_sum_data.npy'].item()
        train['Eswald_array'] = train['id'].apply(lambda x : train_E[x-1])
        test_E = np.load('test_ewald_sum_data.npy.zip')['test_ewald_sum_data.npy'].item()
        test['Eswald_array'] = test['id'].apply(lambda x : test_E[x-1])
        for i in [3,]:
            if i < 3:
                diagonal= np.diagonal
            else:
                def x(x):return x
                diagonal = x #last one only has forces
            train['Eswald_%s_mean'%i] = train['Eswald_array'].apply(lambda x :np.mean(diagonal(x[i])))
            train['Eswald_%s_50'%i] = train['Eswald_array'].apply(lambda x :np.median(diagonal(x[i])))
            train['Eswald_%s_25'%i] = train['Eswald_array'].apply(lambda x :np.percentile(diagonal(x[i]),25))
            train['Eswald_%s_75'%i] = train['Eswald_array'].apply(lambda x :np.percentile(diagonal(x[i]),75))
            train['Eswald_%s_std'%i] = train['Eswald_array'].apply(lambda x :np.median(diagonal(x[i])))
            test['Eswald_%s_mean'%i] = test['Eswald_array'].apply(lambda x :np.mean(diagonal(x[i])))
            test['Eswald_%s_50'%i] = test['Eswald_array'].apply(lambda x :np.median(diagonal(x[i])))
            test['Eswald_%s_25'%i] = test['Eswald_array'].apply(lambda x :np.percentile(diagonal(x[i]),25))
            test['Eswald_%s_75'%i] = test['Eswald_array'].apply(lambda x :np.percentile(diagonal(x[i]),75))
            test['Eswald_%s_std'%i] = test['Eswald_array'].apply(lambda x :np.median(diagonal(x[i])))
        


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

    train[target1] = np.log(1+train[target1])
    train[target2] = np.log(1+train[target2])
    train['predict1'] = 0
    test[target1] = 0
    train['predict2'] = 0
    test[target2] = 0
    from sklearn.cross_validation import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn import linear_model

    cols = ["N('O', 'O', 2)", "('In', 'In')_125_135", "('Al', 'O')_2.2_2.3000000000000003", "('Al', 'Al')_85_95", "N('In', 'Ga', 0)", 'period_mean', "N('Al', 'Al', 0)", "('Al', 'Ga')_115_125", 'dihe_mean', "('In', 'In')_A_50", "N('In', 'In', 0)", "('O', 'O')_85_95", "('Ga', 'In')_95_105", "('Al', 'In')_125_135", "('Al', 'Al')_A_50", "('Al', 'O')_B_25", "Bond_('Ga', 'O')_std", "('Al', 'In')_A_50", "('Al', 'Ga')_A_50", "('In', 'In')_A_std", "Bond_('In', 'O')_mean", "N('Al', 'O', 2)", "('Ga', 'Ga')_95_105", "('Ga', 'Ga')_A_75", "N('In', 'In', 1)", "('Ga', 'Ga')_115_125", "('Al', 'O')_1.8_1.9000000000000001", "('O', 'O')_75_85", "('Al', 'Ga')_85_95", "('O', 'O')_A_mean", 'spacegroup', "('Ga', 'O')_B_75", "('In', 'In')_95_105", 'dihe_nan', "N('Al', 'Ga', 1)", "('In', 'In')_A_mean", "('Al', 'In')_95_105", "N('O', 'O', 1)", 'IonChar_mean', "('Ga', 'O')_2.1_2.2", "N('In', 'In', 4)", "N('Ga', 'Ga', 2)", 'lattice_angle_beta_degree', "N('Al', 'In', 0)", 'IonChar_std', "('In', 'In')_A_75", "('Al', 'In')_105_115", "('Al', 'Al')_A_75", "('Ga', 'In')_115_125", "('O', 'O')_155_165", "('Ga', 'O')_B_50", 'dihe_std', "N('Al', 'O', 0)", "('Ga', 'In')_85_95", "('Ga', 'In')_A_mean", "N('Al', 'Ga', 0)", 'dihe_25', "N('In', 'O', 0)", "N('Ga', 'O', 4)", "('Al', 'Ga')_95_105", "Bond_('Al', 'O')_mean", 'z1', 'z2', "('Ga', 'O')_1.8_1.9000000000000001", 'period_std', "('Ga', 'Ga')_A_mean", "N('Ga', 'Ga', 3)", "('O', 'O')_175_185", "Bond_('In', 'O')_std", 'lattice_vector_3_ang', "('Al', 'O')_B_75", "('Al', 'Ga')_A_mean", "('Al', 'Ga')_125_135", "N('Al', 'Ga', 3)", "('O', 'O')_A_75", "('Al', 'In')_85_95", "('Al', 'Ga')_A_25", "('Ga', 'In')_A_25", "('Al', 'In')_115_125", 'lattice_angle_alpha_degree', "N('In', 'O', 1)", "N('Ga', 'O', 3)", "N('Al', 'O', 1)", "('Ga', 'Ga')_125_135", "N('Ga', 'Ga', 0)", "('Ga', 'Ga')_85_95", "('Ga', 'Ga')_A_25", 'dihe_50', "('Al', 'In')_A_mean", "N('Al', 'In', 2)", "('Ga', 'O')_1.9000000000000001_2.0", "Bond_('Ga', 'O')_mean", "('Ga', 'O')_B_25", "('In', 'O')_2.0_2.1", "('Al', 'In')_A_75", 'percent_atom_in', 'percent_atom_ga', "('In', 'In')_115_125", "('In', 'O')_1.9000000000000001_2.0", "('In', 'O')_2.1_2.2", "('Ga', 'In')_A_50", "N('Al', 'O', 3)", "N('Ga', 'O', 2)", "('O', 'O')_95_105", "('O', 'O')_105_115", "N('In', 'O', 2)", "('Al', 'Al')_125_135", "('In', 'In')_A_25", "N('Al', 'In', 3)", "('O', 'O')_A_25", "N('In', 'Ga', 3)", "('Al', 'Al')_A_25", "('Ga', 'Ga')_A_std", "Bond_('Al', 'O')_std", "('Al', 'O')_B_50", 'Norm3', 'Norm2', 'Norm7', 'Norm5', "('Al', 'O')_2.0_2.1", "('Ga', 'In')_A_std", "('O', 'O')_A_50", "N('O', 'O', 0)", 'vol', "('Al', 'O')_2.1_2.2", "('Al', 'In')_A_std", "N('Ga', 'O', 1)", "('In', 'O')_2.2_2.3000000000000003", "('O', 'O')_A_std", 'lattice_angle_gamma_degree', "N('Al', 'Ga', 2)", "N('In', 'O', 3)", 'dihe_75', "('In', 'O')_1.8_1.9000000000000001", "('In', 'In')_85_95", "N('Al', 'Al', 2)", "('Al', 'O')_1.9000000000000001_2.0", "N('In', 'In', 2)", "N('In', 'Ga', 2)", "('In', 'O')_B_75", "('Al', 'Al')_95_105", 'lattice_vector_2_ang', "('In', 'O')_B_25", "('Al', 'In')_A_25", "('Al', 'Al')_A_mean", "('Al', 'Ga')_A_75", "('In', 'In')_105_115", "('Ga', 'Ga')_A_50", "('Ga', 'O')_2.2_2.3000000000000003", 'lattice_vector_1_ang', "('Ga', 'In')_A_75", "('Al', 'Al')_A_std", "N('Al', 'O', 4)", "N('Ga', 'O', 0)", "N('Al', 'Al', 3)", "('In', 'O')_B_50", "N('In', 'In', 3)", 'percent_atom_al', "('Al', 'Al')_115_125", "('Ga', 'O')_2.0_2.1", "('Al', 'Ga')_A_std"]
    cols = ["('In', 'O', 0)", "('In', 'In')_A_50", 'force2_75', "('In', 'In')_125_135", "('Al', 'Ga')_125_135", "('In', 'In', 0)", 'Norm10', 'lattice_angle_beta_degree', "('Ga', 'Ga', 0)", "('O', 'O')_115_125", 'dihe_126.0_128.0', "('Al', 'Al')_85_95", "('Al', 'Ga')_A_std", 'IonChar_std', "('In', 'In')_75_85", 'lattice_vector_3_ang', "N('In', 'Ga', 0)", 'dihe_64.0_66.0', 'dihe_124.0_126.0', 'dihe_14.0_16.0', "('In', 'In', 4)", 'period_mean', "N('Al', 'Al', 0)", "N('In', 'O', 2)", "('Al', 'Ga')_115_125", 'RS_mean', 'dihe_mean', 'dihe_130.0_132.0', "N('In', 'In', 0)", "N('Al', 'O', 3)", 'dihe_16.0_18.0', "('Al', 'O', 3)", "('Ga', 'In')_95_105", "('Al', 'In')_125_135", "('Al', 'Al')_125_135", "('Al', 'Al')_A_50", 'dihe_58.0_60.0', "('In', 'Ga', 3)", "('Al', 'O')_B_25", "('Al', 'In')_85_95", 'dihe_104.0_106.0', "Bond_('Ga', 'O')_std", 'dihe_118.0_120.0', 'EN_25', 'LUMO_std', 'VOL_mean', "('Al', 'Ga')_A_50", 'dihe_56.0_58.0', "('In', 'In', 1)", 'dihe_110.0_112.0', 'dihe_30.0_32.0', "('In', 'In')_A_25", "('In', 'O', 1)", "('Ga', 'Ga')_95_105", 'dihe_166.0_168.0', "('Ga', 'Ga')_A_75", "('Al', 'Al')_A_75", "('Ga', 'O', 4)", 'HOMO_75', "('Ga', 'In')_125_135", 'dihe_44.0_46.0', "('In', 'O')_2.0_2.1", "N('In', 'In', 1)", 'dihe_42.0_44.0', "('Al', 'Ga')_95_105", "('Ga', 'Ga')_115_125", "('Al', 'Ga')_105_115", 'dihe_132.0_134.0', "('Al', 'O', 4)", "('In', 'O')_B_75", "('Al', 'O')_1.8_1.9000000000000001", "('O', 'O')_75_85", "('Al', 'Ga')_85_95", "('Al', 'In')_A_50", 'dihe_146.0_148.0', "('In', 'Ga', 0)", 'force1_75', 'spacegroup', "('Ga', 'O')_B_75", "('O', 'O')_175_185", 'dihe_nan', 'dihe_136.0_138.0', "('Al', 'In')_A_75", 'dihe_0.0_2.0', "N('Ga', 'O', 3)", 'RD_std', "('Ga', 'Ga', 2)", "N('Ga', 'O', 4)", "N('Al', 'O', 1)", 'dihe_74.0_76.0', "N('Al', 'Ga', 1)", 'dihe_156.0_158.0', "('Ga', 'Ga')_A_std", "('In', 'In')_A_mean", "('Al', 'In')_95_105", "('In', 'In', 2)", 'dihe_106.0_108.0', "('Al', 'Ga')_A_75", 'dihe_84.0_86.0', "('Ga', 'O')_2.1_2.2", "('Ga', 'O')_B_50", 'dihe_66.0_68.0', "N('Ga', 'Ga', 0)", "N('Ga', 'Ga', 2)", "('O', 'O')_85_95", 'dihe_138.0_140.0', 'dihe_154.0_156.0', "N('Al', 'In', 0)", 'M_E', 'IP_std', 'dihe_50', "('In', 'In')_A_75", "('Al', 'In')_105_115", 'RS_std', 'RP_mean', 'dihe_38.0_40.0', "('Ga', 'In')_115_125", 'dihe_100.0_102.0', 'dihe_12.0_14.0', "('O', 'O')_155_165", 'dihe_22.0_24.0', "('O', 'O', 2)", "('Ga', 'In')_A_50", 'dihe_std', "N('Al', 'O', 0)", "('Ga', 'In')_85_95", "('Ga', 'In')_A_mean", "('Al', 'Ga', 2)", 'dihe_152.0_154.0', 'dihe_25', 'dihe_52.0_54.0', "('O', 'O')_A_mean", "('Al', 'Al', 0)", 'dihe_168.0_170.0', 'dihe_34.0_36.0', "N('Al', 'Ga', 0)", "('Ga', 'Ga', 3)", 'dihe_94.0_96.0', 'dihe_50.0_52.0', "('In', 'O')_B_50", 'z2', "('Ga', 'O')_1.8_1.9000000000000001", 'dihe_72.0_74.0', "('Al', 'In')_115_125", "('Ga', 'In')_105_115", "('Ga', 'Ga')_A_mean", 'dihe_116.0_118.0', 'lattice_angle_gamma_degree', "('In', 'In')_95_105", "Bond_('In', 'O')_std", "N('In', 'O', 0)", "N('Al', 'In', 1)", "N('O', 'O', 2)", "('In', 'Ga', 2)", 'dihe_70.0_72.0', 'RD_mean', "('In', 'In', 3)", "('Ga', 'Ga')_105_115", "('Al', 'Ga')_A_mean", 'dihe_32.0_34.0', "('Al', 'O')_2.2_2.3000000000000003", 'dihe_160.0_162.0', "N('Al', 'Ga', 3)", 'dihe_120.0_122.0', "('Al', 'O')_B_50", 'dihe_6.0_8.0', "('Al', 'Ga', 1)", "('O', 'O')_A_75", 'dihe_172.0_174.0', "('Al', 'Ga')_A_25", "('Ga', 'In')_A_25", 'dihe_134.0_136.0', 'lattice_angle_alpha_degree', "N('In', 'O', 1)", 'RP_25', 'dihe_2.0_4.0', 'dihe_86.0_88.0', "('Ga', 'O', 1)", 'dihe_68.0_70.0', "('Ga', 'Ga')_125_135", 'dihe_28.0_30.0', "('Ga', 'Ga')_85_95", "('Ga', 'Ga')_A_25", 'dihe_90.0_92.0', "N('Ga', 'Ga', 3)", "('Al', 'In')_A_mean", "N('Al', 'In', 2)", 'number_of_total_atoms', 'dihe_174.0_176.0', "Bond_('Al', 'O')_mean", 'dihe_62.0_64.0', "('Ga', 'O')_1.9000000000000001_2.0", 'predict1_0.485', "('Al', 'In', 2)", 'z1', 'dihe_92.0_94.0', 'dihe_176.0_178.0', "Bond_('Ga', 'O')_mean", "('Ga', 'O')_B_25", 'dihe_144.0_146.0', "('Al', 'O')_2.1_2.2", 'force2_mean', 'force2_50', 'force2_25', 'dihe_18.0_20.0', 'dihe_162.0_164.0', 'percent_atom_ga', "N('Al', 'Ga', 2)", "('In', 'O')_1.9000000000000001_2.0", "('In', 'O')_2.1_2.2", 'dihe_96.0_98.0', 'dihe_54.0_56.0', 'force1_std', 'VOL_25', "N('Al', 'Al', 3)", "('O', 'O', 1)", 'dihe_88.0_90.0', 'period_std', "('Ga', 'O', 0)", "('Al', 'Ga', 3)", 'dihe_140.0_142.0', "('Al', 'Ga', 0)", "('O', 'O')_95_105", "('O', 'O')_105_115", "('Al', 'Al', 2)", "('Al', 'Al')_105_115", 'dihe_76.0_78.0', "Bond_('In', 'O')_mean", "N('Al', 'In', 3)", 'dihe_102.0_104.0', 'dihe_46.0_48.0', 'dihe_148.0_150.0', "('O', 'O')_A_25", "N('In', 'Ga', 3)", "('Al', 'Al')_A_25", "('Al', 'O', 0)", "('Al', 'In', 3)", 'dihe_20.0_22.0', "Bond_('Al', 'O')_std", 'dihe_112.0_114.0', "N('In', 'In', 4)", 'Norm2', 'dihe_98.0_100.0', 'Norm7', "N('In', 'In', 3)", 'Norm5', "('Al', 'O')_2.0_2.1", "('Ga', 'In')_A_std", 'dihe_82.0_84.0', "('O', 'O')_A_50", "('In', 'O', 2)", 'vol', 'dihe_178.0_180.0', 'dihe_150.0_152.0', "('O', 'O', 0)", "('Al', 'In')_A_std", "N('Ga', 'O', 1)", "('Ga', 'O', 3)", 'LUMO_mean', "('In', 'O')_2.2_2.3000000000000003", 'EA_mean', "('O', 'O')_A_std", 'RD_25', 'dihe_48.0_50.0', "('In', 'In')_115_125", "('Al', 'Al', 3)", 'dihe_75', 'IonChar_mean', "('In', 'In')_85_95", "N('Al', 'Al', 2)", "N('In', 'O', 3)", "('Al', 'O')_1.9000000000000001_2.0", 'dihe_4.0_6.0', "('Al', 'O', 1)", "N('In', 'In', 2)", "N('In', 'Ga', 2)", 'force1_25', "('In', 'O')_1.8_1.9000000000000001", 'dihe_158.0_160.0', 'EA_std', 'force1_50', 'dihe_10.0_12.0', 'lattice_vector_2_ang', 'force1_mean', "('Al', 'Al')_95_105", "('In', 'O')_B_25", 'dihe_26.0_28.0', 'dihe_60.0_62.0', "('Al', 'O', 2)", 'VOL_sum', "('Al', 'Al')_A_mean", 'dihe_80.0_82.0', "('Al', 'O')_B_75", "('In', 'O', 3)", "('In', 'In')_105_115", "N('O', 'O', 1)", "('Ga', 'Ga')_A_50", 'dihe_122.0_124.0', "('Ga', 'O')_2.2_2.3000000000000003", 'lattice_vector_1_ang', 'dihe_36.0_38.0', 'dihe_128.0_130.0', "('Al', 'In')_A_25", "N('Al', 'O', 2)", "('Ga', 'In')_A_75", "('Al', 'Al')_A_std", 'RP_std', "N('Al', 'O', 4)", 'lj_E', "('Ga', 'O', 2)", "N('Ga', 'O', 0)", "N('O', 'O', 0)", 'percent_atom_in', "('Al', 'Al')_75_85", 'HOMO_std', 'dihe_170.0_172.0', "('Al', 'Al')_115_125", 'dihe_40.0_42.0', 'dihe_78.0_80.0', "('In', 'In')_A_std", "('Al', 'In', 0)", 'dihe_114.0_116.0', 'dihe_24.0_26.0', 'percent_atom_al', "('Al', 'In')_75_85", 'dihe_164.0_166.0', "('Ga', 'O')_2.0_2.1", 'dihe_142.0_144.0', "N('Ga', 'O', 2)"]    
    '''
    remove less imprt features *3
    0.023780121738
    0.0764123392778
    0.0500962305079
    '''
    cols = [x for x in cols if x not in ['z1','z2']]
    cols = [x for x in train.keys() if x not in ['id',target1,target2,'predict1','predict2'] and 'array' not in x]
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
    cols_model12 = list(np.copy(cols))
    comps=1
    '''
        train['z1']= 0
        train['z2']= 0
        test['z1'] = 0
        test['z2'] = 0
        for i in range(0,10):
            test_id = [x for x in range(0,2400) if x%10 == i] 
            train_id = [x for x in range(0,2400) if x%10 != i] 
            scaler = StandardScaler()
            regr = linear_model.LinearRegression()
            scaler = scaler.fit(pd.concat([train,test])[ori_cols].values)
            train = train.set_value(test_id,'z1',regr.fit(scaler.transform(train.iloc[train_id][ori_cols].values),
                                   train.iloc[train_id][target1].values).predict(scaler.transform(train.iloc[test_id][ori_cols].values)))
            regr = linear_model.LinearRegression()
            train = train.set_value(test_id,'z2', regr.fit(scaler.transform(train.iloc[train_id][ori_cols].values),
                                   train.iloc[train_id][target2].values).predict(scaler.transform(train.iloc[test_id][ori_cols].values)))
            regr = linear_model.LinearRegression()
            test= test.set_value(range(600),'z1', test['z1']+regr.fit(scaler.transform(train.iloc[train_id][ori_cols].values),
                                  train.iloc[train_id][target1].values).predict(scaler.transform(test[ori_cols].values))/10)
            regr = linear_model.LinearRegression()
            test= test.set_value(range(600),'z2', test['z2']+regr.fit(scaler.transform(train.iloc[train_id][ori_cols].values),
                                  train.iloc[train_id][target2].values).predict(scaler.transform(test[ori_cols].values))/10)
        xgtest = xgb.DMatrix(test[cols].values,missing=np.NAN,feature_names=cols)
    '''

    '''150 3 90 0.050112912902007464'''
    for repeat in range(3,4):
        estimators,leaf,max_features = 150,3,90
        seeds = 10*[1,5516,643,5235,2352,12,5674,19239,41241,1231,151,34,1235,2664,
                 75764,2314,1111,2222,3333,4444,5555,6666,7777,8888,9999,111,222,333,444,555,666,777,888,999,11,22,
                    33,44,55,66,77,88,99,12,23,34,45,56,67,78,89,90]
        seeds = seeds[repeat*20:(repeat+1)*20]
        train['predict1'] = 0
        test[target1] = 0
        train['predict2'] = 0
        test[target2] = 0
        print seeds
        for seed in seeds:
            train = train.sample(2400,random_state=seed).reset_index(drop=True)
            for i in range(0,10):
                test_id = [x for x in range(0,2400) if x%10 == i] 
                train_id = [x for x in range(0,2400) if x%10 != i] 
                scaler = StandardScaler()
                regr = linear_model.LinearRegression()
                ori_cols1 = [x for x in ori_cols if '0.495' not in x and 'Eswald' not in x]
                ori_cols2 = [x for x in ori_cols if 'Eswald' not in x]# if '0.485' not in x]# if 'predict2_0.485' not in x] causes 0.0755 instead of 0.0735
                scaler = scaler.fit(pd.concat([train,test])[ori_cols1].values)
                train['z1'] = regr.fit(scaler.transform(train.iloc[train_id][ori_cols1].values),
                                       train.iloc[train_id][target1].values).predict(scaler.transform(train[ori_cols1].values))
                regr = linear_model.LinearRegression()
                test['z1'] = regr.fit(scaler.transform(train.iloc[train_id][ori_cols1].values),
                                      train.iloc[train_id][target1].values).predict(scaler.transform(test[ori_cols1].values))
                scaler = scaler.fit(pd.concat([train,test])[ori_cols2].values)
                regr = linear_model.LinearRegression()
                train['z2'] = regr.fit(scaler.transform(train.iloc[train_id][ori_cols2].values),
                                       train.iloc[train_id][target2].values).predict(scaler.transform(train[ori_cols2].values))
                regr = linear_model.LinearRegression()
                test['z2'] = regr.fit(scaler.transform(train.iloc[train_id][ori_cols2].values),
                                      train.iloc[train_id][target2].values).predict(scaler.transform(test[ori_cols2].values))
                
        ##    for i in range(0,10):
                test_id = [x for x in range(0,2400) if x%10 == i] 
                train_id = [x for x in range(0,2400) if x%10 != i] 
                from sklearn.decomposition import PCA
                regr = PCA(n_components=comps)
                names = []
                for i in range(comps):
                    names += ['zz'+str(i),]
                for name in names :
                    train[name] = 0
                    test[name] = 0
                scaler = scaler.fit(pd.concat([train,test])[ori_cols].values)
                temp = regr.fit(scaler.transform(pd.concat([train,test])[ori_cols].values)).transform(scaler.transform(train[ori_cols].values))
                train = train.set_value(train.index,names,temp);
                temp = regr.fit(scaler.transform(pd.concat([train,test])[ori_cols].values)).transform(scaler.transform(test[ori_cols].values))
                test = test.set_value(test.index,names,temp)
                if 'z1' not in cols:
                    cols_model12 += ['z1',]  #+names #PCA does not help
                if 'z2' not in cols:
                    cols_model12 += ['z2',]  #+names #PCA does not help
                cols_model12 = [x for x in cols_model12 if x != removed_feature]

                from sklearn.ensemble import RandomForestRegressor
                
                cols = [x for x in cols_model12 if x != removed_feature and  '0.495' not \
                        in x and "Eswald" not in x and 'CNN' not in x ]
                X_train, y_train1,y_train2 = train.iloc[train_id][cols], train.iloc[train_id][target1],\
                                             train.iloc[train_id][target2]
                X_test, y_test1,y_test2 = train.iloc[test_id][cols], train.iloc[test_id][target1],\
                                          train.iloc[test_id][target2]
                X_test2 = test[cols]
                model1 = RandomForestRegressor(n_estimators=estimators,min_samples_leaf =leaf,
                                               max_features= max_features,n_jobs=1)
                model1 = model1.fit(X_train,y_train1)
                #print np.mean((model1.predict(X_train)-y_train1)**2)**.5,np.mean((model1.predict(X_test)-y_test1)**2)**.5
                train = train.set_value(test_id,'predict1',train.iloc[test_id]['predict1']+model1.predict(X_test)/len(seeds))
                test = test.set_value(test.index, target1, test[target1]+model1.predict(X_test2)/(10*len(seeds)))
                dictt = {}
                for i in range(len(cols)):
                    dictt[cols[i]] =  model1.feature_importances_[i]
                dictt_cols1[len(dictt_cols1.keys())+1] = dictt_cols1[1].map(dictt)

 
                cols = [x for x in cols_model12 if x != removed_feature  and 'predict2_0.495' \
                        not in x and 'CNN' not in x and 'Eswald' not in x]
                X_train, y_train1,y_train2 = train.iloc[train_id][cols], train.iloc[train_id][target1],\
                                             train.iloc[train_id][target2]
                X_test, y_test1,y_test2 = train.iloc[test_id][cols], train.iloc[test_id][target1],\
                                          train.iloc[test_id][target2]
                X_test2 = test[cols]
                model2 = RandomForestRegressor(n_estimators=estimators,min_samples_leaf =leaf,
                                               max_features= max_features,n_jobs=1)
                model2 = model1.fit(X_train,y_train2)
                #print np.mean((model2.predict(X_train)-y_train2)**2)**.5,np.mean((model2.predict(X_test)-y_test2)**2)**.5
        
                train = train.set_value(test_id,'predict2',train.iloc[test_id]['predict2']+model2.predict(X_test)/len(seeds))
                test = test.set_value(test.index, target2, test[target2]+model2.predict(X_test2)/(10*len(seeds)))
                dictt = {}
                for i in range(len(cols)):
                    dictt[cols[i]] =  model1.feature_importances_[i]
                dictt_cols2[len(dictt_cols2.keys())+1] = dictt_cols2[1].map(dictt)
        a,b = np.mean((train['predict1']-train[target1])**2)**.5, np.mean((train['predict2']-train[target2])**2)**.5
        print a
        print b
        print estimators,leaf,max_features,(a+b)/2
        log.write(str(seeds)+'\n')
        log.write('%s\n%s\n%s\n'%(a,b,(a+b)/2))
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
        print dictt_cols1[[1,'avg']].sort_values('avg').iloc[:50]
        print dictt_cols2[[1,'avg']].sort_values('avg').iloc[:50]
        print dictt_cols1[[1,'avg']].sort_values('avg').iloc[-50:]
        print dictt_cols2[[1,'avg']].sort_values('avg').iloc[-50:]
    print list(set(list(dictt_cols1[[1,'avg']].sort_values('avg').iloc[-160:][1])+list(dictt_cols2[[1,'avg']].sort_values('avg').iloc[-160:][1])))

log.close()

def write_files():
    train[target1] = np.exp(train[target1])-1
    train[target2] = np.exp(train[target2])-1
       
    permutation_seed = 'RF'
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
    if True:
        test[target1] = np.exp(test[target1])-1
        test[target2] = np.exp(test[target2])-1
        test[['id',target1,target2]].to_csv('submit_test_%s_%s.csv'%(permutation_seed,np.round((a+b)/2,5)),index=0)
    list(set(list(dictt_cols1[[1,'avg']].sort_values('avg').iloc[-70:][1]) + list(dictt_cols2[[1,'avg']].sort_values('avg').iloc[-70:][1])))
    train[target1] = np.log(1+train[target1])
    train[target2] = np.log(1+train[target2])
    test[target1] = np.log1p(test[target1])
    test[target2] = np.log1p(test[target2])
write_files()
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
[41241, 1231, 151, 34, 1235, 2664, 75764, 2314, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 111, 222, 333]
0.024381132559098906
0.07570733725112831
150 3 90 0.05004423490511361
                                                     1       avg
226                                ('In', 'O')_1.5_1.6  0.000000
213                 ('Ga', 'O')_1.6_1.7000000000000002  0.000000
369                                     predict2_0.495  0.000000
368                                     predict1_0.495  0.000000
227                 ('In', 'O')_1.6_1.7000000000000002  0.000000
200                 ('Al', 'O')_1.6_1.7000000000000002  0.016187
222                                ('Ga', 'O')_2.5_2.6  0.066853
193                                 ('O', 'O')_135_145  0.069315
346                                              RP_75  0.084659
209                                ('Al', 'O')_2.5_2.6  0.097837
170                               ('Ga', 'Ga')_165_175  0.123758
356                                              IP_75  0.149563
236                                ('In', 'O')_2.5_2.6  0.172576
228                                ('In', 'O')_1.7_1.8  0.177810
122                               ('Al', 'Ga')_165_175  0.188642
29                                     ('Ga', 'Ga', 1)  0.190076
208  ('Al', 'O')_2.4000000000000004_2.5000000000000004  0.223583
164                                 ('Ga', 'Ga')_75_85  0.235569
349                                              RS_25  0.239783
318                                   dihe_154.0_156.0  0.264779
221  ('Ga', 'O')_2.4000000000000004_2.5000000000000004  0.291666
146                               ('Ga', 'In')_165_175  0.321044
235  ('In', 'O')_2.4000000000000004_2.5000000000000004  0.321337
158                               ('Al', 'Al')_165_175  0.354461
220  ('Ga', 'O')_2.3000000000000003_2.4000000000000004  0.377050
86                                    N('Ga', 'Ga', 1)  0.396478
134                               ('Al', 'In')_165_175  0.411297
362                                              RD_25  0.428010
182                               ('In', 'In')_165_175  0.446568
247                                     dihe_10.0_12.0  0.448726
21                                     ('Al', 'In', 1)  0.452701
25                                     ('Al', 'Al', 1)  0.454732
320                                   dihe_158.0_160.0  0.523665
201                                ('Al', 'O')_1.7_1.8  0.531422
1                                number_of_total_atoms  0.543330
140                                 ('Ga', 'In')_75_85  0.567054
317                                   dihe_152.0_154.0  0.589025
297                                   dihe_112.0_114.0  0.591537
214                                ('Ga', 'O')_1.7_1.8  0.598913
313                                   dihe_144.0_146.0  0.610866
345                                              RP_25  0.635479
250                                     dihe_16.0_18.0  0.646328
207  ('Al', 'O')_2.3000000000000003_2.4000000000000004  0.650708
324                                   dihe_166.0_168.0  0.650735
251                                     dihe_18.0_20.0  0.657161
192                                 ('O', 'O')_115_125  0.659183
367                                            HOMO_75  0.661556
328                                   dihe_174.0_176.0  0.666238
38                                     ('In', 'Ga', 1)  0.690647
234  ('In', 'O')_2.3000000000000003_2.4000000000000004  0.715526
                                                     1       avg
227                 ('In', 'O')_1.6_1.7000000000000002  0.000000
226                                ('In', 'O')_1.5_1.6  0.000000
369                                     predict2_0.495  0.000000
213                 ('Ga', 'O')_1.6_1.7000000000000002  0.000000
200                 ('Al', 'O')_1.6_1.7000000000000002  0.001141
193                                 ('O', 'O')_135_145  0.001976
209                                ('Al', 'O')_2.5_2.6  0.016492
222                                ('Ga', 'O')_2.5_2.6  0.018122
170                               ('Ga', 'Ga')_165_175  0.029282
221  ('Ga', 'O')_2.4000000000000004_2.5000000000000004  0.035534
38                                     ('In', 'Ga', 1)  0.037249
122                               ('Al', 'Ga')_165_175  0.045055
146                               ('Ga', 'In')_165_175  0.045828
182                               ('In', 'In')_165_175  0.050507
208  ('Al', 'O')_2.4000000000000004_2.5000000000000004  0.051974
236                                ('In', 'O')_2.5_2.6  0.054259
134                               ('Al', 'In')_165_175  0.062429
29                                     ('Ga', 'Ga', 1)  0.062699
1                                number_of_total_atoms  0.062718
164                                 ('Ga', 'Ga')_75_85  0.072205
140                                 ('Ga', 'In')_75_85  0.074879
158                               ('Al', 'Al')_165_175  0.077326
95                                    N('In', 'Ga', 1)  0.088136
349                                              RS_25  0.089579
21                                     ('Al', 'In', 1)  0.090779
220  ('Ga', 'O')_2.3000000000000003_2.4000000000000004  0.091323
116                                 ('Al', 'Ga')_75_85  0.103872
86                                    N('Ga', 'Ga', 1)  0.116150
235  ('In', 'O')_2.4000000000000004_2.5000000000000004  0.116306
152                                 ('Al', 'Al')_75_85  0.116702
25                                     ('Al', 'Al', 1)  0.136668
207  ('Al', 'O')_2.3000000000000003_2.4000000000000004  0.149632
128                                 ('Al', 'In')_75_85  0.169779
176                                 ('In', 'In')_75_85  0.174844
46                                     ('In', 'In', 2)  0.217766
214                                ('Ga', 'O')_1.7_1.8  0.227763
313                                   dihe_144.0_146.0  0.235832
250                                     dihe_16.0_18.0  0.238824
167                               ('Ga', 'Ga')_105_115  0.254215
318                                   dihe_154.0_156.0  0.261461
322                                   dihe_162.0_164.0  0.269357
247                                     dihe_10.0_12.0  0.296227
143                               ('Ga', 'In')_105_115  0.322332
276                                     dihe_68.0_70.0  0.323308
325                                   dihe_168.0_170.0  0.329333
155                               ('Al', 'Al')_105_115  0.329461
319                                   dihe_156.0_158.0  0.340532
119                               ('Al', 'Ga')_105_115  0.352074
64                                               Norm0  0.355499
275                                     dihe_66.0_68.0  0.355634
                              1          avg
100              N('O', 'O', 2)    30.297683
76              N('In', 'O', 3)    33.721968
332                         M_E    34.158554
96             N('In', 'Ga', 2)    34.293027
341                   force2_75    34.433152
199            ('Al', 'O')_B_25    34.523756
9     lattice_angle_beta_degree    34.804568
6          lattice_vector_2_ang    36.619769
336                   force1_25    36.715995
334                 force2_mean    41.290471
178         ('In', 'In')_95_105    41.355134
231         ('In', 'O')_2.0_2.1    42.037358
225            ('In', 'O')_B_25    44.352545
183           ('O', 'O')_A_mean    45.939141
5          lattice_vector_1_ang    46.111105
241                     dihe_50    51.574537
79             N('Al', 'In', 2)    52.633488
73              N('In', 'O', 0)    54.077487
371                          z2    54.471689
286              dihe_88.0_90.0    54.574512
56        Bond_('Ga', 'O')_mean    57.696488
4               percent_atom_in    60.555069
337                   force2_25    60.636408
132        ('Al', 'In')_115_125    61.041339
54        Bond_('Al', 'O')_mean    63.389326
174           ('In', 'In')_A_50    70.024827
75              N('In', 'O', 2)    70.433015
363                   LUMO_mean    73.992852
212            ('Ga', 'O')_B_25   102.146069
171         ('In', 'In')_A_mean   104.681910
126           ('Al', 'In')_A_50   133.558565
8    lattice_angle_alpha_degree   137.043996
125           ('Al', 'In')_A_75   179.218266
123         ('Al', 'In')_A_mean   185.390663
180        ('In', 'In')_115_125   185.993116
127           ('Al', 'In')_A_25   200.246792
20              ('Al', 'In', 0)   262.857082
173           ('In', 'In')_A_75   266.366408
300            dihe_118.0_120.0   273.832814
23              ('Al', 'In', 3)   342.703428
124          ('Al', 'In')_A_std   349.580100
366                    HOMO_std   376.042568
196          ('O', 'O')_175_185   552.060630
80             N('Al', 'In', 3)   581.423449
7          lattice_vector_3_ang   637.638364
287              dihe_90.0_92.0   671.671877
77             N('Al', 'In', 0)   766.315577
188            ('O', 'O')_75_85  1345.868040
187             ('O', 'O')_A_25  1361.263657
370                          z1  7288.587430
                              1          avg
5          lattice_vector_1_ang    21.460650
81             N('Al', 'Al', 0)    21.606974
286              dihe_88.0_90.0    22.653057
244                dihe_2.0_4.0    22.946345
100              N('O', 'O', 2)    22.985977
356                       IP_75    23.401489
309            dihe_136.0_138.0    23.724453
99               N('O', 'O', 1)    23.955105
6          lattice_vector_2_ang    25.229927
335                  force1_std    26.971124
336                   force1_25    27.353851
8    lattice_angle_alpha_degree    33.740720
189            ('O', 'O')_85_95    34.275422
255              dihe_26.0_28.0    36.833045
263              dihe_42.0_44.0    36.837197
11                          vol    38.123884
102            N('In', 'In', 1)    39.204721
194          ('O', 'O')_155_165    41.533682
362                       RD_25    48.828969
0                    spacegroup    49.558048
340                   force1_75    49.938398
91              N('Al', 'O', 2)    50.808249
196          ('O', 'O')_175_185    52.082467
183           ('O', 'O')_A_mean    52.279820
331                        lj_E    54.170479
329            dihe_176.0_178.0    55.848454
274              dihe_64.0_66.0    60.554799
334                 force2_mean    60.813642
287              dihe_90.0_92.0    69.190559
300            dihe_118.0_120.0    73.172590
368              predict1_0.495    90.468890
370                          z1   101.490987
89              N('Al', 'O', 0)   116.321354
187             ('O', 'O')_A_25   122.397200
339                   force2_50   124.941517
188            ('O', 'O')_75_85   129.025465
93              N('Al', 'O', 4)   129.704485
360                     RD_mean   149.285688
7          lattice_vector_3_ang   199.932294
344                      RP_std   204.867373
333                 force1_mean   266.655007
4               percent_atom_in   426.366390
364                    LUMO_std   436.245100
338                   force1_50   475.576090
354                      IP_std   909.504277
62                  period_mean  1178.848803
63                   period_std  1502.117283
351                    VOL_mean  1677.815009
353                     VOL_sum  3380.248177
371                          z2  5983.230605

'''

'''
https://media.nature.com/original/nature-assets/npjcompumats/2016/npjcompumats201628/extref/npjcompumats201628-s1.pdf
Average atomic mass
Average column on periodic table
Average  row  on  the  periodic  table
Average atomic radius
verage electronegativity
fraction of valence electrons
'''
