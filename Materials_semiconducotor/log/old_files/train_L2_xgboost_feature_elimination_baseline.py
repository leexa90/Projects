import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
# calculate the volume of the structure


'''
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
dictt_cols1_old = pd.DataFrame(np.load('./log/dictt_cols1.npy'))
dictt_cols2_old = pd.DataFrame(np.load('./log/dictt_cols2.npy'))
removed_features = list(set(list(dictt_cols1_old[0].iloc[-10:])+list(dictt_cols2_old[0].iloc[-10:])))[-3:]
print removed_features
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

    if True:
        print '###doing eswald features###'
        train_E = np.load('train_ewald_sum_data.npy.zip')['train_ewald_sum_data.npy'].item()
        train['Eswald_array'] = train['id'].apply(lambda x : train_E[x-1])
        test_E = np.load('test_ewald_sum_data.npy.zip')['test_ewald_sum_data.npy'].item()
        test['Eswald_array'] = test['id'].apply(lambda x : test_E[x-1])
        for i in range(0,4):
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
    cols = ["('In', 'O', 0)", 'lattice_vector_1_ang', "('In', 'In')_A_50", 'force2_75', "('In', 'In')_125_135", "('Al', 'Ga')_125_135", "('In', 'In', 0)", 'Norm10', 'lattice_angle_beta_degree', "('Ga', 'Ga', 0)", 'dihe_126.0_128.0', "('Al', 'Al')_85_95", "('Al', 'Ga')_A_std", 'IonChar_std', "('In', 'In')_75_85", 'lattice_vector_3_ang', "N('In', 'Ga', 0)", 'dihe_64.0_66.0', 'dihe_124.0_126.0', 'dihe_14.0_16.0', "('In', 'In', 4)", 'period_mean', "N('Al', 'Al', 0)", "N('In', 'O', 2)", "('Al', 'Ga')_115_125", 'RS_mean', 'dihe_mean', 'dihe_130.0_132.0', "N('In', 'In', 0)", "N('Al', 'O', 3)", 'dihe_16.0_18.0', "('Al', 'O', 3)", "('Ga', 'In')_95_105", "('Al', 'In')_125_135", "('Al', 'Al')_125_135", "('Al', 'Al')_A_50", 'dihe_58.0_60.0', "('In', 'Ga', 3)", "('Al', 'O')_B_25", 'dihe_168.0_170.0', "('Al', 'In')_85_95", 'dihe_104.0_106.0', "Bond_('Ga', 'O')_std", 'dihe_118.0_120.0', 'dihe_120.0_122.0', 'LUMO_std', 'VOL_mean', "('Al', 'Ga')_A_50", 'dihe_56.0_58.0', "('In', 'In', 1)", 'dihe_110.0_112.0', 'dihe_30.0_32.0', "('In', 'In')_A_25", "('In', 'O', 1)", "('Ga', 'Ga')_95_105", 'dihe_166.0_168.0', "('Ga', 'Ga')_A_75", "('Al', 'Al')_A_75", "('Ga', 'O', 4)", 'HOMO_75', "('Ga', 'In')_125_135", 'dihe_44.0_46.0', "('In', 'O')_2.0_2.1", "N('In', 'In', 1)", 'dihe_42.0_44.0', "('Al', 'Ga')_95_105", "('Ga', 'Ga')_115_125", "('Al', 'Ga')_105_115", 'dihe_132.0_134.0', "('Al', 'O', 4)", "('In', 'O')_B_75", "('Al', 'O')_1.8_1.9000000000000001", "('O', 'O')_75_85", "('Al', 'Ga')_85_95", "('Al', 'In')_A_50", 'dihe_60.0_62.0', "('In', 'Ga', 0)", 'force1_75', 'spacegroup', "('Ga', 'O')_B_75", "('O', 'O')_175_185", 'dihe_nan', 'dihe_136.0_138.0', "('Al', 'In')_A_75", 'dihe_0.0_2.0', "N('Ga', 'O', 3)", 'RD_std', "('Ga', 'Ga', 2)", "N('Ga', 'O', 4)", "('Ga', 'O', 1)", 'dihe_54.0_56.0', "N('Al', 'Ga', 1)", 'dihe_156.0_158.0', "('Ga', 'Ga')_A_std", "('In', 'In')_A_mean", "('Al', 'In')_95_105", "('In', 'In', 2)", 'dihe_106.0_108.0', "('Al', 'Ga')_A_75", 'dihe_84.0_86.0', "('Ga', 'O')_2.1_2.2", 'dihe_10.0_12.0', 'dihe_122.0_124.0', "N('Ga', 'Ga', 0)", "N('Ga', 'Ga', 2)", "('O', 'O')_85_95", 'dihe_138.0_140.0', "N('Al', 'In', 0)", 'M_E', 'IP_std', 'dihe_50', "('In', 'In')_A_75", "('Al', 'In')_105_115", 'RS_std', 'RP_mean', 'dihe_38.0_40.0', "('Ga', 'In')_115_125", 'dihe_100.0_102.0', 'dihe_12.0_14.0', "('O', 'O')_155_165", 'dihe_22.0_24.0', "('O', 'O', 2)", "('Ga', 'O')_B_50", 'dihe_std', "N('Al', 'O', 0)", "('Ga', 'In')_85_95", "('Ga', 'In')_A_mean", "('Al', 'Ga', 2)", "N('Al', 'Ga', 0)", 'dihe_25', 'dihe_52.0_54.0', "('O', 'O')_A_mean", "('Al', 'Al', 0)", "('In', 'In', 3)", 'dihe_34.0_36.0', 'dihe_152.0_154.0', "('Ga', 'Ga', 3)", 'dihe_94.0_96.0', 'dihe_50.0_52.0', "('In', 'O')_B_50", 'z2', "('Ga', 'O')_1.8_1.9000000000000001", 'dihe_72.0_74.0', "('Al', 'In')_115_125", "('Ga', 'In')_105_115", "('Ga', 'Ga')_A_mean", 'dihe_116.0_118.0', "('In', 'In')_95_105", "Bond_('In', 'O')_std", "N('In', 'O', 0)", "N('Al', 'In', 1)", "N('O', 'O', 2)", "('In', 'Ga', 2)", 'dihe_70.0_72.0', 'RD_mean', "('Ga', 'Ga')_105_115", "('Al', 'Ga')_A_mean", 'dihe_32.0_34.0', "('Al', 'O')_2.2_2.3000000000000003", "N('Al', 'Ga', 3)", "('Al', 'O')_B_50", 'dihe_6.0_8.0', "('Al', 'Ga', 1)", "('O', 'O')_A_75", 'dihe_172.0_174.0', "('Al', 'Ga')_A_25", "('Ga', 'In')_A_25", 'dihe_134.0_136.0', 'lattice_angle_alpha_degree', "N('In', 'O', 1)", 'RP_25', 'dihe_2.0_4.0', 'dihe_86.0_88.0', "N('Al', 'O', 1)", "('Ga', 'Ga')_125_135", 'dihe_28.0_30.0', "('Ga', 'Ga')_85_95", "('Ga', 'Ga')_A_25", 'dihe_90.0_92.0', "N('Ga', 'Ga', 3)", "('Al', 'In')_A_mean", "N('Al', 'In', 2)", 'number_of_total_atoms', 'dihe_174.0_176.0', "Bond_('Al', 'O')_mean", 'dihe_62.0_64.0', "('Ga', 'O')_1.9000000000000001_2.0", "('Al', 'In', 2)", 'z1', 'dihe_92.0_94.0', 'dihe_176.0_178.0', "Bond_('Ga', 'O')_mean", "('Ga', 'O')_B_25", 'dihe_144.0_146.0', "('Al', 'O')_2.1_2.2", 'force2_mean', 'force2_50', 'force2_25', 'dihe_18.0_20.0', 'dihe_162.0_164.0', 'percent_atom_ga', "N('Al', 'Ga', 2)", "('In', 'O')_1.9000000000000001_2.0", "('In', 'O')_2.1_2.2", 'dihe_96.0_98.0', 'dihe_74.0_76.0', 'force1_std', 'VOL_25', "N('Al', 'Al', 3)", "('O', 'O', 1)", 'dihe_88.0_90.0', 'period_std', "('Ga', 'O', 0)", "('Al', 'Ga', 3)", 'dihe_140.0_142.0', "('Al', 'Ga', 0)", "('O', 'O')_95_105", "('O', 'O')_105_115", "('Al', 'Al', 2)", "('Al', 'Al')_105_115", 'dihe_76.0_78.0', "Bond_('In', 'O')_mean", "N('Al', 'In', 3)", 'dihe_102.0_104.0', 'dihe_46.0_48.0', 'dihe_148.0_150.0', "('O', 'O')_A_25", "N('In', 'Ga', 3)", "('Al', 'Al')_A_25", "('Al', 'O', 0)", "('Al', 'In', 3)", 'dihe_20.0_22.0', "Bond_('Al', 'O')_std", 'dihe_112.0_114.0', "N('In', 'In', 4)", 'Norm2', 'dihe_98.0_100.0', 'Norm7', "N('In', 'In', 3)", 'Norm5', "('Al', 'O')_2.0_2.1", "('Ga', 'In')_A_std", 'dihe_82.0_84.0', "('O', 'O')_A_50", "('In', 'O', 2)", 'vol', 'dihe_178.0_180.0', 'dihe_150.0_152.0', 'IP_75', "('O', 'O', 0)", "('Al', 'In')_A_std", "N('Ga', 'O', 1)", "('Ga', 'O', 3)", 'LUMO_mean', "('In', 'O')_2.2_2.3000000000000003", 'EA_mean', "('O', 'O')_A_std", 'RD_25', 'dihe_48.0_50.0', "('In', 'In')_115_125", "('Al', 'Al', 3)", 'dihe_75', 'IonChar_mean', "('In', 'In')_85_95", "N('Al', 'Al', 2)", "N('In', 'O', 3)", "('Al', 'O')_1.9000000000000001_2.0", 'dihe_4.0_6.0', "('Al', 'O', 1)", "N('In', 'In', 2)", "N('In', 'Ga', 2)", 'force1_25', "('In', 'O')_1.8_1.9000000000000001", 'dihe_158.0_160.0', 'EA_std', 'force1_50', "('Al', 'Al')_95_105", 'lattice_vector_2_ang', 'force1_mean', 'lattice_angle_gamma_degree', "('In', 'O')_B_25", 'dihe_26.0_28.0', 'dihe_146.0_148.0', "('Al', 'O', 2)", 'VOL_sum', "('Al', 'Al')_A_mean", 'dihe_80.0_82.0', "('Al', 'O')_B_75", "('In', 'O', 3)", "('In', 'In')_105_115", "N('O', 'O', 1)", "('Ga', 'Ga')_A_50", 'dihe_66.0_68.0', "('Ga', 'O')_2.2_2.3000000000000003", "('Ga', 'In')_A_50", 'dihe_36.0_38.0', 'dihe_128.0_130.0', "('Al', 'In')_A_25", "N('Al', 'O', 2)", "('Ga', 'In')_A_75", "('Al', 'Al')_A_std", 'RP_std', "N('Al', 'O', 4)", 'lj_E', "('Ga', 'O', 2)", "N('Ga', 'O', 0)", "N('O', 'O', 0)", 'percent_atom_in', "('Al', 'Al')_75_85", 'HOMO_std', 'dihe_170.0_172.0', "('Al', 'Al')_115_125", 'dihe_40.0_42.0', 'dihe_78.0_80.0', "('In', 'In')_A_std", "('Al', 'In', 0)", 'dihe_114.0_116.0', 'dihe_24.0_26.0', 'percent_atom_al', 'dihe_164.0_166.0', "('Ga', 'O')_2.0_2.1", 'dihe_142.0_144.0', "N('Ga', 'O', 2)"]
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
    for repeat in range(0,3):
        seeds = 10*[1,5516,643,5235,2352,12,5674,19239,41241,1231,151,34,1235,2664,
                 75764,2314,1111,2222,3333,4444,5555,6666,7777,8888,9999,111,222,333,444,555,666,777,888,999,11,22,
                    33,44,55,66,77,88,99,12,23,34,45,56,67,78,89,90]
        seeds = seeds[repeat*5:(repeat+1)*5]
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
                scaler = scaler.fit(pd.concat([train,test])[ori_cols].values)
                train['z1'] = regr.fit(scaler.transform(train.iloc[train_id][ori_cols].values),
                                       train.iloc[train_id][target1].values).predict(scaler.transform(train[ori_cols].values))
                regr = linear_model.LinearRegression()
                train['z2'] = regr.fit(scaler.transform(train.iloc[train_id][ori_cols].values),
                                       train.iloc[train_id][target2].values).predict(scaler.transform(train[ori_cols].values))
                regr = linear_model.LinearRegression()
                test['z1'] = regr.fit(scaler.transform(train.iloc[train_id][ori_cols].values),
                                      train.iloc[train_id][target1].values).predict(scaler.transform(test[ori_cols].values))
                regr = linear_model.LinearRegression()
                test['z2'] = regr.fit(scaler.transform(train.iloc[train_id][ori_cols].values),
                                      train.iloc[train_id][target2].values).predict(scaler.transform(test[ori_cols].values))
                
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
                temp = regr.fit(scaler.transform(pd.concat([train,test])[ori_cols].values)).transform(scaler.transform(train[ori_cols].values))
                train = train.set_value(train.index,names,temp);
                temp = regr.fit(scaler.transform(pd.concat([train,test])[ori_cols].values)).transform(scaler.transform(test[ori_cols].values))
                test = test.set_value(test.index,names,temp)
                if 'z1' not in cols:
                    cols_model12 += ['z1',]  #+names #PCA does not help
                if 'z2' not in cols:
                    cols_model12 += ['z2',]  #+names #PCA does not help
                cols_model12 = [x for x in cols_model12 if x != removed_feature]

                

                params = {}
                params["objective"] = 'reg:linear' 
                params["eta"] = 0.03
                params["min_child_weight"] = 30
                params["subsample"] = 0.4
                params["colsample_bytree"] = 0.22 # many features here
                params["scale_pos_weight"] = 1
                params["silent"] = 0
                params["max_depth"] = 8
                params['seed']=seed
                #params['maximize'] =True
                params['eval_metric'] =  'rmse'
                if seed %2 == 0:
                    None#params["eta"] = params["eta"]/3
                plst = list(params.items())
                cols = [x for x in cols_model12 if x != removed_feature and  '0.495' not in x and "Eswald" not in x and 'CNN' not in x ]
                X_train, y_train1,y_train2 = train.iloc[train_id][cols], train.iloc[train_id][target1],\
                                             train.iloc[train_id][target2]
                X_test, y_test1,y_test2 = train.iloc[test_id][cols], train.iloc[test_id][target1],\
                                          train.iloc[test_id][target2]
                xgtest = xgb.DMatrix(test[cols].values,missing=np.NAN,feature_names=cols)
                xgtrain = xgb.DMatrix(X_train[cols].values, label=y_train1.values,missing=np.NAN,feature_names=cols)
                xgval = xgb.DMatrix(X_test[cols].values, label=y_test1.values,missing=np.NAN,feature_names=cols)
                watchlist  = [ (xgtrain,'train'),(xgval,'test')]
                model1_a = {}
                model1=xgb.train(plst,xgtrain,5000,watchlist,early_stopping_rounds=200,
                                 evals_result=model1_a,maximize=False,verbose_eval=1000)
                cols = [x for x in cols_model12 if x != removed_feature  and 'predict2_0.495' not in x and 'CNN' not in x]
                X_train, y_train1,y_train2 = train.iloc[train_id][cols], train.iloc[train_id][target1],\
                                             train.iloc[train_id][target2]
                X_test, y_test1,y_test2 = train.iloc[test_id][cols], train.iloc[test_id][target1],\
                                          train.iloc[test_id][target2]
                train = train.set_value(test_id,'predict1',train.iloc[test_id]['predict1']+model1.predict(xgval)/len(seeds))
                test = test.set_value(test.index, target1, test[target1]+model1.predict(xgtest)/(10*len(seeds)))
                dictt_cols1[len(dictt_cols1.keys())+1] = dictt_cols1[1].map(model1.get_fscore())
                xgtest = xgb.DMatrix(test[cols].values,missing=np.NAN,feature_names=cols)
                xgtrain = xgb.DMatrix(X_train[cols].values, label=y_train2.values,missing=np.NAN,feature_names=cols)
                xgval = xgb.DMatrix(X_test[cols].values, label=y_test2.values,missing=np.NAN,feature_names=cols)
                watchlist  = [ (xgtrain,'train'),(xgval,'test')]
                model1_a = {}
                params["eta"] = 0.01
                params["max_depth"] = 8
                if seed %2 == 0:
                    None#params["eta"] = params["eta"]/3
                plst = list(params.items())
                model2=xgb.train(plst,xgtrain,6500,watchlist,early_stopping_rounds=200,
                                 evals_result=model1_a,maximize=False,verbose_eval=1000)
                train = train.set_value(test_id,'predict2',train.iloc[test_id]['predict2']+model2.predict(xgval)/len(seeds))
                test = test.set_value(test.index, target2, test[target2]+model2.predict(xgtest)/(10*len(seeds)))
                dictt_cols2[len(dictt_cols2.keys())+1] = dictt_cols2[1].map(model2.get_fscore())
        a,b = np.mean((train['predict1']-train[target1])**2)**.5, np.mean((train['predict2']-train[target2])**2)**.5
        print a
        print b
        print (a+b)/2
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
##train[target1] = np.exp(train[target1])-1
##train[target2] = np.exp(train[target2])-1
##   
##permutation_seed = 'L2'
##if True:
##    train.to_csv('train_%s_%s.csv'%(permutation_seed,np.round((a+b)/2,5)),index=0)
##    test.to_csv('test_%s_%s.csv'%(permutation_seed,np.round((a+b)/2,5)),index=0)
##    np.save('cols_%s_%s.npy'%(permutation_seed,np.round((a+b)/2,5)),cols)
##if True:
##    name1 = 'predict1_%s'%np.round((a+b)/2,5)
##    name2 = 'predict2_%s'%np.round((a+b)/2,5)
##    test[name1] = test[target1]
##    test[name2] = test[target2]
##    train[name1] = train['predict1']
##    train[name2] = train['predict2']
##    train[['id',name1,name2,target1,target2]].to_csv('model_train_%s_%s.csv'%(permutation_seed,np.round((a+b)/2,5)),index=0)
##    test[['id',name1,name2]].to_csv('model_test_%s_%s.csv'%(permutation_seed,np.round((a+b)/2,5)),index=0)
##if True:
##    test[target1] = np.exp(test[target1])-1
##    test[target2] = np.exp(test[target2])-1
##    test[['id',target1,target2]].to_csv('submit_test_%s_%s.csv'%(permutation_seed,np.round((a+b)/2,5)),index=0)
##list(set(list(dictt_cols1[[1,'avg']].sort_values('avg').iloc[-70:][1]) + list(dictt_cols2[[1,'avg']].sort_values('avg').iloc[-70:][1])))
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
# with eswald 20
0.0242578824265
0.0747007564236
0.049479319425

L2 with 0.498 and normal code *3
0.0242731863625
0.0754817689574
0.0498774776599
*20
0.0240891981651
0.0744876137471
0.0492884059561

benchmark *5
0.03250080113
0.0880216978405
0.0602612494852
with engeinner features *5
0.028781766046
0.0849339614409
0.0568578637434

engineer_features from paper*5
0.026665032169
0.0816681029669
0.054166567568

# engineer_features from paper*5 + bond angles
0.0245717966321
0.0791911408058
0.051881468719

subsample - .4,.2 , minchildweight 30
0.0239410315486
0.0782776287956
0.0511093301721

subsample - .4,.2 , minchildweight 30 + energy
0.023734659522426566
0.07768566908570522
0.05071016430406589


subsample - .4,.2 , minchildweight 30 lr = 0.01,0.03 *20 times
0.023778079611263384
0.0771968674936528
0.05048747355245809

subsample - .4,.2 , minchildweight 30 + energy lr = 0.01,0.03 *20 times
0.023752181966207975
0.07653704132015239
0.05014461164318018

subsample - .4,.2 , minchildweight 30 + energy lr = 0.01,0.03 *20 times
0.023627274977340185
0.07623970604179706
0.04993349050956862

subsample - .4,.2 , minchildweight 30 + energy +dihe all ,lr  = 0.01,0.03 *20 times
0.0236444411868
0.0759608951044
0.0498026681456
                              1         avg
150             ('O', 'O')_A_75  183.417086
153            ('O', 'O')_75_85  190.947214
11                          vol  198.055595
10   lattice_angle_gamma_degree  200.901129
154            ('O', 'O')_85_95  216.069805
148           ('O', 'O')_A_mean  222.199119
58              N('Al', 'O', 4)  228.808007
24                  period_mean  263.938349
206                     dihe_50  265.067483
7          lattice_vector_3_ang  275.732155
64               N('O', 'O', 1)  281.492789
25                   period_std  299.460332
8    lattice_angle_alpha_degree  305.278223
203                    dihe_std  310.953925
207                     dihe_75  323.714476
152             ('O', 'O')_A_25  331.960053
208                        CNN1  347.957447
210                          z1  467.259731
209                        CNN2  511.531939
211                          z2  737.917741
                                      1         avg
97                 ('Al', 'In')_115_125  175.930240
75                      N('Ga', 'O', 4)  183.028885
7                  lattice_vector_3_ang  191.126826
148                   ('O', 'O')_A_mean  193.755681
9             lattice_angle_beta_degree  194.402176
211                                  z2  197.553866
10           lattice_angle_gamma_degree  202.273635
168  ('Al', 'O')_1.9000000000000001_2.0  209.181913
206                             dihe_50  211.853298
202                           dihe_mean  213.543254
152                     ('O', 'O')_A_25  214.879128
205                             dihe_25  221.379413
65                       N('O', 'O', 2)  226.921717
8            lattice_angle_alpha_degree  270.385879
63                       N('O', 'O', 0)  277.364493
207                             dihe_75  284.543782
64                       N('O', 'O', 1)  312.089142
203                            dihe_std  341.195960
208                                CNN1  406.399358
210                                  z1  600.574949
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
