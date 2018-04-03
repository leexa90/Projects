text = []
temp = ''
import pandas as pd
for line in open('MMFF_atom_type.csv'):
    if line.split(',')[0]  != '' and  line.split(',')[1]  != '':
        temp = line.split(',')
        text += [temp[:3],]
    else:
        i =2 
        if line.split(',')[2] != '':
            new_input= list([text[-1][i] +'_'+ line.split(',')[i],])
            text[-1] = text[-1][:i] + new_input + text[-1][i+1:]
        i = 1
        if line.split(',')[1] != '':
            new_input= list([text[-1][i] +'_'+ line.split(',')[i],])
            text[-1] = text[-1][:i] + new_input + text[-1][i+1:]
        print text[-1]
data =  pd.DataFrame(text)
temp = data[1].values
for i in range(len(temp)):
	if temp[i] == '':
		temp[i] = temp[i-1]
data[1] = temp
data.to_csv('MMFF.csv',index=0,header=0)
