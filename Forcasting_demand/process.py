import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('Date_dictionary.csv')
data['date'] = pd.to_datetime(data['date'],infer_datetime_format=True)
data = data.sort_values('date').reset_index(drop= True)
colsI,colsO =[],[]
for i in range(0,70,1):
    colsI += ['area_I_%s' %i,]
    colsO += ['area_O_%s' %i,]
result = {}
for i in range(len(data)):
    result[str(np.datetime64(str(data.iloc[i]['date'])))] = data.iloc[i][colsI].values

#for cast two weeks in advance
data2 = pd.DataFrame(None,
                     columns=['Date','terminal2','0',
                              '1','2','3','4','5','6','7',
                              '8','9','10','11','12','13','14',
                              '-1','-2','-3','-4','-5','-6','7',
                              '-8','-9','-10','-11','-12','-13','-14',
                              '1-7','7-14','14-21','21-28','day'])

initial_date = np.datetime64('2014-03-29 00:00:01')
for i in range(30*8):
    new_date = initial_date+i*24*60*60
    if np.datetime64('2014-08-17 23:59:00') > new_date: #if date is less than final date of dataset
        temp_timeSeries = [[str(new_date),]*70,range(70)]
        for ii in range(15): #get values for particular day before
            temp_timeSeries += [result[str(new_date + ii*24*60*60)],]
        for ii in range(1,15): #get values for particular day before
            temp_timeSeries += [result[str(new_date - ii*24*60*60)],]
        for ranges in [range(0,7),range(7,14),range(14,21),range(21,28)]: #get averages for particular week
            temp_timeSeries += [np.mean([result[str(new_date - ii*24*60*60)] for ii in ranges],0),]
        temp_data = pd.DataFrame(np.stack(temp_timeSeries,-1),
                                 columns=['Date','terminal2','0',
                              '1','2','3','4','5','6','7',
                              '8','9','10','11','12','13','14',
                              '-1','-2','-3','-4','-5','-6','7',
                              '-8','-9','-10','-11','-12','-13','-14',
                              '1-7','7-14','14-21','21-28'])
        data2 = pd.concat([data2,temp_data])

data3=data2.reset_index(drop=True)
data3.to_csv('Demand.csv',index=0)

