import hunspell
import pandas as pd
import nltk
import sys

train = pd.read_csv('train.csv.zip',encoding='utf-8').iloc[::2]
test = pd.read_csv('test.csv.zip',encoding='utf-8').iloc[::2]
data = pd.concat([train,test])
data['comment_text'] = data['comment_text'].fillna('')
tknzr = nltk.TreebankWordTokenizer()
data['tweet_token'] = data['comment_text'].apply(tknzr.tokenize)

if True:
    dictt ={}
    for i in range(len(data)):
        for j in set(data['tweet_token'].iloc[i]):
            try:
                dictt[j.lower().encode('ascii', 'ignore')] += 1
            except :
                dictt[j.lower().encode('ascii', 'ignore')] = 1
    sorted_dictt = sorted([ (dictt[x],x) for x in dictt],reverse=True)
    hobj = hunspell.HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')

    for i in sorted_dictt[:6000]:
        if hobj.spell(i[1].encode('ascii', 'ignore')) == False:
            hobj.add(i[1].encode('ascii', 'ignore'))

a = data.iloc[1]['tweet_token']
'''
https://github.com/blatinier/pyhunspell
'''

def preprocess_tokens (x):
    #remove empty
    # spell check
    result = []
    counter = 0 
    for  i in x:
        i = i.lower().encode('ascii', 'ignore')
        try:
            if hobj.spell(i.encode('ascii', 'ignore')) == True:
                result += [i.encode('ascii', 'ignore'),]
            else:
                if len(i) == 1:
                    result += [i.encode('ascii', 'ignore'),]
                else:
                    try:
                        result += [hobj.suggest(i)[0].encode('ascii', 'ignore'),]
                    except : result += [i.encode('ascii', 'ignore'),]
                    counter += 1
                    #print i,hobj.suggest(i)
        except : result += [i.encode('ascii', 'ignore')]
    return result,counter
            
del train,test,data
del a
import gc
gc.collect()
die
if True:
    train = pd.read_csv('train.csv.zip',encoding='utf-8').iloc[::]
    test = pd.read_csv('test.csv.zip',encoding='utf-8').iloc[::]
    data = pd.concat([train,test])
    data['comment_text'] = data['comment_text'].fillna('')
    f1= open('text10','w')
    for i in range(len(data)):
        f1.write(data['comment_text'].iloc[i].replace('\n','').lower().encode('ascii', 'ignore')+' ')
    f1.close()
if True:
    train = pd.read_csv('train.csv.zip',encoding='utf-8').iloc[::]
    test = pd.read_csv('test.csv.zip',encoding='utf-8').iloc[::]
    data = pd.concat([train,test])
    data['comment_text'] = data['comment_text'].fillna('')
    f1= open('text9','w')
    for i in range(len(data)):
        f1.write(data['comment_text'].iloc[i].replace('\n','').lower().encode('ascii', 'ignore')+' ')
    f1.close()
    data['comment_text'] = data['comment_text'].fillna('')
    data['tweet_token'] = data['comment_text'].apply(tknzr.tokenize)
    data['clean'] =''
    data['counter'] =0
    data = data.set_value(data.index,['clean','counter'], map(preprocess_tokens,data['tweet_token']))
    data.to_csv(name.split('.')[0]+'2.csv',index=0)
