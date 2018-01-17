import pandas as pd
import nltk
import sys
sys.path.append('/media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/deep-learning-workshop/notebooks/5-RNN/glove-python')
import glove
train = pd.read_csv('train.csv.zip')
test = pd.read_csv('test.csv.zip')
data = pd.concat([train,test])
data['comment_text'] = data['comment_text'].fillna('')
tknzr = nltk.TweetTokenizer()
data['tweet_token'] = data['comment_text'].apply(tknzr.tokenize)
##from nltk.tokenize.moses import MosesTokenizer
##tokenizer = MosesTokenizer()
##data['moses'] = data['comment_text'].apply(lambda x : x.decode('utf-8').strip())
##data['moses'] = data['moses'].apply(tokenizer.tokenize)

def word_idx_rnn(list):
    if list is None:
        return None
    return map(lambda word : 2+word_embedding.dictionary.get(word.lower(),-1),list) #0 for mask 1 for unknown

word_embedding =glove.Glove.load_stanford('glove.twitter.27B.25d.txt')
EMBEDDING_DIM = 25
word_embedding_rnn = np.vstack([ 
        np.zeros( (1, EMBEDDING_DIM,), dtype='float32'),   # This is the 'zero' value (used as a mask in Keras)
        np.zeros( (1, EMBEDDING_DIM,), dtype='float32'),   # This is for 'UNK'  (word == 1)
        word_embedding.word_vectors,
    ])
word_embedding_rnn.shape
data['tweet_token2'] = map(word_idx_rnn,data['tweet_token'])
