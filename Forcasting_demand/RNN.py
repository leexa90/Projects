import numpy as np
seed = 1
np.random.seed(seed)
import pandas as pd
import tensorflow as tf
tf.set_random_seed(seed)
import keras as keras

'''
https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
https://machinelearningmastery.com/define-encoder-decoder-sequence-sequence-model-neural-machine-translation-keras/
'''
data = pd.read_csv('Demand.csv')
data['Date'] = pd.to_datetime(data['Date'],infer_datetime_format=True)
data['Day'] = data['Date'].dt.dayofweek
dictt_date = {}
for i in range(0,7):
    dictt_date[i] = (range(0,7)*3+range(i))[-15:-1][::-1] # get 14 days of the week preceeding the date
data['Day2'] = data['Day'].map(dictt_date)
test = data.iloc[-2100:]
data = data.iloc[:-2100]
data = data.sample(n=len(data),random_state=seed).reset_index(drop=True)
val = data.iloc[1::2].reset_index(drop=True)
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense, TimeDistributed
from keras.utils.vis_utils import plot_model
# configure
num_encoder_tokens = 2
num_decoder_tokens = 1
latent_dim = 256
# Define an input sequence and process it.
encoder_inputs = Input(shape=(14, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]
# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
'''
A Dense output layer is used to predict each character.
This Dense is used to produce each character in the output sequence
in a one-shot manner, rather than recursively, at least during
training. This is because the entire target sequence required for
input to the model is known during training.
The Dense does not need to be wrapped in a TimeDistributed layer.
'''

decoder_dense = Dense(1, activation=None)
decoder_outputs2 = decoder_dense(decoder_outputs)
#decoder_outputs = Dense(num_decoder_tokens, activation=None)(decoder_outputs)
# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs2)
print 'inputs,outputs are :',[encoder_inputs, decoder_inputs], decoder_outputs2
# plot the model
plot_model(model, to_file='model.png', show_shapes=True)
model.summary()
model.compile(optimizer='rmsprop', loss='mse')

encoder_input_data = data[['-1', '-2', '-3', '-4', '-5', '-6', '7.1',
                           '-8', '-9', '-10', '-11', '-12', '-13', '-14']].values
decoder_input_data = data[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']].values
decoder_target_data= data[['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']].values

encoder_input_data = np.stack([encoder_input_data,np.stack(data['Day2'].values,0)],-1)
decoder_input_data = np.expand_dims(decoder_input_data,-1)
decoder_target_data = np.expand_dims(decoder_target_data,-1)
if True:
    Tencoder_input_data = test[['-1', '-2', '-3', '-4', '-5', '-6', '7.1',
                               '-8', '-9', '-10', '-11', '-12', '-13', '-14']].values
    Tdecoder_input_data = test[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']].values
    Tdecoder_target_data= test[['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']].values

    Tencoder_input_data = np.stack([Tencoder_input_data,np.stack(test['Day2'].values,0)],-1)
    Tdecoder_input_data = np.expand_dims(Tdecoder_input_data,-1)
    Tdecoder_target_data = np.expand_dims(Tdecoder_target_data,-1)
if True:
for i in range(0,10):
    model.fit([encoder_input_data[::2], decoder_input_data[::2]], decoder_target_data[::2],
              batch_size=32,verbose=2,
              validation_data = ([encoder_input_data[1::2], decoder_input_data[1::2]], decoder_target_data[1::2]),
              epochs=10,
              validation_split=0.2)
    model.evaluate([Tencoder_input_data[::], Tdecoder_input_data[::]], Tdecoder_target_data[::],verbose=2,)

##    model2 =Model(encoder_inputs,   encoder_states) #inference layer
##    plot_model(model2, to_file='model2.png', show_shapes=True)
##    model.save_weights('LSTM.h5')
##    model2.save_weights('LSTM_feature.h5')
##    test.to_csv('test.csv',index=0)
##    val.to_csv('val.csv',index=0)
'''
Epoch 100/100
 - 8s - loss: 1.9554 - val_loss: 14.6113
'''

