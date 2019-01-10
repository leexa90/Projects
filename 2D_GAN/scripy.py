import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
# discriminator
D_InputX = tf.placeholder(tf.float32,(None,2))# input distribution vector
with tf.variable_scope('Discriminator'):
    D_InputX_logit = tf.layers.Dense(20,activation='sigmoid',name='dis1')(D_InputX)
    D_InputX_logit = tf.layers.Dense(20,activation='sigmoid',name='dis2')(D_InputX_logit)
    D_InputX_logit = tf.layers.Dense(1,activation=None,name='dis3')(D_InputX_logit)
D_InputX_prob = tf.nn.sigmoid(D_InputX_logit)

#generator
G_Inputz = tf.placeholder(tf.float32,(None,2))# noise vector
G_Output00 = tf.layers.Dense(2,activation=None)(G_Inputz) # generated samples
G_Output0 = tf.layers.Dense(2,activation=None)(G_Output00) # generated samples
with tf.variable_scope('Discriminator',reuse=True):
    D_G_Inputz_logit = tf.layers.Dense(20,activation='sigmoid',name='dis1')(G_Output0)
    D_G_Inputz_logit = tf.layers.Dense(20,activation='sigmoid',name='dis2')(D_G_Inputz_logit)
    D_G_Inputz_logit = tf.layers.Dense(1,activation=None,name='dis3')(D_G_Inputz_logit)
D_G_Inputz_prob = tf.nn.sigmoid(D_G_Inputz_logit)
    
#cheem way of computing loss due to overflow issues 
lossD = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=np.array([[1.],]*5000).astype(np.float32),logits=D_InputX_logit))+ \
        -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=np.array([[0.],]*5000).astype(np.float32),logits=D_G_Inputz_logit))
lossG = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=np.array([[0.],]*5000).astype(np.float32),logits=D_G_Inputz_logit))
# the mathematical equialvalent of the above two losses are below
lossD2 = tf.reduce_mean(tf.math.log(D_InputX_prob)) + tf.reduce_mean(tf.math.log(1-D_G_Inputz_prob))   
lossG2 = tf.reduce_mean(tf.math.log(1-D_G_Inputz_prob))

Dvar = tf.trainable_variables()[0:6]
Gvar = tf.trainable_variables()[6:]


trainD = tf.train.AdamOptimizer().minimize(-lossD, var_list=Dvar) #gradient accent on Dis
trainG = tf.train.AdamOptimizer().minimize(lossG, var_list=Gvar) #graident decent on Gen

np.random.seed(1)
#z = np.random.multivariate_normal([0,0],np.array([[5,10],[10,12]]),size=(5000)) # noise
#z = np.random.exponential(5,size=(5000,2)) # noise
z = np.random.normal(0,1,size=(5000,2)) #mutlivariate gaussian as x
x = np.random.multivariate_normal([15,15],np.array([[5,10],[10,12]]),size=(5000)) # noise

init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)
loss=[[],[]]
for i in ["%04d" % xxx for xxx in range(5000)]:
    z = np.random.normal(0,1,size=(5000,2)) #mutlivariate gaussian as x

    #z = np.random.normal(0,1,size=(5000,2)) # noise
    if int(i) > 3:
        sess.run(trainD,feed_dict={ G_Inputz : z[:], D_InputX : x[:]})
    sess.run(trainG,feed_dict={ G_Inputz : z[:], D_InputX : x[:]})
    a,b,c,d = sess.run([-lossD,-lossD2,lossG,lossG2],feed_dict={ G_Inputz : z[:], D_InputX : x[:]})
    loss[0] += [a,]
    loss[1] += [c,]
    #a,b = sess.run([-lossD,lossG],feed_dict={ G_Inputz : z[:], D_InputX : x[:]})
    if int(i)%50==0:
        f, ax = plt.subplots(1,2,figsize=(10,5))
        ax[1].plot((-100,5100),(np.log(.5),np.log(.5)),'black')
        ax[1].plot((-100,5100),(-2*np.log(.5),-2*np.log(.5)),'black')
        ax[1].plot(range(len(loss[0])),loss[0],'green',label='Discriminator loss (maximized)')
        ax[1].plot(range(len(loss[1])),loss[1],'orange',label='Generator loss (minimized)')
        plt.text(1500,.1-2*np.log(.5),s='Discriminator loss limit')
        plt.text(1500,.1+1*np.log(.5),s='Generator loss limit')
        ax[1].set_xlabel('epochs');ax[1].set_ylabel('Loss')
        ax[1].set_xlim(0,5000);ax[1].set_ylim(-2,3)
        ax[1].legend(loc='upper left');ax[0].set_title('Loss Curve');
        a,b,c,d = sess.run([-lossD,-lossD2,lossG,lossG2],feed_dict={ G_Inputz : z[:], D_InputX : x[:]})
        print ('lossD:(high is good)',a,b,'lossG:(low is good)',c,d)#,sess.run([Dvar[0],Gvar[0]]))
        e,f = sess.run([D_InputX_prob,D_G_Inputz_prob],feed_dict={ G_Inputz : z[:], D_InputX : x[:]})
        print(np.mean(e),np.mean(f))
        r=sess.run(G_Output0,feed_dict={ G_Inputz : z[:], D_InputX : x[:]})
        print (np.mean(r,0))
        ax[0].plot(r[:,0],r[:,1],'ro',markersize=0.3)
        ax[0].plot(r[:1,0]-1000,r[:1,1],'ro',label = 'Generated Data',markersize=3)
        r=sess.run(G_Inputz,feed_dict={ G_Inputz : z[:], D_InputX : x[:]})
        print (np.mean(r,0))
        ax[0].plot(x[:,0],x[:,1],'bo',markersize=0.3);
        ax[0].plot(x[:1,0]-1000,x[:1,1],'bo',label = 'Real Multivariate Gaussian',markersize=3);
        ax[0].set_xlabel('X1');ax[0].set_ylabel('X2')
        ax[0].legend(loc='upper left');ax[0].set_title('Generated vs Real distribution');
        ax[0].set_xlim(-15,40);ax[0].set_ylim(-15,40);plt.savefig(str(i)+'.png',dpi=300);plt.close();#plt.show()
        
##        a,b = sess.run([D_G_Inputz_prob,D_InputX_prob],feed_dict={ G_Inputz : z[:], D_InputX : x[:]})
##        plt.hist([a[:,0],b[:,0]],label=['Generate','real']);plt.title('disrimiatior probs');plt.legend();plt.close()

#sess.run([D_G_Inputz_prob,D_InputX_prob],feed_dict={ G_Inputz : z[:], D_InputX : x[:]})
#ffmpeg -framerate 1  -pattern_type glob -i '*.png' -filter:v tblend video.mp4

