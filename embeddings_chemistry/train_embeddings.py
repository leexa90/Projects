import numpy  as np
data = np.load('all_bonds.npy')
import time

import numpy as np
import tensorflow as tf
import random
import pandas as pd
import math
z = pd.DataFrame(data,columns=['A','B'])
z1 =z.groupby(['A','B']).apply(len).reset_index()

train = z.iloc[:-len(z)//10]
valid = z.iloc[-len(z)//10:]

# Step 4: Build and train a skip-gram model.
vocabulary_size = 82
batch_size = 128
embedding_size = 8  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
num_sampled = 64  # Number of negative examples to sample.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 50  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

graph = tf.Graph()

with graph.as_default():

  # Input data.
  with tf.name_scope('inputs'):
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels2 = tf.expand_dims(train_labels,-1)
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    with tf.name_scope('embeddings'):
      embeddings = tf.Variable(
          tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
      embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    with tf.name_scope('weights'):
      nce_weights = tf.Variable(
          tf.truncated_normal(
              [vocabulary_size, embedding_size],
              stddev=1.0 / math.sqrt(embedding_size)))
    with tf.name_scope('biases'):
      nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  # Explanation of the meaning of NCE loss:
  #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
  with tf.name_scope('loss'):
    loss = tf.reduce_mean(
        tf.nn.nce_loss(
            weights=nce_weights,
            biases=nce_biases,
            labels=train_labels2,
            inputs=embed,
            num_sampled=num_sampled,
            num_classes=vocabulary_size))

  # Add the loss value as a scalar to summary.
  tf.summary.scalar('loss', loss)

  # Construct the SGD optimizer using a learning rate of 1.0.
  with tf.name_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                            valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # Merge all summaries.
  merged = tf.summary.merge_all()

  # Add variable initializer.
  init = tf.global_variables_initializer()
  init = tf.global_variables_initializer();session = tf.Session();session.run(init)
  writer = tf.summary.FileWriter('.', session.graph)
  # Create a saver.
  saver = tf.train.Saver()
  average_loss = 0
  num_steps = 12
  for step in xrange(num_steps):
      train = train.sample(n=len(train)).reset_index(drop=True)
      for batch_id in range(0,len(train)-128,128):
        batch_inputs, batch_labels = train.iloc[batch_id:batch_id+128]['A'], train.iloc[batch_id:batch_id+128]['B']
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        run_metadata = tf.RunMetadata()
        _, summary, loss_val = session.run(
                                            [optimizer, merged, loss],
                                            feed_dict=feed_dict,
                                            run_metadata=run_metadata)
        feed_dict = {train_inputs: batch_labels, train_labels: batch_inputs}
        _, summary, loss_val = session.run(
                                            [optimizer, merged, loss],
                                            feed_dict=feed_dict,
                                            run_metadata=run_metadata)
      writer.add_summary(summary, step)
      # Add metadata to visualize the graph for the last run.
      if step == (num_steps - 1):
        writer.add_run_metadata(run_metadata, 'step%d' % step)


      if step > 0:
        average_loss /= len(range(0,len(train)-128,128))
        # The average loss is an estimate of the loss over the last 2000 batches.
        print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0
      sim = session.run(similarity)
      if True:
          for i in xrange(valid_size):
            top_k = 8  # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_examples[i]
            for k in xrange(top_k):
              log_str = '%s %s,' % (log_str, nearest[k])
            print(log_str)
          final_embeddings = session.run(normalized_embeddings)
np.save('final_embeddings.npy',final_embeddings)
