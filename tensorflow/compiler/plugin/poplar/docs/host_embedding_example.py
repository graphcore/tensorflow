import numpy as np
import tensorflow as tf

from tensorflow.python.ipu import embedding_ops
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import loops
from tensorflow.python.ipu import cross_replica_optimizer
from tensorflow.python.ipu import scopes
from tensorflow.python.ipu import rnn_ops
from tensorflow.python import ipu
from tensorflow.python import keras

path_to_file = keras.utils.get_file(
    'shakespeare.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
)

# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

# The unique characters in the file
vocab = sorted(set(text))

# Creating a mapping from unique characters to indices
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text]).astype(np.int32)

sequence_length = 100
batch_size = 16
replication_factor = 2

#  Create training examples / targets
ds = tf.data.Dataset.from_tensor_slices(text_as_int)
ds = ds.batch(sequence_length, drop_remainder=True)
ds = ds.shuffle(batch_size * batch_size)
ds = ds.batch(batch_size, drop_remainder=True)
ds = ds.repeat()

# The host side queues
infeed_queue = ipu_infeed_queue.IPUInfeedQueue(ds)

# Set the learning rate
lr = 0.0001

# Create a momentum optimiser for replication
optimizer = cross_replica_optimizer.CrossReplicaOptimizer(
    tf.train.MomentumOptimizer(lr, 0.99))

# Create a host embedding object
embedding = embedding_ops.create_host_embedding(
    "char_embedding",
    shape=[256, 256],
    dtype=tf.float32,
    partition_strategy="TOKEN",
    optimizer_spec=embedding_ops.HostEmbeddingOptimizerSpec(lr))


# PopnnGRU is time-major
def gru(partials):
  gru_ = rnn_ops.PopnnGRU(256)
  partial_t = tf.transpose(partials, [1, 0, 2])
  gru_outputs_t, _ = gru_(partial_t)
  return tf.transpose(gru_outputs_t, [1, 0, 2])


# The main model
def model(sequence):
  # Perform a lookup on the embedding
  partial = embedding.lookup(sequence)

  partial = gru(partial)
  partial = tf.reshape(partial, [partial.shape[0], -1])
  partial = tf.layers.dense(partial, 256)
  return tf.nn.softmax(partial)


# Compute the loss for a given batch of examples
def evaluation(sequence):
  # Use the last element of the sequence as the label to predict
  label = tf.slice(sequence, [0, sequence_length - 1], [-1, 1])
  sequence = tf.slice(sequence, [0, 0], [-1, sequence_length - 1])
  logits = model(sequence)
  return keras.losses.sparse_categorical_crossentropy(label, logits)


# Minimise the loss
def training(loss, sequence):
  loss = evaluation(sequence)
  mean_loss = tf.math.reduce_mean(loss)
  train = optimizer.minimize(loss)
  return mean_loss, train


num_iterations = 1000


# Loop over our infeed queue, training the model
def my_net():
  loss = tf.constant(0.0, shape=[])
  r = loops.repeat(num_iterations, training, [loss], infeed_queue)
  return r


# Compile the model
with scopes.ipu_scope('/device:IPU:0'):
  run_loop = ipu_compiler.compile(my_net, inputs=[])

# Configure the hardware
config = ipu.config.IPUConfig()
config.auto_select_ipus = replication_factor
config.configure_ipu_system()

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(infeed_queue.initializer)

  # Train the model for some iterations
  with embedding.register(sess):
    for i in range(25):
      l = sess.run(run_loop)
      print("Step " + str(i) + ", loss = " + str(l))
