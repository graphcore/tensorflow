from threading import Thread

from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import loops
from tensorflow.python.ipu import scopes
from tensorflow.python.ipu import utils
from tensorflow.python import keras
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# The dataset for feeding the graphs
ds = tf.data.Dataset.from_tensors(tf.constant(1.0, shape=[2, 20]))
ds = ds.repeat()

# The host side queues
infeed_queue = ipu_infeed_queue.IPUInfeedQueue(ds, feed_name="infeed")
outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(feed_name="outfeed")


# The device side main
def body(image):
  partial = keras.layers.Dense(256, activation=tf.nn.relu)(image)
  partial = keras.layers.Dense(128, activation=tf.nn.relu)(partial)
  logits = keras.layers.Dense(10)(partial)
  classes = tf.argmax(input=logits, axis=1, output_type=tf.dtypes.int32)
  outfeed = outfeed_queue.enqueue(classes)
  return outfeed


num_iterations = 100


def my_net():
  r = loops.repeat(100, body, [], infeed_queue)
  return r


with scopes.ipu_scope('/device:IPU:0'):
  run_loop = ipu_compiler.compile(my_net, inputs=[])

# The outfeed dequeue has to happen after the outfeed enqueue and in the same
# thread as the enqueue, since threads have different default graphs
dequeue_outfeed = outfeed_queue.dequeue()

# Configure the hardware
config = utils.create_ipu_config()
config = utils.auto_select_ipus(config, 1)
utils.configure_ipu_system(config)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(infeed_queue.initializer)

  # Function which is executed when continuously dequeuing the outfeed
  def dequeue():
    counter = 0
    # We expect 2*`num_iterations` results because we execute the loop twice
    while counter != num_iterations * 2:
      r = sess.run(dequeue_outfeed)
      # Check if there are any results to process
      if r.size:
        # Print the partial results
        print(r)
        counter += len(r)

  # Run the main loop once to compile the program.
  sess.run(run_loop)
  # Create a thread which will continuously dequeue the outfeed queue and start
  # it
  dequeue_thread = Thread(target=dequeue)
  dequeue_thread.start()
  # Run the main loop
  sess.run(run_loop)
  # Once the main loop has finished, make sure to only finish once the dequeue
  # thread has stopped
  dequeue_thread.join()
