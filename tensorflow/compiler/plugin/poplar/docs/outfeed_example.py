from threading import Thread

from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import loops
from tensorflow.python.ipu import scopes
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python import keras
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# The dataset for feeding the graphs
ds = tf.data.Dataset.from_tensors(tf.constant(1.0, shape=[2, 20]))
ds = ds.repeat()

# Host side queues that handle data transfer to and from the device
infeed_queue = ipu_infeed_queue.IPUInfeedQueue(ds)
outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()


# A simple inference step that predicts classes for an image with a model
# composed of three dense layers and sends the predicted classes from the IPU
# device to the host using an IPUOutfeedQueue
def inference_step(image):
  partial = keras.layers.Dense(256, activation=tf.nn.relu)(image)
  partial = keras.layers.Dense(128, activation=tf.nn.relu)(partial)
  logits = keras.layers.Dense(10)(partial)
  classes = tf.argmax(input=logits, axis=1, output_type=tf.dtypes.int32)
  # Insert an enqueue op into the graph when inference_step is called
  outfeed = outfeed_queue.enqueue(classes)
  return outfeed


NUM_ITERATIONS = 100


def inference_loop():
  r = loops.repeat(NUM_ITERATIONS, inference_step, [], infeed_queue)
  return r


# Build the main graph and encapsulate it into an IPU cluster
with scopes.ipu_scope('/device:IPU:0'):
  run_loop = ipu_compiler.compile(inference_loop, inputs=[])

# Calling outfeed_queue.dequeue() will insert the dequeue op into the graph.
# We must do this after we've inserted the enqueue op and in the same graph as
# the enqueue op. Note that threads have different default graphs, so we call it
# here to ensure both ops are in the same graph.
dequeue_outfeed = outfeed_queue.dequeue()

# Configure the hardware
config = IPUConfig()
config.auto_select_ipus = 1
config.configure_ipu_system()

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(infeed_queue.initializer)

  # Function to continuously execute the dequeue op and print the dequeued data
  def dequeue():
    counter = 0
    # We expect 2*`NUM_ITERATIONS` results because we execute the loop twice
    while counter != NUM_ITERATIONS * 2:
      r = sess.run(dequeue_outfeed)
      # Check if there are any results to process
      if r.size:
        # Print the partial results
        for t in r:
          print("Step:", counter, "classes:", t)
          counter += 1

  # Execute the main loop once to compile the program. We must do this before
  # starting the dequeuing thread, or the TensorFlow runtime will try to dequeue
  # an outfeed that it doesn't know about
  sess.run(run_loop)

  # Start a thread that asynchronously continuously dequeues the outfeed
  dequeue_thread = Thread(target=dequeue)
  dequeue_thread.start()

  # Run the main loop while the outfeed is being dequeued by the second thread
  sess.run(run_loop)

  # After main loop execution, wait for the dequeuing thread to finish
  dequeue_thread.join()
