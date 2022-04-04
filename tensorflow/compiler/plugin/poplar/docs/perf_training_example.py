from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import loops
from tensorflow.python.ipu import ipu_strategy
from tensorflow.python.ipu.config import IPUConfig
import tensorflow as tf

# The dataset for feeding the graphs
ds = tf.data.Dataset.from_tensors(tf.constant(1.0, shape=[800]))
ds = ds.map(lambda x: [x, x])
ds = ds.repeat()

# The host side queues
infeed_queue = ipu_infeed_queue.IPUInfeedQueue(ds)
outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()


# The device side main
def body(counter, x1, x2):
  d1 = x1 + x2
  d2 = x1 - x2
  counter += 1
  outfeed_queue.enqueue({'d1': d1, 'd2': d2})
  return counter


@tf.function(jit_compile=True)
def my_net():
  count = 0
  count = loops.repeat(10, body, [count], infeed_queue)
  return count


# Configure the hardware.
config = IPUConfig()
config.auto_select_ipus = 1
config.configure_ipu_system()

# Initialize the IPU default strategy.
strategy = ipu_strategy.IPUStrategyV1()

with strategy.scope():
  infeed_queue.initializer
  count_out = strategy.run(my_net)
  print("counter", count_out)

  # The outfeed dequeue has to happen after the outfeed enqueue op has been executed.
  result = outfeed_queue.dequeue()

  print("outfeed result", result)
