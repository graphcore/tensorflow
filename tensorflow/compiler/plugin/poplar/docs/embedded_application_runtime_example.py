import tempfile
import os
import numpy as np

from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import loops
from tensorflow.python.ipu import application_compile_op
from tensorflow.python.ipu import embedded_runtime
from tensorflow.python.ipu.config import IPUConfig
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

element_count = 4
loop_count = 16

# The dataset for feeding the graphs.
ds = tf.data.Dataset.from_tensors(tf.constant(1.0, shape=[element_count]))
ds = ds.repeat()

# The host side queues.
infeed_queue = ipu_infeed_queue.IPUInfeedQueue(ds)
outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()


# The device side main.
def body(x):
  # double the input - replace this with application body.
  result = x * 2
  outfeed = outfeed_queue.enqueue({'result': result})
  return outfeed


# Wrap in a loop.
def my_net():
  r = loops.repeat(loop_count, body, [], infeed_queue)
  return r


# Configure the IPU for compilation.
cfg = IPUConfig()
cfg.auto_select_ipus = 1
cfg.configure_ipu_system()

# Setup a temporary directory to store the executable.
tmp_dir_obj = tempfile.TemporaryDirectory()
tmp_dir = tmp_dir_obj.name
poplar_exec_filepath = os.path.join(tmp_dir, "application.poplar_exec")

# Compile the application.
compile_op = application_compile_op.experimental_application_compile_op(
    my_net, output_path=poplar_exec_filepath)

with tf.Session() as sess:
  path = sess.run(compile_op)
  print(f"Poplar executable: {path}")

# Create the start op.
# This creates the poplar engine in a background thread.
inputs = []
engine_name = "my_engine"
ctx = embedded_runtime.embedded_runtime_start(poplar_exec_filepath, inputs,
                                              engine_name)
# Create the call op and the input placeholder.
input_placeholder = tf.placeholder(tf.float32, shape=[element_count])
call_result = embedded_runtime.embedded_runtime_call([input_placeholder], ctx)

# Call the application.
# This should print the even numbers 0 to 30.
for i in range(loop_count):
  with tf.Session() as sess:
    input_data = np.ones(element_count, dtype=np.float32) * i
    print(sess.run(call_result, feed_dict={input_placeholder: input_data}))
