# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from tensorflow.python import ipu
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import loops
from tensorflow.python.ipu import scopes
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# The dataset for feeding the graphs
ds = tf.data.Dataset.from_tensors(tf.constant(1.0, shape=[20000, 64]))
ds = ds.repeat()

# The host side queues
infeed_queue = ipu_infeed_queue.IPUInfeedQueue(ds)
outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()


# The device side main
def body(x):
  # The model looks as following:
  # x = a tensor of shape [20000, 64]
  # w = a tensor of shape [64, 128]
  # partial = tf.matmul(x, w) <- output shape is [20000, 128]
  # result = tf.reduce_mean(partial, axis=1) <- output shape is [20000]
  #
  # If the code generated when calculating `partial` and `result` is too large,
  # we can manually serialize the computation and reuse the code
  w = tf.get_variable(
      "w",
      shape=[64, 128],
      initializer=tf.glorot_uniform_initializer(dtype=tf.float32))

  # We are going to serialize along the 0th dimension of x
  x_shape = tf.shape(x)
  # Split the computation into 10 chunks
  NUM_SPLITS = 10
  SLICE_SIZE = x_shape[0] // NUM_SPLITS

  # An IPU function which works on the part of x
  @ipu.outlined_function
  def func(partial_x, w):
    partial = tf.matmul(partial_x, w)
    partial_result = tf.reduce_mean(partial, axis=1)
    return partial_result

  # A list to store the partials results in
  result_slices = []
  # Loop which works on the serialized slices
  for i in range(NUM_SPLITS):
    # Get the slice
    slice_start = i * SLICE_SIZE
    x_slice = tf.slice(x, [slice_start, 0], [SLICE_SIZE, x_shape[1]])
    # Call the function to generate the partial result
    partial_result = func(x_slice, w)
    result_slices.append(partial_result)

  # Combine the partials results
  result = tf.stack(result_slices)

  outfeed = outfeed_queue.enqueue(result)
  return outfeed


def my_net():
  r = loops.repeat(10, body, [], infeed_queue)
  return r


with scopes.ipu_scope('/device:IPU:0'):
  run_loop = ipu_compiler.compile(my_net, inputs=[])

# The outfeed dequeue has to happen after the outfeed enqueue
dequeue_outfeed = outfeed_queue.dequeue()

# Configure the hardware
config = ipu.config.IPUConfig()
config.auto_select_ipus = 1
config.configure_ipu_system()

with tf.Session() as sess:
  sess.run(infeed_queue.initializer)
  sess.run(tf.global_variables_initializer())
  sess.run(run_loop)
  output = sess.run(dequeue_outfeed)
  print(output)
