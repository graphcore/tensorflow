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
from tensorflow.python.ipu import nn_ops
from tensorflow.python.ipu import normalization_ops
from tensorflow.python.ipu import scopes
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# The dataset for feeding the graphs
ds = tf.data.Dataset.from_tensors(tf.constant(1.0, shape=[128, 128]))
ds = ds.repeat()

# The host side queues
infeed_queue = ipu_infeed_queue.IPUInfeedQueue(ds)
outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()


# The device side main
def body(x):
  w1 = tf.get_variable(
      "w1",
      shape=[128, 128],
      initializer=tf.glorot_uniform_initializer(dtype=tf.float32))
  w2 = tf.get_variable(
      "w2",
      shape=[128, 128],
      initializer=tf.glorot_uniform_initializer(dtype=tf.float32))

  # The model has some repeated structure to it, and we manually convert it into
  # an IPU function
  @ipu.outlined_function
  def func(a, b):
    x = tf.matmul(a, b)
    x = normalization_ops.layer_norm(x)
    x = nn_ops.gelu(x)
    return x

  # Invoke the function twice with different arguments
  x = func(x, w1)
  x = func(x, w2)
  outfeed = outfeed_queue.enqueue(x)
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
  result = sess.run(dequeue_outfeed)
  print(result)
