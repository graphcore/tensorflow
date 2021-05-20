import numpy as np

from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ipu.scopes import ipu_scope
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Configure arguments for targeting the IPU
cfg = IPUConfig()
cfg.auto_select_ipus = 1
cfg.configure_ipu_system()

with tf.device("cpu"):
  pa = tf.placeholder(np.float32, [2], name="a")
  pb = tf.placeholder(np.float32, [2], name="b")
  pc = tf.placeholder(np.float32, [2], name="c")


def basic_graph(pa, pb, pc):
  # Do basic addition with tensors
  o1 = pa + pb
  o2 = pa + pc
  simple_graph_output = o1 + o2
  return simple_graph_output


with ipu_scope("/device:IPU:0"):
  result = basic_graph(pa, pb, pc)

with tf.Session() as sess:
  # Run the graph through the session feeding it an arbitrary dictionary
  result = sess.run(result,
                    feed_dict={
                        pa: [1., 1.],
                        pb: [0., 1.],
                        pc: [1., 5.]
                    })
  print(result)
