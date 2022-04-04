import tensorflow as tf
from tensorflow.python import ipu

# Configure the IPU device.
config = ipu.config.IPUConfig()
config.auto_select_ipus = 1
config.configure_ipu_system()

a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


@tf.function(jit_compile=True)
def matmul_fn(x, y):
  z = tf.matmul(x, y)
  return z


strategy = ipu.ipu_strategy.IPUStrategy()
with strategy.scope():
  c = strategy.run(matmul_fn, args=(a, b))
print(c)
