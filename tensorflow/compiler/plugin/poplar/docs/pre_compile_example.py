from tensorflow.python import ipu
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Create a configuration for a single IPU.
cfg = ipu.utils.create_ipu_config()
cfg = ipu.utils.auto_select_ipus(cfg, num_ipus=1)

# Enable the Pre-compile mode for IPU version 2 with remote buffers enabled.
cfg = ipu.utils.set_ipu_connection_type(
    cfg,
    connection_type=ipu.utils.DeviceConnectionType.PRE_COMPILE,
    ipu_version="ipu2",
    enable_remote_buffers=True)

ipu.utils.configure_ipu_system(cfg)

# The dataset for feeding the graphs
ds = tf.data.Dataset.from_tensors(tf.constant(1.0, shape=[64, 64]))
ds = ds.repeat()

# The host side queues
infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(ds, feed_name="infeed")
outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(feed_name="outfeed")


# The device side main
def body(x):
  w1 = tf.get_variable(
      "w1",
      shape=[64, 64],
      initializer=tf.glorot_uniform_initializer(dtype=tf.float32))
  w2 = tf.get_variable(
      "w2",
      shape=[64, 64],
      initializer=tf.glorot_uniform_initializer(dtype=tf.float32))

  def func(a, b):
    x = tf.matmul(a, b)
    x = ipu.normalization_ops.layer_norm(x)
    x = ipu.nn_ops.gelu(x)
    return x

  x = func(x, w1)
  x = func(x, w2)
  outfeed = outfeed_queue.enqueue(x)
  return outfeed


def my_net():
  r = ipu.loops.repeat(10, body, [], infeed_queue)
  return r


with ipu.scopes.ipu_scope('/device:IPU:0'):
  run_loop = ipu.ipu_compiler.compile(my_net, inputs=[])

# The outfeed dequeue has to happen after the outfeed enqueue
dequeue_outfeed = outfeed_queue.dequeue()

with tf.Session() as sess:
  sess.run(infeed_queue.initializer)
  sess.run(tf.global_variables_initializer())
  sess.run(run_loop)
  print(sess.run(dequeue_outfeed))
