import tensorflow as tf
import numpy as np

# IPU imports
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.python.ipu import utils
from tensorflow.python.ipu.scopes import ipu_scope

# Configure argument for targeting the IPU
cfg = utils.create_ipu_config(profiling=True, use_poplar_text_report=True)
cfg = utils.set_ipu_model_options(cfg, compile_ipu_code=False)
cfg = utils.auto_select_ipus(cfg, 1)
utils.configure_ipu_system(cfg)

with tf.device("cpu"):
  pa = tf.placeholder(np.float32, [2], name="a")
  pb = tf.placeholder(np.float32, [2], name="b")
  pc = tf.placeholder(np.float32, [2], name="c")

  # Create a trace event
  report = gen_ipu_ops.ipu_event_trace()


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
  result = sess.run(
      result, feed_dict={
          pa: [1., 1.],
          pb: [0., 1.],
          pc: [1., 5.]
      })

  # Generate report based on the event run in session
  trace_out = sess.run(report)
  trace_report = utils.extract_all_strings_from_event_trace(trace_out)

  # Write trace report to file
  with open('Trace_Event_Report.rep', "w") as f:
    f.write(trace_report)

  # Print the result
  print(result)
