from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.contrib import ipu
from tensorflow.python.client import session as sl
from tensorflow.python.framework import test_util
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent

datatype = np.float16


def _get_weights(name, shape):
  init = init_ops.truncated_normal_initializer(stddev=0.01)
  return vs.get_variable(name, shape, initializer=init, dtype=datatype)


def _get_biases(name, shape):
  init = init_ops.constant_initializer(0.0)
  return vs.get_variable(name, shape, initializer=init, dtype=datatype)


def layer(x, chans_out, name):

  w = _get_weights(name + "_w", shape=[x.shape[1], chans_out])
  b = _get_biases(name + "_b", shape=[chans_out])

  x = nn_ops.xw_plus_b(x, w, b)
  x = nn_ops.relu(x)
  return x


def inference(x):

  with vs.variable_scope('all', use_resource=True):

    x = layer(x, 32, "l1")
    x = layer(x, 64, "l2")
    x = layer(x, 64, "l3")

  return x


class MatMulSizeTest(test_util.TensorFlowTestCase):
  def testInference(self):
    x = array_ops.placeholder(datatype, shape=[2, 112 * 112 * 4])
    y_ = array_ops.placeholder(datatype, shape=[2, 64])

    with ipu.ops.ipu_scope("/device:IPU:0"):
      logits = inference(x)

      loss = math_ops.reduce_mean(
          nn_ops.softmax_cross_entropy_with_logits_v2(
              logits=logits, labels=array_ops.stop_gradient(y_)))

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    opts = ipu.utils.create_ipu_config(profiling=True)
    ipu.utils.configure_ipu_system(opts)
    sess = sl.Session()

    sess.run(variables.global_variables_initializer())
    sess.run(report)

    data = np.zeros([2, 112 * 112 * 4])
    labels = np.zeros([2, 64])

    sess.run(loss, feed_dict={x: data, y_: labels})
    out = sess.run(report)

    sess.close()

    evts = ipu.utils.extract_all_events(out)
    size = ipu.utils.get_memory_size_from_events(evts)
    self.assertTrue(size < 17740000)

  def testTrainingBs1(self):
    x = array_ops.placeholder(datatype, shape=[1, 112 * 112 * 4])
    y_ = array_ops.placeholder(datatype, shape=[1, 64])

    with ipu.ops.ipu_scope("/device:IPU:0"):
      logits = inference(x)

      loss = math_ops.reduce_mean(
          nn_ops.softmax_cross_entropy_with_logits_v2(
              logits=logits, labels=array_ops.stop_gradient(y_)))

      train = gradient_descent.GradientDescentOptimizer(0.01).minimize(loss)

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    opts = ipu.utils.create_ipu_config(profiling=True)
    ipu.utils.configure_ipu_system(opts)

    sess = sl.Session()

    sess.run(variables.global_variables_initializer())
    sess.run(report)

    data = np.zeros([1, 112 * 112 * 4])
    labels = np.zeros([1, 64])

    sess.run(train, feed_dict={x: data, y_: labels})
    out = sess.run(report)

    sess.close()

    evts = ipu.utils.extract_all_events(out)
    size = ipu.utils.get_memory_size_from_events(evts)
    self.assertTrue(size < 15820000)

  def testTrainingBs2(self):
    x = array_ops.placeholder(datatype, shape=[2, 112 * 112 * 4])
    y_ = array_ops.placeholder(datatype, shape=[2, 64])

    with ipu.ops.ipu_scope("/device:IPU:0"):
      logits = inference(x)

      loss = math_ops.reduce_mean(
          nn_ops.softmax_cross_entropy_with_logits_v2(
              logits=logits, labels=array_ops.stop_gradient(y_)))

      train = gradient_descent.GradientDescentOptimizer(0.01).minimize(loss)

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    opts = ipu.utils.create_ipu_config(profiling=True)
    ipu.utils.configure_ipu_system(opts)

    sess = sl.Session()

    sess.run(variables.global_variables_initializer())
    sess.run(report)

    data = np.zeros([2, 112 * 112 * 4])
    labels = np.zeros([2, 64])

    sess.run(train, feed_dict={x: data, y_: labels})
    out = sess.run(report)

    sess.close()

    evts = ipu.utils.extract_all_events(out)
    size = ipu.utils.get_memory_size_from_events(evts)
    self.assertTrue(size < 26380000)


if __name__ == "__main__":
  googletest.main()
