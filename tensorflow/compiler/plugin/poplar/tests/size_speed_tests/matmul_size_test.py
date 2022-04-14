#  Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  =============================================================================

import numpy as np
import pva

from tensorflow.python.ipu import test_utils as tu
from tensorflow.compiler.tests import xla_test
from tensorflow.python import ipu
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


class MatMulSizeTest(xla_test.XLATestCase):
  def testInference(self):
    cfg = ipu.utils.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 1472
    cfg.optimizations.math.dot_strength = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      x = array_ops.placeholder(datatype, shape=[2, 112 * 112 * 4])
      y_ = array_ops.placeholder(datatype, shape=[2, 64])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        logits = inference(x)

        loss = math_ops.reduce_mean(
            nn_ops.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=array_ops.stop_gradient(y_)))

      sess.run(variables.global_variables_initializer())
      report_helper.clear_reports()

      data = np.zeros([2, 112 * 112 * 4])
      labels = np.zeros([2, 64])

      sess.run(loss, feed_dict={x: data, y_: labels})

    report = pva.openReport(report_helper.find_report())
    self.assert_total_tile_memory(report, 12181182)

  def testInferenceManyLayersRebalancedBroadcastInInitialisation(self):
    cfg = ipu.utils.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = True
    cfg.ipu_model.tiles_per_ipu = 1472
    cfg.optimizations.math.dot_strength = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      x = array_ops.placeholder(datatype, shape=[2, 112 * 112 * 4])
      y = array_ops.placeholder(datatype, shape=[2, 4])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        logits = x
        for idx in range(20):
          # Any value less than tiles per ipu would work here,
          # see ShouldRebalanceTensor().
          logits = layer(logits, 1400, "l%d" % idx)
        logits = layer(logits, 4, "out")

        math_ops.reduce_mean(
            nn_ops.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                        labels=y))

      sess.run(variables.global_variables_initializer())
      report = pva.openReport(report_helper.find_report())
      self.assert_max_tile_memory(report, 162410)
    # Skip the rest of the test, broadcast are returned for global initialisers.

  def testTrainingBs1(self):
    cfg = ipu.utils.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 1472
    cfg.optimizations.math.dot_strength = False
    cfg.configure_ipu_system()

    with self.session() as sess:

      x = array_ops.placeholder(datatype, shape=[1, 112 * 112 * 4])
      y_ = array_ops.placeholder(datatype, shape=[1, 64])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        logits = inference(x)

        loss = math_ops.reduce_mean(
            nn_ops.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=array_ops.stop_gradient(y_)))

        train = gradient_descent.GradientDescentOptimizer(0.01).minimize(loss)

      sess.run(variables.global_variables_initializer())
      report_helper.clear_reports()

      data = np.zeros([1, 112 * 112 * 4])
      labels = np.zeros([1, 64])

      sess.run(train, feed_dict={x: data, y_: labels})

    report = pva.openReport(report_helper.find_report())
    self.assert_total_tile_memory(report, 7142699)

  def testTrainingBs2(self):
    cfg = ipu.utils.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 1472
    cfg.optimizations.math.dot_strength = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      x = array_ops.placeholder(datatype, shape=[2, 112 * 112 * 4])
      y_ = array_ops.placeholder(datatype, shape=[2, 64])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        logits = inference(x)

        loss = math_ops.reduce_mean(
            nn_ops.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=array_ops.stop_gradient(y_)))

        train = gradient_descent.GradientDescentOptimizer(0.01).minimize(loss)

      sess.run(variables.global_variables_initializer())
      report_helper.clear_reports()

      data = np.zeros([2, 112 * 112 * 4])
      labels = np.zeros([2, 64])

      sess.run(train, feed_dict={x: data, y_: labels})

    report = pva.openReport(report_helper.find_report())
    self.assert_total_tile_memory(report, 16126623)

  def testSerializedMatmul(self):
    cfg = ipu.utils.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 1472
    cfg.configure_ipu_system()

    with self.session() as sess:

      def serialized_matmul(lhs, rhs, num_splits):
        lhs_shape = lhs.get_shape().as_list()
        rhs_shape = rhs.get_shape().as_list()
        assert lhs_shape[1] == rhs_shape[0]
        assert (lhs_shape[0] % num_splits) == 0

        @ipu.outlined_function
        def inner_func(lhs_, rhs_):
          return math_ops.matmul(lhs_, rhs_)

        split_size = lhs_shape[0] // num_splits
        result = []
        for i in range(0, num_splits):
          lhs_slice = array_ops.slice(lhs, [i * split_size, 0],
                                      [split_size, lhs_shape[1]])
          result.append(inner_func(lhs_slice, rhs))
        return array_ops.concat(result, axis=0)

      def model(x, y, z):
        return x + serialized_matmul(y, z, num_splits=6)

      B = 30522
      I = 768
      O = 128
      x = array_ops.placeholder(datatype, shape=[B, I])
      y = array_ops.placeholder(datatype, shape=[B, O])
      z = array_ops.placeholder(datatype, shape=[O, I])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        out = ipu.ipu_compiler.compile(model, [x, y, z])

      output = sess.run(out,
                        feed_dict={k: np.ones(k.shape)
                                   for k in (x, y, z)})
      self.assertAllClose(np.full([B, I], 129.0), output[0])

    report = pva.openReport(report_helper.find_report())
    self.assert_total_tile_memory(report, 126942643)
    self.assert_max_tile_memory(report, 86813)


if __name__ == "__main__":
  googletest.main()
