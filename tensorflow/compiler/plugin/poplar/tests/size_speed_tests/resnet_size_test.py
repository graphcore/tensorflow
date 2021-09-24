# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pva

from tensorflow.compiler.tests import xla_test
import tensorflow.compiler.plugin.poplar.tests.test_utils as tu
from tensorflow.python import ipu
from tensorflow.python.ipu.optimizers import gradient_accumulation_optimizer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import momentum

datatype = np.float16


def _get_variable(name, shape, init):
  return vs.get_variable(name, shape, initializer=init, dtype=datatype)


def inference(x):

  with vs.variable_scope('all', use_resource=True):
    x = conv(x, 7, 2, 64)
    x = nn_ops.relu(x)
    x = max_pool(x, ksize=3, stride=2)
    x = block("b1", 64, 1, 2, x)
    x = nn_ops.max_pool(x, [1, x.shape[1], x.shape[2], 1], [1, 1, 1, 1],
                        'VALID')
    x = array_ops.reshape(x, [x.shape[0], x.shape[3]])
    x = fc("fc1", x, 1000)

  return x


def block(name, out_filters, first_stride, count, x):

  for i in range(count):
    sc = x
    shape_in = x.shape
    stride = (first_stride if (i == 0) else 1)

    with vs.variable_scope(name + "/" + str(i) + "/1"):
      x = conv(x, 3, stride, out_filters)
      x = nn_ops.relu(x)

    with vs.variable_scope(name + "/" + str(i) + "/2"):
      x = conv(x, 3, 1, out_filters)

      # shortcut
      if stride != 1:
        sc = array_ops.strided_slice(sc, [0, 0, 0, 0],
                                     sc.shape,
                                     strides=[1, stride, stride, 1])
      pad = int(x.shape[3] - shape_in[3])
      if pad != 0:
        sc = array_ops.pad(sc, paddings=[[0, 0], [0, 0], [0, 0], [0, pad]])

      x = nn_ops.relu(x + sc)

  return x


def fc(name, x, num_units_out):
  num_units_in = x.shape[1]
  weights_initializer = init_ops.truncated_normal_initializer(stddev=0.01)

  with vs.variable_scope(name):
    weights = _get_variable('weights',
                            shape=[num_units_in, num_units_out],
                            init=weights_initializer)
    biases = _get_variable('biases',
                           shape=[num_units_out],
                           init=init_ops.constant_initializer(0.0))

    x = nn_ops.xw_plus_b(x, weights, biases)

  return x


def conv(x, ksize, stride, filters_out):

  filters_in = x.shape[-1]

  wshape = [ksize, ksize, filters_in, filters_out]
  winitializer = init_ops.truncated_normal_initializer(stddev=0.1)
  bshape = [filters_out]
  binitializer = init_ops.zeros_initializer()

  weights = _get_variable('weights', shape=wshape, init=winitializer)
  biases = _get_variable('biases', shape=bshape, init=binitializer)
  stride = [1, stride, stride, 1]
  return nn_ops.conv2d(x, weights, strides=stride, padding='SAME') + biases


def max_pool(x, ksize=3, stride=2):
  return nn_ops.max_pool(x,
                         ksize=[1, ksize, ksize, 1],
                         strides=[1, stride, stride, 1],
                         padding='SAME')


class Resnet18_No_Batchnorm(xla_test.XLATestCase):
  def testInference(self):
    cfg = ipu.utils.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 1472
    cfg.configure_ipu_system()

    with self.session() as sess:
      x = array_ops.placeholder(datatype, shape=[1, 224, 224, 4])
      y_ = array_ops.placeholder(datatype, shape=[1, 1000])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        logits = inference(x)

        loss = math_ops.reduce_mean(
            nn_ops.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=array_ops.stop_gradient(y_)))

      sess.run(variables.global_variables_initializer())
      report_helper.clear_reports()

      data = np.zeros([1, 224, 224, 4])
      labels = np.zeros([1, 1000])

      sess.run(loss, feed_dict={x: data, y_: labels})

    report = pva.openReport(report_helper.find_report())
    self.assert_total_tile_memory(report, 25260426)

  def testTraining(self):
    cfg = ipu.utils.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 1472
    cfg.configure_ipu_system()

    with self.session() as sess:

      x = array_ops.placeholder(datatype, shape=[1, 224, 224, 4])
      y_ = array_ops.placeholder(datatype, shape=[1, 1000])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        logits = inference(x)

        loss = math_ops.reduce_mean(
            nn_ops.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=array_ops.stop_gradient(y_)))

        train = gradient_descent.GradientDescentOptimizer(0.01).minimize(loss)

      sess.run(variables.global_variables_initializer())
      report_helper.clear_reports()

      data = np.zeros([1, 224, 224, 4])
      labels = np.zeros([1, 1000])

      sess.run(train, feed_dict={x: data, y_: labels})

    report = pva.openReport(report_helper.find_report())
    self.assert_total_tile_memory(report, 40144583)

  def testTrainingMomentum(self):
    cfg = ipu.utils.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 1472
    cfg.configure_ipu_system()

    with self.session() as sess:

      x = array_ops.placeholder(datatype, shape=[1, 224, 224, 4])
      y_ = array_ops.placeholder(datatype, shape=[1, 1000])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        logits = inference(x)

        loss = math_ops.reduce_mean(
            nn_ops.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=array_ops.stop_gradient(y_)))

        train = momentum.MomentumOptimizer(0.01, 0.9).minimize(loss)

      sess.run(variables.global_variables_initializer())
      report_helper.clear_reports()

      data = np.zeros([1, 224, 224, 4])
      labels = np.zeros([1, 1000])

      sess.run(train, feed_dict={x: data, y_: labels})

    report = pva.openReport(report_helper.find_report())
    self.assert_total_tile_memory(report, 42421207)

  def testTrainingInLoop(self):
    cfg = ipu.utils.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 1472
    cfg.configure_ipu_system()

    with self.session() as sess:

      x = array_ops.placeholder(datatype, shape=[1, 224, 224, 4])
      y_ = array_ops.placeholder(datatype, shape=[1, 1000])

      with ipu.scopes.ipu_scope("/device:IPU:0"):

        def model(x, l):
          def body(x, label):
            logits = inference(x)

            loss = math_ops.reduce_mean(
                nn_ops.softmax_cross_entropy_with_logits_v2(
                    logits=logits, labels=array_ops.stop_gradient(y_)))
            return x, label, gradient_descent.GradientDescentOptimizer(
                0.01).minimize(loss)

          return ipu.loops.repeat(10, body, (x, l))

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        train = ipu.ipu_compiler.compile(model, inputs=[x, y_])

      sess.run(variables.global_variables_initializer())
      report_helper.clear_reports()

      data = np.zeros([1, 224, 224, 4])
      labels = np.zeros([1, 1000])

      sess.run(train, feed_dict={x: data, y_: labels})

    report = pva.openReport(report_helper.find_report())
    self.assert_total_tile_memory(report, 40593174)

  def testTrainingMomentumInLoop(self):
    cfg = ipu.utils.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 1472
    cfg.configure_ipu_system()

    with self.session() as sess:

      x = array_ops.placeholder(datatype, shape=[1, 224, 224, 4])
      y_ = array_ops.placeholder(datatype, shape=[1, 1000])

      with ipu.scopes.ipu_scope("/device:IPU:0"):

        def model(x, l):
          def body(x, label):
            logits = inference(x)
            loss = math_ops.reduce_mean(
                nn_ops.softmax_cross_entropy_with_logits_v2(
                    logits=logits, labels=array_ops.stop_gradient(label)))
            return x, label, momentum.MomentumOptimizer(0.01,
                                                        0.9).minimize(loss)

          return ipu.loops.repeat(10, body, (x, l))

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        train = ipu.ipu_compiler.compile(model, inputs=[x, y_])

      sess.run(variables.global_variables_initializer())
      report_helper.clear_reports()

      data = np.zeros([1, 224, 224, 4])
      labels = np.zeros([1, 1000])

      sess.run(train, feed_dict={x: data, y_: labels})

    report = pva.openReport(report_helper.find_report())
    self.assert_total_tile_memory(report, 41459358)

  def testTrainingInLoopWithGradientAccumulation(self):
    cfg = ipu.utils.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 1472
    cfg.configure_ipu_system()

    with self.session() as sess:

      x = array_ops.placeholder(datatype, shape=[1, 224, 224, 4])
      y_ = array_ops.placeholder(datatype, shape=[1, 1000])

      with ipu.scopes.ipu_scope("/device:IPU:0"):

        def model(x, l):
          def body(x, label):
            logits = inference(x)

            loss = math_ops.reduce_mean(
                nn_ops.softmax_cross_entropy_with_logits_v2(
                    logits=logits, labels=array_ops.stop_gradient(y_)))
            opt = gradient_accumulation_optimizer.GradientAccumulationOptimizer(
                gradient_descent.GradientDescentOptimizer(0.01), 5)
            return x, label, opt.minimize(loss)

          return ipu.loops.repeat(10, body, (x, l))

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        train = ipu.ipu_compiler.compile(model, inputs=[x, y_])

      sess.run(variables.global_variables_initializer())
      report_helper.clear_reports()

      data = np.zeros([1, 224, 224, 4])
      labels = np.zeros([1, 1000])

      sess.run(train, feed_dict={x: data, y_: labels})

    report = pva.openReport(report_helper.find_report())
    self.assert_total_tile_memory(report, 44877416)

  def testTrainingMomentumInLoopWithGradientAccumulation(self):
    cfg = ipu.utils.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 1472
    cfg.configure_ipu_system()

    with self.session() as sess:

      x = array_ops.placeholder(datatype, shape=[1, 224, 224, 4])
      y_ = array_ops.placeholder(datatype, shape=[1, 1000])

      with ipu.scopes.ipu_scope("/device:IPU:0"):

        def model(x, l):
          def body(x, label):
            logits = inference(x)
            loss = math_ops.reduce_mean(
                nn_ops.softmax_cross_entropy_with_logits_v2(
                    logits=logits, labels=array_ops.stop_gradient(label)))
            opt = gradient_accumulation_optimizer.GradientAccumulationOptimizer(
                momentum.MomentumOptimizer(0.01, 0.9), 10)
            return x, label, opt.minimize(loss)

          return ipu.loops.repeat(10, body, (x, l))

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        train = ipu.ipu_compiler.compile(model, inputs=[x, y_])

      sess.run(variables.global_variables_initializer())
      report_helper.clear_reports()

      data = np.zeros([1, 224, 224, 4])
      labels = np.zeros([1, 1000])

      sess.run(train, feed_dict={x: data, y_: labels})

    report = pva.openReport(report_helper.find_report())
    self.assert_total_tile_memory(report, 45078070)


if __name__ == "__main__":
  googletest.main()
