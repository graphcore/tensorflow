# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

import os
import pva
import numpy as np
import test_utils as tu

from tensorflow.compiler.tests import xla_test
from tensorflow.python.platform import googletest
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops


class IpuXlaConvTest(xla_test.XLATestCase):
  def test3DConv1x1x1_Stride2x1x1_In1x1x5(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [1, 1, 1, 5, 1], name="a")
        pb = array_ops.placeholder(np.float32, [1, 1, 1, 1, 1], name="b")
        output = nn_ops.convolution(pa, pb, strides=[1, 1, 2], padding="VALID")

      fd = {pa: [[[[[1], [2], [3], [4], [5]]]]], pb: [[[[[10]]]]]}
      result = sess.run(output, fd)
      self.assertAllClose(result, [[[[[10], [30], [50]]]]])

  def test3DConv3x3x3_Pad1x1x1(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [1, 14, 14, 14, 64], name="a")
        pb = array_ops.placeholder(np.float32, [3, 3, 3, 64, 128], name="b")
        output = nn_ops.convolution(pa, pb, padding="SAME")

      fd = {
          pa: np.zeros([1, 14, 14, 14, 64]),
          pb: np.zeros([3, 3, 3, 64, 128])
      }
      result = sess.run(output, fd)
      self.assertAllClose(result, np.zeros([1, 14, 14, 14, 128]))

  def test3DConv3x3x3_WithBias(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [1, 14, 14, 14, 16], name="a")
        pb = array_ops.placeholder(np.float32, [3, 3, 3, 16, 32], name="b")
        bi = array_ops.placeholder(np.float32, [32], name="b")
        output = nn_ops.convolution(pa, pb, padding="SAME")
        output = nn_ops.bias_add(output, bi)

      fd = {
          pa: np.zeros([1, 14, 14, 14, 16]),
          pb: np.zeros([3, 3, 3, 16, 32]),
          bi: np.zeros([32]),
      }
      result = sess.run(output, fd)
      self.assertAllClose(result, np.zeros([1, 14, 14, 14, 32]))

    report = pva.openReport(report_helper.find_report())
    ok = ['convolution/convolution.*/Conv_3x3x3', 'BiasAdd/fusion/Op/Add']
    self.assert_all_compute_sets_and_list(report, ok)

  def test3DConv8x8x8_WithBias(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        inp = array_ops.placeholder(np.float32, [1, 84, 84, 84, 2], name="inp")
        wei = array_ops.placeholder(np.float32, [8, 8, 8, 2, 4], name="wei")
        bia = array_ops.placeholder(np.float32, [4], name="bia")
        output = nn_ops.conv3d(inp,
                               wei,
                               strides=[1, 4, 4, 4, 1],
                               padding="VALID")
        output = nn_ops.bias_add(output, bia)

      fd = {
          inp: np.zeros([1, 84, 84, 84, 2]),
          wei: np.zeros([8, 8, 8, 2, 4]),
          bia: np.zeros([4]),
      }
      result = sess.run(output, fd)
      self.assertAllClose(result, np.zeros([1, 20, 20, 20, 4]))

    report = pva.openReport(report_helper.find_report())
    ok = [
        'Conv3D/convolution.*/Conv_8x8x8_stride4x4x4', 'BiasAdd/fusion/Op/Add'
    ]
    self.assert_all_compute_sets_and_list(report, ok)

  def test3DConv1x1x1_WithBias(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        inp = array_ops.placeholder(np.float32, [1, 1, 1, 1, 4], name="inp")
        wei = array_ops.placeholder(np.float32, [1, 1, 1, 4, 8], name="wei")
        bia = array_ops.placeholder(np.float32, [8], name="bia")
        output = nn_ops.conv3d(inp,
                               wei,
                               strides=[1, 1, 1, 1, 1],
                               padding="VALID")
        output = output + bia

      fd = {
          inp: np.zeros([1, 1, 1, 1, 4]),
          wei: np.zeros([1, 1, 1, 4, 8]),
          bia: np.zeros([8]),
      }
      result = sess.run(output, fd)
      self.assertAllClose(result, np.zeros([1, 1, 1, 1, 8]))

    report = pva.openReport(report_helper.find_report())
    ok = ['Conv3D/convolution.*/Conv_1x1', 'add/fusion/Op/Add']
    self.assert_all_compute_sets_and_list(report, ok)

  def test3DConvBackpropInput(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        ins = constant_op.constant([2, 8, 8, 8, 3], np.int32)
        fil = array_ops.placeholder(np.float32, [2, 2, 2, 3, 5], name="inp")
        bck = array_ops.placeholder(np.float32, [2, 8, 8, 8, 5], name="wei")

        output = nn_ops.conv3d_backprop_input_v2(ins,
                                                 fil,
                                                 bck,
                                                 strides=[1, 1, 1, 1, 1],
                                                 padding="SAME")

      fd = {
          fil: np.zeros([2, 2, 2, 3, 5]),
          bck: np.zeros([2, 8, 8, 8, 5]),
      }
      result = sess.run(output, fd)
      self.assertAllClose(result, np.zeros([2, 8, 8, 8, 3]))

    report = pva.openReport(report_helper.find_report())
    ok = [
        'Conv3DBackpropInputV2/conv-with-reverse/Conv_2x2x2',
    ]
    self.assert_all_compute_sets_and_list(report, ok)

  def test3DConvBackpropFilter(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        inp = array_ops.placeholder(np.float32, [2, 8, 8, 8, 3])
        fil = constant_op.constant([2, 2, 2, 3, 5], np.int32)
        bck = array_ops.placeholder(np.float32, [2, 8, 8, 8, 5], name="wei")

        output = nn_ops.conv3d_backprop_filter_v2(inp,
                                                  fil,
                                                  bck,
                                                  strides=[1, 1, 1, 1, 1],
                                                  padding="SAME")

      fd = {
          inp: np.zeros([2, 8, 8, 8, 3]),
          bck: np.zeros([2, 8, 8, 8, 5]),
      }
      result = sess.run(output, fd)
      self.assertAllClose(result, np.zeros([2, 2, 2, 3, 5]))

    report = pva.openReport(report_helper.find_report())
    ok = ['Conv3DBackpropFilterV2/convolution.*/Conv_8x8x8']
    self.assert_all_compute_sets_and_list(report, ok)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
