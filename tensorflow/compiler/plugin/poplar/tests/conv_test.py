# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
import numpy as np
import pva
import test_utils as tu

from tensorflow.compiler.tests import xla_test
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.platform import googletest
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops


class IpuXlaConvTest(xla_test.XLATestCase):

  data_formats = ['NHWC', 'NCHW']

  def _ip_shp(self, nhwc, fmt):
    if fmt == 'NHWC':
      return nhwc
    return [nhwc[0], nhwc[3], nhwc[1], nhwc[2]]

  def testConv1x1_Stride2x1_In1x5(self):
    for fmt in self.data_formats:
      with self.session() as sess:
        with ops.device("/device:IPU:0"):
          pa = array_ops.placeholder(np.float32,
                                     self._ip_shp([1, 1, 5, 1], fmt),
                                     name="a")
          pb = array_ops.placeholder(np.float32, [1, 1, 1, 1], name="b")
          output = nn_ops.convolution(pa,
                                      pb,
                                      strides=[1, 2],
                                      padding="VALID",
                                      data_format=fmt,
                                      name='cnv1')

        fd = {
            pa: np.zeros(self._ip_shp([1, 1, 5, 1], fmt)),
            pb: np.zeros([1, 1, 1, 1])
        }
        result = sess.run(output, fd)
        self.assertAllClose(result, np.zeros(self._ip_shp([1, 1, 3, 1], fmt)))

  def testConv3x3_Pad1x1(self):
    for fmt in self.data_formats:
      with self.session() as sess:
        with ops.device("/device:IPU:0"):
          pa = array_ops.placeholder(np.float32,
                                     self._ip_shp([1, 14, 14, 64], fmt),
                                     name="a")
          pb = array_ops.placeholder(np.float32, [3, 3, 64, 128], name="b")
          output = nn_ops.convolution(pa,
                                      pb,
                                      padding="SAME",
                                      data_format=fmt,
                                      name='cnv2')

          fd = {
              pa: np.zeros(self._ip_shp([1, 14, 14, 64], fmt)),
              pb: np.zeros([3, 3, 64, 128])
          }
          result = sess.run(output, fd)
          self.assertAllClose(result,
                              np.zeros(self._ip_shp([1, 14, 14, 128], fmt)))

  def testConv3x3_WithBias(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    for fmt in self.data_formats:
      report_helper.clear_reports()
      with self.session() as sess:
        with ops.device("/device:IPU:0"):
          pa = array_ops.placeholder(np.float32,
                                     self._ip_shp([1, 14, 14, 64], fmt),
                                     name="a")
          pb = array_ops.placeholder(np.float32, [3, 3, 64, 128], name="b")
          bi = array_ops.placeholder(np.float32, [128], name="b")
          output = nn_ops.convolution(pa,
                                      pb,
                                      padding="SAME",
                                      data_format=fmt,
                                      name='cnv3')
          output = nn_ops.bias_add(output, bi, data_format=fmt, name='ba3')

        fd = {
            pa: np.zeros(self._ip_shp([1, 14, 14, 64], fmt)),
            pb: np.zeros([3, 3, 64, 128]),
            bi: np.zeros([128]),
        }
        result = sess.run(output, fd)
        self.assertAllClose(result,
                            np.zeros(self._ip_shp([1, 14, 14, 128], fmt)))

      report = pva.openReport(report_helper.find_report())
      ok = ['__seed*', 'cnv3*/convolution.*/Conv_3x3', 'ba3*/fusion/Op/Add']

      self.assert_compute_sets_contain_list(report, ok)

  def testConv8x8_WithBias(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    for fmt in self.data_formats:
      report_helper.clear_reports()
      with self.session() as sess:
        with ops.device("/device:IPU:0"):
          inp = array_ops.placeholder(np.float32,
                                      self._ip_shp([1, 84, 84, 4], fmt),
                                      name="inp")
          wei = array_ops.placeholder(np.float32, [8, 8, 4, 16], name="wei")
          bia = array_ops.placeholder(np.float32, [16], name="bia")
          output = nn_ops.conv2d(inp,
                                 wei,
                                 strides=self._ip_shp([1, 4, 4, 1], fmt),
                                 padding="VALID",
                                 data_format=fmt,
                                 name='cnv4')
          output = nn_ops.bias_add(output, bia, data_format=fmt, name='ba4')

        fd = {
            inp: np.zeros(self._ip_shp([1, 84, 84, 4], fmt)),
            wei: np.zeros([8, 8, 4, 16]),
            bia: np.zeros([16]),
        }
        result = sess.run(output, fd)
        self.assertAllClose(result, np.zeros(self._ip_shp([1, 20, 20, 16],
                                                          fmt)))

      report = pva.openReport(report_helper.find_report())
      ok = [
          '__seed*', 'host-exchange-local-copy-',
          'cnv4*/convolution.*/Conv_8x8_stride4x4', 'ba4*/fusion/Op/Add'
      ]
      self.assert_all_compute_sets_and_list(report, ok)

  def testConv1x1_WithBias(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    for fmt in self.data_formats:
      report_helper.clear_reports()
      with self.session() as sess:
        with ops.device("/device:IPU:0"):
          inp = array_ops.placeholder(np.float32,
                                      self._ip_shp([1, 1, 1, 4], fmt),
                                      name="inp")
          wei = array_ops.placeholder(np.float32, [1, 1, 4, 16], name="wei")
          bia = array_ops.placeholder(np.float32, [16], name="bia")
          output = nn_ops.conv2d(inp,
                                 wei,
                                 strides=[1, 1, 1, 1],
                                 padding="VALID",
                                 data_format=fmt,
                                 name='cnv5')
          output = nn_ops.bias_add(output, bia, data_format=fmt, name='ba5')

        fd = {
            inp: np.zeros(self._ip_shp([1, 1, 1, 4], fmt)),
            wei: np.zeros([1, 1, 4, 16]),
            bia: np.zeros([16]),
        }
        result = sess.run(output, fd)
        self.assertAllClose(result, np.zeros(self._ip_shp([1, 1, 1, 16], fmt)))

      report = pva.openReport(report_helper.find_report())
      ok = [
          '__seed*', 'Copy_', 'cnv5*/convolution.*/Conv_1x1',
          'ba5*/fusion/Op/Add'
      ]
      self.assert_all_compute_sets_and_list(report, ok)

  def testConvBackpropInput(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        ins = constant_op.constant([2, 8, 8, 3], np.int32)
        fil = array_ops.placeholder(np.float32, [2, 2, 3, 5], name="inp")
        bck = array_ops.placeholder(np.float32, [2, 8, 8, 5], name="wei")

        output = nn_ops.conv2d_backprop_input(ins,
                                              fil,
                                              bck,
                                              strides=[1, 1, 1, 1],
                                              padding="SAME")

      fd = {
          fil: np.zeros([2, 2, 3, 5]),
          bck: np.zeros([2, 8, 8, 5]),
      }
      result = sess.run(output, fd)
      self.assertAllClose(result, np.zeros([2, 8, 8, 3]))

    report = pva.openReport(report_helper.find_report())
    # pylint: disable=line-too-long
    ok = [
        '__seed*',
        'Copy_',
        'Conv2DBackpropInput/conv-with-reverse/Conv_2x2',
    ]
    # pylint: enable=line-too-long
    self.assert_all_compute_sets_and_list(report, ok)

  def testConvBackpropFilter(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        inp = array_ops.placeholder(np.float32, [2, 8, 8, 3])
        fil = constant_op.constant([2, 2, 3, 5], np.int32)
        bck = array_ops.placeholder(np.float32, [2, 8, 8, 5], name="wei")

        output = nn_ops.conv2d_backprop_filter(inp,
                                               fil,
                                               bck,
                                               strides=[1, 1, 1, 1],
                                               padding="SAME")

      fd = {
          inp: np.zeros([2, 8, 8, 3]),
          bck: np.zeros([2, 8, 8, 5]),
      }
      result = sess.run(output, fd)
      self.assertAllClose(result, np.zeros([2, 2, 3, 5]))

    report = pva.openReport(report_helper.find_report())
    ok = [
        '__seed*',
        '[cC]opy',
        'Conv2DBackpropFilter/convolution.*/Conv_8x8',
    ]
    self.assert_all_compute_sets_and_list(report, ok)

  def testDepthwiseConv3x2(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [1, 2, 2, 3], name="a")
        pb = array_ops.placeholder(np.float32, [1, 1, 3, 2], name="b")
        pc = array_ops.placeholder(np.float32, [6], name="c")
        c = nn.depthwise_conv2d(pa, pb, strides=[1, 1, 1, 1], padding="SAME")
        output = c + pc

      fd = {
          pa: [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]],
          pb: [[[[6, 5], [4, 3], [2, 1]]]],
          pc: [1, 1, 1, 1, 1, 1]
      }
      result = sess.run(output, fd)
      self.assertAllClose(
          result, [[[[7, 6, 9, 7, 7, 4], [25, 21, 21, 16, 13, 7]],
                    [[43, 36, 33, 25, 19, 10], [61, 51, 45, 34, 25, 13]]]])

    report = pva.openReport(report_helper.find_report())
    ok = [
        '__seed*', 'host-exchange-local-copy-',
        'depthwise/convolution.*/Conv_1x1', 'add/fusion*/Add'
    ]
    self.assert_all_compute_sets_and_list(report, ok)

  def testDepthwiseConv3x1(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [1, 2, 2, 3], name="a")
        pb = array_ops.placeholder(np.float32, [1, 1, 3, 1], name="b")
        pc = array_ops.placeholder(np.float32, [3], name="c")
        c = nn.depthwise_conv2d(pa, pb, strides=[1, 1, 1, 1], padding="SAME")
        output = c + pc

      fd = {
          pa: [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]],
          pb: [[[[6], [4], [2]]]],
          pc: [1, 1, 1]
      }
      result = sess.run(output, fd)
      self.assertAllClose(
          result, [[[[7, 9, 7], [25, 21, 13]], [[43, 33, 19], [61, 45, 25]]]])

    report = pva.openReport(report_helper.find_report())
    # pylint: disable=line-too-long
    ok = [
        '__seed*', 'host-exchange-local-copy-',
        'depthwise/convolution.*/Conv_1x1', 'add/fusion*/Add'
    ]
    # pylint: enable=line-too-long
    self.assert_all_compute_sets_and_list(report, ok)

  def testDepthwiseConvBackpropInput(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = constant_op.constant([1, 8, 8, 3], dtype=np.int32)  # input sizes
        filt = array_ops.placeholder(np.float32, [3, 3, 3, 2], name="filt")
        outb = array_ops.placeholder(np.float32, [1, 8, 8, 6], name="outb")
        c = nn.depthwise_conv2d_native_backprop_input(pa,
                                                      filt,
                                                      outb,
                                                      strides=[1, 1, 1, 1],
                                                      padding="SAME")

      fd = {filt: np.zeros([3, 3, 3, 2]), outb: np.zeros([1, 8, 8, 6])}
      result = sess.run(c, fd)
      self.assertAllClose(result, np.zeros([1, 8, 8, 3]))

    report = pva.openReport(report_helper.find_report())
    # pylint: disable=line-too-long
    ok = [
        '__seed*', 'copy*OnTileCopy-',
        'DepthwiseConv2dNativeBackpropInput/conv-with-reverse/Conv_3x3/Convolve',
        'DepthwiseConv2dNativeBackpropInput/conv-with-reverse/Conv_3x3/WeightsTranspose'
    ]
    # pylint: enable=line-too-long
    self.assert_all_compute_sets_and_list(report, ok)

  def testDepthwiseConvBackpropInput1x1(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = constant_op.constant([1, 8, 8, 3], dtype=np.int32)  # input sizes
        pb = array_ops.placeholder(np.float32, [1, 1, 3, 2], name="b")
        pc = array_ops.placeholder(np.float32, [1, 8, 8, 6], name="c")
        c = nn.depthwise_conv2d_native_backprop_input(pa,
                                                      pb,
                                                      pc,
                                                      strides=[1, 1, 1, 1],
                                                      padding="SAME")

      fd = {pb: np.zeros([1, 1, 3, 2]), pc: np.zeros([1, 8, 8, 6])}
      result = sess.run(c, fd)
      self.assertAllClose(result, np.zeros([1, 8, 8, 3]))

    report = pva.openReport(report_helper.find_report())
    # pylint: disable=line-too-long
    ok = [
        '__seed*',
        'DepthwiseConv2dNativeBackpropInput/conv-with-reverse/*Transpose',
        'DepthwiseConv2dNativeBackpropInput/conv-with-reverse/Conv_1x1',
    ]
    # pylint: enable=line-too-long

    self.assert_compute_sets_contain_list(report, ok)

  def testDataLayout(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa1 = array_ops.placeholder(np.float32, [1, 14, 14, 64], name="a")
        pb1 = array_ops.placeholder(np.float32, [3, 3, 64, 128], name="b")
        bi1 = array_ops.placeholder(np.float32, [128], name="b")
        op1 = nn_ops.convolution(pa1, pb1, padding="SAME", data_format='NHWC')
        op1 = nn_ops.bias_add(op1, bi1, data_format='NHWC')

        pa2 = array_ops.placeholder(np.float32, [1, 64, 14, 14], name="a")
        pb2 = array_ops.placeholder(np.float32, [3, 3, 64, 128], name="b")
        bi2 = array_ops.placeholder(np.float32, [128], name="b")
        op2 = nn_ops.convolution(pa2, pb2, padding="SAME", data_format='NCHW')
        op2 = nn_ops.bias_add(op2, bi2, data_format='NCHW')

      fd = {
          pa1: np.zeros([1, 14, 14, 64]),
          pb1: np.zeros([3, 3, 64, 128]),
          bi1: np.zeros([128]),
          pa2: np.zeros([1, 64, 14, 14]),
          pb2: np.zeros([3, 3, 64, 128]),
          bi2: np.zeros([128]),
      }
      result = sess.run(op1, fd)
      self.assertAllClose(result, np.zeros([1, 14, 14, 128]))

      report = pva.openReport(report_helper.find_report())
      mem_nhwc = sum(tile.memory.total.excludingGaps
                     for tile in report.compilation.tiles)
      report_helper.clear_reports()

      result = sess.run(op2, fd)
      self.assertAllClose(result, np.zeros([1, 128, 14, 14]))

      report = pva.openReport(report_helper.find_report())
      mem_nchw = sum(tile.memory.total.excludingGaps
                     for tile in report.compilation.tiles)

      self.assertTrue((mem_nhwc - mem_nchw) / mem_nhwc > -0.1)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
