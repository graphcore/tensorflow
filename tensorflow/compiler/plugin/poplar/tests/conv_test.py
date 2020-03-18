# Copyright 2017, 2018, 2019 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import test_utils as tu

from tensorflow.compiler.tests import xla_test
from tensorflow.python import ipu
from tensorflow.python.platform import googletest
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.training import gradient_descent


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
    for fmt in self.data_formats:
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

        report = tu.ReportJSON(self, sess)
        report.reset()

        fd = {
            pa: np.zeros(self._ip_shp([1, 14, 14, 64], fmt)),
            pb: np.zeros([3, 3, 64, 128]),
            bi: np.zeros([128]),
        }
        result = sess.run(output, fd)
        self.assertAllClose(result,
                            np.zeros(self._ip_shp([1, 14, 14, 128], fmt)))

        report.parse_log()

        ok = [
            '__seed*', 'Copy_*actsRearranged', 'host-exchange-local-copy-',
            'cnv3*/convolution.*/Conv_3x3', 'ba3*/fusion/Op/Add'
        ]

        report.assert_all_compute_sets_and_list(ok)

  def testConv8x8_WithBias(self):
    for fmt in self.data_formats:
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

        report = tu.ReportJSON(self, sess)
        report.reset()

        fd = {
            inp: np.zeros(self._ip_shp([1, 84, 84, 4], fmt)),
            wei: np.zeros([8, 8, 4, 16]),
            bia: np.zeros([16]),
        }
        result = sess.run(output, fd)
        self.assertAllClose(result, np.zeros(self._ip_shp([1, 20, 20, 16],
                                                          fmt)))

        report.parse_log()

        ok = [
            '__seed*', 'host-exchange-local-copy-',
            'Copy_{*_input,*_weights}_to_{*actsRearranged,*weightsRearranged}',
            'cnv4*/convolution.*/Conv_8x8_stride4x4', 'ba4*/fusion/Op/Add'
        ]
        report.assert_all_compute_sets_and_list(ok)

  def testConv1x1_WithBias(self):
    for fmt in self.data_formats:
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

        report = tu.ReportJSON(self, sess)
        report.reset()

        fd = {
            inp: np.zeros(self._ip_shp([1, 1, 1, 4], fmt)),
            wei: np.zeros([1, 1, 4, 16]),
            bia: np.zeros([16]),
        }
        result = sess.run(output, fd)
        self.assertAllClose(result, np.zeros(self._ip_shp([1, 1, 1, 16], fmt)))

        report.parse_log()

        ok = [
            '__seed*', 'host-exchange-local-copy-', 'Copy_',
            'cnv5*/convolution.*/Conv_1x1', 'ba5*/fusion/Op/Add'
        ]
        report.assert_all_compute_sets_and_list(ok)

  def testConvBackpropInput(self):
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

      report = tu.ReportJSON(self, sess)
      report.reset()

      fd = {
          fil: np.zeros([2, 2, 3, 5]),
          bck: np.zeros([2, 8, 8, 5]),
      }
      result = sess.run(output, fd)
      self.assertAllClose(result, np.zeros([2, 8, 8, 3]))

      report.parse_log()

      ok = [
          '__seed*', 'Copy_', 'Conv2DBackpropInput/fusion*/Conv_2x2',
          'Conv2DBackpropInput/fusion*/WeightTranspose'
      ]
      report.assert_all_compute_sets_and_list(ok)

  def testConvBackpropFilter(self):
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

      report = tu.ReportJSON(self, sess)
      report.reset()

      fd = {
          inp: np.zeros([2, 8, 8, 3]),
          bck: np.zeros([2, 8, 8, 5]),
      }
      result = sess.run(output, fd)
      self.assertAllClose(result, np.zeros([2, 2, 3, 5]))

      report.parse_log()

      ok = [
          '__seed*',
          'Copy_',
          'Conv2DBackpropFilter/convolution.*/Conv_8x8',
      ]
      report.assert_all_compute_sets_and_list(ok)

  def testDepthwiseConv3x2(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [1, 2, 2, 3], name="a")
        pb = array_ops.placeholder(np.float32, [1, 1, 3, 2], name="b")
        pc = array_ops.placeholder(np.float32, [6], name="c")
        c = nn.depthwise_conv2d(pa, pb, strides=[1, 1, 1, 1], padding="SAME")
        output = c + pc

      report = tu.ReportJSON(self, sess)
      report.reset()

      fd = {
          pa: [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]],
          pb: [[[[6, 5], [4, 3], [2, 1]]]],
          pc: [1, 1, 1, 1, 1, 1]
      }
      result = sess.run(output, fd)
      self.assertAllClose(
          result, [[[[7, 6, 9, 7, 7, 4], [25, 21, 21, 16, 13, 7]],
                    [[43, 36, 33, 25, 19, 10], [61, 51, 45, 34, 25, 13]]]])

      report.parse_log()

      ok = [
          '__seed*', 'host-exchange-local-copy-',
          'depthwise/convolution.*/Conv_1x1', 'add/fusion*/Add'
      ]
      report.assert_all_compute_sets_and_list(ok)

  def testDepthwiseConv3x1(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [1, 2, 2, 3], name="a")
        pb = array_ops.placeholder(np.float32, [1, 1, 3, 1], name="b")
        pc = array_ops.placeholder(np.float32, [3], name="c")
        c = nn.depthwise_conv2d(pa, pb, strides=[1, 1, 1, 1], padding="SAME")
        output = c + pc

      report = tu.ReportJSON(self, sess)
      report.reset()

      fd = {
          pa: [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]],
          pb: [[[[6], [4], [2]]]],
          pc: [1, 1, 1]
      }
      result = sess.run(output, fd)
      self.assertAllClose(
          result, [[[[7, 9, 7], [25, 21, 13]], [[43, 33, 19], [61, 45, 25]]]])

      report.parse_log()

      # pylint: disable=line-too-long
      ok = [
          '__seed*', 'host-exchange-local-copy-', 'Copy_',
          'depthwise/convolution.*/Conv_1x1',
          'Copy_depthwise/convolution.*/Conv_1x1/partials_to_depthwise/convolution.*/Conv_1x1/partials[[]cloned[]]',
          'add/fusion*/Add'
      ]
      # pylint: enable=line-too-long
      report.assert_all_compute_sets_and_list(ok)

  def testDepthwiseConvBackpropInput(self):
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

      report = tu.ReportJSON(self, sess)
      report.reset()

      fd = {filt: np.zeros([3, 3, 3, 2]), outb: np.zeros([1, 8, 8, 6])}
      result = sess.run(c, fd)
      self.assertAllClose(result, np.zeros([1, 8, 8, 3]))

      report.parse_log()

      ok = [
          '__seed*',
          'DepthwiseConv2dNativeBackpropInput/fusion*/WeightTranspose',
          'DepthwiseConv2dNativeBackpropInput/fusion*/Conv_3x3', 'Copy_'
      ]
      report.assert_all_compute_sets_and_list(ok)

  def testDepthwiseConvBackpropInput1x1(self):
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

      report = tu.ReportJSON(self, sess)
      report.reset()

      fd = {pb: np.zeros([1, 1, 3, 2]), pc: np.zeros([1, 8, 8, 6])}
      result = sess.run(c, fd)
      self.assertAllClose(result, np.zeros([1, 8, 8, 3]))

      report.parse_log()

      ok = [
          '__seed*',
          'Copy_',
          'DepthwiseConv2dNativeBackpropInput/fusion*/WeightTranspose',
          'DepthwiseConv2dNativeBackpropInput/fusion*/Conv_1x1',
      ]

      report.assert_all_compute_sets_and_list(ok)

  def testDepthwiseConvBackpropFilter(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [1, 6, 6, 3], name="a")
        pb = constant_op.constant([3, 3, 3, 2], dtype=np.int32)  # filter sizes
        pc = array_ops.placeholder(np.float32, [1, 6, 6, 6], name="c")
        c = nn.depthwise_conv2d_native_backprop_filter(pa,
                                                       pb,
                                                       pc,
                                                       strides=[1, 1, 1, 1],
                                                       padding="SAME")

      report = tu.ReportJSON(self, sess)
      report.reset()

      fd = {pa: np.zeros([1, 6, 6, 3]), pc: np.zeros([1, 6, 6, 6])}
      result = sess.run(c, fd)
      self.assertAllClose(result, np.zeros([3, 3, 3, 2]))

      report.parse_log()

      ok = [
          '__seed*', 'Copy_',
          'DepthwiseConv2dNativeBackpropFilter/fusion*/Conv_6x6'
      ]
      report.assert_all_compute_sets_and_list(ok)

  def testDepthwiseConvBackpropFilter1x1(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [1, 6, 6, 3], name="a")
        pb = constant_op.constant([1, 1, 3, 2], dtype=np.int32)  # filter sizes
        pc = array_ops.placeholder(np.float32, [1, 6, 6, 6], name="c")
        c = nn.depthwise_conv2d_native_backprop_filter(pa,
                                                       pb,
                                                       pc,
                                                       strides=[1, 1, 1, 1],
                                                       padding="SAME")

      report = tu.ReportJSON(self, sess)
      report.reset()

      fd = {pa: np.zeros([1, 6, 6, 3]), pc: np.zeros([1, 6, 6, 6])}
      result = sess.run(c, fd)
      self.assertAllClose(result, np.zeros([1, 1, 3, 2]))

      report.parse_log()

      ok = [
          '__seed*', 'Copy_',
          'DepthwiseConv2dNativeBackpropFilter/fusion*/Conv_6x6'
      ]
      report.assert_all_compute_sets_and_list(ok)

  def testDepthwiseConvBackpropFilter1x1WithRelu(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [1, 6, 6, 3], name="a")
        pb = constant_op.constant([1, 1, 3, 2], dtype=np.int32)  # filter sizes
        pc = array_ops.placeholder(np.float32, [1, 6, 6, 6], name="c")
        c = nn.depthwise_conv2d_native_backprop_filter(pa,
                                                       pb,
                                                       pc,
                                                       strides=[1, 1, 1, 1],
                                                       padding="SAME")
        c = nn.relu(c)

      report = tu.ReportJSON(self, sess)
      report.reset()

      fd = {pa: np.zeros([1, 6, 6, 3]), pc: np.zeros([1, 6, 6, 6])}
      result = sess.run(c, fd)
      self.assertAllClose(result, np.zeros([1, 1, 3, 2]))

      report.parse_log()

      ok = [
          '__seed*', 'Copy_',
          'DepthwiseConv2dNativeBackpropFilter/fusion*/Conv_6x6',
          'Relu/custom-call*/Nonlinearity'
      ]
      report.assert_all_compute_sets_and_list(ok)

  def testDataLayout(self):
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

      report = tu.ReportJSON(self, sess)
      report.reset()

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

      report.parse_log()
      mem_nhwc = report.get_total_tile_memory()

      result = sess.run(op2, fd)
      self.assertAllClose(result, np.zeros([1, 128, 14, 14]))

      report.parse_log()
      mem_nchw = report.get_total_tile_memory()

      self.assertTrue((mem_nhwc - mem_nchw) / mem_nhwc > -0.1)

  def testGroupedConvolutions(self):
    """
    Check we can compile a graph with grouped convolutions, ie. where the
    input and output features are divided into K groups, with the N outputs
    in the kth group just depending on the M inputs in the kth group.
    """
    ndims = 2
    M = 3
    N = 5
    K = 7  # input features per group, output features per group, number of groups
    with self.session() as sess:
      with variable_scope.variable_scope("vs", use_resource=True):
        with ops.device("cpu"):
          inp = array_ops.placeholder(np.float32, [1] + [24] * ndims + [M * K],
                                      name="input")
          bias = array_ops.placeholder(np.float32, [N * K], name="bias")
        with ops.device("/device:IPU:0"):
          weights = variable_scope.get_variable("weights",
                                                [8] * ndims + [M, N * K])
          output = nn.convolution(inp,
                                  weights,
                                  strides=[1] + [4] * ndims + [1],
                                  padding="VALID",
                                  name='cnv')
          output = nn.bias_add(output, bias, name='bias_add')
          loss = math_ops.reduce_sum(math_ops.square(output))
          optimizer = gradient_descent.GradientDescentOptimizer(0.0005)
          train = optimizer.minimize(loss)

      train = ipu.ipu_compiler.compile(lambda: (loss, train), [])

      report = tu.ReportJSON(self, sess)

      sess.run(variables.global_variables_initializer())

      report.reset()

      fd = {
          inp: np.random.random_sample([1] + [24] * ndims + [M * K]),
          bias: np.random.random_sample([N * K])
      }
      sess.run(train, fd)

      report.parse_log()

      ok = [
          '__seed*',
          'Copy_',
          'host-exchange*',
          'vs/cnv/convolution*/Conv_8x8_stride4x4/Convolve',
          'vs/cnv/convolution*/Conv_8x8_stride4x4/Reduce',
          'vs/gradients/vs/cnv_grad/Conv2DBackpropFilter/fusion*/Conv_5x5',
          'vs/gradients/vs/cnv_grad/Conv2DBackpropFilter/fusion*/AddTo',
          'vs/gradients/vs/Square_grad/Mul/fusion*/Op/Multiply',
          'vs/bias_add/fusion/Op/Add',
          'vs/Sum/reduce',
          'vs/Square/multiply*/Op/Multiply',
      ]
      report.assert_all_compute_sets_and_list(ok)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
