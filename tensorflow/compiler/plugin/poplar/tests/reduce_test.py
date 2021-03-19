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

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn


class IpuXlaConvTest(xla_test.XLATestCase):
  def testReductionMeanDim12(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [2, 7, 7, 32], name="a")
        output = math_ops.reduce_mean(pa, axis=[1, 2])

        fd = {pa: np.ones([2, 7, 7, 32])}
        result = sess.run(output, fd)
        self.assertAllClose(result, np.ones([2, 32]))

  def testReductionMeanDim03(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [2, 7, 7, 32], name="a")
        output = math_ops.reduce_mean(pa, axis=[0, 3])

        fd = {pa: np.ones([2, 7, 7, 32])}
        result = sess.run(output, fd)
        self.assertAllClose(result, np.ones([7, 7]))

  def testReductionMeanDim13(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [2, 7, 7, 32], name="a")
        output = math_ops.reduce_mean(pa, axis=[1, 3])

        fd = {pa: np.ones([2, 7, 7, 32])}
        result = sess.run(output, fd)
        self.assertAllClose(result, np.ones([2, 7]))

  def testReductionMeanDim23(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [2, 7, 7, 32], name="a")
        output = math_ops.reduce_mean(pa, axis=[2, 3])

        fd = {pa: np.ones([2, 7, 7, 32])}
        result = sess.run(output, fd)
        self.assertAllClose(result, np.ones([2, 7]))

  def testAvgPoolSamePaddingWithStridesF32(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [1, 1, 10, 10], name="a")
        output = nn.avg_pool(pa,
                             ksize=[1, 1, 5, 5],
                             strides=[1, 1, 2, 2],
                             data_format='NCHW',
                             padding='SAME',
                             name="avg")

        fd = {pa: np.ones([1, 1, 10, 10])}
        result = sess.run(output, fd)
        self.assertAllClose(result, np.ones([1, 1, 5, 5]))

  def testAvgPoolSamePaddingWithStridesF16(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float16, [1, 1, 10, 10], name="a")
        output = nn.avg_pool(pa,
                             ksize=[1, 1, 5, 5],
                             strides=[1, 1, 2, 2],
                             data_format='NCHW',
                             padding='SAME')

        fd = {pa: np.ones([1, 1, 10, 10])}
        result = sess.run(output, fd)
        self.assertAllClose(result, np.ones([1, 1, 5, 5]))

  def testAvgPoolValidPaddingWithStridesF32(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [1, 1, 10, 10], name="a")
        output = nn.avg_pool(pa,
                             ksize=[1, 1, 5, 5],
                             strides=[1, 1, 2, 2],
                             data_format='NCHW',
                             padding='VALID')

        fd = {pa: np.ones([1, 1, 10, 10])}
        result = sess.run(output, fd)
        self.assertAllClose(result, np.ones([1, 1, 3, 3]))

  def testAvgPoolValidPaddingWithStridesF16(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float16, [1, 1, 10, 10], name="a")
        output = nn.avg_pool(pa,
                             ksize=[1, 1, 5, 5],
                             strides=[1, 1, 2, 2],
                             data_format='NCHW',
                             padding='VALID')

        fd = {pa: np.ones([1, 1, 10, 10])}
        result = sess.run(output, fd)
        self.assertAllClose(result, np.ones([1, 1, 3, 3]))

  def testMaxPoolSamePaddingWithStridesF32(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [1, 1, 10, 10], name="a")
        output = nn.max_pool(pa,
                             ksize=[1, 1, 5, 5],
                             strides=[1, 1, 2, 2],
                             data_format='NCHW',
                             padding='SAME',
                             name="max")

        fd = {pa: np.ones([1, 1, 10, 10])}
        result = sess.run(output, fd)
        self.assertAllClose(result, np.ones([1, 1, 5, 5]))

  def testMaxPoolValidPaddingWithStridesF32(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [1, 1, 10, 10], name="a")
        output = nn.max_pool(pa,
                             ksize=[1, 1, 5, 5],
                             strides=[1, 1, 2, 2],
                             data_format='NCHW',
                             padding='VALID',
                             name="max")

        fd = {pa: np.ones([1, 1, 10, 10])}
        result = sess.run(output, fd)
        self.assertAllClose(result, np.ones([1, 1, 3, 3]))

  def testAvgPoolSamePaddingWithStridesF32Dim12(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [1, 10, 10, 1], name="a")
        output = nn.avg_pool(pa,
                             ksize=[1, 5, 5, 1],
                             strides=[1, 2, 2, 1],
                             data_format='NHWC',
                             padding='SAME',
                             name="avg")

        fd = {pa: np.ones([1, 10, 10, 1])}
        result = sess.run(output, fd)
        self.assertAllClose(result, np.ones([1, 5, 5, 1]))

  def testAvgPoolValidPaddingWithStridesF32Dim12(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [1, 10, 10, 1], name="a")
        output = nn.avg_pool(pa,
                             ksize=[1, 5, 5, 1],
                             strides=[1, 2, 2, 1],
                             data_format='NHWC',
                             padding='VALID',
                             name="avg")

        fd = {pa: np.ones([1, 10, 10, 1])}
        result = sess.run(output, fd)
        self.assertAllClose(result, np.ones([1, 3, 3, 1]))

  def testAvgPool3DSamePaddingWithStridesF32(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [1, 1, 10, 10, 10], name="a")
        output = nn.avg_pool3d(pa,
                               ksize=[1, 1, 5, 5, 5],
                               strides=[1, 1, 2, 2, 2],
                               data_format='NCDHW',
                               padding='SAME',
                               name="avg")

        fd = {pa: np.ones([1, 1, 10, 10, 10])}
        result = sess.run(output, fd)
        self.assertAllClose(result, np.ones([1, 1, 5, 5, 5]))

  def testAvgPool3DSamePaddingWithStridesF16(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float16, [1, 1, 10, 10, 10], name="a")
        output = nn.avg_pool3d(pa,
                               ksize=[1, 1, 5, 5, 5],
                               strides=[1, 1, 2, 2, 2],
                               data_format='NCDHW',
                               padding='SAME')

        fd = {pa: np.ones([1, 1, 10, 10, 10])}
        result = sess.run(output, fd)
        self.assertAllClose(result, np.ones([1, 1, 5, 5, 5]))

  def testAvgPool3DValidPaddingWithStridesF32(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [1, 1, 10, 10, 10], name="a")
        output = nn.avg_pool3d(pa,
                               ksize=[1, 1, 5, 5, 5],
                               strides=[1, 1, 2, 2, 2],
                               data_format='NCDHW',
                               padding='VALID')

        fd = {pa: np.ones([1, 1, 10, 10, 10])}
        result = sess.run(output, fd)
        self.assertAllClose(result, np.ones([1, 1, 3, 3, 3]))

  def testAvgPool3DValidPaddingWithStridesF16(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float16, [1, 1, 10, 10, 10], name="a")
        output = nn.avg_pool3d(pa,
                               ksize=[1, 1, 5, 5, 5],
                               strides=[1, 1, 2, 2, 2],
                               data_format='NCDHW',
                               padding='VALID')

        fd = {pa: np.ones([1, 1, 10, 10, 10])}
        result = sess.run(output, fd)
        self.assertAllClose(result, np.ones([1, 1, 3, 3, 3]))

  def testMaxPool3DSamePaddingWithStridesF32(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [1, 1, 10, 10, 10], name="a")
        output = nn.max_pool3d(pa,
                               ksize=[1, 1, 5, 5, 5],
                               strides=[1, 1, 2, 2, 2],
                               data_format='NCDHW',
                               padding='SAME',
                               name="max")

        fd = {pa: np.ones([1, 1, 10, 10, 10])}
        result = sess.run(output, fd)
        self.assertAllClose(result, np.ones([1, 1, 5, 5, 5]))

  def testMaxPool3DValidPaddingWithStridesF32(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [1, 1, 10, 10, 10], name="a")
        output = nn.max_pool3d(pa,
                               ksize=[1, 1, 5, 5, 5],
                               strides=[1, 1, 2, 2, 2],
                               data_format='NCDHW',
                               padding='VALID',
                               name="max")

        fd = {pa: np.ones([1, 1, 10, 10, 10])}
        result = sess.run(output, fd)
        self.assertAllClose(result, np.ones([1, 1, 3, 3, 3]))

  def testAvgPool3DSamePaddingWithStridesF32Dim123(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [1, 10, 10, 12, 1], name="a")
        output = nn.avg_pool3d(pa,
                               ksize=[1, 5, 5, 7, 1],
                               strides=[1, 2, 2, 2, 1],
                               data_format='NDHWC',
                               padding='SAME',
                               name="avg")

        fd = {pa: np.ones([1, 10, 10, 12, 1])}
        result = sess.run(output, fd)
        self.assertAllClose(result, np.ones([1, 5, 5, 6, 1]))

  def testAvgPool3DValidPaddingWithStridesF32Dim123(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [1, 10, 12, 10, 1], name="a")
        output = nn.avg_pool3d(pa,
                               ksize=[1, 5, 5, 5, 1],
                               strides=[1, 2, 2, 2, 1],
                               data_format='NDHWC',
                               padding='VALID',
                               name="avg")

        fd = {pa: np.ones([1, 10, 12, 10, 1])}
        result = sess.run(output, fd)
        self.assertAllClose(result, np.ones([1, 3, 4, 3, 1]))


if __name__ == "__main__":
  googletest.main()
