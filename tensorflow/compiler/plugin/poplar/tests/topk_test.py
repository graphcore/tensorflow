# Copyright 2019 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import os
import numpy as np
import test_utils as tu

from tensorflow.python.platform import googletest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops

TYPES = [np.float16, np.float32, np.int32]
TESTCASES = [{"testcase_name": np.dtype(x).name, "dtype": x} for x in TYPES]


def _get_random_input(dtype, shape):
  if np.issubdtype(dtype, np.integer):
    info_fn = np.iinfo
    random_fn = np.random.random_integers
  else:
    info_fn = np.finfo
    random_fn = np.random.uniform
  return random_fn(
      info_fn(dtype).min, info_fn(dtype).max, size=shape).astype(dtype)


class ArgTopK(test_util.TensorFlowTestCase, parameterized.TestCase):
  @parameterized.named_parameters(*TESTCASES)
  def testTopKBasic(self, dtype):
    def model(a):
      return nn.top_k(a, k=10, sorted=True)

    with ops.device('cpu'):
      pa = array_ops.placeholder(dtype, [100])

    with ops.device("/device:IPU:0"):
      out = model(pa)

    tu.configure_ipu_system()

    input = _get_random_input(dtype, (100))
    # IPU Run
    with tu.ipu_session() as sess:
      fd = {pa: input}
      ipu_result = sess.run(out, fd)

    with ops.device("/device:CPU:0"):
      out = model(pa)

    with tu.ipu_session() as sess:
      fd = {pa: input}
      cpu_result = sess.run(out, fd)
      self.assertAllClose(cpu_result, ipu_result)

  @parameterized.named_parameters(*TESTCASES)
  def testArgMaxMultiDimensional(self, dtype):
    def model(a, k):
      return nn.top_k(a, k=k, sorted=True)

    with ops.device('cpu'):
      pa = array_ops.placeholder(dtype, [1, 2, 3, 4, 5, 6])
      p_k = array_ops.placeholder(np.int32, shape=())

    with ops.device("/device:IPU:0"):
      out = model(pa, p_k)

    tu.configure_ipu_system()

    for k in range(6):
      input = _get_random_input(dtype, (1, 2, 3, 4, 5, 6))
      with ops.device("/device:IPU:0"):
        out = model(pa, p_k)

      with tu.ipu_session() as sess:
        fd = {pa: input, p_k: k}
        ipu_result = sess.run(out, fd)

      with ops.device("/device:CPU:0"):
        out = model(pa, p_k)

      with tu.ipu_session() as sess:
        fd = {pa: input, p_k: k}
        cpu_result = sess.run(out, fd)
        self.assertAllClose(cpu_result, ipu_result)

  @parameterized.named_parameters(*TESTCASES)
  def testTopkSort(self, dtype):
    def model(a, k):
      return nn.top_k(a, k=k, sorted=True)

    with ops.device('cpu'):
      pa = array_ops.placeholder(dtype, [100])
      p_k = array_ops.placeholder(np.int32, shape=())

    with ops.device("/device:IPU:0"):
      out = model(pa, p_k)

    tu.configure_ipu_system()

    input = _get_random_input(dtype, (100))
    with ops.device("/device:IPU:0"):
      out = model(pa, p_k)

    with tu.ipu_session() as sess:
      fd = {pa: input, p_k: 100}
      ipu_result = sess.run(out, fd)

    with ops.device("/device:CPU:0"):
      out = model(pa, p_k)

    with tu.ipu_session() as sess:
      fd = {pa: input, p_k: 100}
      cpu_result = sess.run(out, fd)

    self.assertAllClose(cpu_result, ipu_result)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = (
      '--tf_xla_min_cluster_size=1 ' + os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
