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

import os
from absl.testing import parameterized
import numpy as np
import test_utils as tu

from tensorflow.compiler.tests import xla_test
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python import ipu
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn

TYPES = [np.float16, np.float32, np.int32]
TESTCASES = [{"testcase_name": np.dtype(x).name, "dtype": x} for x in TYPES]


def _get_random_input(dtype, shape):
  if np.issubdtype(dtype, np.integer):
    info_fn = np.iinfo
    random_fn = np.random.random_integers
  else:
    info_fn = np.finfo
    random_fn = np.random.uniform

  n = len(np.empty(shape).flatten())
  s = set()
  while len(s) < n:
    data = random_fn(info_fn(dtype).min, info_fn(dtype).max,
                     size=n).astype(dtype)
    s.update(data.flatten())

  data = np.array(list(s), dtype=dtype)
  data = data[0:n]
  return data.reshape(shape)


class ArgTopK(xla_test.XLATestCase, parameterized.TestCase):
  configured = False

  def __configureIPU(self):
    if not self.configured:
      cfg = ipu.config.IPUConfig()
      cfg.ipu_model.compile_ipu_code = False
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()
      self.configured = True

  @parameterized.named_parameters(*TESTCASES)
  def testTopKBasic(self, dtype):
    def model(a):
      return nn.top_k(a, k=10, sorted=True)

    input_ = _get_random_input(dtype, (100))

    def executeModel(device):
      with self.session() as sess:
        with ops.device('cpu'):
          pa = array_ops.placeholder(dtype, [100])

        with ops.device(device):
          out = model(pa)

        fd = {pa: input_}
        return sess.run(out, fd)

    ipu_result = executeModel('/device:IPU:0')
    cpu_result = executeModel('/device:CPU:0')
    self.assertAllClose(cpu_result, ipu_result)

  @parameterized.named_parameters(*TESTCASES)
  def testArgMaxMultiDimensional(self, dtype):
    def model(a, k):
      return nn.top_k(a, k=k, sorted=True)

    def executeModel(input_, k, device):
      with self.session() as sess:
        with ops.device('cpu'):
          pa = array_ops.placeholder(dtype, [1, 2, 3, 4, 5, 6])
          p_k = array_ops.placeholder(np.int32, shape=())

        with ops.device(device):
          out = model(pa, p_k)

        fd = {pa: input_, p_k: k}
        return sess.run(out, fd)

    for k in range(6):
      input_ = _get_random_input(dtype, (1, 2, 3, 4, 5, 6))
      ipu_result = executeModel(input_, k, '/device:IPU:0')
      cpu_result = executeModel(input_, k, '/device:CPU:0')
      self.assertAllClose(cpu_result, ipu_result)

  @parameterized.named_parameters(*TESTCASES)
  def testTopkSort(self, dtype):
    def model(a, k):
      return nn.top_k(a, k=k, sorted=True)

    input_ = _get_random_input(dtype, (100))

    def executeModel(device):
      with self.session() as sess:
        with ops.device('cpu'):
          pa = array_ops.placeholder(dtype, [100])
          p_k = array_ops.placeholder(np.int32, shape=())

        with ops.device(device):
          out = model(pa, p_k)

        fd = {pa: input_, p_k: 100}
        return sess.run(out, fd)

    ipu_result = executeModel('/device:IPU:0')
    cpu_result = executeModel('/device:CPU:0')
    self.assertAllClose(cpu_result, ipu_result)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
