# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python import ipu
from tensorflow.python.platform import googletest
from tensorflow.python.framework.errors_impl import InternalError
from tensorflow.python.ops import check_ops


class AssertTest(xla_test.XLATestCase):
  # Overriding abstract method.
  def cached_session(self):
    return 0

  # Overriding abstract method.
  def test_session(self):
    return 0

  def run_assert(self, value):
    with self.session() as sess:
      with sess.as_default():
        pa = array_ops.placeholder(np.float32, [])

        def model(x):
          zero = array_ops.constant(0, np.float32)
          c = check_ops.assert_none_equal(x, zero, [x])
          with ops.control_dependencies([c]):
            return array_ops.identity(x)

        with ops.device("/device:IPU:0"):
          compiled_graph = ipu.ipu_compiler.compile(model, [pa])

          cfg = ipu.config.IPUConfig()
          cfg.configure_ipu_system()

        return sess.run(compiled_graph, {pa: value})

  def testFalse(self):
    try:
      self.run_assert(0)
    except InternalError:
      return
    raise Exception('Assert exception should have been raised')

  def testTrue(self):
    try:
      self.run_assert(1)
    except InternalError as e:
      raise Exception(
          'No exception should not have been raised. Caught exception: ' +
          str(e))


if __name__ == "__main__":
  googletest.main()
