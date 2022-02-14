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
# =============================================================================
import numpy as np

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

from tensorflow.python import ipu


class TestReductions(test_util.TensorFlowTestCase):
  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def testReduceMean(self):
    shape = [2, 10000]
    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float16, shape)
      output = math_ops.reduce_mean(pa, axis=[1])

    config = ipu.config.IPUConfig()
    config.auto_select_ipus = [1]
    config.floating_point_behaviour.inv = True
    config.floating_point_behaviour.div0 = True
    config.floating_point_behaviour.oflo = True
    config.floating_point_behaviour.esr = \
      ipu.config.StochasticRoundingBehaviour.ON
    config.floating_point_behaviour.nanoo = True
    tu.add_hw_ci_connection_options(config)
    config.configure_ipu_system()

    with tu.ipu_session() as sess:
      val = np.finfo(np.float16).max / 2
      result = sess.run(output, {pa: np.full(shape, val)})
      self.assertAllClose(result, np.full([2], val))


if __name__ == "__main__":
  googletest.main()
