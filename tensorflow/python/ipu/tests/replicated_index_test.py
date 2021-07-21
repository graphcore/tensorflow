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
from tensorflow.python.ipu.config import IPUConfig

from tensorflow.python import ipu
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.client import session as sl
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables


class TestReplicatedIndex(test_util.TensorFlowTestCase):
  @tu.test_uses_ipus(num_ipus=2)
  @test_util.deprecated_graph_mode_only
  def testReplicatedIndex(self):
    def my_graph(inp):
      with ops.device("/device:IPU:0"):
        idx = ipu.replication_ops.replication_index()
        x = array_ops.slice(inp, [idx], [1])
        return ipu.cross_replica_ops.cross_replica_sum(x)

    with ops.device('cpu'):
      inp = array_ops.placeholder(np.float32, [2], name="data")

    out = ipu.ipu_compiler.compile(my_graph, [inp])

    cfg = IPUConfig()
    cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    with sl.Session() as sess:
      sess.run(variables.global_variables_initializer())

      result = sess.run(out, {inp: [2, 4]})
      self.assertAllClose(result[0], [6.0])


if __name__ == "__main__":
  googletest.main()
