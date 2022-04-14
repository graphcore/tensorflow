# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.python.ipu import test_utils as tu
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.ipu import internal_ops
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import pipelining_ops
from tensorflow.python.ipu import utils
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope


class IterationCounterTest(test_util.TensorFlowTestCase):
  @test_util.deprecated_graph_mode_only
  def testIterationCounter(self):
    gradient_accumulation_count = 10
    repeat_count = 3

    dataset = Dataset.range(gradient_accumulation_count * repeat_count)
    dataset = dataset.map(lambda i: math_ops.cast(i, np.int32))
    dataset = dataset.batch(batch_size=1, drop_remainder=True)

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    def stage1(x):
      with variable_scope.variable_scope("vs", use_resource=True):
        c1 = internal_ops.get_current_iteration_counter()
      return x, c1

    def stage2(x, c1):
      with variable_scope.variable_scope("vs", use_resource=True):
        c2 = internal_ops.get_current_iteration_counter()
      return x, c1, c2

    def my_net():
      return pipelining_ops.pipeline(
          [stage1, stage2],
          gradient_accumulation_count=gradient_accumulation_count,
          repeat_count=repeat_count,
          infeed_queue=infeed_queue,
          outfeed_queue=outfeed_queue,
          device_mapping=[0, 0])

    with ops.device("/device:IPU:0"):
      r = ipu_compiler.compile(my_net, inputs=[])

    dequeue = outfeed_queue.dequeue()

    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    with tu.ipu_session() as sess:
      sess.run(infeed_queue.initializer)
      sess.run(r)
      _, c1, c2 = sess.run(dequeue)

    expected_numpy = np.tile(np.arange(gradient_accumulation_count),
                             reps=repeat_count)

    self.assertAllEqual(c1, c2)
    self.assertAllEqual(c1, expected_numpy)


if __name__ == "__main__":
  googletest.main()
