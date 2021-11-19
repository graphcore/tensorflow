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

import json
import multiprocessing
import os

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import loops
from tensorflow.python.ipu import config
from tensorflow.python.ipu import scopes
from tensorflow.python.platform import test


class ExceptionTest(test_util.TensorFlowTestCase):
  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_soft_error(self):
    def inner_process():
      gcda_options = os.environ.get("GCDA_OPTIONS", "{}")
      gcda_options_json = json.loads(gcda_options)
      gcda_options_json["simulateSOCError.errorList"] = "IPUSOFTERR"
      gcda_options_json["simulateSOCError.triggerCount"] = 1
      gcda_options_json["simulateSOCError.rearmOnReset"] = True
      gcda_options = json.dumps(gcda_options_json)

      with test.mock.patch.dict("os.environ", {"GCDA_OPTIONS": gcda_options}), \
        session.Session() as sess:

        cfg = config.IPUConfig()
        cfg.auto_select_ipus = 1
        tu.add_hw_ci_connection_options(cfg)
        cfg.compilation_poplar_options = {'target.syncPollPeriodUs': '100'}
        cfg.configure_ipu_system()

        # The dataset for feeding the graphs.
        shape = (16, 16)
        ds = dataset_ops.Dataset.from_tensors(
            constant_op.constant(1.0, shape=shape))
        ds = ds.repeat()

        # The host side queues.
        infeed_queue = ipu_infeed_queue.IPUInfeedQueue(ds)
        outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

        def body(x):
          x = x @ x
          return outfeed_queue.enqueue(12.0 / x)

        # Wrap in a loop.
        def my_net():
          r = loops.repeat(10, body, [], infeed_queue)
          return r

        with scopes.ipu_scope('/device:IPU:0'):
          res = ipu_compiler.compile(my_net, inputs=[])

        with self.assertRaisesRegex(
            errors.InternalError,
            r"\[Poplar\]\[Load engine\] recoverable_runtime_error: "
            r"\[Recovery action: IPU_RESET\] CIUERRVR\.IPUSOFTERR error on "
            r"IPU 0 [\s\S]* IPU will be reset the next time a program is "
            r"executed."):
          sess.run(infeed_queue.initializer)
          sess.run(res)
          sess.run(res)

    process = multiprocessing.Process(target=inner_process)
    process.start()
    process.join()
    self.assertEqual(process.exitcode, 0)


if __name__ == "__main__":
  test.main()
