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

import multiprocessing
import os
import tempfile
import numpy as np

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import test_util
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import loops
from tensorflow.python.ipu import config
from tensorflow.python.ipu.ops import application_compile_op
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.ipu import embedded_runtime


class TestApplicationCompileInProcess(test_util.TensorFlowTestCase):
  @tu.test_uses_ipus(num_ipus=2)
  @test_util.deprecated_graph_mode_only
  def test_from_cache(self):
    with tempfile.TemporaryDirectory() as tmp_folder:

      def inner_process(pid):
        # Set up a cache - executable will be re-used between the processes.
        os.environ["TF_POPLAR_FLAGS"] = \
          f"--executable_cache_path={tmp_folder} " \
          + os.environ.get("TF_POPLAR_FLAGS", "")

        cfg = config.IPUConfig()
        cfg.auto_select_ipus = 1
        tu.add_hw_ci_connection_options(cfg)
        cfg.configure_ipu_system()

        with session.Session() as sess:
          dataset = dataset_ops.Dataset.from_tensor_slices(
              np.ones(10, dtype=np.float32))
          infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
          outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

          def body(x):
            outfed = outfeed_queue.enqueue(x * x)
            return outfed

          def my_net():
            return loops.repeat(2, body, [], infeed_queue)

          result = application_compile_op.experimental_application_compile_op(
              my_net, output_path=f"{tmp_folder}/{pid}.poplar_exec)")
          compiled_path = sess.run(result)

        config.reset_ipu_configuration()

        engine_name = f'engine_{self.id()}{pid}'
        ctx = embedded_runtime.embedded_runtime_start(compiled_path, [],
                                                      engine_name)

        input_data = array_ops.placeholder(np.float32, shape=[])
        result = embedded_runtime.embedded_runtime_call([input_data], ctx)

        with session.Session() as sess:
          res = sess.run(result, {input_data: 2.})
          self.assertEqual(res, [4.])

      processes = []
      for i in range(2):
        processes.append(
            multiprocessing.Process(target=inner_process, args=[i]))
        processes[-1].start()

      for process in processes:
        process.join()
        self.assertEqual(process.exitcode, 0)


if __name__ == "__main__":
  test.main()
