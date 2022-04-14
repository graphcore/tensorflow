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

import os
import tempfile
import tensorflow as tf
from tensorflow.python.ipu import test_utils as tu
from tensorflow.python import ipu
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.compiler.tests import xla_test
from tensorflow.python.platform import googletest
import popef


class DummyInputTest(xla_test.XLATestCase):
  # Overriding abstract method.
  def cached_session(self):
    return 0

  # Overriding abstract method.
  def test_session(self):
    return 0

  @test_util.run_v2_only
  @tu.test_uses_ipus(num_ipus=1, allow_ipu_model=False)
  def testDummyInputDeduplication(self):
    @tf.function(jit_compile=True)
    def jsh_f(tensor_count, factor1, factor2):
      def body(x):
        return outfeed_queue.enqueue(x + factor1 + factor2)

      ipu.loops.repeat(tensor_count, body, infeed_queue=infeed_queue)

    # Enable the cache
    with tempfile.TemporaryDirectory() as cache_dir:
      poplar_flags = "--executable_cache_path={} {}".format(
          cache_dir, os.environ.get("TF_POPLAR_FLAGS", ""))

      with test.mock.patch.dict("os.environ",
                                {"TF_POPLAR_FLAGS": poplar_flags}):
        # Create IPU configuration
        config = ipu.config.IPUConfig()
        config.auto_select_ipus = 1
        config.device_connection.enable_remote_buffers = True
        config.device_connection.type = ipu.utils.DeviceConnectionType.ON_DEMAND
        config.configure_ipu_system()

        # Input dataset
        tensor_count = 3
        input_tensors = tf.reshape(tf.range(4 * tensor_count, dtype=tf.int32),
                                   (tensor_count, 2, 2))
        dataset = tf.data.Dataset.from_tensor_slices(input_tensors)

        # These will end up as dummy_inputs
        factors = [
            tf.constant([[1, 2], [3, 4]], dtype=tf.int32),
            tf.constant([[5, 6], [7, 8]], dtype=tf.int32)
        ]

        # Run the model
        strategy = ipu.ipu_strategy.IPUStrategy()
        with strategy.scope():
          infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
          outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()
          # pylint: disable=pointless-statement
          variables.global_variables_initializer
          # pylint: disable=pointless-statement
          infeed_queue.initializer
          strategy.run(jsh_f, args=[tensor_count] + factors)
          outfeed_queue.dequeue()

      # Load the executable just saved to the cache
      cache_files = [
          x for x in os.listdir(cache_dir) if x.endswith(".poplar_exec")
      ]
      self.assertEqual(len(cache_files), 1)
      r = popef.Reader()
      r.parseFile(os.path.join(cache_dir, cache_files[0]))
      self.assertEqual(len(r.metadata()), 1)
      m = r.metadata()[0]
      # Read the anchor names
      anchor_names = [anch.name() for anch in m.anchors()]
      # Assert that there are no duplicates
      self.assertEqual(len(anchor_names), len(set(anchor_names)))


if __name__ == "__main__":
  googletest.main()
