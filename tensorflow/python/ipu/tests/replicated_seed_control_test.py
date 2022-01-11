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

from tensorflow.python import ipu
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.framework import random_seed
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.compiler.plugin.poplar.ops import gen_popops_ops
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.platform import googletest

# This test rounds a 1000 element float32 by casting to a float16. Stochastic
# rounding will produce one of 2 values for each element.
#
# We test:
# - that 2 replicated devices do not have the same rounding with the initial
#   seed
# - that repeatedly rounding produces different values
# - setting the seed to a specific value generates the same rounding
# - setting the seed to different values generates different rounding
# - setting the seed after generating multiple rounding events restarts the
#   same sequence


@test_util.deprecated_graph_mode_only
class TestSeedControl(test_util.TensorFlowTestCase):
  @tu.test_uses_ipus(num_ipus=4)
  @test_util.deprecated_graph_mode_only
  def test_reset_seed(self):
    # The host side
    inp = array_ops.placeholder(np.float32, [1000])

    # The device side (0)
    with ipu.scopes.ipu_scope('/device:IPU:0'):
      out0 = math_ops.cast(inp, dtype=np.float16)

    # The device side (1)
    with ipu.scopes.ipu_scope('/device:IPU:1'):
      out1 = math_ops.cast(inp, dtype=np.float16)

    # Configure the hardware
    config = ipu.config.IPUConfig()
    config.auto_select_ipus = [2, 2]
    tu.add_hw_ci_connection_options(config)
    config.floating_point_behaviour.inv = True
    config.floating_point_behaviour.div0 = True
    config.floating_point_behaviour.oflo = True
    config.floating_point_behaviour.esr = True
    config.floating_point_behaviour.nanoo = True
    config.compilation_poplar_options = {'target.deterministicWorkers': 'true'}
    config.configure_ipu_system()

    with session_lib.Session() as sess:

      in_data = np.array([0.1] * 1000)

      # Each device doesn't generate same initial rounding
      res1 = sess.run(out0, {inp: in_data}).astype(np.float32)
      res2 = sess.run(out1, {inp: in_data}).astype(np.float32)
      self.assertNotAllEqual(res1, res2)

      # Device 0 doesn't generate the same rounding repeatedly
      res1 = sess.run(out0, {inp: in_data}).astype(np.float32)
      res2 = sess.run(out0, {inp: in_data}).astype(np.float32)
      self.assertNotAllEqual(res1, res2)
      res2 = sess.run(out0, {inp: in_data}).astype(np.float32)
      self.assertNotAllEqual(res1, res2)
      res2 = sess.run(out0, {inp: in_data}).astype(np.float32)
      self.assertNotAllEqual(res1, res2)

      # Device 1 doesn't generate the same rounding repeatedly
      res1 = sess.run(out1, {inp: in_data}).astype(np.float32)
      res2 = sess.run(out1, {inp: in_data}).astype(np.float32)
      self.assertNotAllEqual(res1, res2)

      # Device 0 can reset seed to generate the same result
      ipu.utils.reset_ipu_seed(1)
      res1 = sess.run(out0, {inp: in_data}).astype(np.float32)
      ipu.utils.reset_ipu_seed(1)
      res2 = sess.run(out0, {inp: in_data}).astype(np.float32)
      self.assertAllEqual(res1, res2)
      ipu.utils.reset_ipu_seed(1)
      res2 = sess.run(out0, {inp: in_data}).astype(np.float32)
      self.assertAllEqual(res1, res2)

      # Device 0 can reset with different seeds
      ipu.utils.reset_ipu_seed(1)
      res1 = sess.run(out0, {inp: in_data}).astype(np.float32)
      ipu.utils.reset_ipu_seed(2)
      res2 = sess.run(out0, {inp: in_data}).astype(np.float32)
      self.assertNotAllEqual(res1, res2)

      # Device 0 can do multiple steps and then have the seed reset
      ipu.utils.reset_ipu_seed(1)
      res1 = sess.run(out0, {inp: in_data}).astype(np.float32)
      ipu.utils.reset_ipu_seed(1)
      res2 = sess.run(out0, {inp: in_data}).astype(np.float32)
      self.assertAllEqual(res1, res2)
      ipu.utils.reset_ipu_seed(1)
      res2 = sess.run(out0, {inp: in_data}).astype(np.float32)
      self.assertAllEqual(res1, res2)
      ipu.utils.reset_ipu_seed(1)
      res2 = sess.run(out0, {inp: in_data}).astype(np.float32)
      self.assertAllEqual(res1, res2)

  @tu.test_uses_ipus(num_ipus=2)
  @test_util.deprecated_graph_mode_only
  def test_identical_replica_seeds(self):
    inp = array_ops.placeholder(np.float32, [1000])
    outfeed = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

    with ipu.scopes.ipu_scope('/device:IPU:0'):
      device_op = outfeed.enqueue(math_ops.cast(inp, dtype=np.float16))

    dequeue_op = outfeed.dequeue()

    config = ipu.config.IPUConfig()
    config.auto_select_ipus = [2]
    tu.add_hw_ci_connection_options(config)
    config.floating_point_behaviour.inv = True
    config.floating_point_behaviour.div0 = True
    config.floating_point_behaviour.oflo = True
    config.floating_point_behaviour.esr = True
    config.floating_point_behaviour.nanoo = True
    # Portable determinism is required as the replicas use different IPUs.
    config.compilation_poplar_options = {
        'target.deterministicWorkers': 'portable'
    }
    config.configure_ipu_system()

    with session_lib.Session() as sess:

      in_data = np.ones(inp.shape) * 0.1

      # Each replica does not generate same initial rounding.
      sess.run(device_op, {inp: in_data})
      res1, res2 = sess.run(dequeue_op)[0].astype(np.float32)
      self.assertNotAllEqual(res1, res2)

      # Each replica does not generate same rounding after seeding.
      ipu.utils.reset_ipu_seed(1)
      sess.run(device_op, {inp: in_data})
      res1, res2 = sess.run(dequeue_op)[0].astype(np.float32)
      self.assertNotAllEqual(res1, res2)

      # Each replica does generate same rounding after identical seeding.
      ipu.utils.reset_ipu_seed(1, experimental_identical_replicas=True)
      sess.run(device_op, {inp: in_data})
      res1, res2 = sess.run(dequeue_op)[0].astype(np.float32)
      self.assertAllEqual(res1, res2)

      # When run again, it should not generate the same rounding as last time.
      sess.run(device_op, {inp: in_data})
      res1_second, res2_second = sess.run(dequeue_op)[0].astype(np.float32)
      self.assertNotAllEqual(res1, res1_second)
      self.assertNotAllEqual(res2, res2_second)

  @tu.test_uses_ipus(num_ipus=2)
  @test_util.deprecated_graph_mode_only
  def test_experimental_identical_seeds(self):
    inp = array_ops.placeholder(np.float32, [10])

    # This should produce the same result on each IPU
    # as the stochastic rounding should be performed
    # using the same seed.
    with ipu.scopes.ipu_scope('/device:IPU:0'):
      cast1 = math_ops.cast(inp, dtype=np.float16)
      out0, = gen_popops_ops.ipu_all_gather([cast1], replication_factor=2)

    # Configure the hardware
    config = ipu.config.IPUConfig()
    config.auto_select_ipus = [2]
    config.experimental.enable_prng_stability = True
    tu.add_hw_ci_connection_options(config)
    # Enable stochastic rounding
    config.floating_point_behaviour.esr = True
    config.configure_ipu_system()

    in_data = np.array([0.1] * 10)

    with session_lib.Session() as sess:
      res = sess.run(out0, {inp: in_data}).astype(np.float32)
      # Compare the result of each IPU
      self.assertAllEqual(res[0], res[1])

  @tu.test_uses_ipus(num_ipus=4)
  @test_util.deprecated_graph_mode_only
  def test_experimental_identical_compute(self):
    config = ipu.config.IPUConfig()
    config.auto_select_ipus = [4]
    config.experimental.enable_prng_stability = True
    tu.add_hw_ci_connection_options(config)
    # Enable stochastic rounding
    config.floating_point_behaviour.esr = True
    config.configure_ipu_system()

    random_seed.set_seed(1234)
    dataset = dataset_ops.Dataset.from_tensor_slices([[[0.1, 0.2, 0.3],
                                                       [0.4, 0.5, 0.6]]])
    dataset = dataset.repeat()

    infeed = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

    noise = random_ops.random_uniform([2, 3], dtype=np.float32, seed=1)

    def my_net():
      # Perform some arbitrary simple compute that invokes SR.
      def body(noise, infeed_value):
        infeed_value = ipu.cross_replica_ops.assume_equal_across_replicas(
            infeed_value)

        noise_f16 = math_ops.cast(noise, dtype=np.float16)
        infeed_val_f16 = math_ops.cast(infeed_value, dtype=np.float16)

        result = infeed_val_f16 + noise_f16

        const = constant_op.constant(0.01,
                                     shape=result.shape,
                                     dtype=np.float32)
        const_f16 = math_ops.cast(const, dtype=np.float16)

        result = result * const_f16
        result_f32 = math_ops.cast(result, dtype=np.float32)

        out = outfeed.enqueue(result_f32)
        return (result_f32, out)

      r = ipu.loops.repeat(10, body, [noise], infeed)
      return r

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[])

    outqueue = outfeed.dequeue()
    with session_lib.Session() as sess:
      sess.run(infeed.initializer)
      sess.run(res)
      results = sess.run(outqueue)

    # This contains a value per replica for each iteration of the loop. We want to test that
    # for each iteration those values are all equal, since each replica is using the same data and
    # we've enabled experimental prng stability.
    self.assertEqual(len(results), 10)
    for replica_results in results:
      replicas_equal = [
          replica_results[0].tolist() == result.tolist()
          for result in replica_results
      ]
      self.assertTrue(all(replicas_equal))


if __name__ == "__main__":
  googletest.main()
