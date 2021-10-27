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
# =============================================================================

import os
import numpy as np
import pva

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.platform import googletest
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python import ipu


class FifoTest(test_util.TensorFlowTestCase):
  @tu.skip_on_hw
  @test_util.deprecated_graph_mode_only
  def testFifoCompleteLoop(self):
    def my_net(x):
      body = lambda z: ipu.internal_ops.fifo(z, 5)
      return ipu.loops.repeat(6, body, [x])

    with ipu.scopes.ipu_scope('/device:IPU:0'):
      x = array_ops.placeholder(np.float32, shape=[2])
      run_loop = ipu.ipu_compiler.compile(my_net, inputs=[x])

    config = ipu.config.IPUConfig()
    config.auto_select_ipus = 1
    tu.add_hw_ci_connection_options(config)
    config.configure_ipu_system()

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())
      res = sess.run(run_loop, {x: np.ones([2])})
      self.assertAllClose(res, np.ones([1, 2]))

  @tu.skip_on_hw
  @test_util.deprecated_graph_mode_only
  def testFifo(self):
    def my_net(x):
      body = lambda z: ipu.internal_ops.fifo(z, 5)
      return ipu.loops.repeat(3, body, [x])

    with ipu.scopes.ipu_scope('/device:IPU:0'):
      x = array_ops.placeholder(np.float32, shape=[2])
      run_loop = ipu.ipu_compiler.compile(my_net, inputs=[x])

    config = ipu.config.IPUConfig()
    config.auto_select_ipus = 1
    tu.add_hw_ci_connection_options(config)
    config.configure_ipu_system()

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())
      res = sess.run(run_loop, {x: np.ones([2])})
      self.assertAllClose(res, np.zeros([1, 2]))

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def testOffloaded(self):
    def dataset_fn():
      return tu.create_single_increasing_dataset(10, shape=[1])

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset_fn())
    outfeed_queue1 = ipu.ipu_outfeed_queue.IPUOutfeedQueue()
    outfeed_queue2 = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

    def body(x):
      x1 = ipu.internal_ops.fifo(x, 5, offload=True)
      x2 = ipu.internal_ops.fifo(x, 1, offload=True)
      outfeed1 = outfeed_queue1.enqueue(x1)
      outfeed2 = outfeed_queue2.enqueue(x2)
      return outfeed1, outfeed2

    def my_net():
      r = ipu.loops.repeat(10, body, [], infeed_queue)
      return r

    with ipu.scopes.ipu_scope('/device:IPU:0'):
      run_loop = ipu.ipu_compiler.compile(my_net, inputs=[])

    dequeue_outfeed1 = outfeed_queue1.dequeue()
    dequeue_outfeed2 = outfeed_queue2.dequeue()

    # Configure the hardware
    config = ipu.config.IPUConfig()
    config.auto_select_ipus = 1
    tu.add_hw_ci_connection_options(config)
    config.configure_ipu_system()

    with tu.ipu_session() as sess:
      sess.run(infeed_queue.initializer)
      sess.run(run_loop)
      result1 = sess.run(dequeue_outfeed1)
      result2 = sess.run(dequeue_outfeed2)

      self.assertEqual(len(result1), 10)
      self.assertEqual(len(result2), 10)

      self.assertAllClose(result1[5:], [[x] for x in range(5)])
      self.assertAllClose(result2[1:], [[x] for x in range(9)])

  @tu.skip_on_hw
  @test_util.deprecated_graph_mode_only
  def testFifoMultiDevice(self):
    def my_net(x):
      def body(z):
        with ipu.scopes.ipu_shard(-1):
          return ipu.internal_ops.fifo(z, 5)

      return ipu.loops.repeat(3, body, [x])

    with ipu.scopes.ipu_scope('/device:IPU:0'):
      x = array_ops.placeholder(np.float32, shape=[1024])
      run_loop = ipu.ipu_compiler.compile(my_net, inputs=[x])

    config = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(config, output_execution_profile=True)
    tu.enable_ipu_events(config)

    config.auto_select_ipus = 2
    config.ipu_model.tiles_per_ipu = 16
    tu.add_hw_ci_connection_options(config)
    config.configure_ipu_system()

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())
      report_helper.clear_reports()
      res = sess.run(run_loop, {x: np.ones([1024])})
      self.assertAllClose(res, np.zeros([1, 1024]))

    report = pva.openReport(report_helper.find_report())

    # There should be no exchange between the IPUs.
    self.assert_global_exchange_percentage(report, 0)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
