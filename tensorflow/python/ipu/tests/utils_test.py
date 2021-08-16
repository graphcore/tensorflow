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

import json
import os
import tempfile
import numpy as np

from tensorflow.compiler.plugin.poplar.driver.config_pb2 import IpuOptions, IpuDeviceConnectionType, IpuExecutionProfileType
from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.compiler.plugin.poplar.driver import threestate_pb2
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.python import ipu
from tensorflow.python.client import session as sl
from tensorflow.python.framework import test_util
from tensorflow.python.framework import ops
from tensorflow.python.keras import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent
from tensorflow.compat.v1 import disable_v2_behavior

disable_v2_behavior()


def count_compile_end_events(events):
  fn = (lambda x: 1 if x.type == IpuTraceEvent.COMPILE_END and len(
      x.compile_end.tensor_map) > 10 else 0)
  return sum(map(fn, events))


class ContribIpuOpsTest(test_util.TensorFlowTestCase):
  @test_util.deprecated_graph_mode_only
  def testSummary(self):
    with ops.device("/device:IPU:0"):
      a = array_ops.placeholder(np.float32, [1], name="a")
      b = array_ops.placeholder(np.float32, [1], name="b")
      out = a + b

    summary = ipu.summary_ops.ipu_compile_summary('comp', [out])

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    with sl.Session() as sess:
      fd = {
          a: [1.0],
          b: [2.0],
      }
      result, s = sess.run([out, summary], fd)
      self.assertAllClose(result, [3.0])
      self.assertTrue(len(s) > 100)

  @test_util.deprecated_graph_mode_only
  def testBypassUtilsCreator(self):
    cfg = IpuOptions()
    with self.assertRaisesRegex(Exception,
                                "Badly initialized IpuOptions object"):
      ipu.utils.configure_ipu_system(cfg)

  @test_util.deprecated_graph_mode_only
  def testCreateConfig(self):
    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.auto_select_ipus(cfg, [1, 1])
    self.assertTrue(isinstance(cfg, IpuOptions))
    self.assertEqual(len(cfg.device_config), 2)

    cfg = ipu.utils.set_floating_point_behaviour_options(cfg)

    self.assertFalse(cfg.enable_matmul_combiner)
    self.assertEqual(cfg.remote_buffer_merging_mode,
                     threestate_pb2.THREESTATE_UNDEFINED)
    cfg = ipu.utils.set_optimization_options(
        cfg,
        combine_matmuls=True,
        merge_remote_buffers=ipu.utils.MergeRemoteBuffersBehaviour.MERGE)
    self.assertTrue(cfg.enable_matmul_combiner)
    self.assertEqual(cfg.remote_buffer_merging_mode,
                     threestate_pb2.THREESTATE_ON)

    cfg = ipu.utils.set_optimization_options(
        cfg, triangular_solve_expander_block_size=42)
    self.assertEqual(cfg.triangular_solve_expander_block_size, 42)

    cfg = ipu.utils.set_optimization_options(cfg, cholesky_block_size=42)
    self.assertEqual(cfg.cholesky_block_size, 42)

    self.assertFalse(cfg.use_stable_norm_statistics)
    cfg = ipu.utils.set_norm_options(cfg, use_stable_statistics=True)
    self.assertTrue(cfg.use_stable_norm_statistics)

    self.assertFalse(cfg.convolution_options)
    cfg = ipu.utils.set_convolution_options(cfg,
                                            {"tempMemoryBudget": "1000000"})
    self.assertTrue(cfg.convolution_options)

    self.assertFalse(cfg.clear_matmul_pass_type)
    cfg = ipu.utils.set_matmul_options(cfg, clear_pass_type=True)
    self.assertTrue(cfg.clear_matmul_pass_type)

    self.assertFalse(cfg.pooling_options)
    cfg = ipu.utils.set_pooling_options(
        cfg, pooling_options={"poolUseIntrospectiveMapping": "false"})
    self.assertTrue(cfg.pooling_options)

    self.assertFalse(cfg.profiling.graph_options)
    cfg = ipu.utils.set_report_options(
        cfg,
        graph_options={"reportOption1": "false"},
        execution_options={"reportOption1": "false"})
    self.assertTrue(cfg.profiling.graph_options)

    self.assertFalse(cfg.speed_size_config.allow_recompute)
    cfg = ipu.utils.set_recomputation_options(cfg)
    self.assertTrue(cfg.speed_size_config.allow_recompute)

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.auto_select_ipus(cfg, [4, 4])
    self.assertTrue(isinstance(cfg, IpuOptions))
    self.assertTrue(len(cfg.device_config), 2)
    self.assertEqual(cfg.device_config[0].auto_count, 4)
    self.assertEqual(cfg.device_config[1].auto_count, 4)

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.auto_select_ipus(cfg, [4, 4])
    self.assertTrue(isinstance(cfg, IpuOptions))
    self.assertEqual(len(cfg.device_config), 2)
    self.assertEqual(cfg.device_config[0].auto_count, 4)
    self.assertEqual(cfg.device_config[1].auto_count, 4)

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.select_ipus(cfg, [2, 3])
    self.assertTrue(isinstance(cfg, IpuOptions))
    self.assertEqual(len(cfg.device_config), 2)
    self.assertEqual(cfg.device_config[0].cfg_index, 2)
    self.assertEqual(cfg.device_config[1].cfg_index, 3)

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_compilation_options(cfg, {'A': 'B', 'C': 'D'})
    self.assertEqual(len(cfg.compilation_options), 2)
    self.assertEqual(cfg.compilation_options[0].option, "A")
    self.assertEqual(cfg.compilation_options[0].value, "B")
    self.assertEqual(cfg.compilation_options[1].option, "C")
    self.assertEqual(cfg.compilation_options[1].value, "D")

    cfg = ipu.utils.create_ipu_config()
    folder_name = "/tmp/my_folder"
    cfg = ipu.utils.set_serialization_options(cfg, folder_name)
    self.assertEqual(cfg.serialization_folder, folder_name)

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_connection_type(
        cfg, ipu.utils.DeviceConnectionType.NEVER, ipu_version="ipu1")
    self.assertEqual(cfg.device_connection_type, IpuDeviceConnectionType.NEVER)
    self.assertEqual(cfg.ipu_version, "ipu1")
    self.assertFalse(cfg.enable_remote_buffers_without_device)

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_connection_type(cfg, enable_remote_buffers=True)
    self.assertTrue(cfg.enable_remote_buffers_without_device)

    with self.assertRaises(Exception):
      cfg = ipu.utils.create_ipu_config()
      cfg = ipu.utils.set_ipu_connection_type(
          cfg, ipu.utils.DeviceConnectionType.NEVER)

    with self.assertRaises(Exception):
      cfg = ipu.utils.create_ipu_config()
      cfg = ipu.utils.select_ipus(cfg, [4, 4])

    with self.assertRaises(Exception):
      cfg = ipu.utils.create_ipu_config(profiling=True, enable_ipu_events=True)

    with self.assertRaises(Exception):
      cfg = ipu.utils.create_ipu_config(profiling=False,
                                        profile_execution=True)

    cfg = ipu.utils.create_ipu_config(profiling=True, profile_execution=True)
    self.assertEqual(cfg.profiling.execution_trace_type,
                     IpuExecutionProfileType.DEVICE_PROFILE)

    cfg = ipu.utils.create_ipu_config(profiling=True, profile_execution=False)
    self.assertEqual(cfg.profiling.execution_trace_type,
                     IpuExecutionProfileType.NO_PROFILE)

    cfg = ipu.utils.create_ipu_config(
        profiling=True,
        profile_execution=ipu.utils.ExecutionProfileType.IPU_PROFILE)
    self.assertEqual(cfg.profiling.execution_trace_type,
                     IpuExecutionProfileType.IPU_PROFILE)

    with self.assertRaises(Exception):
      cfg = ipu.utils.create_ipu_config(profiling=True,
                                        profile_execution="IPU")

  @test_util.deprecated_graph_mode_only
  def testEventFetchAndStringDecode(self):
    with ops.device("/device:IPU:0"):
      a = array_ops.placeholder(np.float32, [1], name="a")
      b = array_ops.placeholder(np.float32, [1], name="b")
      out = a + b

    events = gen_ipu_ops.ipu_event_trace()

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    with sl.Session() as sess:
      # Discard any existing events
      sess.run(events)

      fd = {
          a: [1.0],
          b: [2.0],
      }
      result = sess.run(out, fd)
      self.assertAllClose(result, [3.0])

      # 1x compile begin, 1x compile end, 1x load engine, 1x execute
      e = sess.run(events)
      self.assertEqual(len(e), 4)

      dump = ipu.utils.extract_all_strings_from_event_trace(e)
      self.assertTrue(len(dump) > 100)

  @test_util.deprecated_graph_mode_only
  def testIpuWhileScope(self):
    # 1: design is targetted at the device
    # 2: variables are resource variables
    # 3: training a while_loop is possible
    def my_net(a, b):
      c = variable_scope.get_variable('c', initializer=[1.0])
      self.assertTrue("ResourceVariable" in str(type(c)))

      lstm_cell = rnn_cell.LSTMCell(1, forget_bias=1.0)
      outputs, _ = rnn.dynamic_rnn(lstm_cell, a, dtype=np.float32)

      logits = outputs[-1] * c
      self.assertEqual(logits.device, "/device:IPU:0")

      res = array_ops.reshape(logits, [1, 8, 1])

      l = losses.mean_squared_error(res, b)

      optimizer = gradient_descent.GradientDescentOptimizer(0.1)
      train = optimizer.minimize(l)

      return [l, train]

    with ops.device('cpu'):
      a = array_ops.placeholder(np.float32, [1, 8, 1], name="a")
      b = array_ops.placeholder(np.float32, [1, 8, 1], name="b")

    with ipu.scopes.ipu_scope("/device:IPU:0"):

      l = ipu.ipu_compiler.compile(my_net, inputs=[a, b])

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    with sl.Session() as sess:
      # Initialize and then discard events relating to initialization
      sess.run(variables.global_variables_initializer())

      fd = {
          a: [[[1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.]]],
          b: [[[1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.]]],
      }

      l_initial = sess.run([l], fd)

      for _ in range(100):
        _ = sess.run([l], fd)

      l_final = sess.run([l], fd)

      self.assertTrue(l_initial > l_final)

  @test_util.deprecated_graph_mode_only
  def testInitializerDeviceChange(self):

    inp = array_ops.placeholder(np.float32, [1, 8, 8, 4])
    with ipu.scopes.ipu_scope("/device:IPU:0"):
      layers.Conv2D(8, 1)(inp)

    events = gen_ipu_ops.ipu_event_trace()

    ipu.utils.move_variable_initialization_to_cpu()

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    with sl.Session() as sess:
      # Discard any pending events
      sess.run(events)

      # Run initializer (should be on CPU)
      sess.run(variables.global_variables_initializer())

      e = sess.run(events)
      self.assertEqual(len(e), 0)  # no begin/end, no load/execute

  @test_util.deprecated_graph_mode_only
  def testInitializerThatUsesIPUResourceDeviceChange(self):
    var_val = np.ones([1, 8, 8, 4])

    def graph():
      var = variables.Variable(var_val, dtype=np.float32)
      # Initialize a variable with an initializer graph that contains a read of
      # another variable resource.
      var_copy = variables.Variable(var.initialized_value())

      # Make an IPU op that also uses the variable resource
      x = var * var_copy
      return x

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      # Cluster the IPU ops together
      compile_op = ipu.ipu_compiler.compile(graph)

    events = gen_ipu_ops.ipu_event_trace()

    # Any variable resources should not be pushed
    # to CPU by move_variable_initialization_to_cpu otherwise colocation will be
    # violated when an IPU op uses that resource too
    ipu.utils.move_variable_initialization_to_cpu()

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    with sl.Session() as sess:
      # Discard any pending events
      sess.run(events)

      # Run initializer (should be on CPU)
      sess.run(variables.global_variables_initializer())

      # Run the graph (should be on IPU)
      sess.run(compile_op)

      e = sess.run(events)
      evts = ipu.utils.extract_all_events(e)
      # Should only be 1 compile event - initialization should be on CPU
      self.assertEqual(count_compile_end_events(evts), 1)

  @test_util.deprecated_graph_mode_only
  def testVarsInitializedByStreamsAreLoggedAsOnDevice(self):
    # This verifies that when an initialization graph has no ops in it (it is
    # a pass through of streaming inputs to initialized resources) then the
    # resources are logged as resources on the device so that a future execution
    # sees them as valid and on device
    w_val1 = np.array([1, 2, 3, 4])
    w_val2 = np.array([4, 3, 2, 1])
    w_val3 = np.array([9, 0, 9, 0])
    with ops.device("/device:IPU:1"):
      with variable_scope.variable_scope("vs", use_resource=True):
        w1 = variable_scope.get_variable(
            "w1",
            shape=[4],
            dtype=np.float32,
            initializer=init_ops.constant_initializer(w_val1,
                                                      dtype=np.float32))
        w2 = variable_scope.get_variable(
            "w2",
            shape=[4],
            dtype=np.float32,
            initializer=init_ops.constant_initializer(w_val2,
                                                      dtype=np.float32))
        w3 = variable_scope.get_variable(
            "w3",
            shape=[4],
            dtype=np.float32,
            initializer=init_ops.constant_initializer(w_val3,
                                                      dtype=np.float32))

      y = w1 + w2 + w3

    ipu.utils.move_variable_initialization_to_cpu()

    cfg = ipu.utils.create_ipu_config(profiling=False)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    cfg = ipu.utils.auto_select_ipus(cfg, 2)
    ipu.utils.configure_ipu_system(cfg)

    with sl.Session() as sess:
      sess.run(variables.global_variables_initializer())

      xs = [
          np.array([7, 3, 5, 9], dtype=np.float32),
          np.array([1, 8, 3, 4], dtype=np.float32),
          np.array([9, 2, 2, 6], dtype=np.float32)
      ]
      for _ in xs:
        out = sess.run(y)
        self.assertAllClose(out, w_val1 + w_val2 + w_val3)

  @test_util.deprecated_graph_mode_only
  def testMultiScopeTest(self):
    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, [2, 2])
      y = array_ops.placeholder(np.float32, [2, 2])
      report = gen_ipu_ops.ipu_event_trace()

    with ipu.scopes.ipu_scope('/device:IPU:0'):
      z = math_ops.matmul(x, y)
    with ipu.scopes.ipu_scope('/device:IPU:0'):
      z2 = math_ops.matmul(x, z)

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    with sl.Session() as sess:
      sess.run(report)
      result = sess.run(z2, {x: np.ones([2, 2]), y: np.ones([2, 2])})

      self.assertAllEqual(result, [[4, 4], [4, 4]])

      rep = sess.run(report)
      evts = ipu.utils.extract_all_types_from_event_trace(rep)

      num_compiles = 0
      num_executions = 0
      for e in evts:
        if e == IpuTraceEvent.COMPILE_END:
          num_compiles += 1
        if e == IpuTraceEvent.EXECUTE:
          num_executions += 1

      self.assertEqual(num_compiles, 1)
      self.assertEqual(num_executions, 1)

  @test_util.deprecated_graph_mode_only
  def testResetSeedTest(self):
    # This tests that the API can be called - full testing must be performed
    # on hardware because the IPU_MODEL doesn't have full random number support.
    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, [2, 2])

    with ipu.scopes.ipu_scope('/device:IPU:0'):
      z = math_ops.cast(x, dtype=np.float16)

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    with sl.Session() as sess:
      result = sess.run(z, {x: [[1., 1.], [1., 1.]]})
      self.assertAllEqual(result, [[1., 1.], [1., 1.]])

      ipu.utils.reset_ipu_seed(1)

      result = sess.run(z, {x: [[2., 2.], [2., 2.]]})
      self.assertAllEqual(result, [[2., 2.], [2., 2.]])

  @test_util.deprecated_graph_mode_only
  def testCreateConfig1(self):
    cfg = ipu.utils.create_ipu_config()
    self.assertFalse(cfg.profiling.graph_options)
    self.assertFalse(cfg.profiling.execution_options)
    cfg = ipu.utils.set_report_options(
        cfg, graph_options={"reportOption1": "false"})
    self.assertTrue(cfg.profiling.graph_options)
    self.assertFalse(cfg.profiling.execution_options)

  @test_util.deprecated_graph_mode_only
  def testCreateConfig2(self):
    cfg = ipu.utils.create_ipu_config()
    self.assertFalse(cfg.profiling.graph_options)
    self.assertFalse(cfg.profiling.execution_options)
    cfg = ipu.utils.set_report_options(
        cfg, execution_options={"reportOption2": "false"})
    self.assertFalse(cfg.profiling.graph_options)
    self.assertTrue(cfg.profiling.execution_options)

  @test_util.deprecated_graph_mode_only
  def testCreateConfig3(self):
    cfg = ipu.utils.create_ipu_config()
    self.assertFalse(cfg.profiling.graph_options)
    self.assertFalse(cfg.profiling.execution_options)
    cfg = ipu.utils.set_report_options(
        cfg,
        graph_options={"reportOption3a": "false"},
        execution_options={"reportOption3b": "false"})
    self.assertTrue(cfg.profiling.graph_options)
    self.assertTrue(cfg.profiling.execution_options)

  @test_util.deprecated_graph_mode_only
  def testCreateConfig4(self):
    cfg = ipu.utils.create_ipu_config()
    self.assertFalse(cfg.profiling.graph_options)
    self.assertFalse(cfg.profiling.execution_options)
    cfg = ipu.utils.set_report_options(cfg)
    self.assertFalse(cfg.profiling.graph_options)
    self.assertFalse(cfg.profiling.execution_options)

  @test_util.deprecated_graph_mode_only
  def testGclOptions(self):
    cfg = ipu.utils.create_ipu_config()
    self.assertEqual(len(cfg.gcl_options), 0)

    with self.assertRaisesRegex(TypeError,
                                "`gcl_options` must be a dictionary"):
      ipu.utils.set_gcl_options(cfg, gcl_options=123)

    cfg = ipu.utils.set_gcl_options(cfg,
                                    gcl_options={"maxBytesPerTile": "128"})

    self.assertEqual(len(cfg.gcl_options), 1)
    self.assertEqual(cfg.gcl_options[0].option, "maxBytesPerTile")
    self.assertEqual(cfg.gcl_options[0].value, "128")

  @test_util.deprecated_graph_mode_only
  def testIoTileOptions(self):
    cfg = ipu.utils.create_ipu_config()
    self.assertEqual(cfg.num_io_tiles, 0)

    cfg = ipu.utils.set_io_tile_options(cfg, num_io_tiles=32)

    self.assertEqual(cfg.num_io_tiles, 32)


if __name__ == "__main__":
  googletest.main()
