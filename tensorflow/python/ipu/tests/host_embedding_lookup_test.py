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
# ==============================================================================

import numpy as np
import pva

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.compiler.plugin.poplar.ops import gen_pop_datastream_ops
from tensorflow.python import ipu
from tensorflow.python.client import session as sl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import googletest
from tensorflow.python.ipu import embedding_ops
from tensorflow.python.training import training as train
from tensorflow import __version__ as version

TF1 = version.split('.')[0] == '1'


class HostEmbeddingLookupTest(test_util.TensorFlowTestCase):
  @tu.test_may_use_ipus_or_model(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def testDIENShape(self):
    shape = [10000000, 20]  # 740MB at float32
    lookup_count = 4096

    def my_net(i):

      # lookup
      out = gen_pop_datastream_ops.ipu_device_embedding_lookup(
          i,
          embedding_id="host_embedding",
          embedding_shape=shape,
          dtype=np.float32)

      #update
      gen_pop_datastream_ops.ipu_device_embedding_update_add(
          out, out, i, embedding_id="host_embedding", embedding_shape=shape)

      self.assertEqual(out.shape, (lookup_count, shape[1]))
      return out

    with ops.device('cpu'):
      i = array_ops.placeholder(np.int32, [lookup_count])
      w = variable_scope.get_variable("foo",
                                      dtype=np.float32,
                                      shape=shape,
                                      use_resource=False)

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      r = ipu.ipu_compiler.compile(my_net, inputs=[i])

    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.ipu_model.compile_ipu_code = False
    if tu.has_ci_ipus():
      tu.add_hw_ci_connection_options(cfg)
    else:
      report_helper = tu.ReportHelper()
      report_helper.set_autoreport_options(cfg)
    cfg.configure_ipu_system()
    with sl.Session() as sess:
      i_h = np.arange(0, lookup_count).reshape([lookup_count])

      sess.run(variables.global_variables_initializer())
      sess.run(
          gen_pop_datastream_ops.ipu_host_embedding_register(
              w, "host_embedding"))
      result = sess.run([r], {i: i_h})
      v = sess.run(
          gen_pop_datastream_ops.ipu_host_embedding_deregister(
              w, "host_embedding"))

      # Since we updated with the same activations, we expect to see a 2x
      self.assertAllClose(result[0][0] * 2, np.take(v, i_h, axis=0))
      self.assertEqual(result[0][0].shape, (lookup_count, shape[1]))

    if not tu.has_ci_ipus():
      report = pva.openReport(report_helper.find_report())
      self.assert_max_tile_memory(report, 46124, tolerance=0.3)

  @tu.test_may_use_ipus_or_model(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def testAGIShape(self):
    shape = [100000, 200]
    lookup_count = 4096

    def my_net(i):
      # lookup
      out = gen_pop_datastream_ops.ipu_device_embedding_lookup(
          i,
          embedding_id="host_embedding",
          embedding_shape=shape,
          dtype=np.float32)

      #update
      gen_pop_datastream_ops.ipu_device_embedding_update_add(
          out, out, i, embedding_id="host_embedding", embedding_shape=shape)

      self.assertEqual(out.shape, (lookup_count, shape[1]))
      return out

    with ops.device('cpu'):
      i = array_ops.placeholder(np.int32, [lookup_count])
      w = variable_scope.get_variable("foo",
                                      dtype=np.float32,
                                      shape=shape,
                                      use_resource=False)

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      r = ipu.ipu_compiler.compile(my_net, inputs=[i])

    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.ipu_model.compile_ipu_code = False
    if tu.has_ci_ipus():
      tu.add_hw_ci_connection_options(cfg)
    else:
      report_helper = tu.ReportHelper()
      report_helper.set_autoreport_options(cfg)

    cfg.configure_ipu_system()

    with sl.Session() as sess:
      i_h = np.arange(0, lookup_count).reshape([lookup_count])

      sess.run(variables.global_variables_initializer())
      sess.run(
          gen_pop_datastream_ops.ipu_host_embedding_register(
              w, "host_embedding"))
      result = sess.run([r], {i: i_h})
      v = sess.run(
          gen_pop_datastream_ops.ipu_host_embedding_deregister(
              w, "host_embedding"))

      # Since we updated with the same activations, we expect to see a 2x
      self.assertAllClose(result[0][0] * 2, np.take(v, i_h, axis=0))
      self.assertEqual(result[0][0].shape, (lookup_count, shape[1]))

    if not tu.has_ci_ipus():
      report = pva.openReport(report_helper.find_report())
      self.assert_max_tile_memory(report, 440684, tolerance=0.3)

  @tu.test_may_use_ipus_or_model(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def testTrainNoExec(self):
    shape = [100000, 200]
    lookup_count = 4096

    host_embedding = embedding_ops.create_host_embedding(
        "my_host_embedding",
        shape,
        np.float32,
        optimizer_spec=embedding_ops.HostEmbeddingOptimizerSpec(0.5))

    def my_net(i):
      out = host_embedding.lookup(i)

      return out

    with ops.device('cpu'):
      i = array_ops.placeholder(np.int32, [lookup_count])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      r = ipu.ipu_compiler.compile(my_net, inputs=[i])

    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.ipu_model.compile_ipu_code = False
    if tu.has_ci_ipus():
      tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()
    with sl.Session() as sess:
      i_h = np.arange(0, lookup_count).reshape([lookup_count])

      sess.run(variables.global_variables_initializer())

      with host_embedding.register(sess):
        # training=False should ignore the number of expected updates.
        result = sess.run([r], {i: i_h})

      v = sess.run(host_embedding.get_embedding_tensor())
      # Check the lookup result, but we are really interested that it doesn't hang.
      self.assertAllClose(result[0][0], np.take(v, i_h, axis=0))

  @tu.test_may_use_ipus_or_model(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def testNoLookup(self):
    shape = [100000, 200]
    lookup_count = 4096

    host_embedding = embedding_ops.create_host_embedding(
        "my_host_embedding",
        shape,
        np.float32,
        optimizer_spec=embedding_ops.HostEmbeddingOptimizerSpec(0.5))

    def my_net(i):
      return i

    with ops.device('cpu'):
      i = array_ops.placeholder(np.int32, [lookup_count])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      r = ipu.ipu_compiler.compile(my_net, inputs=[i])

    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.ipu_model.compile_ipu_code = False
    if tu.has_ci_ipus():
      tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()
    with sl.Session() as sess:
      i_h = np.arange(0, lookup_count).reshape([lookup_count])

      sess.run(variables.global_variables_initializer())

      with host_embedding.register(sess):
        result = sess.run([r], {i: i_h})

      # Check the indices are correct, but the real test is no timeout.
      self.assertAllClose(result[0][0], i_h)

  @tu.test_may_use_ipus_or_model(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def testNoTrain(self):
    shape = [4, 1024]
    lookup_count = 2

    # Training with the host embedding optimizer_spec
    # set to None *or* with learning rate 0.0 should
    # disable updates on the host embeddings.
    optimizer_spec = embedding_ops.HostEmbeddingOptimizerSpec(0.0)
    if TF1:
      initializer = init_ops.ones_initializer()
    else:
      initializer = array_ops.ones(shape, np.float32)
    host_embedding = embedding_ops.create_host_embedding(
        "my_host_embedding",
        shape,
        np.float32,
        initializer=initializer,
        optimizer_spec=optimizer_spec)

    def my_net(i):
      lookup = host_embedding.lookup(i)
      weight = variable_scope.get_variable(
          "w",
          shape=(shape[1], 1),
          dtype=np.float32,
          initializer=init_ops.zeros_initializer(),
          trainable=True,
      )
      activations = math_ops.matmul(lookup, weight)
      loss = math_ops.reduce_mean(
          math_ops.square(activations - np.ones(i.shape[0])))
      optimizer = train.GradientDescentOptimizer(learning_rate=0.0001)
      grads_and_vars = optimizer.compute_gradients(loss)
      train_op = optimizer.apply_gradients(grads_and_vars)
      with ops.control_dependencies([train_op]):
        return loss

    with ops.device('cpu'):
      i = array_ops.placeholder(np.int32, [lookup_count])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      r = ipu.ipu_compiler.compile(my_net, inputs=[i])

    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.ipu_model.compile_ipu_code = False
    if tu.has_ci_ipus():
      tu.add_hw_ci_connection_options(cfg)
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg, output_execution_profile=True)

    cfg.configure_ipu_system()
    with sl.Session() as sess:

      i_h = np.arange(1, 1 + lookup_count).reshape([lookup_count])

      sess.run(variables.global_variables_initializer())

      with host_embedding.register(sess):

        v = sess.run(host_embedding.get_embedding_tensor())
        embeddings_before = np.take(v, i_h, axis=0)

        result = sess.run([r], {i: i_h})
        loss = result[0][0]
        v = sess.run(host_embedding.get_embedding_tensor())
        embeddings_after = np.take(v, i_h, axis=0)

        self.assertEqual(loss, 1.0)
        self.assertAllEqual(embeddings_before, embeddings_after)

        result = sess.run([r], {i: i_h})
        loss = result[0][0]

        v = sess.run(host_embedding.get_embedding_tensor())
        embeddings_after = np.take(v, i_h, axis=0)

        reports = report_helper.find_reports()
        r = pva.openReport(reports[-1])

        # Sum stream copy bytes back to the host.
        run_streamcopy_out = None
        if r.execution.runs:
          run = r.execution.runs[0]
          run_streamcopy_out = 0
          for step in run.steps:
            if step.program.type == pva.Program.Type.StreamCopyMid:
              run_streamcopy_out += sum([ipu.dataOut for ipu in step.ipus])

        # When not training,
        #  - final loss should change since the matrix is still trainable
        #  - embeddings should remain unchanged since this are not trained
        #  - IpuDeviceEmbeddingLookupTrainable should not be used
        #  - Streamcopy out bytes should NOT include lookup grads
        self.assertNotEqual(loss, 1.0)
        self.assertAllEqual(embeddings_before, embeddings_after)
        if run_streamcopy_out is not None:
          approx_lookup_grads = lookup_count * shape[-1] * 4
          self.assertTrue(run_streamcopy_out < approx_lookup_grads)
        self.assert_compute_sets_not_in_blacklist(
            r, ["IpuDeviceEmbeddingLookupTrainable"])


if __name__ == "__main__":
  googletest.main()
