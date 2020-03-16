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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python import ipu
from tensorflow.python.client import session as sl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import googletest
from tensorflow.python.ipu import embedding_ops
from tensorflow.python.training import gradient_descent as gd

from tensorflow.compiler.plugin.poplar.ops import gen_pop_datastream_ops


class HostEmbeddingLookupTest(test_util.TensorFlowTestCase):
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
          out, i, embedding_id="host_embedding", embedding_shape=shape)

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

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)
    with sl.Session() as sess:
      i_h = np.arange(0, lookup_count).reshape([lookup_count])

      report = tu.ReportJSON(self, sess, configure_device=False)

      sess.run(variables.global_variables_initializer())
      report.reset()
      result = sess.run([
          r,
          gen_pop_datastream_ops.ipu_host_embedding(
              w, embedding_id="host_embedding")
      ], {i: i_h})

      # Since we updated with the same activations, we expect to see a 2x
      self.assertAllClose(result[0][0] * 2, np.take(result[1], i_h, axis=0))
      self.assertEqual(result[0][0].shape, (lookup_count, shape[1]))
      report.parse_log()
      report.assert_max_tile_memory(4056, tolerance=0.3)

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
          out, i, embedding_id="host_embedding", embedding_shape=shape)

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

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)
    with sl.Session() as sess:
      i_h = np.arange(0, lookup_count).reshape([lookup_count])

      report = tu.ReportJSON(self, sess, configure_device=False)
      sess.run(variables.global_variables_initializer())
      report.reset()
      result = sess.run([
          r,
          gen_pop_datastream_ops.ipu_host_embedding(
              w, embedding_id="host_embedding")
      ], {i: i_h})

      # Since we updated with the same activations, we expect to see a 2x
      self.assertAllClose(result[0][0] * 2, np.take(result[1], i_h, axis=0))
      self.assertEqual(result[0][0].shape, (lookup_count, shape[1]))
      report.parse_log()
      report.assert_max_tile_memory(9136, tolerance=0.3)

  @test_util.deprecated_graph_mode_only
  def testTrainNoExec(self):
    shape = [100000, 200]
    lookup_count = 4096

    host_embedding = embedding_ops.create_host_embedding(
        "my_host_embedding",
        shape,
        np.float32,
        optimizer_spec=embedding_ops.HostEmbeddingOptimizerSpec(0.5))

    # Inject an update that would cause a hang
    host_embedding._update_count = 1  # pylint: disable=W0212

    def my_net(i):
      out = host_embedding.lookup(i)

      return out

    with ops.device('cpu'):
      i = array_ops.placeholder(np.int32, [lookup_count])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      r = ipu.ipu_compiler.compile(my_net, inputs=[i])

    cfg = ipu.utils.create_ipu_config(profiling=True,
                                      always_rearrange_copies_on_the_host=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)
    with sl.Session() as sess:
      i_h = np.arange(0, lookup_count).reshape([lookup_count])

      report = tu.ReportJSON(self, sess, configure_device=False)
      sess.run(variables.global_variables_initializer())
      report.reset()

      # training=False should ignore the number of expected updates.
      result = sess.run(
          [r, host_embedding(iteration_count=1, training=False)], {i: i_h})

      # Check the lookup resolt, but we are really interested that it doesn't hang.
      self.assertAllClose(result[0][0], np.take(result[1], i_h, axis=0))

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

    cfg = ipu.utils.create_ipu_config(profiling=True,
                                      always_rearrange_copies_on_the_host=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)
    with sl.Session() as sess:
      i_h = np.arange(0, lookup_count).reshape([lookup_count])

      report = tu.ReportJSON(self, sess, configure_device=False)
      sess.run(variables.global_variables_initializer())
      report.reset()
      result = sess.run([r, host_embedding(iteration_count=1)], {i: i_h})

      # Check the indices are correct, but the real test is no timeout.
      self.assertAllClose(result[0][0], i_h)


if __name__ == "__main__":
  googletest.main()
