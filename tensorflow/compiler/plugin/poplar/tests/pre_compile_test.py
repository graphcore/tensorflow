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

import contextlib
import multiprocessing
import os
import tempfile
import numpy as np
from tensorflow.python.ipu import test_utils as tu

from tensorflow.compiler.tests import xla_test
from tensorflow.python import ipu
from tensorflow.python import ops
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.estimator import model_fn
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import metrics_impl
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test

from tensorflow.compiler.plugin.poplar.ops import gen_pop_datastream_ops


def _options_function(opts):
  opts.ipu_model.compile_ipu_code = False
  opts.device_connection.version = 'ipu1'
  opts.device_connection.type = ipu.utils.DeviceConnectionType.PRE_COMPILE


@contextlib.contextmanager
def _temporary_executable_cache():
  with tempfile.TemporaryDirectory() as temp_dir:
    # Use a nonexistent subdirectory that must be created
    cache_dir = os.path.join(temp_dir, "cache")
    poplar_flags = "--executable_cache_path={} {}".format(
        cache_dir, os.environ.get("TF_POPLAR_FLAGS", ""))
    # Disable the IPU model
    poplar_flags = poplar_flags.replace("--use_ipu_model", "")
    with test.mock.patch.dict("os.environ", {"TF_POPLAR_FLAGS": poplar_flags}):
      yield


class TestPreCompileMode(xla_test.XLATestCase):  # pylint: disable=abstract-method
  def _run_in_new_process(self, fn):
    q = multiprocessing.Queue()

    def process_fn():
      try:
        q.put(fn())
      except:
        # To avoid dead lock on q.get()
        q.put(None)
        raise

    p = multiprocessing.Process(target=process_fn)
    p.start()
    # Get from queue before joining to avoid deadlock if the queue is full and
    # blocks the producer.
    ret = q.get()
    p.join()
    self.assertEqual(p.exitcode, 0)
    return ret

  @test_util.deprecated_graph_mode_only
  def test_basic_model(self):
    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    _options_function(cfg)

    def build_and_run_model():
      cfg.configure_ipu_system()

      def my_net(x):
        return x * x

      v = array_ops.placeholder(dtype=np.float32, shape=(2,))
      with ipu.scopes.ipu_scope("/device:IPU:0"):
        [result] = ipu.ipu_compiler.compile(my_net, inputs=[v])

      with session.Session() as sess:
        res = sess.run(result, {v: [1.0, 2.0]})
        num_reports = len(report_helper.find_reports())
        report_helper.clear_reports()
        return res, num_reports

    with _temporary_executable_cache():
      result0, num_reports_0 = self._run_in_new_process(build_and_run_model)
      result1, num_reports_1 = self._run_in_new_process(build_and_run_model)
      self.assertAllClose(result0, result1)
      self.assertAllClose(result1, [0.0, 0.0])
      self.assertEqual(1, num_reports_0)
      self.assertEqual(0, num_reports_1)

  @test_util.deprecated_graph_mode_only
  def test_model_with_infeed_and_outfeed(self):
    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    _options_function(cfg)

    def build_and_run_model():
      cfg.configure_ipu_system()

      dataset = dataset_ops.Dataset.from_tensor_slices(
          np.ones(10, dtype=np.float32))
      infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
      outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

      def body(v, x):
        v = v + x
        outfed = outfeed_queue.enqueue(v)
        return v, outfed

      def my_net(v):
        return ipu.loops.repeat(10, body, v, infeed_queue)

      v = array_ops.placeholder(np.float32, shape=())
      with ipu.scopes.ipu_scope("/device:IPU:0"):
        result = ipu.ipu_compiler.compile(my_net, inputs=[v])

      dequeued = outfeed_queue.dequeue()

      report_helper.clear_reports()
      with session.Session() as sess:
        sess.run(infeed_queue.initializer)
        res = sess.run(result, {v: 0.0})
        deq = sess.run(dequeued)
        num_reports = len(report_helper.find_reports())
        return res, deq, num_reports

    with _temporary_executable_cache():
      result0, dequeued0, num_reports_0 = self._run_in_new_process(
          build_and_run_model)
      result1, dequeued1, num_reports_1 = self._run_in_new_process(
          build_and_run_model)
      self.assertEqual(dequeued0.shape, dequeued1.shape)
      self.assertEqual(result0, result1)
      self.assertEqual(1, num_reports_0)
      self.assertEqual(0, num_reports_1)

  @test_util.deprecated_graph_mode_only
  def test_new_graph_in_same_process(self):
    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    _options_function(cfg)

    def build_and_run_model():
      cfg.configure_ipu_system()

      def my_net(x):
        return x * x

      v = array_ops.placeholder(dtype=np.float32, shape=(2,))
      with ipu.scopes.ipu_scope("/device:IPU:0"):
        [result] = ipu.ipu_compiler.compile(my_net, inputs=[v])

      report_helper.clear_reports()
      with session.Session() as sess:
        res = sess.run(result, {v: [1.0, 2.0]})
        num_reports = len(report_helper.find_reports())
        return res, num_reports

    with _temporary_executable_cache():
      # Since each Graph will have its own XLA compilation cache,
      # the cache we test is the last-level Poplar executable cache.

      with ops.Graph().as_default():
        result0, num_reports_0 = build_and_run_model()

      with ops.Graph().as_default():
        result1, num_reports_1 = build_and_run_model()

      self.assertAllEqual(result0, result1)
      self.assertEqual(1, num_reports_0)
      self.assertEqual(0, num_reports_1)

  def test_ipu_estimator(self):
    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.auto_select_ipus = 1
    _options_function(cfg)

    def my_model_fn(features, labels, mode):
      loss = features + labels
      # Make different graphs for train and eval
      if mode == model_fn.ModeKeys.TRAIN:
        train_op = array_ops.identity(loss)
        return model_fn.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
      elif mode == model_fn.ModeKeys.EVAL:
        eval_metric_ops = {"metric": metrics_impl.mean(features * labels)}
        return model_fn.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      eval_metric_ops=eval_metric_ops)
      else:
        raise NotImplementedError(mode)

    def my_input_fn():
      dataset = dataset_ops.Dataset.from_tensor_slices((
          [[0], [1]],
          [[2], [3]],
      ))
      return dataset.batch(1, drop_remainder=True)

    def build_and_run_model():
      ipu_config = ipu.ipu_run_config.IPURunConfig(iterations_per_loop=2,
                                                   ipu_options=cfg)

      run_config = ipu.ipu_run_config.RunConfig(ipu_run_config=ipu_config)
      estimator = ipu.ipu_estimator.IPUEstimator(model_fn=my_model_fn,
                                                 config=run_config)

      self.assert_num_reports(report_helper, 0)

      # Compile the training graph
      estimator.train(input_fn=my_input_fn, steps=2)
      self.assert_num_reports(report_helper, 1)

      # Re-use cached training graph
      estimator.train(input_fn=my_input_fn, steps=2)
      self.assert_num_reports(report_helper, 1)

      # Compile the evaluation graph
      estimator.evaluate(input_fn=my_input_fn, steps=2)
      self.assert_num_reports(report_helper, 2)

      # Re-use cached evaluation graph
      estimator.evaluate(input_fn=my_input_fn, steps=2)
      self.assert_num_reports(report_helper, 2)

      # Re-use cached training graph
      estimator.train(input_fn=my_input_fn, steps=2)
      self.assert_num_reports(report_helper, 2)

    with _temporary_executable_cache():
      self._run_in_new_process(build_and_run_model)

  @test_util.deprecated_graph_mode_only
  def test_unhashable_op(self):
    def build_and_run_model():
      cfg = ipu.config.IPUConfig()
      _options_function(cfg)
      cfg.configure_ipu_system()

      cwd = os.getcwd()
      outputs = {
          "output_types": [dtypes.float32],
          "output_shapes": [tensor_shape.TensorShape([128])],
      }

      lib_path = os.path.join(cwd, "tensorflow/compiler/plugin/poplar",
                              "libadd_incrementing_custom_with_metadata.so")

      def my_net(x, y):
        x = ipu.custom_ops.precompiled_user_op([x, y],
                                               lib_path,
                                               op_name="UnhashableTest",
                                               outs=outputs)
        return x

      with ipu.scopes.ipu_scope('/device:IPU:0'):
        x = array_ops.placeholder(np.float32, shape=[128])
        y = array_ops.placeholder(np.float32, shape=[128])

        result = ipu.ipu_compiler.compile(my_net, inputs=[x, y])

      with session.Session() as sess:
        excepted = False
        try:
          sess.run(result, feed_dict={x: np.ones([128]), y: np.ones([128])})
        except errors.FailedPreconditionError as e:
          excepted = "cannot be safely pre-compiled" in e.message
        return excepted

    with _temporary_executable_cache():
      # Expect no compilation due to exception.
      self.assertTrue(self._run_in_new_process(build_and_run_model))

  @tu.skip_with_asan("non-deterministic dlopen user ops addresses with asan")
  @test_util.deprecated_graph_mode_only
  def test_hashable_op(self):
    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    _options_function(cfg)

    def build_and_run_model():
      cfg.configure_ipu_system()

      cwd = os.getcwd()
      outputs = {
          "output_types": [dtypes.float32],
          "output_shapes": [tensor_shape.TensorShape([128])],
      }

      lib_path = os.path.join(cwd, "tensorflow/compiler/plugin/poplar",
                              "libadd_incrementing_custom_with_metadata.so")

      def my_net(x, y):
        x = ipu.custom_ops.precompiled_user_op([x, y],
                                               lib_path,
                                               op_name="HashableTest",
                                               outs=outputs)
        return x

      with ipu.scopes.ipu_scope('/device:IPU:0'):
        x = array_ops.placeholder(np.float32, shape=[128])
        y = array_ops.placeholder(np.float32, shape=[128])

        result = ipu.ipu_compiler.compile(my_net, inputs=[x, y])

      report_helper.clear_reports()
      with session.Session() as sess:
        sess.run(result, feed_dict={x: np.ones([128]), y: np.ones([128])})
        num_reports = len(report_helper.find_reports())
        return num_reports

    with _temporary_executable_cache():
      num_reports_0 = self._run_in_new_process(build_and_run_model)
      num_reports_1 = self._run_in_new_process(build_and_run_model)
      # Expect no second compilation as the executable should be cached
      self.assertEqual(1, num_reports_0)
      self.assertEqual(0, num_reports_1)

  @test_util.deprecated_graph_mode_only
  def test_host_embeddings(self):
    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    _options_function(cfg)

    def build_and_run_model():
      cfg.configure_ipu_system()

      shape = [100, 20]
      lookup_count = 20

      def my_net(i):
        # lookup
        out = gen_pop_datastream_ops.ipu_device_embedding_lookup(
            i,
            embedding_id="host_embedding",
            embedding_shape=shape,
            dtype=np.float32)

        # update
        gen_pop_datastream_ops.ipu_device_embedding_update_add(
            out, out, i, embedding_id="host_embedding", embedding_shape=shape)

        # notify
        gen_pop_datastream_ops.ipu_device_embedding_notify(
            embedding_id="host_embedding")

        return out

      with ops.device('cpu'):
        i = array_ops.placeholder(np.int32, [lookup_count])
        w = variable_scope.get_variable("foo",
                                        dtype=np.float32,
                                        shape=shape,
                                        use_resource=False)

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        r = ipu.ipu_compiler.compile(my_net, inputs=[i])

      with session.Session() as sess:
        i_h = np.arange(0, lookup_count).reshape([lookup_count])

        sess.run(variables.global_variables_initializer())
        report_helper.clear_reports()
        sess.run(
            gen_pop_datastream_ops.ipu_host_embedding_register(
                w, "host_embedding", optimizer="SGD+GA"))
        result = sess.run([r], {i: i_h})
        sess.run(
            gen_pop_datastream_ops.ipu_host_embedding_deregister(
                w, "host_embedding"))

        num_reports = len(report_helper.find_reports())
        return result, num_reports

    with _temporary_executable_cache():
      result0, num_reports_0 = self._run_in_new_process(build_and_run_model)
      result1, num_reports_1 = self._run_in_new_process(build_and_run_model)
      self.assertFalse(np.any(result0))
      self.assertAllEqual(result0, result1)
      # Expect no second compilation as the executable should be cached
      self.assertEqual(1, num_reports_0)
      self.assertEqual(0, num_reports_1)


if __name__ == "__main__":
  googletest.main()
