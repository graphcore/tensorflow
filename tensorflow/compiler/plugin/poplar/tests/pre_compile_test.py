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
from tensorflow.python.ipu.config import IPUConfig
import glob
import multiprocessing
import os
import tempfile
import numpy as np
import test_utils as tu

from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.compiler.plugin.poplar.tests.test_utils import ReportJSON, count_ipu_compilations
from tensorflow.compiler.tests import xla_test
from tensorflow.python import ipu
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.estimator import model_fn
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import metrics_impl
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.python.summary import summary_iterator

from tensorflow.compiler.plugin.poplar.ops import gen_pop_datastream_ops


def _options_function(opts):
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


def _count_ipu_compilations_in_summary(summary):
  count = 0
  for val in summary.value:
    if val.tag == "ipu_trace":
      count += count_ipu_compilations(val.tensor.string_val)
  return count


def _count_ipu_compilations_in_dir(model_dir):
  count = 0
  for event_file in glob.glob(os.path.join(model_dir, "event*")):
    for event in summary_iterator.summary_iterator(event_file):
      count += _count_ipu_compilations_in_summary(event.summary)
  return count


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
    def build_and_run_model():
      def my_net(x):
        return x * x

      v = array_ops.placeholder(dtype=np.float32, shape=(2,))
      with ipu.scopes.ipu_scope("/device:IPU:0"):
        [result] = ipu.ipu_compiler.compile(my_net, inputs=[v])

      with session.Session() as sess:
        report = ReportJSON(self, sess, set_opts_fn=_options_function)
        res = sess.run(result, {v: [1.0, 2.0]})
        events = report.get_event_trace(sess)
        return res, events

    with _temporary_executable_cache():
      result0, report0 = self._run_in_new_process(build_and_run_model)
      result1, report1 = self._run_in_new_process(build_and_run_model)
      self.assertAllClose(result0, result1)
      self.assertAllClose(result1, [0.0, 0.0])
      self.assertEqual(1, count_ipu_compilations(report0))
      self.assertEqual(0, count_ipu_compilations(report1))

  @test_util.deprecated_graph_mode_only
  def test_model_with_infeed_and_outfeed(self):
    def build_and_run_model():
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

      with session.Session() as sess:
        report = ReportJSON(self, sess, set_opts_fn=_options_function)
        sess.run(infeed_queue.initializer)
        res = sess.run(result, {v: 0.0})
        deq = sess.run(dequeued)
        events = report.get_event_trace(sess)
        return res, deq, events

    with _temporary_executable_cache():
      result0, dequeued0, events0 = self._run_in_new_process(
          build_and_run_model)
      result1, dequeued1, events1 = self._run_in_new_process(
          build_and_run_model)
      self.assertEqual(dequeued0.shape, dequeued1.shape)
      self.assertEqual(result0, result1)
      self.assertEqual(1, count_ipu_compilations(events0))
      self.assertEqual(0, count_ipu_compilations(events1))

  @test_util.deprecated_graph_mode_only
  def test_new_graph_in_same_process(self):
    def build_and_run_model():
      def my_net(x):
        return x * x

      v = array_ops.placeholder(dtype=np.float32, shape=(2,))
      with ipu.scopes.ipu_scope("/device:IPU:0"):
        [result] = ipu.ipu_compiler.compile(my_net, inputs=[v])

      with session.Session() as sess:
        report = ReportJSON(self, sess, set_opts_fn=_options_function)

        res = sess.run(result, {v: [1.0, 2.0]})
        events = report.get_event_trace(sess)
        return res, events

    with _temporary_executable_cache():
      # Since each Graph will have its own XLA compilation cache,
      # the cache we test is the last-level Poplar executable cache.

      with ops.Graph().as_default():
        result0, events0 = build_and_run_model()

      with ops.Graph().as_default():
        result1, events1 = build_and_run_model()

      self.assertAllEqual(result0, result1)
      self.assertEqual(1, count_ipu_compilations(events0))
      self.assertEqual(0, count_ipu_compilations(events1))

  def test_ipu_estimator(self):
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
      ipu_options = IPUConfig()
      ipu_options._profiling.profiling = True  # pylint: disable=protected-access
      ipu_options.auto_select_ipus = 1
      _options_function(ipu_options)

      ipu_config = ipu.ipu_run_config.IPURunConfig(iterations_per_loop=2,
                                                   ipu_options=ipu_options,
                                                   compile_summary=True)

      run_config = ipu.ipu_run_config.RunConfig(ipu_run_config=ipu_config)
      estimator = ipu.ipu_estimator.IPUEstimator(model_fn=my_model_fn,
                                                 config=run_config)

      log_dir = estimator.model_dir
      self.assertEqual(0, _count_ipu_compilations_in_dir(log_dir))

      # Compile the training graph
      estimator.train(input_fn=my_input_fn, steps=2)
      self.assertEqual(1, _count_ipu_compilations_in_dir(log_dir))

      # Re-use cached training graph
      estimator.train(input_fn=my_input_fn, steps=2)
      self.assertEqual(1, _count_ipu_compilations_in_dir(log_dir))

      # Compile the evaluation graph
      estimator.evaluate(input_fn=my_input_fn, steps=2)
      self.assertEqual(2, _count_ipu_compilations_in_dir(log_dir))

      # Re-use cached evaluation graph
      estimator.evaluate(input_fn=my_input_fn, steps=2)
      self.assertEqual(2, _count_ipu_compilations_in_dir(log_dir))

      # Re-use cached training graph
      estimator.train(input_fn=my_input_fn, steps=2)
      self.assertEqual(2, _count_ipu_compilations_in_dir(log_dir))

    with _temporary_executable_cache():
      self._run_in_new_process(build_and_run_model)

  @test_util.deprecated_graph_mode_only
  def test_unhashable_op(self):
    def build_and_run_model():
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
        ReportJSON(self, sess, set_opts_fn=_options_function)
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
    def build_and_run_model():
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

      with session.Session() as sess:
        report = ReportJSON(self, sess, set_opts_fn=_options_function)
        sess.run(result, feed_dict={x: np.ones([128]), y: np.ones([128])})
        events = report.get_event_trace(sess)
        return events

    with _temporary_executable_cache():
      events0 = self._run_in_new_process(build_and_run_model)
      events1 = self._run_in_new_process(build_and_run_model)
      # Expect no second compilation as the executable should be cached
      self.assertEqual(1, count_ipu_compilations(events0))
      self.assertEqual(0, count_ipu_compilations(events1))

  @test_util.deprecated_graph_mode_only
  def test_host_embeddings(self):
    def build_and_run_model():
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

        report = ReportJSON(self, sess, set_opts_fn=_options_function)

        sess.run(variables.global_variables_initializer())
        report.reset()
        sess.run(
            gen_pop_datastream_ops.ipu_host_embedding_register(
                w, "host_embedding", optimizer="SGD+GA"))
        result = sess.run([r], {i: i_h})
        sess.run(
            gen_pop_datastream_ops.ipu_host_embedding_deregister(
                w, "host_embedding"))

        events = report.get_event_trace(sess)
        return result, events

    with _temporary_executable_cache():
      result0, events0 = self._run_in_new_process(build_and_run_model)
      result1, events1 = self._run_in_new_process(build_and_run_model)
      self.assertFalse(np.any(result0))
      self.assertAllEqual(result0, result1)
      # Expect no second compilation as the executable should be cached
      self.assertEqual(1, count_ipu_compilations(events0))
      self.assertEqual(0, count_ipu_compilations(events1))


if __name__ == "__main__":
  googletest.main()
