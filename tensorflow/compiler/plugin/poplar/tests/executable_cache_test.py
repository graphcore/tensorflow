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

import contextlib
import glob
import multiprocessing
import numpy as np
import os
import popef
import tempfile

from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.compiler.plugin.poplar.ops import gen_sendrecv_ops
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.compiler.tests import xla_test
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.python import ipu
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.estimator import model_fn
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ipu.ops import application_compile_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import metrics_impl
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.python.summary import summary_iterator

offline_compilation_needed = "--use_ipu_model" in os.environ.get(
    "TF_POPLAR_FLAGS", "")


def _extra_ipu_config(cfg):
  if offline_compilation_needed:
    cfg.ipu_model.compile_ipu_code = False
    cfg.device_connection.version = 'ipu1'
    cfg.device_connection.type = ipu.utils.DeviceConnectionType.NEVER
    return
  tu.add_hw_ci_connection_options(cfg)


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


class TestExecutableCache(xla_test.XLATestCase):  # pylint: disable=abstract-method
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

  @tu.test_uses_ipus(num_ipus=1, allow_ipu_model=True)
  @test_util.deprecated_graph_mode_only
  def test_basic_model(self):
    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    _extra_ipu_config(cfg)

    def build_and_run_model():
      cfg.configure_ipu_system()

      def my_net(x):
        return x * x

      v = array_ops.placeholder(dtype=np.float32, shape=(2,))
      with ipu.scopes.ipu_scope("/device:IPU:0"):
        [result] = ipu.ipu_compiler.compile(my_net, inputs=[v])

      with session.Session() as sess:
        try:
          res = sess.run(result, {v: [1.0, 2.0]})
        except errors.InvalidArgumentError as e:
          if offline_compilation_needed and "compilation only" in e.message:
            res = []
          else:
            raise
      num_reports = len(report_helper.find_reports())
      report_helper.clear_reports()
      return res, num_reports

    with _temporary_executable_cache():
      result0, num_reports_0 = self._run_in_new_process(build_and_run_model)
      result1, num_reports_1 = self._run_in_new_process(build_and_run_model)
      self.assertAllEqual(result0, result1)
      self.assertEqual(num_reports_0, 1)
      self.assertEqual(num_reports_1, 0)

  @tu.test_uses_ipus(num_ipus=1, allow_ipu_model=True)
  @test_util.deprecated_graph_mode_only
  def test_model_with_infeed_and_outfeed(self):
    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    _extra_ipu_config(cfg)

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
        [result] = ipu.ipu_compiler.compile(my_net, inputs=[v])
      with ops.control_dependencies([result]):
        dequeued = outfeed_queue.dequeue()

      with session.Session() as sess:
        sess.run(infeed_queue.initializer)
        try:
          res, deq = sess.run([result, dequeued], {v: 0.0})
        except errors.InvalidArgumentError as e:
          if offline_compilation_needed and "compilation only" in e.message:
            res = []
            deq = []
          else:
            raise
      num_reports = len(report_helper.find_reports())
      report_helper.clear_reports()
      return res, deq, num_reports

    with _temporary_executable_cache():
      result0, dequeued0, num_reports_0 = self._run_in_new_process(
          build_and_run_model)
      result1, dequeued1, num_reports_1 = self._run_in_new_process(
          build_and_run_model)
      self.assertAllEqual(dequeued0, dequeued1)
      self.assertEqual(result0, result1)
      self.assertEqual(num_reports_0, 1)
      self.assertEqual(num_reports_1, 0)

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_model_with_send_to_host_op(self):
    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    _extra_ipu_config(cfg)

    def build_and_run_model():
      cfg.configure_ipu_system()

      def my_net(x):
        return gen_sendrecv_ops.ipu_send_to_host(x,
                                                 tensor_name="test_tensor",
                                                 send_device="/device:IPU:0",
                                                 send_device_incarnation=0,
                                                 recv_device="/device:CPU:0")

      v = array_ops.placeholder(np.float32, shape=())
      with ipu.scopes.ipu_scope("/device:IPU:0"):
        send_op = ipu.ipu_compiler.compile(my_net, inputs=[v])
      with ops.device("/device:CPU:0"):
        recv_op = gen_sendrecv_ops.ipu_recv_at_host(
            T=np.float32,
            tensor_name="test_tensor",
            send_device="/device:IPU:0",
            send_device_incarnation=0,
            recv_device="/device:CPU:0")

      with session.Session() as sess:
        _, received = sess.run([send_op, recv_op], feed_dict={v: 1.0})
      num_reports = len(report_helper.find_reports())
      report_helper.clear_reports()
      return received, num_reports

    with _temporary_executable_cache():
      received0, num_reports_0 = self._run_in_new_process(build_and_run_model)
      received1, num_reports_1 = self._run_in_new_process(build_and_run_model)
      self.assertEqual(received0, received1)
      self.assertEqual(received0, 1.0)
      self.assertEqual(num_reports_0, 1)
      self.assertEqual(num_reports_1, 0)

  @tu.test_uses_ipus(num_ipus=1, allow_ipu_model=True)
  @test_util.deprecated_graph_mode_only
  def test_new_graph_in_same_process(self):
    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    _extra_ipu_config(cfg)

    def build_and_run_model():
      cfg.configure_ipu_system()

      def my_net(x):
        return x * x

      v = array_ops.placeholder(dtype=np.float32, shape=(2,))
      with ipu.scopes.ipu_scope("/device:IPU:0"):
        [result] = ipu.ipu_compiler.compile(my_net, inputs=[v])

      with session.Session() as sess:
        try:
          res = sess.run(result, {v: [1.0, 2.0]})
        except errors.InvalidArgumentError as e:
          if offline_compilation_needed and "compilation only" in e.message:
            res = []
          else:
            raise
      num_reports = len(report_helper.find_reports())
      report_helper.clear_reports()
      return res, num_reports

    with _temporary_executable_cache():
      # Since each Graph will have its own XLA compilation cache,
      # the cache we test is the last-level Poplar executable cache.

      with ops.Graph().as_default():
        result0, num_reports_0 = build_and_run_model()

      with ops.Graph().as_default():
        result1, num_reports_1 = build_and_run_model()

      self.assertAllEqual(result0, result1)
      self.assertEqual(num_reports_0, 1)
      self.assertEqual(num_reports_1, 0)

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
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
      cfg = ipu.config.IPUConfig()
      report_helper = tu.ReportHelper()
      report_helper.set_autoreport_options(cfg)
      _extra_ipu_config(cfg)
      cfg.auto_select_ipus = 1

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

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_model_with_outside_compilation_scope(self):
    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    _extra_ipu_config(cfg)

    def build_and_run_model():
      cfg.configure_ipu_system()

      def my_net(x):
        y = x * x
        with ipu.scopes.outside_compilation_scope():
          z = y * 2.0
        return z + 2.0

      v = array_ops.placeholder(np.float32, shape=())
      with ipu.scopes.ipu_scope("/device:IPU:0"):
        [result] = ipu.ipu_compiler.compile(my_net, inputs=[v])

      with session.Session() as sess:
        received = sess.run(result, feed_dict={v: 1.0})
      num_reports = len(report_helper.find_reports())
      report_helper.clear_reports()
      return received, num_reports

    with _temporary_executable_cache():
      received0, num_reports_0 = self._run_in_new_process(build_and_run_model)
      received1, num_reports_1 = self._run_in_new_process(build_and_run_model)
      self.assertEqual(received0, received1)
      self.assertEqual(received0, 4.0)
      self.assertEqual(num_reports_0, 1)
      self.assertEqual(num_reports_1, 0)

  @tu.test_uses_ipus(num_ipus=1, allow_ipu_model=True)
  @test_util.deprecated_graph_mode_only
  def test_unhashable_op(self):
    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    _extra_ipu_config(cfg)

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
                                               op_name="UnhashableTest",
                                               outs=outputs)
        return x

      with ipu.scopes.ipu_scope('/device:IPU:0'):
        x = array_ops.placeholder(np.float32, shape=[128])
        y = array_ops.placeholder(np.float32, shape=[128])

        result = ipu.ipu_compiler.compile(my_net, inputs=[x, y])

      with session.Session() as sess:
        try:
          sess.run(result, feed_dict={x: np.ones([128]), y: np.ones([128])})
        except errors.InvalidArgumentError as e:
          if offline_compilation_needed and "compilation only" in e.message:
            pass
          else:
            raise
      num_reports = len(report_helper.find_reports())
      report_helper.clear_reports()
      return num_reports

    with _temporary_executable_cache():
      num_reports_0 = self._run_in_new_process(build_and_run_model)
      num_reports_1 = self._run_in_new_process(build_and_run_model)
      # Expect second compilation as the executable should not be cached
      self.assertEqual(num_reports_0, 1)
      self.assertEqual(num_reports_1, 1)

  @tu.skip_with_asan("non-deterministic dlopen user ops addresses with asan")
  @tu.test_uses_ipus(num_ipus=1, allow_ipu_model=True)
  @test_util.deprecated_graph_mode_only
  def test_hashable_op(self):
    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    _extra_ipu_config(cfg)

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

      with session.Session() as sess:
        try:
          sess.run(result, feed_dict={x: np.ones([128]), y: np.ones([128])})
        except errors.InvalidArgumentError as e:
          if offline_compilation_needed and "compilation only" in e.message:
            pass
          else:
            raise
      num_reports = len(report_helper.find_reports())
      report_helper.clear_reports()
      return num_reports

    with _temporary_executable_cache():
      num_reports_0 = self._run_in_new_process(build_and_run_model)
      num_reports_1 = self._run_in_new_process(build_and_run_model)
      # Expect no second compilation as the executable should be cached
      self.assertEqual(num_reports_0, 1)
      self.assertEqual(num_reports_1, 0)

  @tu.test_uses_ipus(num_ipus=1, allow_ipu_model=True)
  @test_util.deprecated_graph_mode_only
  def test_export_with_cache(self):
    cfg = ipu.config.IPUConfig()
    _extra_ipu_config(cfg)

    def export_model(poplar_exec_output_path):
      cfg.configure_ipu_system()

      dataset = dataset_ops.Dataset.from_tensors(
          np.ones((64, 64), dtype=np.float16))
      dataset = dataset.repeat(10)
      infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
      outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

      def body(x):
        x = Flatten()(x)
        x = Dense(256)(x)
        return outfeed_queue.enqueue(x)

      def my_net():
        return ipu.loops.repeat(10, body, [], infeed_queue)

      with session.Session() as sess:
        compile_op = application_compile_op.experimental_application_compile_op(
            my_net,
            output_path=poplar_exec_output_path,
            freeze_variables=False)
        sess.run(variables.global_variables_initializer())
        sess.run(compile_op)

    with tempfile.TemporaryDirectory() as temp_dir:
      poplar_exec_precache = os.path.join(temp_dir, "precache.popef")
      poplar_exec_postcache = os.path.join(temp_dir, "postcache.popef")
      with _temporary_executable_cache():
        self._run_in_new_process(lambda: export_model(poplar_exec_precache))
        self._run_in_new_process(lambda: export_model(poplar_exec_postcache))
      # Check the anchors are the same pre and post cache
      r_precache = popef.Reader()
      r_precache.parseFile(poplar_exec_precache)
      self.assertEqual(len(r_precache.metadata()), 1)
      anchor_names_precache = sorted(
          a.name() for a in r_precache.metadata()[0].anchors())
      r_postcache = popef.Reader()
      r_postcache.parseFile(poplar_exec_postcache)
      self.assertEqual(len(r_postcache.metadata()), 1)
      anchor_names_postcache = sorted(
          a.name() for a in r_postcache.metadata()[0].anchors())
      self.assertEqual(anchor_names_precache, anchor_names_postcache)


if __name__ == "__main__":
  googletest.main()
