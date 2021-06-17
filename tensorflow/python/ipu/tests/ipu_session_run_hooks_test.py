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
# =============================================================================
import time

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import loops
from tensorflow.python.ipu.ipu_session_run_hooks import IPULoggingTensorHook
from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
from tensorflow.python.training.monitored_session import MonitoredTrainingSession
from tensorflow.python import ops


@test_util.deprecated_graph_mode_only
class IPULoggingTensorHookTest(test_util.TensorFlowTestCase):
  def mock_log(self, *args, **kwargs):
    del kwargs
    self.logged_message = args

  def test_illegal_args(self):
    with self.assertRaisesRegex(
        ValueError, "Cannot provide both every_n_iter and every_n_secs"):
      IPULoggingTensorHook(every_n_iter=5, every_n_secs=5)

    with self.assertRaisesRegex(
        ValueError,
        "Either every_n_iter, every_n_secs or at_end should be provided"):
      IPULoggingTensorHook()

  def test_illegal_log_types(self):
    hook = IPULoggingTensorHook(at_end=True)

    with self.assertRaisesRegex(TypeError, "Expected `tf.Tensor`"):
      hook.log("foo")

    with self.assertRaisesRegex(TypeError, "Expected `tf.Tensor`"):
      hook.log([1.0])

  def test_missing_log_call(self):
    hook = IPULoggingTensorHook(at_end=True)

    with self.assertRaisesRegex(RuntimeError,
                                "Did you forget to call the log function"):
      hook.begin()

  def test_log_twice_not_supported(self):
    hook = IPULoggingTensorHook(at_end=True)

    with ipu_scope("/device:IPU:0"):
      t = constant_op.constant(0.0)
      hook.log(t)
      with self.assertRaisesRegex(
          RuntimeError,
          "Cannot use this hook object's log function more than once"):
        return hook.log(t)

  def test_print_tensor(self):
    hook = IPULoggingTensorHook(at_end=True)

    def model():
      t = constant_op.constant(42.0, name="foo")
      return hook.log(t)

    with ipu_scope("/device:IPU:0"):
      compiled_model = ipu_compiler.compile(model)

    with test.mock.patch.object(tf_logging, "info", self.mock_log):
      with MonitoredTrainingSession(hooks=[hook]) as mon_sess:
        mon_sess.run(compiled_model)

    self.assertRegex(str(self.logged_message), "foo:0 = 42.0")

  def test_print_list(self):
    hook = IPULoggingTensorHook(at_end=True)

    def model():
      t1 = constant_op.constant(42.0, name="foo")
      t2 = constant_op.constant(43.0, name="bar")
      return hook.log([t1, t2])

    with ipu_scope("/device:IPU:0"):
      compiled_model = ipu_compiler.compile(model)

    with test.mock.patch.object(tf_logging, "info", self.mock_log):
      with MonitoredTrainingSession(hooks=[hook]) as mon_sess:
        mon_sess.run(compiled_model)

    self.assertRegex(str(self.logged_message), "foo:0 = 42.0, bar:0 = 43.0")

  def test_print_dict(self):
    hook = IPULoggingTensorHook(at_end=True)

    def model():
      t1 = constant_op.constant(42.0)
      t2 = constant_op.constant(43.0)
      return hook.log({"foo": t1, "bar": t2})

    with ipu_scope("/device:IPU:0"):
      compiled_model = ipu_compiler.compile(model)

    with test.mock.patch.object(tf_logging, "info", self.mock_log):
      with MonitoredTrainingSession(hooks=[hook]) as mon_sess:
        mon_sess.run(compiled_model)

    self.assertRegex(str(self.logged_message), "foo = 42.0, bar = 43.0")

  def test_print_formatter(self):
    def formatter(args):
      self.assertIsInstance(args, dict)
      return "foobar: {}".format(args)

    hook = IPULoggingTensorHook(at_end=True, formatter=formatter)

    def model():
      t1 = constant_op.constant(42.0, name="foo")
      t2 = constant_op.constant(43.0, name="bar")
      return hook.log([t1, t2])

    with ipu_scope("/device:IPU:0"):
      compiled_model = ipu_compiler.compile(model)

    with test.mock.patch.object(tf_logging, "info", self.mock_log):
      with MonitoredTrainingSession(hooks=[hook]) as mon_sess:
        mon_sess.run(compiled_model)

    self.assertRegex(str(self.logged_message),
                     r"foobar: \{'foo:0': 42.0, 'bar:0': 43.0\}")

  def test_print_at_end_only(self):
    hook = IPULoggingTensorHook(at_end=True)

    def model():
      t = constant_op.constant(42.0, name="foo")
      return hook.log(t)

    with ipu_scope("/device:IPU:0"):
      compiled_model = ipu_compiler.compile(model)

    with test.mock.patch.object(tf_logging, "info", self.mock_log):
      with MonitoredTrainingSession(hooks=[hook]) as mon_sess:
        self.logged_message = ""
        for _ in range(3):
          mon_sess.run(compiled_model)
          self.assertEqual(str(self.logged_message).find("foo"), -1)

    self.assertRegex(str(self.logged_message), "foo:0 = 42.0")

  def test_print_all_at_end(self):
    hook = IPULoggingTensorHook(
        at_end=True, logging_mode=IPULoggingTensorHook.LoggingMode.ALL)

    def body(v):
      logging_op = hook.log({"foo": v})
      with ops.control_dependencies([logging_op]):
        return v + 1

    def model():
      return loops.repeat(2, body, inputs=[1.0])

    with ipu_scope("/device:IPU:0"):
      compiled_model = ipu_compiler.compile(model)

    with test.mock.patch.object(tf_logging, "info", self.mock_log):
      with MonitoredTrainingSession(hooks=[hook]) as mon_sess:
        for _ in range(2):
          mon_sess.run(compiled_model)

    self.assertRegex(str(self.logged_message), r"foo = \[1. 2. 1. 2.\]")

  def test_print_every_n_iter(self):
    hook = IPULoggingTensorHook(every_n_iter=2)

    def model():
      step = variables.Variable(0)
      return hook.log({"step": step.assign_add(1).value()})

    with ipu_scope("/device:IPU:0"):
      compiled_model = ipu_compiler.compile(model)

    with test.mock.patch.object(tf_logging, "info", self.mock_log):
      # Test re-using the hook.
      for _ in range(2):
        with MonitoredTrainingSession(hooks=[hook]) as mon_sess:
          mon_sess.run(compiled_model)
          self.assertRegex(str(self.logged_message), "step = 1")

          self.logged_message = ""
          mon_sess.run(compiled_model)
          self.assertEqual(self.logged_message, "")

          mon_sess.run(compiled_model)
          self.assertRegex(str(self.logged_message), "step = 3")

  @test.mock.patch.object(time, "time")
  def test_print_every_n_secs(self, mock_time):
    hook = IPULoggingTensorHook(every_n_secs=0.5)

    def model():
      return hook.log({"log": constant_op.constant(0)})

    with ipu_scope("/device:IPU:0"):
      compiled_model = ipu_compiler.compile(model)

    with test.mock.patch.object(tf_logging, "info", self.mock_log):
      with MonitoredTrainingSession(hooks=[hook]) as mon_sess:
        mock_time.return_value = 1.0
        mon_sess.run(compiled_model)
        self.assertRegex(str(self.logged_message), "log = 0")

        self.logged_message = ""
        mock_time.return_value = 1.49
        mon_sess.run(compiled_model)
        self.assertEqual(self.logged_message, "")

        mock_time.return_value = 1.5
        mon_sess.run(compiled_model)
        self.assertRegex(str(self.logged_message), "log = 0")

  def test_two_hooks(self):
    hook1 = IPULoggingTensorHook(every_n_iter=1)
    hook2 = IPULoggingTensorHook(
        every_n_iter=2, logging_mode=IPULoggingTensorHook.LoggingMode.ALL)

    def model():
      step = variables.Variable(0)
      updated = step.assign_add(1).value()
      return hook1.log({"hook1": updated}), hook2.log({"hook2": updated})

    with ipu_scope("/device:IPU:0"):
      compiled_model = ipu_compiler.compile(model)

    logged_messages = []

    def mock_log(*args, **kwargs):
      del kwargs
      logged_messages.append(str(args))

    with MonitoredTrainingSession(hooks=[hook1, hook2]) as mon_sess:
      with test.mock.patch.object(tf_logging, "info", mock_log):
        mon_sess.run(compiled_model)
        self.assertEqual(len(logged_messages), 2)
        self.assertRegex(logged_messages[0], "hook1 = 1")
        self.assertRegex(logged_messages[1], r"hook2 = \[1\]")

        mon_sess.run(compiled_model)
        self.assertEqual(len(logged_messages), 3)
        self.assertRegex(logged_messages[2], "hook1 = 2")

        mon_sess.run(compiled_model)
        self.assertEqual(len(logged_messages), 5)
        self.assertRegex(logged_messages[3], "hook1 = 3")
        self.assertRegex(logged_messages[4], r"hook2 = \[2 3\]")


if __name__ == "__main__":
  test.main()
