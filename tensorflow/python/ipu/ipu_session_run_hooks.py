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
# ===================================================================
"""
Session run hooks
~~~~~~~~~~~~~~~~~
"""
import numpy as np

from tensorflow.python import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import session_run_hook
from tensorflow.python.training.basic_session_run_hooks import NeverTriggerTimer
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer


class IPULoggingTensorHook(session_run_hook.SessionRunHook):
  """Prints the given tensors every N local steps, every N seconds, or at end.

  This is a version of `tf.estimator.LoggingTensorHook` that supports logging
  from inside a function compiled for the IPU. The implementation uses an IPU
  outfeed in order to send the tensors from the compiled function to the host.

  The tensors will be printed to the log, with `INFO` severity.
  """

  LoggingMode = ipu_outfeed_queue.IPUOutfeedMode

  def __init__(self,
               every_n_iter=None,
               every_n_secs=None,
               at_end=False,
               formatter=None,
               logging_mode=LoggingMode.LAST):
    """Initializes the hook.

    Args:
      every_n_iter: `int`, print the tensor values once every N steps.
      every_n_secs: `int` or `float`, print the tensor values once every N
        seconds. Exactly one of `every_n_iter` and `every_n_secs` should be
        provided (unless `at_end` is True).
      at_end: `bool` specifying whether to print the tensor values at the
        end of the run.
      formatter: function that takes a dict with tensor names and values and
        returns a string. If None, uses default formatting.
      logging_mode: `IPULoggingTensorHook.LoggingMode` that determines the
        behaviour when enqueuing multiple tensor values between dequeues
        (e.g. print all of them or only the last one).
    """
    if (every_n_iter is not None) and (every_n_secs is not None):
      raise ValueError("Cannot provide both every_n_iter and every_n_secs")
    if every_n_iter is None and every_n_secs is None and not at_end:
      raise ValueError(
          "Either every_n_iter, every_n_secs or at_end should be provided")

    only_log_at_end = (at_end and (every_n_iter is None)
                       and (every_n_secs is None))

    self._timer = (NeverTriggerTimer() if only_log_at_end else
                   SecondOrStepTimer(every_secs=every_n_secs,
                                     every_steps=every_n_iter))
    self._log_at_end = at_end
    self._formatter = formatter

    self._outfeed = ipu_outfeed_queue.IPUOutfeedQueue(
        outfeed_mode=logging_mode)

    self._dequeue_op = None
    self._deleter_op = None
    self._iter_count = 0

  def log(self, tensors):
    """Logs the given `tensors`.

    Args:
      tensors: either a dict from string to `tf.Tensor`, a list/tuple of
        `tf.Tensor` objects, or a `tf.Tensor`.

    Returns:
      The logging operation. It might be necessary to add a control dependency
      on this operation, or include it in the training operation using
      `tf.group()`, to avoid it from being pruned from the graph.
    """
    if self._outfeed.enqueued:
      raise RuntimeError(
          "Cannot use this hook object's log function more than once. Either "
          "enqueue all tensors in a single log call using a list or dict, or "
          "create another hook.")

    tensor_dict = self._convert_to_dict(tensors)
    return self._outfeed.enqueue(tensor_dict)

  def _convert_to_dict(self, tensors):
    if isinstance(tensors, dict):
      return tensors

    if not isinstance(tensors, (list, tuple)):
      tensors = [tensors]

    for t in tensors:
      if not tensor_util.is_tensor(t):
        raise TypeError("Expected `tf.Tensor`, got {}".format(type(t)))

    return {t.name: t for t in tensors}

  def begin(self):
    if not self._outfeed.enqueued:
      raise RuntimeError("This logging hook's outfeed was not enqueued. "
                         "Did you forget to call the log function?")

    assert self._dequeue_op is None
    assert self._deleter_op is None

    with ops.device("cpu"):
      self._dequeue_op = self._outfeed.dequeue()
      self._deleter_op = self._outfeed.deleter

    self._iter_count = 0

  def end(self, session):
    if self._log_at_end:
      values = session.run(self._dequeue_op)
      self._log_values(values)

    session.run(self._deleter_op)
    self._deleter_op = None
    self._dequeue_op = None
    self._timer.reset()

  def _log_values(self, tensor_values, elapsed_secs=None):
    assert isinstance(tensor_values, dict)

    original = np.get_printoptions()
    np.set_printoptions(suppress=True)
    elapsed_secs, _ = self._timer.update_last_triggered_step(self._iter_count)
    if self._formatter:
      logging.info(self._formatter(tensor_values))
    else:
      stats = ["{} = {}".format(k, v) for k, v in tensor_values.items()]
      if elapsed_secs is not None:
        logging.info("%s (%.3f sec)", ", ".join(stats), elapsed_secs)
      else:
        logging.info("%s", ", ".join(stats))
    np.set_printoptions(**original)

  def after_run(self, run_context, run_values):
    del run_values

    if self._timer.should_trigger_for_step(self._iter_count):
      values = run_context.session.run(self._dequeue_op)
      self._log_values(values)

    self._iter_count += 1
