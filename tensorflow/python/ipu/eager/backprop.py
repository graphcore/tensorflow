# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

from collections import namedtuple
from threading import Lock

from tensorflow.python.eager.backprop import GradientTape
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients

_CapturedGradientStackFrame = namedtuple("StackFrame", "tape_id gradients")


class _CapturedGradientStack:
  """
  An internal stack data structure intended for use by
  `tensorflow.python.ipu.ops.grad_util_ops.capture_upstream_gradients`,
  `tensorflow.python.ipu.eager.backprop.GradientCaptureContext` and
  `tensorflow.python.ipu.eager.backprop.GradientCaptureTape`.

  Each element on the stack is a "grad frame" which is a named tuple containing
  an ID for each tape or context and a dictionary of gradients associated with
  that tape or context.
  """
  def __init__(self):
    self.__grad_stack = []
    self.__mutex = Lock()

  def __check_stack_access(self):
    if not self.__grad_stack:
      raise RuntimeError(
          "Attempting to access grad frame in an empty stack. The "
          "ipu.eager.backprop._CapturedGradientStack is for "
          "use only by ipu.eager.backprop.GradientCaptureTape or "
          "ipu.eager.backprop.GradientCaptureContext.")

  def push_grad_frame(self, tape):
    self.__mutex.acquire()
    try:
      self.__grad_stack.append(_CapturedGradientStackFrame(id(tape), dict()))
    finally:
      self.__mutex.release()

  def pop_grad_frame(self):
    self.__mutex.acquire()
    try:
      self.__check_stack_access()
      frame = self.__grad_stack.pop()
    finally:
      self.__mutex.release()

    return frame.gradients

  def push_captured_grad(self, tag, grad):
    self.__mutex.acquire()
    try:
      self.__check_stack_access()
      frame = self.__grad_stack[-1]
      if tag in frame:
        raise ValueError("Gradient exists in the captured grads frame.")

      frame.gradients[tag] = grad
    finally:
      self.__mutex.release()

  @property
  def tape_count(self):
    self.__mutex.acquire()
    try:
      num_tapes = len(self.__grad_stack)
    finally:
      self.__mutex.release()

    return num_tapes

  @property
  def top_frame_tape_id(self):
    self.__mutex.acquire()
    try:
      self.__check_stack_access()
      frame = self.__grad_stack[-1]
    finally:
      self.__mutex.release()

    return frame.tape_id


_captured_grad_stack = _CapturedGradientStack()


def _push_captured_grad(op_tag, grad):
  _captured_grad_stack.push_captured_grad(str(op_tag, 'utf-8'), grad)


def num_gradient_collection_tapes():
  return _captured_grad_stack.tape_count


class GradientCaptureContext(object):
  """A context manager under which operations wrapped with
  `tensorflow.python.ipu.ops.grad_util_ops.capture_upstream_gradients`
  have any incoming gradients captured and stored during the process of
  computing gradients w.r.t variables.

  Example:

  .. code-block:: python
    opt = SGD()
    x = Variable(3.0)
    with GradientCaptureContext() as gcc:
      o = x**2
      p = capture_upstream_gradients(o, tag="tanh_grad")
      y = math_ops.tanh(p)
      grads_wrt_vars = opt.get_gradients(y, x)

    captured_grads = gcc.captured_gradients
    tanh_grad = grads_wrt_vars["tanh_grad"]
  """
  def __init__(self):
    self.__captured_grads = None

  def __enter__(self):
    _captured_grad_stack.push_grad_frame(self)
    return self

  def __exit__(self, typ, value, traceback):
    pass

  def __del__(self):
    if num_gradient_collection_tapes() > 0:
      if _captured_grad_stack.top_frame_tape_id == id(self):
        _ = _captured_grad_stack.pop_grad_frame()

  def _update_grads(self):
    if num_gradient_collection_tapes() > 0:
      if _captured_grad_stack.top_frame_tape_id == id(self):
        self.__captured_grads = _captured_grad_stack.pop_grad_frame()

  @property
  def captured_gradients(self):
    """
    Incoming gradients of operations wrapped with
    `tensorflow.python.ipu.ops.grad_util_ops.capture_upstream_gradients`
    during the computation of gradients w.r.t variables.
    """
    self._update_grads()
    return self.__captured_grads

  @captured_gradients.setter
  def captured_gradients(self, _):
    raise ValueError("captured_gradients is read only.")


class GradientCaptureTape(GradientTape, GradientCaptureContext):
  """
  A specialised GradientTape under which operations wrapped with
  `tensorflow.python.ipu.ops.grad_util_ops.capture_upstream_gradients`
  have any incoming gradients captured and stored during the process of
  computing gradients w.r.t variables.

  Example:

  .. code-block:: python
    x = Variable(3.0)
    with GradientCaptureTape() as tape:
      o = x**2
      p = capture_upstream_gradients(o, tag="tanh_grad")
      y = math_ops.tanh(p)
      grads_wrt_vars = tape.gradient(y, x)

    captured_grads = tape.captured_gradients
    tanh_grad = grads_wrt_vars["tanh_grad"]
  """
  def __init__(self, persistent=False, watch_accessed_variables=True):
    GradientTape.__init__(self,
                          persistent=persistent,
                          watch_accessed_variables=watch_accessed_variables)

    GradientCaptureContext.__init__(self)

  def gradient(self,
               target,
               sources,
               output_gradients=None,
               unconnected_gradients=UnconnectedGradients.NONE):
    grads_wrt_vars = GradientTape.gradient(
        self,
        target,
        sources,
        output_gradients=output_gradients,
        unconnected_gradients=unconnected_gradients)

    self._update_grads()
    return grads_wrt_vars

  def __enter__(self):
    GradientTape.__enter__(self)
    GradientCaptureContext.__enter__(self)
    return self
