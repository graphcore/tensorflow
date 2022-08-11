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

_GradientDataQuery = namedtuple("GradientDataQuery",
                                "stack_index gd_index tag grad_index grad")
_GradientData = namedtuple("GradientData", "tag gradients")
_CapturedGradientStackFrame = namedtuple("StackFrame", "tape_id gradient_data")


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
      self.__grad_stack.append(_CapturedGradientStackFrame(id(tape), []))
    finally:
      self.__mutex.release()

  def pop_grad_frame(self):
    self.__mutex.acquire()
    try:
      self.__check_stack_access()
      frame = self.__grad_stack.pop()
    finally:
      self.__mutex.release()

    return {gd.tag: gd.gradients for gd in frame.gradient_data}

  def push_captured_grad(self, tag, grad):
    self.__check_stack_access()
    frame = self.__grad_stack[-1]

    if tag in [gd.tag for gd in frame.gradient_data]:
      raise ValueError("Gradient exists in the captured grads frame.")

    frame.gradient_data.append(_GradientData(tag, grad))

  def get_all_grads_with_src_op(self, op):
    if not self.tape_count:
      return None

    grad_queries = []
    for stack_idx, stack_element in enumerate(self.__grad_stack):
      for gd_idx, gd in enumerate(stack_element.gradient_data):
        for o in op.outputs:
          match_idx = [n for n, g in enumerate(gd.gradients) \
            if id(g) == id(o)]

          for n in match_idx:
            grad_queries.append(
                _GradientDataQuery(stack_idx, gd_idx, gd.tag, n,
                                   gd.gradients[n]))

    return grad_queries

  def replace_grads(self, query):
    if not isinstance(query, _GradientDataQuery):
      raise ValueError(
          "Can only update captured gradient with a _GradientDataQuery "
          "instance, as returned by get_all_grads_with_src_op.")

    stack_idx = query.stack_index
    if stack_idx >= self.tape_count:
      raise ValueError(
          f"Stack index {stack_idx} is out of bounds for stack of "
          f"size {self.tape_count}.")

    grad_data = self.__grad_stack[stack_idx].gradient_data

    gd_idx = query.gd_index
    if gd_idx >= len(grad_data):
      raise ValueError(
          f"_CapturedGradientStackFrame #{stack_idx}, has {len(grad_data)} "
          f"_GradientData elements. Index {gd_idx} is out of bounds.")

    gd = grad_data[gd_idx]
    tag = query.tag
    if tag != gd.tag:
      raise ValueError(
          f"_CapturedGradientStackFrame #{stack_idx}, with _GradientData "
          f"at index {gd_idx} has tag {tag}, whilst the query update tag "
          f"is {gd.tag}.")

    grad_idx = query.grad_index
    if grad_idx >= len(gd.gradients):
      raise ValueError(
          f"_CapturedGradientStackFrame #{stack_idx}, with _GradientData "
          f"at index {gd_idx} has {len(gd.gradients)} gradients, whilst "
          f"the query update has a gradient index of {grad_idx}.")

    grad = query.grad
    original_grad = gd.gradients[grad_idx]
    if grad.shape != original_grad.shape:
      raise ValueError(
          f"_CapturedGradientStackFrame #{stack_idx}, with _GradientData "
          f"at index {gd_idx} and gradient index {grad_idx} has gradient "
          f"with shape {original_grad.shape}, whilst the query update gradient "
          f"has shape {grad.shape}.")

    gd.gradients[grad_idx] = grad

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

  @property
  def top_frame_gradient_data(self):
    if self.tape_count == 0:
      return None

    self.__mutex.acquire()
    try:
      gd = self.__grad_stack[-1].gradient_data
    finally:
      self.__mutex.release()

    return gd


_captured_grad_stack = _CapturedGradientStack()


def _push_captured_grad(op_tag, grad):
  _captured_grad_stack.push_captured_grad(str(op_tag, 'utf-8'), grad)


def _get_all_captured_grads_with_src_op(src_op_name):
  return _captured_grad_stack.get_all_grads_with_src_op(src_op_name)


def _replace_grads(gdq):
  _captured_grad_stack.replace_grads(gdq)


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
