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
import popdist

import numpy as np
from absl.testing import parameterized

from tensorflow.python.ipu.distributed import host_collective_ops
from tensorflow.python.framework import test_util, dtypes
from tensorflow.python.platform import test
from tensorflow.python.distribute import reduce_util
from tensorflow.python.framework import errors
from tensorflow.python.framework import constant_op
from tensorflow.python.eager import def_function

TYPES = (dtypes.float16, dtypes.float32, dtypes.int8, dtypes.int16,
         dtypes.int32, dtypes.int64)
TESTCASES = [{
    "testcase_name": np.dtype(x.as_numpy_dtype).name,
    "dtype": x
} for x in TYPES]


class HostCollectiveOpsTest(test_util.TensorFlowTestCase,
                            parameterized.TestCase):  # pylint: disable=abstract-method
  @classmethod
  def setUpClass(cls):
    popdist.init()

  @parameterized.named_parameters(*TESTCASES)
  def test_allgather(self, dtype):
    x = constant_op.constant(popdist.getInstanceIndex(), dtype=dtype)
    self.assertAllEqual(
        host_collective_ops.allgather(x),
        np.array([i for i in range(popdist.getNumInstances())],
                 dtype=dtype.as_numpy_dtype))

  @parameterized.named_parameters(*TESTCASES)
  def test_allreduce_sum(self, dtype):
    x = constant_op.constant(popdist.getInstanceIndex(), dtype=dtype)
    self.assertAllEqual(
        host_collective_ops.allreduce(x, reduce_util.ReduceOp.SUM),
        sum(range(popdist.getNumInstances())))

  @parameterized.named_parameters(*TESTCASES)
  def test_allreduce_mean(self, dtype):
    x = constant_op.constant(popdist.getInstanceIndex(), dtype=dtype)
    self.assertAllEqual(
        host_collective_ops.allreduce(x, reduce_util.ReduceOp.MEAN),
        dtype.as_numpy_dtype(
            sum(range(popdist.getNumInstances())) / popdist.getNumInstances()))

  @parameterized.named_parameters(*TESTCASES)
  def test_broadcast(self, dtype):
    x = constant_op.constant(42 if popdist.getInstanceIndex() == 0 else 0,
                             dtype=dtype)
    self.assertAllEqual(host_collective_ops.broadcast(x), 42)

  def test_all_allgather_different_order(self):
    # Call collective on `x` first and `y` afterwards.
    @def_function.function()
    def body_instance_even(x, y):
      res_x = host_collective_ops.allgather(x)
      res_y = host_collective_ops.allgather(y)

      return (res_x, res_y)

    # Call collective on `y` first and `x` afterwards.
    @def_function.function()
    def body_instance_odd(x, y):
      res_y = host_collective_ops.allgather(y)
      res_x = host_collective_ops.allgather(x)

      return (res_x, res_y)

    x = constant_op.constant(popdist.getInstanceIndex(), dtype=dtypes.float32)
    y = constant_op.constant(
        [popdist.getInstanceIndex(),
         popdist.getInstanceIndex()],
        dtype=dtypes.int32)

    is_even = popdist.getInstanceIndex() % 2 == 0

    # Test that we can call collectives in any order as long as our tensors have names.
    (res_x,
     res_y) = body_instance_even(x, y) if is_even else body_instance_odd(x, y)

    self.assertAllEqual(
        res_x,
        np.array([i for i in range(popdist.getNumInstances())],
                 dtype=np.float32))
    self.assertAllEqual(
        res_y,
        np.array([[i, i] for i in range(popdist.getNumInstances())],
                 dtype=np.float32))

  def test_allreduce_different_order(self):
    # Call collective on `x` first and `y` afterwards.
    @def_function.function()
    def body_instance_even(x, y):
      res_x = host_collective_ops.allreduce(x, reduce_util.ReduceOp.SUM)
      res_y = host_collective_ops.allreduce(y, reduce_util.ReduceOp.SUM)

      return (res_x, res_y)

    # Call collective on `y` first and `x` afterwards.
    @def_function.function()
    def body_instance_odd(x, y):
      res_y = host_collective_ops.allreduce(y, reduce_util.ReduceOp.SUM)
      res_x = host_collective_ops.allreduce(x, reduce_util.ReduceOp.SUM)

      return (res_x, res_y)

    x = constant_op.constant(popdist.getInstanceIndex(), dtype=dtypes.float32)
    y = constant_op.constant(
        [popdist.getInstanceIndex(),
         popdist.getInstanceIndex()],
        dtype=dtypes.int32)

    is_even = popdist.getInstanceIndex() % 2 == 0

    # Test that we can call collectives in any order as long as our tensors have names.
    (res_x,
     res_y) = body_instance_even(x, y) if is_even else body_instance_odd(x, y)

    expected_value = sum(range(popdist.getNumInstances()))
    self.assertAllEqual(res_x, expected_value)
    self.assertAllEqual(res_y, [expected_value, expected_value])

  def test_broadcast_different_order(self):
    # Call collective on `x` first and `y` afterwards.
    @def_function.function()
    def body_instance_even(x, y):
      res_x = host_collective_ops.broadcast(x)
      res_y = host_collective_ops.broadcast(y)

      return (res_x, res_y)

    # Call collective on `y` first and `x` afterwards.
    @def_function.function()
    def body_instance_odd(x, y):
      res_y = host_collective_ops.broadcast(y)
      res_x = host_collective_ops.broadcast(x)

      return (res_x, res_y)

    initial_value = 42 if popdist.getInstanceIndex() == 0 else 0
    x = constant_op.constant(initial_value, dtype=dtypes.float32)
    y = constant_op.constant([initial_value, initial_value],
                             dtype=dtypes.int32)

    is_even = popdist.getInstanceIndex() % 2 == 0

    # Test that we can call collectives in any order as long as our tensors have names.
    (res_x,
     res_y) = body_instance_even(x, y) if is_even else body_instance_odd(x, y)

    self.assertAllEqual(res_x, 42)
    self.assertAllEqual(res_y, [42, 42])

  def test_allgather_different_dtype(self):
    dtype = dtypes.float32 if popdist.getInstanceIndex(
    ) % 2 == 0 else dtypes.int32
    x = constant_op.constant(popdist.getInstanceIndex(), dtype=dtype)

    try:
      host_collective_ops.allgather(x)
    except errors.UnknownError as e:
      self.assertAllEqual(
          True,
          "Tensor layouts did not match on all instances" in e.message,
      )

      return

    self.fail()

  def test_allgather_different_shape(self):
    value = 1 if popdist.getInstanceIndex() % 2 == 0 else [1, 1]
    x = constant_op.constant(value, dtype=dtypes.int32)

    try:
      host_collective_ops.allgather(x)
    except errors.UnknownError as e:
      self.assertAllEqual(
          True,
          "Tensor layouts did not match on all instances" in e.message,
      )

      return

    self.fail()

  def test_allreduce_different_dtype(self):
    dtype = dtypes.float32 if popdist.getInstanceIndex(
    ) % 2 == 0 else dtypes.int32
    x = constant_op.constant(popdist.getInstanceIndex(), dtype=dtype)

    try:
      host_collective_ops.allreduce(x, reduce_util.ReduceOp.SUM)
    except errors.UnknownError as e:
      self.assertAllEqual(
          True,
          "Tensor layouts did not match on all instances" in e.message,
      )

      return

    self.fail()

  def test_allreduce_different_shape(self):
    value = 1 if popdist.getInstanceIndex() % 2 == 0 else [1, 1]
    x = constant_op.constant(value, dtype=dtypes.int32)

    try:
      host_collective_ops.allreduce(x, reduce_util.ReduceOp.SUM)
    except errors.UnknownError as e:
      self.assertAllEqual(
          True,
          "Tensor layouts did not match on all instances" in e.message,
      )

      return

    self.fail()

  def test_broadcast_different_dtype(self):
    dtype = dtypes.float32 if popdist.getInstanceIndex(
    ) % 2 == 0 else dtypes.int32
    x = constant_op.constant(popdist.getInstanceIndex(), dtype=dtype)

    try:
      host_collective_ops.broadcast(x)
    except errors.UnknownError as e:
      self.assertAllEqual(
          True,
          "Tensor layouts did not match on all instances" in e.message,
      )

      return

    self.fail()

  def test_broadcast_different_shape(self):
    value = 1 if popdist.getInstanceIndex() % 2 == 0 else [1, 1]
    x = constant_op.constant(value, dtype=dtypes.int32)

    try:
      host_collective_ops.broadcast(x)
    except errors.UnknownError as e:
      self.assertAllEqual(
          True,
          "Tensor layouts did not match on all instances" in e.message,
      )

      return

    self.fail()


if __name__ == "__main__":
  test.main()
