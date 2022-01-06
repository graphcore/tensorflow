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
"""
Slicing operators
~~~~~~~~~~~~~~~~~
"""

from tensorflow.compiler.plugin.poplar.ops import gen_popops_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


def sequence_slice(dst, src, num_elems, src_offsets, dst_offsets, zero_unused):
  """This op targets the PopLibs SequenceSlice operation.

  The SequenceSlice operation takes specified elements from the source
  tensor and inserts them at specified locations in the destination tensor.

  The parameters of the slice operation are defined by the number of elements
  to take for each slice `num_elems`, the offset in the source tensor from
  which to take them `src_offsets`, and the offset in the destination tensor
  from which the elements should be placed `dst_offsets`.

  For each slice, an element count, source offset and destination offset must
  be provided. The i-th entry of `num_elems` corresponds to the i-th entry of
  `src_offsets` and the i-th entry of `dst_offsets`.

  For example:

  .. code-block:: none

    from tensorflow.python.framework.ops import array_ops
    from tensorflow.python.ipu.ops.slicing_ops import sequence_slice

    src = [[0, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2, 2],
           [3, 3, 3, 3, 3, 3],
           [4, 4, 4, 4, 4, 4],
           [5, 5, 5, 5, 5, 5]]

    num_elems = [2, 2]
    src_offsets = [2, 1]
    dst_offsets = [0, 4]

    dst = array_ops.zeros([6, 6])
    dst = sequence_slice(dst, src, num_elems, src_offsets, dst_offsets, False)

  Following which, the contents of the destination tensor `dst` are as follows:

  .. code-block:: none

    [[2. 2. 2. 2. 2. 2.]
     [3. 3. 3. 3. 3. 3.]
     [0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0.]
     [1. 1. 1. 1. 1. 1.]
     [2. 2. 2. 2. 2. 2.]]

  In this example, the first slice takes two elements from index 2 of the
  source tensor and inserts them at index 0 of the destination tensor. The
  second slice also takes two elements, but from index 1 of the source tensor,
  inserting them at index 4 in the destination tensor.

  Args:
    dst: The destination tensor which will be updated, must be of at least
      rank 2 with inner dimensions matching that of `src`.
    src: The source tensor from which the values are accessed, must be of at
      least rank 2 with inner dimensions matching that of `dst`.
    num_elems: A list (or rank 1 tensor) of the number of elements to copy.
    src_offsets: A list (or rank 1 tensor) of first elements to read from src.
    dst_offsets: A list (or rank 1 tensor) of first elements to write to dst.
    zero_unused: Whether to zero unreferenced dst elements.

  Returns:
    The destination tensor dst.
  """
  dst = ops.convert_to_tensor(dst)
  src = ops.convert_to_tensor(src)
  num_elems = ops.convert_to_tensor(num_elems)
  src_offsets = ops.convert_to_tensor(src_offsets)
  dst_offsets = ops.convert_to_tensor(dst_offsets)

  n = num_elems.shape[0]
  if n == 0 or src_offsets.shape[0] != n or dst_offsets.shape[0] != n:
    raise ValueError("num_elems, src_offsets and dst_offsets must have "
                     "matching, nonzero shapes of rank 1.")

  if src.shape[1:] != dst.shape[1:]:
    raise ValueError("src and dst inner shapes must match.")

  return gen_popops_ops.ipu_sequence_slice(dst=dst,
                                           src=src,
                                           num_elems=num_elems,
                                           src_offsets=src_offsets,
                                           dst_offsets=dst_offsets,
                                           zero_unused=zero_unused)


def sequence_slice_unpack(src, num_elems, src_offsets, total_elements):
  """This op specialises the PopLibs SequenceSlice operation for
  sequence unpacking.

  The SequenceSliceUnpack operation unpacks specified elements from the
  source tensor and inserts them contiguously into the resulting tensor.

  The parameters of the slice operation are defined by the number of elements
  to take for each slice `num_elems` and the offset in the source tensor from
  which to take them `src_offsets`.

  For each slice, an element count and source offset must be provided. The i-th
  entry of `num_elems` corresponds to the i-th entry of `src_offsets`.

  For example:

  .. code-block:: none

    from tensorflow.python.ipu.ops.slicing_ops import sequence_slice_unpack

    src = [[0, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2, 2],
           [3, 3, 3, 3, 3, 3],
           [4, 4, 4, 4, 4, 4],
           [5, 5, 5, 5, 5, 5]]

    num_elems = [2, 2]
    src_offsets = [2, 1]
    total_elements = 4

    dst = sequence_slice_unpack(src, num_elems, src_offsets,
                                False, total_elements)

  Following which, the contents of the destination tensor `dst` are as follows:

  .. code-block:: none

    [[2. 2. 2. 2. 2. 2.]
     [3. 3. 3. 3. 3. 3.]
     [1. 1. 1. 1. 1. 1.]
     [2. 2. 2. 2. 2. 2.]]

  In this example, the first slice takes two elements from index 2 of the
  source tensor and inserts them at index 0 of the output tensor. The
  second slice also takes two elements, but from index 1 of the source tensor,
  inserting them at index 2 in the output tensor.

  Args:
    src: The source tensor from which the values are accessed, must be of at
      least rank 2.
    num_elems: A list (or rank 1 tensor) of the number of elements to copy.
    src_offsets: A list (or rank 1 tensor) of first elements to read from src.
    total_elements: Total number of elements to slice.

  Returns:
    The unpacked sequences.
  """
  src = ops.convert_to_tensor(src)
  num_elems = ops.convert_to_tensor(num_elems)
  src_offsets = ops.convert_to_tensor(src_offsets)

  n = num_elems.shape[0]
  if n == 0 or src_offsets.shape[0] != n:
    raise ValueError("num_elems and src_offsets must have "
                     "matching, nonzero shapes of rank 1.")

  dst_offsets = math_ops.cumsum(array_ops.concat([[0], num_elems[:-1]], 0))

  return gen_popops_ops.ipu_sequence_slice_unpack(
      src=src,
      num_elems=num_elems,
      src_offsets=src_offsets,
      dst_offsets=dst_offsets,
      total_elements=total_elements)


def sequence_slice_pack(dst, src, num_elems, dst_offsets, zero_unused):
  """This op specialises the PopLibs SequenceSlice operation for
  sequence packing.

  The SequenceSlicePack operation takes a contiguous tensor of sequences (
  such as the output of `sequence_slice_unpack`) and packs its elements
  into specified locations in the destination tensor.

  The parameters of the slice operation are defined by the number of elements
  to take for each slice `num_elems` and the offset in the destination tensor
  into which the elements should be placed, `dst_offsets`.

  For each slice, an element count and destination offset must
  be provided. The i-th entry of `num_elems` corresponds to the i-th entry of
  `dst_offsets`.

  For example:

  .. code-block:: none

    from tensorflow.python.framework.ops import array_ops
    from tensorflow.python.ipu.ops.slicing_ops import sequence_slice_pack

    src = [[2, 2, 2, 2, 2, 2],
           [3, 3, 3, 3, 3, 3],
           [1, 1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2, 2]]

    num_elems = [2, 2]
    dst_offsets = [2, 1]

    dst = array_ops.zeros([6, 6])
    dst = sequence_slice_pack(dst, src, num_elems, dst_offsets,
                              False)

  Following which, the contents of the destination tensor `dst` are as follows:

  .. code-block:: none

    [[0. 0. 0. 0. 0. 0.]
     [1. 1. 1. 1. 1. 1.]
     [2. 2. 2. 2. 2. 2.]
     [3. 3. 3. 3. 3. 3.]
     [0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0.]]

  In this example, the first slice takes the first two elements of the
  source tensor and inserts them at index 2 in the destination tensor. The
  second slice takes the next two elements in the source tensor, and inserts
  them at index 1 of destination tensor.

  Args:
    dst: The destination tensor which will be updated, must be of at least
      rank 2 with inner dimensions matching that of `src`.
    src: The source tensor from which the values are accessed, must be of
      at least rank 2.
    num_elems: A list (or rank 1 tensor) of the number of elements to copy.
    dst_offsets: A list (or rank 1 tensor) of first elements to write to dst.
    zero_unused: Whether to zero unreferenced dst elements.

  Returns:
    The packed sequences.
  """
  dst = ops.convert_to_tensor(dst)
  src = ops.convert_to_tensor(src)
  num_elems = ops.convert_to_tensor(num_elems)
  dst_offsets = ops.convert_to_tensor(dst_offsets)

  n = num_elems.shape[0]
  if n == 0 or dst_offsets.shape[0] != n:
    raise ValueError("num_elems and dst_offsets must have "
                     "matching, nonzero shapes of rank 1.")

  if src.shape[1:] != dst.shape[1:]:
    raise ValueError("src and dst inner shapes must match.")

  # Compute src offsets - output offset for slice n is the
  # number of slices up to slice n-1.
  src_offsets = math_ops.cumsum(array_ops.concat([[0], num_elems[:-1]], 0))

  # Equal to a simple sequence slice.
  return gen_popops_ops.ipu_sequence_slice(dst=dst,
                                           src=src,
                                           num_elems=num_elems,
                                           src_offsets=src_offsets,
                                           dst_offsets=dst_offsets,
                                           zero_unused=zero_unused)
