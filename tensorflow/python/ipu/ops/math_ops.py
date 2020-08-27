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
"""
IPU specific maths operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.python.ipu.ops import functional_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import custom_gradient


def serialized_matmul(a,
                      b,
                      serialization_factor,
                      serialization_dimension,
                      transpose_a=False,
                      transpose_b=False,
                      name=None):
  """Multiplies matrix a by matrix b, producing a * b, with the multiplication
  being serialized on one of the dimensions.

  Serializing a matrix multiplication operation can reduce the code size of the
  multiplication at the expense of extra computation due to copying of tensors.

  The inputs must, following any transpositions, be tensors of rank >= 2 where
  the inner 2 dimensions specify valid matrix multiplication dimensions, and any
  further outer dimensions specify matching batch size.

  Either matrix can be transposed on the fly by setting one of the corresponding
  flag to True. These are False by default.

  Given the tensor `a` with shape `[..., m, k]` and tensor `b` with
  shape [..., k, n] *after* the transpositions, the matrix multiplication can be
  serialized as follows:

  * Along the columns dimension of `a` (the `m`-dimension), by setting
    `serialization_dimension` to `a_columns`.
  * Along the rows dimension of `a` and the columns dimension of `b`
    (the `k`-dimension), by setting `serialization_dimension` to
    `a_rows_b_columns`.
  * Along the rows dimension of `b` (the `m`-dimension), by setting
    `serialization_dimension` to `b_rows`.

  Note that taking a gradient of a serialized matrix multiplication means that
  the backward propagation of the matrix multiply will also be serialized.

  Note that adjoining and sparse matrices are not supported.

  Args:
    a: `tf.Tensor` of type float16, float32, int32 and rank >= 2.
    b: `tf.Tensor` with same type and rank as a.
    serialization_factor: An integer indicating the number of smaller matrix
      multiplies this operation is broken up into. Must divide the dimension
      along which the operation is serialized on.
    serialization_dimension: A string, must be one of `a_columns`,
      `a_rows_b_columns` or `b_rows`. Indicates the dimension along which the
      operation is serialzed on.
    transpose_a: If True, a is transposed before multiplication.
    transpose_b: If True, b is transposed before multiplication.
    name: Name for the operation (optional).

  Returns:
    A `tf.Tensor` of the same type as a and b where each inner-most matrix is
    the product of the corresponding matrices in a and b, e.g. if all transpose
    attributes are False:

    output[..., i, j] = sum_k (a[..., i, k] * b[..., k, j]), for all indices i,
    j.
  """
  serialization_factor = int(serialization_factor)
  if serialization_factor < 1:
    raise ValueError(
        'serialization_factor has to be at least 1, but was {}.'.format(
            serialization_factor))

  name = name or "serialized_matmul"

  if len(a.shape.as_list()) < 2:
    raise ValueError('Expected tensor \'a\' to have a minimum rank of 2')

  if len(b.shape.as_list()) < 2:
    raise ValueError('Expected tensor \'b\' to have a minimum rank of 2')

  # Function which slices a tensor along a single dimension.
  def slice_tensor(tensor, slice_dim, start_idx, slice_size):
    shape = tensor.shape.as_list()
    slice_starts = [
        start_idx if d == slice_dim else 0 for d in range(len(shape))
    ]
    slice_sizes = [
        slice_size if d == slice_dim else size for d, size in enumerate(shape)
    ]
    return array_ops.slice(tensor, slice_starts, slice_sizes)

  # Function which removes broadcasted dimensions
  def remove_broadcasting_dimensions(tensor,
                                     output_shape=None,
                                     output_reduction_axis=None):
    if output_reduction_axis is not None:
      tensor = array_ops.reshape(
          math_ops.reduce_sum(tensor, output_reduction_axis), output_shape)
    return tensor

  # Define the function for splitting matmul on columns of a.
  def matmul_split_a_cols(lhs,
                          rhs,
                          transpose_lhs,
                          transpose_rhs,
                          name,
                          output_shape=None,
                          output_reduction_axis=None):
    name = name + "SplitAColumns"

    @functional_ops.function
    def inner_func(lhs_, rhs_):
      return math_ops.matmul(lhs_,
                             rhs_,
                             transpose_lhs,
                             transpose_rhs,
                             name=name)

    lhs_shape = lhs.shape.as_list()
    # Get the slice dimension, taking transpose into account.
    slice_dim = len(lhs_shape) - (1 if transpose_lhs else 2)
    slice_dim_size = lhs_shape[slice_dim]

    if (slice_dim_size % serialization_factor) != 0:
      raise ValueError(
          'Expected \'serialization_factor\' ({}) to divide the columns '
          'dimension of \'a\' ({}).'.format(serialization_factor,
                                            slice_dim_size))

    slice_size = slice_dim_size // serialization_factor

    output_slice_shape = None
    # Given the output shape, find the shape of the slice.
    if output_reduction_axis is not None:
      assert output_shape
      output_slice_shape = list(output_shape)
      output_slice_shape[-2] = slice_size

    result = []
    for i in range(0, serialization_factor):
      lhs_slice = slice_tensor(lhs, slice_dim, i * slice_size, slice_size)
      output_slice = inner_func(lhs_slice, rhs)

      # Collapse any batch dimensions.
      output_slice = remove_broadcasting_dimensions(output_slice,
                                                    output_slice_shape,
                                                    output_reduction_axis)

      result.append(output_slice)
    return array_ops.concat(result, axis=-2)

  # Define the function for splitting matmul on rows of a/columns of b.
  def matmul_split_a_rows_b_cols(lhs,
                                 rhs,
                                 transpose_lhs,
                                 transpose_rhs,
                                 name,
                                 output_shape=None,
                                 output_reduction_axis=None):
    name = name + 'SplitARowsBColumns'

    @functional_ops.function
    def inner_func(lhs_, rhs_):
      return math_ops.matmul(lhs_,
                             rhs_,
                             transpose_lhs,
                             transpose_rhs,
                             name=name)

    lhs_shape = lhs.shape.as_list()
    rhs_shape = rhs.shape.as_list()

    # Get the slice dimensions, taking transpose into account.
    lhs_slice_dim = len(lhs_shape) - (2 if transpose_lhs else 1)
    lhs_slice_dim_size = lhs_shape[lhs_slice_dim]

    rhs_slice_dim = len(rhs_shape) - (1 if transpose_rhs else 2)
    rhs_slice_dim_size = rhs_shape[rhs_slice_dim]

    if lhs_slice_dim_size != rhs_slice_dim_size:
      raise ValueError(
          'Expected the number of columns in \'a\' ({}) to match the number of '
          'rows in \'b\' ({}).'.format(lhs_slice_dim_size, rhs_slice_dim_size))

    if (lhs_slice_dim_size % serialization_factor) != 0:
      raise ValueError(
          'Expected \'serialization_factor\' ({}) to divide the rows dimension '
          'of \'a\' ({}).'.format(serialization_factor, lhs_slice_dim_size))

    slice_size = lhs_slice_dim_size // serialization_factor
    # Do the first slice.
    lhs_slice = slice_tensor(lhs, lhs_slice_dim, 0, slice_size)
    rhs_slice = slice_tensor(rhs, rhs_slice_dim, 0, slice_size)
    result = inner_func(lhs_slice, rhs_slice)

    for i in range(1, serialization_factor):
      lhs_slice = slice_tensor(lhs, lhs_slice_dim, i * slice_size, slice_size)
      rhs_slice = slice_tensor(rhs, rhs_slice_dim, i * slice_size, slice_size)
      result += inner_func(lhs_slice, rhs_slice)

    # Collapse any batch dimensions.
    result = remove_broadcasting_dimensions(result, output_shape,
                                            output_reduction_axis)

    return result

  # Define the function for splitting matmul on rows of b.
  def matmul_split_b_rows(lhs,
                          rhs,
                          transpose_lhs,
                          transpose_rhs,
                          name,
                          output_shape=None,
                          output_reduction_axis=None):
    name = name + 'SplitBRows'

    @functional_ops.function
    def inner_func(lhs_, rhs_):
      return math_ops.matmul(lhs_,
                             rhs_,
                             transpose_lhs,
                             transpose_rhs,
                             name=name)

    # Get the slice dimension, taking transpose into account.
    rhs_shape = rhs.shape.as_list()
    slice_dim = len(rhs_shape) - (2 if transpose_rhs else 1)
    slice_dim_size = rhs_shape[slice_dim]

    if (slice_dim_size % serialization_factor) != 0:
      raise ValueError(
          'Expected \'serialization_factor\' ({}) to divide the rows dimension '
          'of \'b\' ({}).'.format(serialization_factor, slice_dim_size))

    slice_size = slice_dim_size // serialization_factor

    output_slice_shape = None
    # Given the output shape, find the shape of the slice.
    if output_reduction_axis is not None:
      assert output_shape
      output_slice_shape = list(output_shape)
      output_slice_shape[-1] = slice_size

    result = []
    for i in range(0, serialization_factor):
      rhs_slice = slice_tensor(rhs, slice_dim, i * slice_size, slice_size)
      output_slice = inner_func(lhs, rhs_slice)

      # Collapse any batch dimensions.
      output_slice = remove_broadcasting_dimensions(output_slice,
                                                    output_slice_shape,
                                                    output_reduction_axis)

      result.append(output_slice)
    return array_ops.concat(result, axis=-1)

  # Define the fwd/bwd functions in such a way that the backprop operations
  # are split too and minimize the number of transposes done.
  if serialization_dimension == 'a_columns':
    fwd_fn = matmul_split_a_cols
    grad_b_fn = matmul_split_a_rows_b_cols
    if not transpose_a and not transpose_b:
      grad_a_fn = matmul_split_a_cols
    elif not transpose_a and transpose_b:
      grad_a_fn = matmul_split_a_cols
    elif transpose_a and not transpose_b:
      grad_a_fn = matmul_split_b_rows
    elif transpose_a and transpose_b:
      grad_a_fn = matmul_split_b_rows

  elif serialization_dimension == 'a_rows_b_columns':
    fwd_fn = matmul_split_a_rows_b_cols
    if not transpose_a and not transpose_b:
      grad_a_fn = matmul_split_b_rows
      grad_b_fn = matmul_split_a_cols
    elif not transpose_a and transpose_b:
      grad_a_fn = matmul_split_b_rows
      grad_b_fn = matmul_split_b_rows
    elif transpose_a and not transpose_b:
      grad_a_fn = matmul_split_a_cols
      grad_b_fn = matmul_split_a_cols
    elif transpose_a and transpose_b:
      grad_a_fn = matmul_split_a_cols
      grad_b_fn = matmul_split_b_rows

  elif serialization_dimension == 'b_rows':
    fwd_fn = matmul_split_b_rows
    grad_a_fn = matmul_split_a_rows_b_cols
    if not transpose_a and not transpose_b:
      grad_b_fn = matmul_split_b_rows
    elif not transpose_a and transpose_b:
      grad_b_fn = matmul_split_a_cols
    elif transpose_a and not transpose_b:
      grad_b_fn = matmul_split_b_rows
    elif transpose_a and transpose_b:
      grad_b_fn = matmul_split_a_cols

  else:
    raise ValueError('Invalid serialization_dimension={}, expected one of: '
                     '\'a_columns\', \'a_rows_b_columns\', \'b_rows\'.'.format(
                         serialization_dimension))

  @custom_gradient.custom_gradient
  def _matmul(lhs, rhs):
    def grad_fn(grad):
      grad_lhs_name = name + 'GradA'
      grad_rhs_name = name + 'GradB'

      # Reduce along the broadcasted batch dimensions, if broadcasting is
      # required.
      lhs_shape = lhs.shape.as_list()
      rhs_shape = rhs.shape.as_list()

      lhs_reduction = None
      rhs_reduction = None
      if lhs_shape != rhs_shape:
        lhs_reduction, rhs_reduction = gen_array_ops.broadcast_gradient_args(
            lhs_shape[:-2], rhs_shape[:-2])

      if not transpose_a and not transpose_b:
        grad_lhs = grad_a_fn(grad, rhs, False, True, grad_lhs_name, lhs_shape,
                             lhs_reduction)
        grad_rhs = grad_b_fn(lhs, grad, True, False, grad_rhs_name, rhs_shape,
                             rhs_reduction)
      elif not transpose_a and transpose_b:
        grad_lhs = grad_a_fn(grad, rhs, False, False, grad_lhs_name, lhs_shape,
                             lhs_reduction)
        grad_rhs = grad_b_fn(grad, lhs, True, False, grad_rhs_name, rhs_shape,
                             rhs_reduction)
      elif transpose_a and not transpose_b:
        grad_lhs = grad_a_fn(rhs, grad, False, True, grad_lhs_name, lhs_shape,
                             lhs_reduction)
        grad_rhs = grad_b_fn(lhs, grad, False, False, grad_rhs_name, rhs_shape,
                             rhs_reduction)
      elif transpose_a and transpose_b:
        grad_lhs = grad_a_fn(rhs, grad, True, True, grad_lhs_name, lhs_shape,
                             lhs_reduction)
        grad_rhs = grad_b_fn(grad, lhs, True, True, grad_rhs_name, rhs_shape,
                             rhs_reduction)
      return [grad_lhs, grad_rhs]

    return fwd_fn(lhs, rhs, transpose_a, transpose_b, name), grad_fn

  return _matmul(a, b)
