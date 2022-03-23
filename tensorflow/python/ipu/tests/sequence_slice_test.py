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
# =============================================================================

import numpy as np
from absl.testing import parameterized

from tensorflow.python import ipu
from tensorflow.python.client import session as sl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ipu import loops
from tensorflow.python.ipu.ops.slicing_ops import sequence_slice
from tensorflow.python.ipu.ops.slicing_ops import sequence_slice_unpack
from tensorflow.python.ipu.ops.slicing_ops import sequence_slice_pack
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent


def get_sequences(num_sequences, sequence_length, dtype):
  seqs = [np.ones(sequence_length) * i for i in range(num_sequences)]
  return np.vstack(seqs).astype(dtype)


TEST_CASES = [
    # Test different dtypes without zeroing.
    {
        'testcase_name': "TestCase1",
        'sequences': get_sequences(6, 6, np.float16),
        'output_shape': [6, 6],
        'num_elems': [2, 2],
        'src_offsets': [2, 1],
        'dst_offsets': [0, 4],
        'zero_unused': False
    },
    {
        'testcase_name': "TestCase2",
        'sequences': get_sequences(6, 6, np.float32),
        'output_shape': [6, 6],
        'num_elems': [2, 2],
        'src_offsets': [2, 1],
        'dst_offsets': [0, 4],
        'zero_unused': False
    },
    {
        'testcase_name': "TestCase3",
        'sequences': get_sequences(6, 6, np.int32),
        'output_shape': [6, 6],
        'num_elems': [2, 2],
        'src_offsets': [2, 1],
        'dst_offsets': [0, 4],
        'zero_unused': False
    },
    # Test different dtypes with zeroing.
    {
        'testcase_name': "TestCase4",
        'sequences': get_sequences(6, 6, np.float16),
        'output_shape': [6, 6],
        'num_elems': [2, 2],
        'src_offsets': [2, 1],
        'dst_offsets': [0, 4],
        'zero_unused': True
    },
    {
        'testcase_name': "TestCase5",
        'sequences': get_sequences(6, 6, np.float32),
        'output_shape': [6, 6],
        'num_elems': [2, 2],
        'src_offsets': [2, 1],
        'dst_offsets': [0, 4],
        'zero_unused': True
    },
    {
        'testcase_name': "TestCase6",
        'sequences': get_sequences(6, 6, np.int32),
        'output_shape': [6, 6],
        'num_elems': [2, 2],
        'src_offsets': [2, 1],
        'dst_offsets': [0, 4],
        'zero_unused': True
    },
    # Test simple identity case.
    {
        'testcase_name': "TestCase7",
        'sequences': get_sequences(3, 3, np.float16),
        'output_shape': [3, 3],
        'num_elems': [1, 1, 1],
        'src_offsets': [0, 1, 2],
        'dst_offsets': [0, 1, 2],
        'zero_unused': True
    },
    # Test larger output shape.
    {
        'testcase_name': "TestCase8",
        'sequences': get_sequences(3, 3, np.float16),
        'output_shape': [9, 3],
        'num_elems': [1, 1, 1],
        'src_offsets': [0, 1, 2],
        'dst_offsets': [1, 3, 5],
        'zero_unused': True
    },
]


class PopOpsSequenceSliceTest(test_util.TensorFlowTestCase,
                              parameterized.TestCase):
  @parameterized.named_parameters(*TEST_CASES)
  @test_util.deprecated_graph_mode_only
  def testSlice(self, sequences, output_shape, num_elems, src_offsets,
                dst_offsets, zero_unused):
    with sl.Session() as sess:
      cfg = ipu.config.IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      with ops.device('cpu'):
        input_ph = array_ops.placeholder(sequences.dtype, sequences.shape)
        dst_ph = array_ops.placeholder(sequences.dtype, output_shape)

      with ipu.scopes.ipu_scope("/device:IPU:0"):

        def f(y, x):
          return sequence_slice(y, x, num_elems, src_offsets, dst_offsets,
                                zero_unused)

        f_compiled = ipu.ipu_compiler.compile(f, inputs=[dst_ph, input_ph])

        dst = np.zeros(output_shape, dtype=sequences.dtype)
        res = sess.run(f_compiled, {dst_ph: dst, input_ph: sequences})

    dst_out = res[0]
    for i, j, k in zip(num_elems, src_offsets, dst_offsets):
      src_row = sequences[j:j + i, :]
      dst_row = dst_out[k:k + i, :]
      self.assertAllEqual(src_row, dst_row)

  @parameterized.named_parameters(*TEST_CASES)
  @test_util.deprecated_graph_mode_only
  def testSliceBackwardPass(self, sequences, output_shape, num_elems,
                            src_offsets, dst_offsets, zero_unused):
    if sequences.dtype == np.int32:
      return

    with sl.Session() as sess:
      cfg = ipu.config.IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      with ops.device('cpu'):
        input_ph = array_ops.placeholder(sequences.dtype, sequences.shape)
        dst_ph = array_ops.placeholder(sequences.dtype, output_shape)

      with ipu.scopes.ipu_scope("/device:IPU:0"):

        opt = gradient_descent.GradientDescentOptimizer(learning_rate=0.1)

        def f(y, x):
          out = sequence_slice(y, x, num_elems, src_offsets, dst_offsets,
                               zero_unused)

          cost = math_ops.square(out)
          grads = opt.compute_gradients(cost, x)

          return out, grads

        f_compiled = ipu.ipu_compiler.compile(f, inputs=[dst_ph, input_ph])

        dst = np.zeros(output_shape, dtype=sequences.dtype)
        res = sess.run([f_compiled], {dst_ph: dst, input_ph: sequences})

    grad_out = res[0][1][0][0]
    self.assertEqual(grad_out.shape, sequences.shape)

  @parameterized.named_parameters(*TEST_CASES)
  @test_util.deprecated_graph_mode_only
  def testUnpack(
      self,
      sequences,
      output_shape,  #pylint: disable=unused-argument
      num_elems,
      src_offsets,
      dst_offsets,  #pylint: disable=unused-argument
      zero_unused):  #pylint: disable=unused-argument
    with sl.Session() as sess:
      cfg = ipu.config.IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      with ops.device('cpu'):
        input_ph = array_ops.placeholder(sequences.dtype, sequences.shape)

      total_elements = sum(num_elems)

      with ipu.scopes.ipu_scope("/device:IPU:0"):

        def f(x):
          return sequence_slice_unpack(x, num_elems, src_offsets,
                                       total_elements)

        f_compiled = ipu.ipu_compiler.compile(f, inputs=[input_ph])
        res = sess.run(f_compiled, {input_ph: sequences})

    out = res[0]
    k = 0
    for i, j in zip(num_elems, src_offsets):
      out_slice = out[k:k + i, :]
      src_slice = sequences[j:j + i, :]
      self.assertAllEqual(out_slice, src_slice)
      k += i

  @parameterized.named_parameters(*TEST_CASES)
  @test_util.deprecated_graph_mode_only
  def testUnpackBackwardPass(
      self,
      sequences,
      output_shape,  #pylint: disable=unused-argument
      num_elems,
      src_offsets,
      dst_offsets,  #pylint: disable=unused-argument
      zero_unused):  #pylint: disable=unused-argument
    if sequences.dtype == np.int32:
      return

    with sl.Session() as sess:
      cfg = ipu.config.IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      with ops.device('cpu'):
        input_ph = array_ops.placeholder(sequences.dtype, sequences.shape)

      total_elements = sum(num_elems)

      with ipu.scopes.ipu_scope("/device:IPU:0"):

        opt = gradient_descent.GradientDescentOptimizer(learning_rate=0.1)

        def f(x):
          out = sequence_slice_unpack(x, num_elems, src_offsets,
                                      total_elements)

          cost = math_ops.square(out)
          grads = opt.compute_gradients(cost, x)

          return out, grads

        f_compiled = ipu.ipu_compiler.compile(f, inputs=[input_ph])

        res = sess.run([f_compiled], {input_ph: sequences})

    grad_out = res[0][1][0][0]
    self.assertEqual(grad_out.shape, sequences.shape)

  @parameterized.named_parameters(*TEST_CASES)
  @test_util.deprecated_graph_mode_only
  def testPack(
      self,
      sequences,
      output_shape,  #pylint: disable=unused-argument
      num_elems,
      src_offsets,
      dst_offsets,  #pylint: disable=unused-argument
      zero_unused):
    with sl.Session() as sess:
      cfg = ipu.config.IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      # First unpack the sequences contiguously.
      with ops.device('cpu'):
        input_ph = array_ops.placeholder(sequences.dtype, sequences.shape)

      total_elements = sum(num_elems)

      with ipu.scopes.ipu_scope("/device:IPU:0"):

        def f(x):
          return sequence_slice_unpack(x, num_elems, src_offsets,
                                       total_elements)

        f_compiled = ipu.ipu_compiler.compile(f, inputs=[input_ph])
        res_unpack = sess.run(f_compiled, {input_ph: sequences})

      # Now pack the result of the unpack operation.
      with ops.device('cpu'):
        pack_src_ph = array_ops.placeholder(sequences.dtype,
                                            res_unpack[0].shape)
        pack_dst_ph = array_ops.placeholder(sequences.dtype, sequences.shape)

      with ipu.scopes.ipu_scope("/device:IPU:0"):

        def g(y, x):
          # Reuse src_offsets here as dst_offsets.
          # Pack is the inverse of unpack.
          return sequence_slice_pack(y, x, num_elems, src_offsets, zero_unused)

        pack_dst = np.zeros(sequences.shape, dtype=sequences.dtype)
        g_compiled = ipu.ipu_compiler.compile(
            g, inputs=[pack_dst_ph, pack_src_ph])
        res_pack = sess.run(g_compiled, {
            pack_dst_ph: pack_dst,
            pack_src_ph: res_unpack[0]
        })

    dst_out = res_pack[0]
    for i, j in zip(num_elems, src_offsets):
      src_row = sequences[j:j + i, :]
      dst_row = dst_out[j:j + i, :]
      self.assertAllEqual(src_row, dst_row)

  @parameterized.named_parameters(*TEST_CASES)
  @test_util.deprecated_graph_mode_only
  def testPackBackwardPass(
      self,
      sequences,
      output_shape,
      num_elems,
      src_offsets,
      dst_offsets,  #pylint: disable=unused-argument
      zero_unused):
    if sequences.dtype == np.int32:
      return

    with sl.Session() as sess:
      cfg = ipu.config.IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      with ops.device('cpu'):
        input_ph = array_ops.placeholder(sequences.dtype, sequences.shape)
        dst_ph = array_ops.placeholder(sequences.dtype, output_shape)

      with ipu.scopes.ipu_scope("/device:IPU:0"):

        opt = gradient_descent.GradientDescentOptimizer(learning_rate=0.1)

        def f(y, x):
          out = sequence_slice_pack(y, x, num_elems, src_offsets, zero_unused)

          cost = math_ops.square(out)
          grads = opt.compute_gradients(cost, x)

          return out, grads

        f_compiled = ipu.ipu_compiler.compile(f, inputs=[dst_ph, input_ph])

        dst = np.zeros(output_shape, dtype=sequences.dtype)
        res = sess.run([f_compiled], {dst_ph: dst, input_ph: sequences})

    grad_out = res[0][1][0][0]
    self.assertEqual(grad_out.shape, sequences.shape)

  @test_util.deprecated_graph_mode_only
  def testMismatchedArgsSlice(self):
    sequences = get_sequences(3, 3, np.float16)
    num_elems = [0, 1]
    src_offsets = [0]
    dst_offsets = [0, 1]

    input_ph = array_ops.placeholder(sequences.dtype, sequences.shape)
    dst_ph = array_ops.placeholder(sequences.dtype, [3, 3])

    def f(y, x):
      return sequence_slice(y, x, num_elems, src_offsets, dst_offsets, False)

    with self.assertRaisesRegex(
        ValueError, "num_elems, src_offsets and dst_offsets must have "):
      _ = ipu.ipu_compiler.compile(f, inputs=[dst_ph, input_ph])

  @test_util.deprecated_graph_mode_only
  def testMismatchedArgsUnpack(self):
    sequences = get_sequences(3, 3, np.float16)
    num_elems = [0, 1]
    src_offsets = [0]

    input_ph = array_ops.placeholder(sequences.dtype, sequences.shape)

    def f(x):
      return sequence_slice_unpack(x, num_elems, src_offsets, 0)

    with self.assertRaisesRegex(
        ValueError, "num_elems and src_offsets must have matching"):
      _ = ipu.ipu_compiler.compile(f, inputs=[input_ph])

  @test_util.deprecated_graph_mode_only
  def testMismatchedArgsPack(self):
    sequences = get_sequences(3, 3, np.float16)
    num_elems = [0]
    dst_offsets = [0, 1]

    input_ph = array_ops.placeholder(sequences.dtype, sequences.shape)
    dst_ph = array_ops.placeholder(sequences.dtype, [3, 3])

    def f(y, x):
      return sequence_slice_pack(y, x, num_elems, dst_offsets, False)

    with self.assertRaisesRegex(
        ValueError, "num_elems and dst_offsets must have matching"):
      _ = ipu.ipu_compiler.compile(f, inputs=[dst_ph, input_ph])

  @test_util.deprecated_graph_mode_only
  def testShapeMismatchSlice(self):
    sequences = get_sequences(3, 3, np.float16)
    num_elems = [0, 1]
    src_offsets = [0, 1]
    dst_offsets = [0, 1]

    input_ph = array_ops.placeholder(sequences.dtype, sequences.shape)
    dst_ph = array_ops.placeholder(sequences.dtype, [3, 2])

    def f(y, x):
      return sequence_slice(y, x, num_elems, src_offsets, dst_offsets, False)

    with self.assertRaisesRegex(ValueError,
                                "src and dst inner shapes must match."):
      _ = ipu.ipu_compiler.compile(f, inputs=[dst_ph, input_ph])

  @test_util.deprecated_graph_mode_only
  def testShapeMismatchPack(self):
    sequences = get_sequences(3, 3, np.float16)
    num_elems = [0, 1]
    dst_offsets = [0, 1]

    input_ph = array_ops.placeholder(sequences.dtype, sequences.shape)
    dst_ph = array_ops.placeholder(sequences.dtype, [3, 2])

    def f(y, x):
      return sequence_slice_pack(y, x, num_elems, dst_offsets, False)

    with self.assertRaisesRegex(ValueError,
                                "src and dst inner shapes must match."):
      _ = ipu.ipu_compiler.compile(f, inputs=[dst_ph, input_ph])

  @test_util.deprecated_graph_mode_only
  def testSequenceSliceRepeatLoop(self):
    sequences = get_sequences(1024, 128, np.float16)

    # Each loop iteration is to take 4 slices of 4 elements.
    num_elems = np.array([4] * 4, dtype=np.int32)

    # Initial src offsets.
    # src and dst should match after the loop.
    src_offsets = np.cumsum(num_elems[0:-1], dtype=np.int32)
    src_offsets = np.insert(src_offsets, 0, 0)
    dst_offsets = src_offsets

    with sl.Session() as sess:
      cfg = ipu.config.IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      with ops.device('cpu'):
        dst_ph = array_ops.placeholder(sequences.dtype, sequences.shape)
        src_ph = array_ops.placeholder(sequences.dtype, sequences.shape)
        num_elems_ph = array_ops.placeholder(num_elems.dtype, num_elems.shape)
        src_offsets_ph = array_ops.placeholder(src_offsets.dtype,
                                               src_offsets.shape)
        dst_offsets_ph = array_ops.placeholder(dst_offsets.dtype,
                                               dst_offsets.shape)

      with ipu.scopes.ipu_scope("/device:IPU:0"):

        def loop_body(dst, src, num_elems, src_offsets, dst_offsets):
          res = sequence_slice(dst, src, num_elems, src_offsets, dst_offsets,
                               False)

          # Compute offsets for next iteration.
          src_offsets_upd = src_offsets + 16
          dst_offsets_upd = src_offsets_upd

          return res, src, num_elems, src_offsets_upd, dst_offsets_upd

        def f(dst, src, num_elems, src_offsets, dst_offsets):
          return loops.repeat(
              64,
              loop_body,
              inputs=[dst, src, num_elems, src_offsets, dst_offsets])

        f_compiled = ipu.ipu_compiler.compile(f,
                                              inputs=[
                                                  dst_ph, src_ph, num_elems_ph,
                                                  src_offsets_ph,
                                                  dst_offsets_ph
                                              ])

        dst = np.zeros_like(sequences)
        res = sess.run(
            f_compiled, {
                dst_ph: dst,
                src_ph: sequences,
                num_elems_ph: num_elems,
                src_offsets_ph: src_offsets,
                dst_offsets_ph: dst_offsets
            })

    self.assertAllEqual(sequences, res[0])


if __name__ == "__main__":
  googletest.main()
