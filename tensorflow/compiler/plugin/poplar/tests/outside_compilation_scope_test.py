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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pva
import sys

from absl.testing import parameterized
from tensorflow.python.ipu import test_utils as tu
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import loops
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ipu.scopes import ipu_scope, outside_compilation_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest


class OutsideCompilationScopeTest(  # pylint: disable=abstract-method
    xla_test.XLATestCase, parameterized.TestCase):
  @combinations.generate(
      combinations.combine(
          dtype=[dtypes.float16, dtypes.float32, dtypes.int32]))
  def testScalarInputOutput(self, dtype):
    with self.session() as sess:

      def device_fn(x):
        with ipu_scope("/device:IPU:0"):
          x = math_ops.cast(x * x, dtype=dtype)
          with outside_compilation_scope():
            # Use float64 which is not supported on IPU
            x = math_ops.cast(x, dtype=dtypes.float64)
            x *= constant_op.constant(2.0, dtype=dtypes.float64)
            x = math_ops.cast(x, dtype=dtype)
          x += constant_op.constant(2, dtype=dtype)
        return x

      inputs = array_ops.placeholder(dtype=dtypes.float32, shape=())
      [device_out] = ipu_compiler.compile(device_fn, inputs=[inputs])

      opts = IPUConfig()
      opts.configure_ipu_system()

      # Make sure the communication channels are reusable.
      for i in range(3):
        result = sess.run(device_out, feed_dict={inputs: i})
        self.assertEqual(i * i * 2 + 2, result)

  def testNoInputScalarOutput(self):
    with self.session() as sess:

      def device_fn(x):
        with ipu_scope("/device:IPU:0"):
          x = x * x
          with outside_compilation_scope():
            y = constant_op.constant(2.0, dtype=dtypes.float32)
            y = y * y
          z = x + y
        return z

      inputs = array_ops.placeholder(dtype=dtypes.float32, shape=())
      [device_out] = ipu_compiler.compile(device_fn, inputs=[inputs])

      opts = IPUConfig()
      opts.configure_ipu_system()

      result = sess.run(device_out, feed_dict={inputs: 2.0})
      self.assertEqual(8.0, result)

  def testNoInputTwoOutputs(self):
    with self.session() as sess:

      def device_fn(x):
        with ipu_scope("/device:IPU:0"):
          x = x * x
          with outside_compilation_scope():
            y = constant_op.constant(2.0, dtype=dtypes.float32)
            z = constant_op.constant(3.0, dtype=dtypes.float32)
            a = y * z
            b = z + y
          return x + a + b

      inputs = array_ops.placeholder(dtype=dtypes.float32, shape=())
      [device_out] = ipu_compiler.compile(device_fn, inputs=[inputs])

      opts = IPUConfig()
      opts.configure_ipu_system()

      result = sess.run(device_out, feed_dict={inputs: 2.0})
      self.assertEqual(15.0, result)

  def testScalarInputNoOutput(self):
    with self.session() as sess:

      def device_fn(x):
        with ipu_scope("/device:IPU:0"):
          x = x * x
          with outside_compilation_scope():
            logging_ops.print_v2(x, output_stream=sys.stdout, end="")
          return x

      inputs = array_ops.placeholder(dtype=dtypes.float32, shape=())
      [device_out] = ipu_compiler.compile(device_fn, inputs=[inputs])

      opts = IPUConfig()
      opts.configure_ipu_system()

      with self.captureWritesToStream(sys.stdout) as printed:
        _ = sess.run(device_out, feed_dict={inputs: 2.0})
      self.assertEqual("4", printed.contents())

  def testTwoInputsTwoOutputs(self):
    with self.session() as sess:

      def device_fn(x1, x2):
        with ipu_scope("/device:IPU:0"):
          x1 *= x1
          x2 *= x2
          with outside_compilation_scope():
            x1 += 1.0
            x2 += 2.0
          x1 *= 1.0
          x2 *= 2.0
          return x1, x2

      input1 = array_ops.placeholder(dtype=dtypes.float32, shape=())
      input2 = array_ops.placeholder(dtype=dtypes.float32, shape=())
      out1, out2 = ipu_compiler.compile(device_fn, inputs=[input1, input2])

      opts = IPUConfig()
      opts.optimizations.maximum_send_recv_cluster_size = 8
      opts.configure_ipu_system()

      res1, res2 = sess.run([out1, out2], feed_dict={input1: 1.0, input2: 2.0})
      self.assertEqual(2.0, res1)
      self.assertEqual(12.0, res2)

  def testCombineStreamCopies(self):
    with self.session() as sess:

      def with_outside_scope(x1, x2):
        with ipu_scope("/device:IPU:0"):
          x1 *= 1.0
          x2 *= 2.0
          with outside_compilation_scope():
            y1 = constant_op.constant(1.0, dtype=dtypes.float32)
            y1 += x1
            y2 = constant_op.constant(2.0, dtype=dtypes.float32)
            y2 += x2
          x1 += y1
          x2 += y2
          return x1, x2

      def without_outside_scope(x1, x2):
        with ipu_scope("/device:IPU:0"):
          x1 *= 1.0
          x2 *= 2.0
          y1 = constant_op.constant(1.0, dtype=dtypes.float32)
          y1 += x1
          y2 = constant_op.constant(2.0, dtype=dtypes.float32)
          y2 += x2
          x1 += y1
          x2 += y2
          return x1, x2

      input1 = array_ops.placeholder(dtype=dtypes.float32, shape=(2,))
      input2 = array_ops.placeholder(dtype=dtypes.float32, shape=(1,))

      compiled_with_outside_scope = ipu_compiler.compile(
          with_outside_scope, inputs=[input1, input2])

      compiled_without_outside_scope = ipu_compiler.compile(
          without_outside_scope, inputs=[input1, input2])

      cfg = IPUConfig()
      report_helper = tu.ReportHelper()
      report_helper.set_autoreport_options(cfg)
      cfg.optimizations.maximum_send_recv_cluster_size = 12
      cfg.configure_ipu_system()

      def count_stream_copies(compiled_func):
        report_helper.clear_reports()

        out1, out2 = sess.run(compiled_func, {
            input1: [1.0, 1.0],
            input2: [1.0]
        })
        self.assertAllEqual(out1, [3.0, 3.0])
        self.assertAllEqual(out2, [6.0])

        report = pva.openReport(report_helper.find_report())

        main_program = next(
            p for p in report.compilation.programs
            if p.type == pva.Program.Type.OnEveryTileSwitch).children[1]

        stream_copies = [
            p for p in main_program.children
            if p.type == pva.Program.Type.StreamCopyBegin
        ]
        return len(stream_copies)

      num_copies_without_outside_scope = count_stream_copies(
          compiled_without_outside_scope)
      num_copies_with_outside_scope = count_stream_copies(
          compiled_with_outside_scope)

      # There should be at most two new SendToHost/RecvFromHost stream copies.
      self.assertLessEqual(num_copies_with_outside_scope,
                           num_copies_without_outside_scope + 2)

  def testSentTensorIsUsedAfterReceive(self):
    with self.session() as sess:

      def device_fn(x):
        with ipu_scope("/device:IPU:0"):
          x *= x  # 4

          with outside_compilation_scope():
            y = x + 1.0  # 5

          # Use `x` after receiving `y` and make sure that we still have the correct
          # value of `x` (i.e. it is not overwritten by the receive, in which case
          # we would get 25).
          z = x * y  # 20

          return z

      inputs = array_ops.placeholder(dtype=dtypes.float32, shape=())
      [out] = ipu_compiler.compile(device_fn, inputs=[inputs])

      opts = IPUConfig()
      opts.configure_ipu_system()

      res = sess.run(out, feed_dict={inputs: 2.0})
      self.assertEqual(20.0, res)

  def testVectorInputOutput(self):
    with self.session() as sess:

      def device_fn(x):
        with ipu_scope("/device:IPU:0"):
          x = x + x
          with outside_compilation_scope():
            # Use float64 which is not supported on IPU
            x = math_ops.cast(x, dtype=dtypes.float64)
            c = constant_op.constant(2.0, dtype=dtypes.float64, shape=(2,))
            x += c
            x = math_ops.cast(x, dtype=dtypes.float32)
          x = x + 2.0
        return x

      inputs = array_ops.placeholder(dtype=dtypes.float32, shape=(2,))
      [device_out] = ipu_compiler.compile(device_fn, inputs=[inputs])

      opts = IPUConfig()
      opts.configure_ipu_system()
      result = sess.run(device_out, feed_dict={inputs: [1.0, 2.0]})
      self.assertEqual((2,), result.shape)
      self.assertAllEqual([6.0, 8.0], result)

  def testMultipleScopes(self):
    with self.session() as sess:

      def device_fn(x):
        with ipu_scope("/device:IPU:0"):
          x *= 2.0
          with outside_compilation_scope():
            x *= 2.0
          x *= 2.0
          with outside_compilation_scope():
            x *= 2.0
          x *= 2.0
        return x

      inputs = array_ops.placeholder(dtype=dtypes.float32, shape=())
      [device_out] = ipu_compiler.compile(device_fn, inputs=[inputs])

      opts = IPUConfig()
      opts.configure_ipu_system()
      result = sess.run(device_out, feed_dict={inputs: 2.0})
      self.assertEqual(64.0, result)

  def testEnclosedInLoopNotSupported(self):
    with self.session() as sess:

      def body(v):
        with outside_compilation_scope():
          v += 1.0
        return v

      def my_net(v):
        r = loops.repeat(2, body, inputs=[v])
        return r

      with ipu_scope("/device:IPU:0"):
        [res] = ipu_compiler.compile(my_net, inputs=[0.0])

      opts = IPUConfig()
      opts.configure_ipu_system()

      with self.assertRaisesRegex(
          errors_impl.UnimplementedError,
          r"`outside_compilation_scope` enclosed in control flow "
          r"\(loop or cond\) is not supported"):
        sess.run(res)

  def testCommonSubexpressionInScope(self):
    with self.session() as sess:

      # Verify that the scope works when it contains (potentially)
      # eliminated common subexpressions.
      def device_fn(x):
        with ipu_scope("/device:IPU:0"):
          x *= x
          with outside_compilation_scope():
            a = x + 1.0
            b = x + 1.0
            x = a + b
          x += 1.0
          return x

      inputs = array_ops.placeholder(dtype=dtypes.float32, shape=())
      [out] = ipu_compiler.compile(device_fn, inputs=[inputs])

      opts = IPUConfig()
      opts.configure_ipu_system()
      self.assertEqual(5.0, sess.run(out, feed_dict={inputs: 1.0}))

  def testNotInXlaContextShouldRaiseException(self):
    with self.assertRaisesRegex(ValueError, "only allowed in XLA context"):
      with outside_compilation_scope():
        pass

  def testNestedScopesShouldRaiseException(self):
    def device_fn():
      with ipu_scope("/device:IPU:0"):
        with outside_compilation_scope():
          with outside_compilation_scope():
            pass

    with self.assertRaisesRegex(ValueError, "Illegal nesting"):
      ipu_compiler.compile(device_fn, inputs=[])


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
