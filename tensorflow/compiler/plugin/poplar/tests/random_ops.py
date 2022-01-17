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
# =============================================================================

import numpy as np
from absl.testing import parameterized
import test_utils as tu
from tensorflow.compiler.tests import xla_test
from tensorflow.python.platform import googletest

import tensorflow as tf
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_stateless_random_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python import ipu


class PoprandSameSeedTest(xla_test.XLATestCase):
  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def testRandomNumbers(self):
    def my_net():
      x = variable_scope.get_variable(
          op.__name__ + 'x',
          shape=[3],
          dtype=dtypes.int32,
          initializer=init_ops.constant_initializer(5),
          use_resource=True)
      x_as_array = tensor_array_ops.TensorArray(dtypes.int32,
                                                3,
                                                element_shape=[]).unstack(x)

      def get_random_numbers():
        def loop_body(i, _):
          epsilon = op([5],
                       seed=array_ops.stack([0, x_as_array.read(i)]),
                       dtype=dtypes.float32)
          return math_ops.add(i, 1), epsilon

        condition = lambda i, _: i < 1

        _, epsilon = control_flow_ops.while_loop(
            cond=condition,
            body=loop_body,
            loop_vars=(constant_op.constant(0, dtype=dtypes.int32),
                       constant_op.constant(0.0,
                                            shape=[5],
                                            dtype=dtypes.float32)),
            parallel_iterations=1,
            maximum_iterations=1,
            back_prop=True)

        return epsilon

      output = []

      for i in range(0, 6):
        output.append(get_random_numbers())
      return output

    ops = [
        gen_stateless_random_ops.stateless_random_normal,
        gen_stateless_random_ops.stateless_truncated_normal,
        gen_stateless_random_ops.stateless_random_uniform
    ]
    for op in ops:
      with ipu.scopes.ipu_scope('/device:IPU:0'):
        model = ipu.ipu_compiler.compile(my_net)

      cfg = ipu.config.IPUConfig()
      tu.add_hw_ci_connection_options(cfg)
      cfg.configure_ipu_system()

      with session.Session() as sess:
        sess.run(variables.global_variables_initializer())
        res = sess.run(model)

        first_val = res[0]
        for i in res:
          np.testing.assert_equal(i, first_val)


# pylint: disable=abstract-method
class ShuffleOpTest(xla_test.XLATestCase, parameterized.TestCase):
  @parameterized.parameters([tf.int32, tf.float32, tf.float16])
  @test_util.deprecated_graph_mode_only
  def testShuffleOp(self, dtype):
    cfg = ipu.config.IPUConfig()
    cfg.configure_ipu_system()

    ds = tf.data.Dataset.from_tensors(
        tf.constant(1, shape=[32, 32], dtype=dtype))
    ds = ds.repeat()

    # The host side queues
    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

    def my_net():
      x = variable_scope.get_variable(
          'x',
          shape=[32, 32],
          dtype=dtype,
          initializer=init_ops.constant_initializer(5),
          use_resource=True)

      def random_shuffle(x):
        x = tf.random.shuffle(x)
        return x

      output = random_shuffle(x)
      outfeed = outfeed_queue.enqueue(output)
      return outfeed

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      xla_graph = ipu.ipu_compiler.compile(my_net, inputs=[])

    dequeue_outfeed = outfeed_queue.dequeue()

    with session.Session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(xla_graph)
      sess.run(dequeue_outfeed)


if __name__ == "__main__":
  googletest.main()
