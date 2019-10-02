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
import tensorflow as tf
from tensorflow.compiler.tests import xla_test
from tensorflow.python.platform import googletest

from tensorflow.python.ops import variables
from tensorflow.python import ipu


class PoprandSameSeedTest(xla_test.XLATestCase):
  def testUserOpMetadata(self):
    def my_net():
      x = tf.get_variable(
          op.__name__ + 'x',
          shape=[3],
          dtype=tf.int32,
          initializer=tf.constant_initializer(5),
          use_resource=True)
      x_as_array = tf.TensorArray(tf.int32, 3, element_shape=[]).unstack(x)

      def get_random_numbers():
        def loop_body(i, _):
          epsilon = op([5],
                       seed=tf.stack([0, x_as_array.read(i)]),
                       dtype=tf.float32)
          return tf.add(i, 1), epsilon

        condition = lambda i, _: i < 1

        _, epsilon = tf.while_loop(
            cond=condition,
            body=loop_body,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.constant(0.0, shape=[5], dtype=tf.float32)),
            parallel_iterations=1,
            maximum_iterations=1,
            back_prop=True)

        return epsilon

      output = []

      for i in range(0, 6):
        output.append(get_random_numbers())
      return output

    ops = [
        tf.random.stateless_normal, tf.random.stateless_truncated_normal,
        tf.random.stateless_uniform
    ]
    for op in ops:
      with ipu.scopes.ipu_scope('/device:IPU:0'):
        model = ipu.ipu_compiler.compile(my_net)

      cfg = ipu.utils.create_ipu_config()
      ipu.utils.configure_ipu_system(cfg)

      with tf.Session() as sess:
        sess.run(variables.global_variables_initializer())
        res = sess.run(model)

        first_val = res[0]
        for i in res:
          np.testing.assert_equal(i, first_val)


if __name__ == "__main__":
  googletest.main()
