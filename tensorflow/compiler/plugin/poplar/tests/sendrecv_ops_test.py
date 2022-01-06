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
import numpy as np

from absl.testing import parameterized
from tensorflow.compiler.plugin.poplar.ops import gen_sendrecv_ops
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import utils
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest


class IpuSendRecvOpsTest(xla_test.XLATestCase, parameterized.TestCase):  # pylint: disable=abstract-method
  @combinations.generate(
      combinations.combine(
          dtype=[dtypes.float16, dtypes.float32, dtypes.int32]))
  def testSendScalar(self, dtype):
    with self.session() as sess:

      def device_fn(x):
        return gen_sendrecv_ops.ipu_send_to_host(x,
                                                 tensor_name="test_tensor",
                                                 send_device="/device:IPU:0",
                                                 send_device_incarnation=0,
                                                 recv_device="/device:CPU:0")

      inputs = array_ops.placeholder(dtype=dtype, shape=())

      with ipu_scope("/device:IPU:0"):
        send_op = ipu_compiler.compile(device_fn, inputs=[inputs])

      with ops.device("/device:CPU:0"):
        recv_op = gen_sendrecv_ops.ipu_recv_at_host(
            T=dtype,
            tensor_name="test_tensor",
            send_device="/device:IPU:0",
            send_device_incarnation=0,
            recv_device="/device:CPU:0")

      opts = IPUConfig()
      opts.configure_ipu_system()

      sent, received = sess.run([send_op, recv_op], feed_dict={inputs: 1})

      self.assertIsNone(sent)  # Send op has no output
      self.assertEqual(dtype, received.dtype)
      self.assertEqual(0, len(received.shape))
      self.assertEqual(1, received)

  def testSendFromTwoEngines(self):
    with self.session() as sess:

      def make_device_fn(i):
        def device_fn(x):
          return gen_sendrecv_ops.ipu_send_to_host(
              x,
              tensor_name="tensor_{}".format(i),
              send_device="/device:IPU:0",
              send_device_incarnation=0,
              recv_device="/device:CPU:0")

        return device_fn

      input_1 = array_ops.placeholder(dtype=dtypes.float32, shape=())
      input_2 = array_ops.placeholder(dtype=dtypes.float32, shape=())

      with ipu_scope("/device:IPU:0"):
        send_1 = ipu_compiler.compile(make_device_fn(1), inputs=[input_1])
        send_2 = ipu_compiler.compile(make_device_fn(2), inputs=[input_2])

      with ops.device("/device:CPU:0"):
        recv_1 = gen_sendrecv_ops.ipu_recv_at_host(T=dtypes.float32,
                                                   tensor_name="tensor_1",
                                                   send_device="/device:IPU:0",
                                                   send_device_incarnation=0,
                                                   recv_device="/device:CPU:0")
        recv_2 = gen_sendrecv_ops.ipu_recv_at_host(T=dtypes.float32,
                                                   tensor_name="tensor_2",
                                                   send_device="/device:IPU:0",
                                                   send_device_incarnation=0,
                                                   recv_device="/device:CPU:0")

      opts = IPUConfig()
      opts.configure_ipu_system()

      # Test it a couple of times to verify the communication channel is reusable.
      for i in range(2):
        _, _, result_1, result_2 = sess.run([send_1, send_2, recv_1, recv_2],
                                            feed_dict={
                                                input_1: i,
                                                input_2: i + 1
                                            })
        self.assertEqual(i, result_1)
        self.assertEqual(i + 1, result_2)

  @combinations.generate(
      combinations.combine(dtype=[dtypes.float16, dtypes.float32]))
  def testSendMatrices(self, dtype):
    with self.session() as sess:
      L = 3

      def device_fn(x):
        for i in range(L):
          x = math_ops.matmul(x, x)
          if i < L - 1:
            gen_sendrecv_ops.ipu_send_to_host(x,
                                              tensor_name="x_{}".format(i),
                                              send_device="/device:IPU:0",
                                              send_device_incarnation=0,
                                              recv_device="/device:CPU:0")
        return x

      N = 2
      inputs = array_ops.placeholder(dtype=dtype, shape=(N, N))

      with ipu_scope("/device:IPU:0"):
        [device_out] = ipu_compiler.compile(device_fn, inputs=[inputs])

      received = []
      with ops.device("/device:CPU:0"):
        for i in range(L - 1):
          received.append(
              gen_sendrecv_ops.ipu_recv_at_host(T=dtype,
                                                tensor_name="x_{}".format(i),
                                                send_device="/device:IPU:0",
                                                send_device_incarnation=0,
                                                recv_device="/device:CPU:0"))

      opts = IPUConfig()
      opts.configure_ipu_system()

      received_values, device_value = sess.run(
          [received, device_out], feed_dict={inputs: np.ones((N, N))})

      self.assertAllClose(2 * np.ones((N, N)), received_values[0])
      self.assertAllClose(8 * np.ones((N, N)), received_values[1])
      self.assertAllClose(128 * np.ones((N, N)), device_value)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
