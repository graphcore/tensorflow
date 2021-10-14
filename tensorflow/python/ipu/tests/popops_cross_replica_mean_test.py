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

import numpy as np

from absl.testing import parameterized

from tensorflow.python.client import session
from tensorflow.python import ipu
from tensorflow.python.framework import test_util
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.data.ops import dataset_ops


def cpu_cross_replica_mean(data, replica_group_size=0):
  nrows = data.shape[0]
  if replica_group_size == 0:
    replica_group_size = nrows

  out = np.zeros(shape=data.shape, dtype=data.dtype)
  acc = np.zeros(shape=data.shape[1:], dtype='float64')

  for i, row in enumerate(data):
    acc += row
    if (i + 1) % replica_group_size == 0:
      acc = acc / replica_group_size
      out[(i - replica_group_size + 1):(i + 1), :] = np.tile(
          acc, (replica_group_size, 1))
      acc = np.zeros(shape=data.shape[1:], dtype='float64')

  return out


def create_data(nrows, ncols, replica_group_size=0):
  input_ = np.arange(nrows * ncols, dtype='float32').reshape((nrows, ncols))
  output = cpu_cross_replica_mean(input_, replica_group_size)
  return input_, output


def configure_ipu(num_ipus):
  '''
  Configure the IPU
  '''
  cfg = ipu.config.IPUConfig()

  cfg.auto_select_ipus = num_ipus
  tu.add_hw_ci_connection_options(cfg)
  cfg.configure_ipu_system()


class CrossReplicaMeanTest(test_util.TensorFlowTestCase,
                           parameterized.TestCase):
  def _run_test(self, input_, replica_group_size=None):
    dataset = dataset_ops.Dataset.from_tensor_slices(input_)
    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

    def body(x):
      mean = ipu.ops.cross_replica_ops.cross_replica_mean(
          x, replica_group_size)
      outfed = outfeed_queue.enqueue(mean)
      return outfed

    def model():
      return ipu.loops.repeat(1, body, [], infeed_queue)

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      compiled_model = ipu.ipu_compiler.compile(model)

    outfed = outfeed_queue.dequeue()

    with session.Session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(infeed_queue.initializer)

      sess.run(compiled_model)
      output = sess.run(outfed)[0]
      return output

  @parameterized.named_parameters(
      {
          'testcase_name': 'Simple',
          'num_replicas': 4,
          'num_samples': 4,
          'replica_group_size': 0,
          'input_val': None
      },
      {
          'testcase_name': 'ReplicaGroups',
          'num_replicas': 4,
          'num_samples': 4,
          'replica_group_size': 2,
          'input_val': None
      },
      {
          'testcase_name': 'Overflow',
          'num_replicas': 4,
          'num_samples': 4,
          'replica_group_size': 0,
          'input_val': 65504
      },
  )
  @tu.test_uses_ipus(num_ipus=4)
  @test_util.deprecated_graph_mode_only
  def test(self, num_replicas, num_samples, replica_group_size, input_val):
    configure_ipu(num_replicas)

    if input_val is None:
      input_, ref_output = \
        create_data(num_replicas, num_samples, replica_group_size)
    else:
      input_ = \
        np.array([[input_val] * num_samples] * num_replicas, dtype='float16')
      ref_output = cpu_cross_replica_mean(input_)
    test_output = self._run_test(input_, replica_group_size)

    self.assertAllClose(ref_output, test_output)


if __name__ == "__main__":
  googletest.main()
