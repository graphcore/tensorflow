# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

import test_utils as tu

from tensorflow.compiler.tests import xla_test
from tensorflow.python import ipu
from tensorflow.python.framework import dtypes
from tensorflow.python.ipu.optimizers import sharded_optimizer
from tensorflow.python.keras import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent as gd


class WhileLoopShardedTest(xla_test.XLATestCase):
  def testSimpleXlaCompileTrainingInLoopWithParam(self):
    with self.session() as sess:
      dataset = tu.create_dual_increasing_dataset(3)

      infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)

      def my_net(lr):
        def my_model(lr, loss, x, y):
          with ipu.scopes.ipu_scope("/device:IPU:0"):
            with ipu.scopes.ipu_shard(0):
              x = layers.Conv2D(8,
                                3,
                                padding='same',
                                name="conv1",
                                use_bias=False)(x)
              x = math_ops.reduce_max(x, axis=[1, 2])
              cross_entropy = nn.softmax_cross_entropy_with_logits_v2(
                  logits=x, labels=array_ops.stop_gradient(y))
              loss = math_ops.reduce_mean(cross_entropy)

            with ipu.scopes.ipu_shard(1):
              optim = sharded_optimizer.ShardedOptimizer(
                  gd.GradientDescentOptimizer(lr))
              train = optim.minimize(cross_entropy)

              return [lr, loss, train]

        loss = 0.0
        return ipu.loops.repeat(2, my_model, [lr, loss], infeed_queue)

      lr = array_ops.placeholder(dtypes.float32, [])
      out = ipu.ipu_compiler.compile(my_net, inputs=[lr])

      cfg = ipu.config.IPUConfig()
      cfg.ipu_model.compile_ipu_code = False
      cfg.auto_select_ipus = 2
      cfg.configure_ipu_system()
      tu.move_variable_initialization_to_cpu()

      sess.run(infeed_queue.initializer)
      sess.run(variables.global_variables_initializer())
      sess.run(out[0], {lr: 0.1})


if __name__ == "__main__":
  googletest.main()
