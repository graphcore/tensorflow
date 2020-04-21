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
# =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import test_utils as tu

from tensorflow.python import ipu
from tensorflow.compiler.tests import xla_test
from tensorflow.python.platform import googletest
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables


def _matmulAndEmbeddingFwd(ids):
  VOCAB_SIZE = 20000
  EMBEDDED_SIZE = 128
  with variable_scope.variable_scope("embedding_and_matmul",
                                     use_resource=True):
    shared_params = variable_scope.get_variable("shared_weights",
                                                [VOCAB_SIZE, EMBEDDED_SIZE],
                                                dtype=np.float16)

    out = ipu.embedding_ops.embedding_lookup(shared_params, ids)
    logits = math_ops.matmul(out, shared_params, transpose_b=True)
  return logits


class AllocationFinderPriorityTest(xla_test.XLATestCase):
  def testMatmulAndEmbedding(self):
    with self.session() as sess:

      def model(ids):
        return _matmulAndEmbeddingFwd(ids)

      ids_ph = array_ops.placeholder(np.int32, shape=[50])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        output = ipu.ipu_compiler.compile(model, [ids_ph])

      report = tu.ReportJSON(self, sess, compile_ipu_code=True)
      tu.move_variable_initialization_to_cpu()
      sess.run(variables.global_variables_initializer())
      report.reset()

      sess.run(output, {ids_ph: np.ones([50])})

      report.parse_log()
      report.assert_total_tile_memory(56722994)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=2 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
