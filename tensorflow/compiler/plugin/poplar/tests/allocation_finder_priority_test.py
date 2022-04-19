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
from absl.testing import parameterized
import pva
from tensorflow.python.ipu import test_utils as tu

from tensorflow.python import ipu
from tensorflow.compiler.tests import xla_test
from tensorflow.python.platform import googletest
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables


def _matmulAndEmbeddingFwd(ids, transpose_matmul):
  VOCAB_SIZE = 20000
  EMBEDDED_SIZE = 128
  shape = [VOCAB_SIZE, EMBEDDED_SIZE] \
     if transpose_matmul else [EMBEDDED_SIZE, VOCAB_SIZE]
  with variable_scope.variable_scope("embedding_and_matmul",
                                     use_resource=True):
    shared_params = variable_scope.get_variable("shared_weights",
                                                shape,
                                                dtype=np.float16)

  if transpose_matmul:
    out = ipu.embedding_ops.embedding_lookup(shared_params, ids)
    logits = math_ops.matmul(out, shared_params, transpose_b=True)
  else:
    out = ipu.embedding_ops.embedding_lookup(
        array_ops.transpose(shared_params), ids)
    logits = math_ops.matmul(out, shared_params)
  return logits


# pylint: disable=abstract-method
class AllocationFinderPriorityTest(xla_test.XLATestCase,
                                   parameterized.TestCase):

  # Run the same test with the transpose on either the embedding or the matmul.
  # The memory usage should be similar.
  @parameterized.parameters([True, False])
  def testMatmulAndEmbedding(self, transpose_matmul):
    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.configure_ipu_system()

    with self.session() as sess:

      def model(ids):
        return _matmulAndEmbeddingFwd(ids, transpose_matmul=transpose_matmul)

      ids_ph = array_ops.placeholder(np.int32, shape=[50])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        output = ipu.ipu_compiler.compile(model, [ids_ph])

      tu.move_variable_initialization_to_cpu()
      sess.run(variables.global_variables_initializer())

      sess.run(output, {ids_ph: np.ones([50])})

    report = pva.openReport(report_helper.find_report())
    self.assert_total_tile_memory(report, 17649232)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=2 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
