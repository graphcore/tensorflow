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

import os
import numpy as np
import pva

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python import ipu
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables


class RecomputeSuggestionTest(test_util.TensorFlowTestCase):
  @test_util.deprecated_graph_mode_only
  def testRecomputeSuggestion(self):
    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.allow_recompute = True
    cfg.configure_ipu_system()

    def my_model(a):
      b = array_ops.constant(np.random.rand(5, 5),
                             dtype=np.float32,
                             name="W_ih")
      c = array_ops.constant(np.random.rand(5, 5),
                             dtype=np.float32,
                             name="W_ho")
      d = a + b
      ipu.internal_ops.print_tensor(d)  # block some optimisation
      e = d + c
      ipu.internal_ops.print_tensor(e)  # block some optimisation
      f = ipu.internal_ops.recompute(e)
      g = f + f
      ipu.internal_ops.print_tensor(g)  # block some optimisation
      output = g + f

      return [output]

    with ops.device("cpu"):
      inp = array_ops.placeholder(np.float32, [5, 5], name="a")

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      out = ipu.ipu_compiler.compile(my_model, inputs=[inp])

    with tu.ipu_session(
        disable_grappler_optimizers=['arithmetic_optimization']) as sess:
      sess.run(variables.global_variables_initializer())

      sess.run(out, {inp: np.ones([5, 5])})

    report = pva.openReport(report_helper.find_report())
    # 5 adds in a graph that only defined 4
    ok = [
        '__seed*',
        'add_1/add.1/Op/Add',
        'add_2/add.10/Op/Add',
        'add_1/add.1.clone.1/Op/Add',
        'add/add.4/Op/Add',
        'add_1/add.1.clone/Op/Add',
        'add_3/add.12/Op/Add',
    ]
    self.assert_all_compute_sets_and_list(report, ok)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
