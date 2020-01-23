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
# ==============================================================================
"""
Gradients for Popops cross replica operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.compiler.plugin.poplar.ops import gen_popops_ops
from tensorflow.python.framework import ops


@ops.RegisterGradient("IpuAllToAll")
def _ipu_all_to_all_grad(op, grads):
  """Gradients for the IpuAllToAll op."""
  return [
      gen_popops_ops.ipu_all_to_all(
          grads,
          split_dimension=op.get_attr("concat_dimension"),
          concat_dimension=op.get_attr("split_dimension"),
          number_of_replicas=op.get_attr("number_of_replicas"),
          name=op.get_attr("name") + "_grad")
  ]
