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
"""
Summary opertions for IPUs
~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.core.framework import summary_pb2
from tensorflow.python.framework import ops
from tensorflow.python.summary.summary import tensor_summary

def ipu_compile_summary(name, op, collections=None):
  """Create an IPU compiler summary operation.

  Args:
    name: A name for the summary
    op: An operation to make this summary dependent upon
    collections: Optional collections to add the summary into
    
  Returns:
    The new summary operation

  """

  with ops.device("cpu"):
    with ops.control_dependencies([op]):

      reports = gen_ipu_ops.ipu_event_trace()

      summary_metadata = summary_pb2.SummaryMetadata(
          plugin_data=summary_pb2.SummaryMetadata.PluginData(
              plugin_name="ipu"))

      t_summary = tensor_summary(
          name='ipu_trace',
          tensor=reports,
          summary_metadata=summary_metadata,
          collections=collections,
          display_name=name)

  return t_summary
