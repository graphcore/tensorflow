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
Summary operations for IPUs
~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.core.framework import summary_pb2
from tensorflow.python.framework import ops
from tensorflow.python.summary.summary import tensor_summary
from tensorflow import executing_eagerly


def ipu_compile_summary(name, op_list, collections=None):
  """Create an IPU compiler summary operation.

  Args:
    name: A name for the summary.
    op_list: An operation or list of operations to make this summary dependent
             upon.
    collections: Optional collections to add the summary into.

  Returns:
    The new summary operation

  """

  if not isinstance(op_list, list):
    op_list = [op_list]

  with ops.device("cpu"):
    with ops.control_dependencies(op_list):

      reports = gen_ipu_ops.ipu_event_trace()

      summary_metadata = summary_pb2.SummaryMetadata(
          plugin_data=summary_pb2.SummaryMetadata.PluginData(
              plugin_name="ipu"))

      t_summary = tensor_summary(name='ipu_trace',
                                 tensor=reports,
                                 summary_metadata=summary_metadata,
                                 collections=collections,
                                 display_name=name)

  return t_summary


def get_ipu_reports():
  """Extracts all reports and converts them from EagerTensor to array of events.

  Returns:
    A two dimensional numpy.ndarray of IPUTraceEvents protobufs.
  """

  # make sure we are running in eager mode (default in tf2)
  if not executing_eagerly():
    raise ValueError("Eager execution mode should be used.")

  # retrieve all reports as an eager tensor
  reports = gen_ipu_ops.ipu_event_trace()

  # convert from eager tensor to numpy array
  if isinstance(reports, ops.EagerTensor):
    reports = reports.numpy()

  return reports
