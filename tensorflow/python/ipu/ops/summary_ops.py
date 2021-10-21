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

from tensorflow.python.util import deprecation


@deprecation.deprecated(
    None,
    "ipu_compile_summary is deprecated and will be removed in a future release."
    " Use the PopVision suite of analysis tools to profile IPU programs.")
def ipu_compile_summary(name, op_list, collections=None):
  """DEPRECATED. Create an IPU compiler summary operation.

  This function is deprecated and is no longer functional. It will be removed
  in a future release. Use the PopVision suite of analysis tools to profile IPU
  programs.

  Args:
    name: A name for the summary.
    op_list: An operation or list of operations to make this summary dependent
             upon.
    collections: Optional collections to add the summary into.

  Returns:
    The new summary operation

  """
  raise NotImplementedError(
      "ipu_compile_summary is deprecated, is no longer functional and will be"
      " removed in a future release. Use the PopVision suite of analysis tools"
      " to profile IPU programs.")
