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
Keras Pipelined Model interfaces for IPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.python.ipu.keras.extensions.functional_extensions import PipelineStage  # pylint: disable=unused-import


class PipelineSequential:
  def __init__(self, *args, **kwargs):
    raise RuntimeError(
        "PipelineSequential has been deleted. Use tf.keras.Sequential within a "
        "`tensorflow.python.ipu.ipu_strategy.IPUStrategy` and use the "
        "`set_pipeline_stage_assignment()` function instead. See the "
        "documentation for full details and examples.")


class PipelineModel:
  def __init__(self, *args, **kwargs):
    raise RuntimeError(
        "PipelineModel has been deleted. Use tf.keras.Model within a "
        "`tensorflow.python.ipu.ipu_strategy.IPUStrategy` and either use "
        "`tensorflow.python.ipu.keras.PipelineStage` scopes or the "
        "`get_pipeline_stage_assignment()` and `set_pipeline_stage_assignment()` "
        "functions instead. See the documentation for full details and examples."
    )
