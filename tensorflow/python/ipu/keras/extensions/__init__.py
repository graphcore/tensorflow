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
# =============================================================================
"""
IPU specific Keras extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# pylint: disable=unused-import
from tensorflow.python.ipu.keras.extensions.functional_extensions import PipelineStage
from tensorflow.python.ipu.keras.extensions.functional_extensions import FunctionalLayerPipelineStageAssignment
from tensorflow.python.ipu.keras.extensions.functional_extensions import FunctionalExtension
from tensorflow.python.ipu.keras.extensions.sequential_extensions import SequentialLayerPipelineStageAssignment
from tensorflow.python.ipu.keras.extensions.sequential_extensions import SequentialExtension
from tensorflow.python.ipu.keras.extensions.model_extensions import ModelLayerPipelineStageAssignment
from tensorflow.python.ipu.keras.extensions.model_extensions import ModelExtension
# pylint: enable=unused-import
