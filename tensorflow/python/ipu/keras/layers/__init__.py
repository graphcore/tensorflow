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
"""
Keras layer specializations for the Graphcore IPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.python.ipu.keras.layers.assume_equal_across_replicas import AssumeEqualAcrossReplicas
from tensorflow.python.ipu.keras.layers.ctc import CTCInferenceLayer
from tensorflow.python.ipu.keras.layers.ctc import CTCPredictionsLayer
from tensorflow.python.ipu.keras.layers.dropout import Dropout
from tensorflow.python.ipu.keras.layers.effective_transformer import EffectiveTransformer
from tensorflow.python.ipu.keras.layers.embedding_lookup import Embedding
from tensorflow.python.ipu.keras.layers.normalization import GroupNorm, InstanceNorm, LayerNorm
from tensorflow.python.ipu.keras.layers.normalization import GroupNormalization, InstanceNormalization, LayerNormalization
from tensorflow.python.ipu.keras.layers.recomputation import RecomputationCheckpoint
from tensorflow.python.ipu.keras.layers.math import SerialDense
