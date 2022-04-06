# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
from tensorflow.python.ipu import keras_extensions
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.ipu.keras.extensions import functional_extensions
from tensorflow.python.ipu.keras.extensions import sequential_extensions
from tensorflow.python.ipu.keras.extensions import model_extensions
from tensorflow.python.keras.engine import functional
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.engine import training

# Insert the extensions for the (old) Keras classes.
# Note: insert Sequential before Functional as Sequential models inherit from
# Functional models.
keras_extensions._extensions_manager._register_extension(
    sequential.Sequential, base_layer.TFKerasExtension,
    sequential_extensions.SequentialExtension)
keras_extensions._extensions_manager._register_extension(
    functional.Functional, base_layer.TFKerasExtension,
    functional_extensions.FunctionalExtension)
keras_extensions._extensions_manager._register_extension(
    training.Model, base_layer.TFKerasExtension,
    model_extensions.ModelExtension)
