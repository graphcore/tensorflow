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
# ==============================================================================
from collections import OrderedDict

from tensorflow.python.eager import context
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.ipu.keras.extensions import functional_extensions
from tensorflow.python.ipu.keras.extensions import sequential_extensions
from tensorflow.python.keras.engine import functional
from tensorflow.python.keras.engine import sequential


class KerasExtensions:
  _enable_legacy_iterators = True

  def __init__(self,
               enable_dataset_iterators=True,
               enable_keras_extensions=True):
    self._enable_iterators = enable_dataset_iterators
    self._enable_keras_extensions = enable_keras_extensions
    self._keras_extensions = OrderedDict()

    # Insert Sequential before Functional as Sequential models inherit from
    # Functional models.
    self._register_keras_extension(sequential.Sequential,
                                   sequential_extensions.SequentialExtension)
    self._register_keras_extension(functional.Functional,
                                   functional_extensions.FunctionalExtension)

  def _enable_dataset_iterators(self):
    return context.executing_eagerly() and self._enable_iterators

  def _create_dataset_iterator(self, dataset):
    assert self._enable_dataset_iterators()
    return ipu_infeed_queue.IPUOwnedIterator(dataset=dataset)  # pylint: disable=protected-access

  def _register_keras_extension(self, class_type, extension):
    self._keras_extensions[class_type] = extension

  def _delete_keras_extension(self, class_type):
    self._keras_extensions.pop(class_type, None)

  def _patch_keras_extension(self, instance):
    if not self._enable_keras_extensions:
      return

    for class_type, extension in self._keras_extensions.items():
      if isinstance(instance, class_type):
        if isinstance(instance, base_layer.KerasExtension):
          if not isinstance(instance, extension):
            raise RuntimeError(
                "KerasExtension patching failed - already patched with a "
                "different extension.")
          break

        # Patch in the extension.
        # Note that we keep the name as Keras sometimes does __name__ checks.
        cls = instance.__class__
        instance.__class__ = cls.__class__(cls.__name__, (cls, extension), {})
        extension.__init__(instance)
        break
