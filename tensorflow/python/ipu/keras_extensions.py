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


class _ExtensionsManager:
  def __init__(self):
    self._extensions = OrderedDict()

  def _register_extension(self, class_type, extension_type, extension):
    self._extensions[(class_type, extension_type)] = extension

  def _delete_extension(self, class_type, extension_type):
    self._extensions.pop((class_type, extension_type), None)

  def __iter__(self):
    for key, value in self._extensions.items():
      yield key, value


_extensions_manager = _ExtensionsManager()


class KerasExtensions:
  _enable_legacy_iterators = True

  def __init__(self,
               enable_dataset_iterators=True,
               enable_keras_extensions=True):
    self._enable_iterators = enable_dataset_iterators
    self._enable_keras_extensions = enable_keras_extensions
    self._keras_extensions = OrderedDict()

    for (class_type, extension_type), extension in _extensions_manager:
      self._register_keras_extension(class_type, extension_type, extension)

  def _enable_dataset_iterators(self):
    return context.executing_eagerly() and self._enable_iterators

  def _create_dataset_iterator(self, dataset):
    assert self._enable_dataset_iterators()
    return ipu_infeed_queue.IPUOwnedIterator(dataset=dataset)  # pylint: disable=protected-access

  def _register_keras_extension(self, class_type, extension_type, extension):
    self._keras_extensions[(class_type, extension_type)] = extension

  def _get_keras_extension(self, class_type, extension_type):
    try:
      return self._keras_extensions[(class_type, extension_type)]
    except KeyError:
      return None

  def _delete_keras_extension(self, class_type, extension_type):
    self._keras_extensions.pop((class_type, extension_type), None)

  def _patch_keras_extension(self, instance):
    if not self._enable_keras_extensions:
      return

    for (class_type,
         extension_type), extension_cls in self._keras_extensions.items():
      if isinstance(instance, class_type):
        if isinstance(instance, extension_type):
          if not isinstance(instance, extension_cls):
            raise RuntimeError(
                "KerasExtension patching failed - already patched with a "
                "different extension.")
          break

        # Create a patched version of the instance class with the extension as
        # a subclass.
        # Note that we keep the name as Keras sometimes does __name__ checks.
        instance_cls = type(instance)
        patched_cls = type(instance_cls.__name__,
                           (instance_cls, extension_cls), {})

        # Generate a new constructor which also calls the extension constructor
        # in case a new instance of the patched class gets constructed, for
        # example if we call from_config on the instance.
        # Generate this within another function to avoid capturing issues.
        patched_cls.__init__ = self.create_patched_init(
            instance_cls, extension_cls)

        # Change the class of the instance to be the patched class.
        instance.__class__ = patched_cls
        instance.__original_class__ = instance_cls

        # Call the extension constructor as it has not been called yet on this
        # instance.
        extension_cls.__init__(instance)
        break

  @staticmethod
  def create_patched_init(instance_cls, extension_cls):
    def patched_init(self, *args, **kwargs):
      instance_cls.__init__(self, *args, **kwargs)
      extension_cls.__init__(self)

    return patched_init
