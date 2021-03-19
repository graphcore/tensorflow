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

import inspect
import sys

from collections import OrderedDict

from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras import layers


class _LayerInstanceDescriptor:
  """Represents an instance of a keras.layers.Layer derived class in terms
  of the arguments passed to it's constructor.
  """
  def __init__(self, layer_instance):
    # Sanity check.
    if not isinstance(layer_instance, Layer):
      raise TypeError("_LayerInstanceDescriptor may only be used with classes "
                      "derived from keras.Layer.")

    # Pull out the layers configuration.
    assert hasattr(layer_instance, '_stored_init_args') and\
      hasattr(layer_instance, '_stored_init_kwargs')

    self._init_args = layer_instance._stored_init_args  # pylint: disable=protected-access
    self._init_kwargs = layer_instance._stored_init_kwargs  # pylint: disable=protected-access

    # Get the class descriptor for the layer type.
    self._class_descriptor = _LayerClassDescriptor(type(layer_instance))

    # Verify that the config satisfies all the requirements of the classes
    # __init__. I.e. it can be constructed with the configuration pulled
    # from the instance.
    assert len(self._init_args) == len(self._class_descriptor.get_init_args())
    assert all(k in self._class_descriptor.get_init_kwargs()
               for k in self._init_kwargs)

  def get_init_args(self):
    return self._init_args

  def get_init_kwargs(self):
    return self._init_kwargs

  def get_class_descriptor(self):
    return self._class_descriptor


class _LayerClassDescriptor:
  """Represents an instance of a keras.layers.Layer derived class in terms of
  it's __init__ and __call__ arguments.
  """
  def __init__(self, layer_class):
    # Sanity check.
    if not issubclass(layer_class, Layer):
      raise TypeError("_LayerClassDescriptor may only be used with classes "
                      "derived from keras.Layer.")

    # Add args from base.
    s = inspect.signature(Layer.__init__)
    self._init_positional, self._init_keyword = self._get_args(s)

    s = inspect.signature(Layer.call)
    self._call_positional, self._call_keyword = self._get_args(s)

    # Get init arguments.
    s = inspect.signature(layer_class.__init__)
    pos, kw = self._get_args(s)
    self._init_positional += [a for a in pos if not a in self._init_positional]
    if 'kwargs' in self._init_positional:
      self._init_positional.remove('kwargs')

    self._init_keyword.update(kw)

    # Get call arguments.
    s = inspect.signature(layer_class.call)
    pos, kw = self._get_args(s)
    self._call_positional += [a for a in pos if not a in self._call_positional]
    self._call_keyword.update(kw)

    # Store the type.
    self._type = layer_class

  def _get_args(self, signature):
    positional = []
    keyword = OrderedDict()

    for name, arg in signature.parameters.items():
      if name == "self":
        continue

      if arg.default == inspect.Signature.empty:
        positional.append(name)
        continue

      keyword[name] = arg.default

    return positional, keyword

  def get_init_args(self):
    return self._init_positional

  def get_init_kwargs(self):
    return list(self._init_keyword.keys())

  def get_call_args(self):
    return self._call_positional

  def get_call_kwargs(self):
    return list(self._call_keyword.keys())

  def get_type(self):
    return self._type


class IPULayerReplacer:
  """This class determines which layers are available in ipu.keras.layers
  upon construction. When invoking __call__ with a single keras.layers
  instance, if it is substitutable with an ipu.keras.layers Layer, then
  an instance of the replacement layer is returned.
  """
  def __init__(self):
    """Determines which layers are available in ipu.keras.layers and builds
    an index of the requirements for each.
    """
    module_path = "tensorflow.python.ipu.keras.layers"
    self._ipu_layers = self._get_layers_in_module(module_path)

    self._blacklist = ["GroupNorm", "InstanceNorm", "SerialDense"]

  def __call__(self, layer_instance):
    """Substitutes a keras.layers.Layer instance with an ipu.keras.layers.Layer
    instance, if such a replacement is available. The substitute layer instance
    is returned.

    Arguments:
      layer_instance: An instance of a keras.layers.Layer derived layer.
    """
    # If we don't have an IPU implementation of this layer, then return
    # the original instance.
    layer_name = layer_instance.__class__.__name__
    if not layer_name in self._ipu_layers or layer_name in self._blacklist:
      return layer_instance

    # Currently, onlu GRU's using implementation=1 are supported. Once IPU GRU
    # is updated to allow implementation=2, this block can be removed.
    if isinstance(layer_instance, layers.GRU) and\
      layer_instance.get_config()['implementation'] == 2:
      return layer_instance

    # Get the descriptors for the given instance and the IPU Keras layer
    # class that it's type corresponds to.
    instance_descriptor = _LayerInstanceDescriptor(layer_instance)
    ipu_class_descriptor = self._ipu_layers[layer_name]

    # Verify that the substitution can be made. This should be the case for
    # Keras and IPU Keras layers that have a consistent API.
    valid_sub = self._can_be_constructed(instance_descriptor,
                                         ipu_class_descriptor)
    valid_sub &= self._can_be_called(instance_descriptor, ipu_class_descriptor)
    if not valid_sub:
      raise RuntimeError(
          "Could not substitute Keras %s layer for it's IPU "
          "counterpart. This is likely due to an API change, such that the"
          "Keras and IPU Keras API's for %s are no longer consistent."\
          % (layer_name, layer_name))

    # Create the replacement layer.
    layer_type = ipu_class_descriptor.get_type()
    layer = layer_type(*instance_descriptor.get_init_args(),
                       **instance_descriptor.get_init_kwargs())

    # If layer_instance is built, copy it's weights over.
    if layer_instance.built and layer_instance.trainable_weights:
      assert hasattr(layer_instance, '_stored_input_shape')
      layer.build(layer_instance._stored_input_shape)  # pylint: disable=protected-access
      layer.set_weights(layer_instance.get_weights())
    return layer

  def get_ipu_layer_names(self):
    """Returns a list of the found ipu.keras.layers Layer classes."""
    layer_names = list(self._ipu_layers.keys())
    layer_names.sort()
    return layer_names

  def get_blacklisted_ipu_layer_names(self):
    """Returns a list of the blacklisted ipu.keras.layers Layer classes."""
    return self._blacklist

  @staticmethod
  def _get_layers_in_module(module_path):
    """Generic helper function to generate instances of _LaterClassDescriptor
    for each keras.layers.Layer derived class in a python module.

    Arguments:
      module_path: A module in which to search for Layer instances.
    """
    modules = inspect.getmembers(sys.modules[module_path])

    layers_found = OrderedDict()
    for name, member in modules:
      if inspect.isclass(member) and issubclass(member, Layer):
        layers_found[name] = _LayerClassDescriptor(member)

    return layers_found

  @staticmethod
  def _can_be_constructed(instance_descriptor, class_descriptor):
    """Determines if, given an instance of _LayerInstanceDescriptor, the
    layer described by the given instance of _LayerClassDescriptor can be
    instantiated.

    Arguments:
      instance_descriptor: An instance of _LayerInstanceDescriptor.
      class_descriptor: An instance of _LayerClassDescriptor.
    """
    assert isinstance(instance_descriptor, _LayerInstanceDescriptor)
    assert isinstance(class_descriptor, _LayerClassDescriptor)

    # Verify that the instance has all required init args for the class
    # from which this descriptor was generated.
    class_args = class_descriptor.get_init_args() +\
      class_descriptor.get_init_kwargs()
    instance_args = instance_descriptor.get_class_descriptor().get_init_args()\
      + list(instance_descriptor.get_init_kwargs().keys())

    if not all(k in class_args for k in instance_args):
      return False
    return True

  @staticmethod
  def _can_be_called(instance_descriptor, class_descriptor):
    """Determines if, given an instance of _LayerInstanceDescriptor, the
    layer described by the given instance of _LayerClassDescriptor can be
    called as the instance would.

    Arguments:
      instance_descriptor: An instance of _LayerInstanceDescriptor.
      class_descriptor: An instance of _LayerClassDescriptor.
    """
    assert isinstance(instance_descriptor, _LayerInstanceDescriptor)
    assert isinstance(class_descriptor, _LayerClassDescriptor)

    # Verify that the instance has all required call args for the class
    # from which this descriptor was generated.
    class_args = class_descriptor.get_call_args() +\
      class_descriptor.get_call_kwargs()

    instance_class_descriptor = instance_descriptor.get_class_descriptor()
    instance_args = instance_class_descriptor.get_call_args() +\
      instance_class_descriptor.get_call_kwargs()

    if not all(k in class_args for k in instance_args):
      return False
    return True
