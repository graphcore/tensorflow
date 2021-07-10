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
IPU specific Keras Functional Model extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.python.ipu.keras.extensions import model_extensions
from tensorflow.python.training.tracking import base as trackable


class FunctionalExtension(model_extensions.ModelExtension):  # pylint: disable=abstract-method
  def _get_shard_count(self):
    return 1

  def _get_config_supported(self):
    return True

  def _get_config_delegate(self):
    # Get the Keras config.
    config = self.get_config(__extension_delegate=False)
    # Get the ModelExtension config and merge it in.
    extension_config = self._get_base_config()
    config.update(extension_config)
    return config

  def _deserialize_from_config_supported(self, config):
    del config
    return True

  @trackable.no_automatic_dependency_tracking
  def _deserialize_from_config_delegate(self, config):
    self._from_base_config(config)

  def set_gradient_accumulation_options(
      self,
      gradient_accumulation_steps=None,
      experimental_normalize_gradients=None,
      **gradient_accumulation_optimizer_kwargs):
    # pylint:disable=line-too-long
    """Sets the gradient accumulation options for non-pipelined models which are
    to be used when training a model.

    When set, and `gradient_accumulation_steps > 1`, the optimizer which the
    current model has been compiled with is wrapped in
    :class:`~tensorflow.python.ipu.optimizers.GradientAccumulationOptimizerV2`.
    This means that instead of performing the weight update for every step,
    gradients across multiple steps are accumulated. After
    `gradient_accumulation_steps` steps have been processed, the accumulated
    gradients are used to compute the weight update.

    This feature of neural networks allows us to simulate bigger batch sizes.
    For example if we have a model of batch size 16 and we accumulate the
    gradients for 4 steps, this simulates an input batch of size 64.

    When training a data-parallel model, enabling gradient accumulation also
    reduces the communication overhead as the all-reduce of gradients is now
    performed every `gradient_accumulation_steps` steps instead of every step.

    See the :ref:`gradient-accumulation` section in the documention for more
    details.

    Args:
      gradient_accumulation_steps: An integer which indicates the number of
        steps the gradients will be accumulated for. This value needs to divide
        the `steps_per_execution` value the model has been compiled with and
        also be divisible by the replication factor if the model is running
        in a data-parallel fashion. This value is saved/loaded when the model
        is saved/loaded.
      experimental_normalize_gradients: If set to `True`, the gradients for each
        step are first scaled by `1/gradient_accumulation_steps` before being
        added to the gradient accumulation buffer. Note that this option is
        experimental and the behavior might change in future releases. This
        value is saved/loaded when the model is saved/loaded.
      gradient_accumulation_optimizer_kwargs: All remaining keyword arguments
        are forwarded to
        :class:`~tensorflow.python.ipu.optimizers.GradientAccumulationOptimizerV2`.
        See the optimizer for all the available arguments. Must not contain
        `opt` or `num_mini_batches` as keys. Note that this dictionary is not
        serializable, which means that when the model is being saved, these
        values are not saved. When restoring/loading a model, please call
        `set_gradient_accumulation_options` again.
    """
    # pylint:enable=line-too-long
    self._set_gradient_accumulation_options_impl(
        gradient_accumulation_steps, experimental_normalize_gradients,
        gradient_accumulation_optimizer_kwargs)
