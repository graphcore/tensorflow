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
IPU specific Keras Model extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.python.keras.engine import base_layer


class ModelExtension(base_layer.KerasExtension):
  def _fit_supported(self, *args, **kwargs):  # pylint:disable=unused-argument
    return False

  # pylint:disable=unused-argument
  def _fit_delegate(self,
                    x=None,
                    y=None,
                    batch_size=None,
                    epochs=1,
                    verbose=1,
                    callbacks=None,
                    validation_split=0.,
                    validation_data=None,
                    shuffle=True,
                    class_weight=None,
                    sample_weight=None,
                    initial_epoch=0,
                    steps_per_epoch=None,
                    validation_steps=None,
                    validation_batch_size=None,
                    validation_freq=1,
                    max_queue_size=10,
                    workers=1,
                    use_multiprocessing=False):
    # pylint:enable=unused-argument
    raise NotImplementedError

  def _predict_supported(self, *args, **kwargs):  # pylint:disable=unused-argument
    return False

  # pylint:disable=unused-argument
  def _predict_delegate(self,
                        x,
                        batch_size=None,
                        verbose=0,
                        steps=None,
                        callbacks=None,
                        max_queue_size=10,
                        workers=1,
                        use_multiprocessing=False):
    # pylint:enable=unused-argument
    raise NotImplementedError

  def _evaluate_supported(self, *args, **kwargs):  # pylint:disable=unused-argument
    return False

  # pylint:disable=unused-argument
  def _evaluate_delegate(self,
                         x=None,
                         y=None,
                         batch_size=None,
                         verbose=1,
                         sample_weight=None,
                         steps=None,
                         callbacks=None,
                         max_queue_size=10,
                         workers=1,
                         use_multiprocessing=False,
                         return_dict=False):
    # pylint:enable=unused-argument
    raise NotImplementedError
