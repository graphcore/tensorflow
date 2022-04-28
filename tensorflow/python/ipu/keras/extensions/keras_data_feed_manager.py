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
from tensorflow.python.eager import context
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.util import deprecation


class InfeedManager:
  """Class for re-using infeed names.

  Re-using infeed names for different execution modes means that the internal
  tf.function does not need to be retraced."""
  @deprecation.deprecated(None, "Use `tf.keras` instead of `tf.python.keras`.")
  def __init__(self):
    self._infeed_names = dict()
    self._infeeds = dict()

  def get_infeed(self, mode, dataset, infeed_kwargs):
    kwargs = dict(infeed_kwargs)
    if not mode in self._infeed_names:
      self._infeed_names[mode] = ipu_infeed_queue._generate_unique_name()  # pylint: disable=protected-access

    # Re-use the infeed name.
    kwargs[ipu_infeed_queue._internal_id] = self._infeed_names[mode]  # pylint: disable=pointless-statement,protected-access

    # De-register the existing infeed.
    if mode in self._infeeds:
      with context.eager_mode():
        self._infeeds[mode]._infeed_queue.deleter  # pylint: disable=pointless-statement,protected-access

    # Create a new infeed.
    self._infeeds[mode] = ipu_infeed_queue.IPUIterator(dataset, **kwargs)
    return self._infeeds[mode]

  def reset(self):
    self._infeed_names = dict()
    for infeed in self._infeeds.values():
      with context.eager_mode():
        infeed._infeed_queue.deleter  # pylint: disable=pointless-statement,protected-access
    self._infeeds = dict()

  def __del__(self):
    self.reset()


class OutfeedManager:
  """Class for re-using outfeeds.

  Re-using outfeeds for different execution modes means that the internal
  tf.function does not need to be retraced."""
  @deprecation.deprecated(None, "Use `tf.keras` instead of `tf.python.keras`.")
  def __init__(self):
    self._outfeeds = dict()

  def get_outfeed(self, mode, outfeed_kwargs):
    if not mode in self._outfeeds:
      self._outfeeds[mode] = ipu_outfeed_queue.IPUOutfeedQueue(
          **outfeed_kwargs)
    return self._outfeeds[mode]

  def reset(self):
    for outfeed in self._outfeeds.values():
      # Delete the outfeed queue.
      with context.eager_mode():
        outfeed.deleter  # pylint: disable=pointless-statement
    self._outfeeds = dict()

  def __del__(self):
    self.reset()
