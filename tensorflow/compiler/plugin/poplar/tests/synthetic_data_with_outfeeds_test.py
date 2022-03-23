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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.compiler.tests import xla_test

from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import loops
from tensorflow.python.ipu import scopes
from tensorflow.python.ipu import utils
from tensorflow.python.ipu.config import IPUConfig
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class SyntheticDataWithOutfeeds(xla_test.XLATestCase):
  def testNoSyntheticData(self):
    poplar_flags = os.environ.get("TF_POPLAR_FLAGS", "")
    poplar_flags += " --use_ipu_model"

    with test.mock.patch.dict("os.environ", {"TF_POPLAR_FLAGS": poplar_flags}):

      # The device side main
      def body(x1, x2):
        d1 = x1 + x2
        d2 = x1 - x2
        outfeed = outfeed_queue.enqueue({'d1': d1, 'd2': d2})
        return outfeed

      def my_net():
        r = loops.repeat(3, body, [], infeed_queue)
        return r

      with ops.device('cpu'):
        # The dataset for feeding the graphs
        ds = tf.data.Dataset.from_tensors(tf.constant(1.0, shape=[10]))
        ds = ds.map(lambda x: [x, x])
        ds = ds.repeat()

        # The host side queues
        infeed_queue = ipu_infeed_queue.IPUInfeedQueue(ds)
        outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

      with scopes.ipu_scope('/device:IPU:0'):
        run_loop = ipu_compiler.compile(my_net, inputs=[])

      # The outfeed dequeue has to happen after the outfeed enqueue
      dequeue_outfeed = outfeed_queue.dequeue()

      # Configure the hardware
      config = IPUConfig()
      config.auto_select_ipus = 1
      config.configure_ipu_system()

      with tf.Session() as sess:
        sess.run(infeed_queue.initializer)
        sess.run(run_loop)
        result = sess.run(dequeue_outfeed)
        self.assertAllEqual(np.full([10], 2.0), result['d1'][0])

  def testSyntheticDataWithOutfeedModeAll(self):
    poplar_flags = os.environ.get("TF_POPLAR_FLAGS", "")
    poplar_flags += " --use_ipu_model"
    poplar_flags += " --use_synthetic_data"
    poplar_flags += " --synthetic_data_initializer=random"

    with test.mock.patch.dict("os.environ", {"TF_POPLAR_FLAGS": poplar_flags}):

      # The device side main
      def body(x1, x2):
        d1 = x1 + x2
        d2 = x1 - x2
        outfeed = outfeed_queue.enqueue({'d1': d1, 'd2': d2})
        return outfeed

      def my_net():
        r = loops.repeat(3, body, [], infeed_queue)
        return r

      with ops.device('cpu'):
        # The dataset for feeding the graphs
        ds = tf.data.Dataset.from_tensors(tf.constant(1.0, shape=[10]))
        ds = ds.map(lambda x: [x, x])
        ds = ds.repeat()

        # The host side queues
        infeed_queue = ipu_infeed_queue.IPUInfeedQueue(ds)
        outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

      with scopes.ipu_scope('/device:IPU:0'):
        run_loop = ipu_compiler.compile(my_net, inputs=[])

      # The outfeed dequeue has to happen after the outfeed enqueue
      dequeue_outfeed = outfeed_queue.dequeue()

      # Configure the hardware
      config = IPUConfig()
      config.auto_select_ipus = 1
      config.configure_ipu_system()

      with tf.Session() as sess:
        sess.run(infeed_queue.initializer)
        sess.run(run_loop)
        result = sess.run(dequeue_outfeed)
        self.assertAllEqual(len(result['d1']), 0)

  def testSyntheticDataWithOutfeedModeLast(self):
    poplar_flags = os.environ.get("TF_POPLAR_FLAGS", "")
    poplar_flags += " --use_ipu_model"
    poplar_flags += " --use_synthetic_data"

    with test.mock.patch.dict("os.environ", {"TF_POPLAR_FLAGS": poplar_flags}):

      # The device side main
      def body(x1, x2):
        d1 = x1 + x2
        d2 = x1 - x2
        outfeed = outfeed_queue.enqueue({'d1': d1, 'd2': d2})
        return outfeed

      def my_net():
        r = loops.repeat(3, body, [], infeed_queue)
        return r

      with ops.device('cpu'):
        # The dataset for feeding the graphs
        ds = tf.data.Dataset.from_tensors(tf.constant(1.0, shape=[10]))
        ds = ds.map(lambda x: [x, x])
        ds = ds.repeat()

        # The host side queues
        infeed_queue = ipu_infeed_queue.IPUInfeedQueue(ds)
        outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(
            outfeed_mode=ipu_outfeed_queue.IPUOutfeedMode.LAST)

      with scopes.ipu_scope('/device:IPU:0'):
        run_loop = ipu_compiler.compile(my_net, inputs=[])

      # The outfeed dequeue has to happen after the outfeed enqueue
      dequeue_outfeed = outfeed_queue.dequeue()

      # Configure the hardware
      config = IPUConfig()
      config.auto_select_ipus = 1
      config.configure_ipu_system()

      with tf.Session() as sess:
        sess.run(infeed_queue.initializer)
        sess.run(run_loop)
        with self.assertRaisesRegex(
            errors.FailedPreconditionError,
            r'when using synthethic data. This is not supported.'):
          sess.run(dequeue_outfeed)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
