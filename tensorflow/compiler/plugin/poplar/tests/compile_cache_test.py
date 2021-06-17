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

import threading
import numpy as np

from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.python.data.ops import dataset_ops
from tensorflow.compiler.tests import xla_test
from tensorflow.python import ipu
from tensorflow.python import ops
from tensorflow.python.client import session
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops


class CompileCacheTest(xla_test.XLATestCase):  # pylint: disable=abstract-method
  @staticmethod
  def _run_in_threads(fns):
    barrier = threading.Barrier(len(fns))

    def wrapper_fn(results, index):
      try:
        results[index] = fns[index](barrier)
      except:  # pylint: disable=bare-except
        results[index] = None

    threads = [None] * len(fns)
    results = [None] * len(fns)
    for i in range(len(fns)):
      threads[i] = threading.Thread(target=wrapper_fn, args=(results, i))
      threads[i].start()

    for thread in threads:
      thread.join()

    return tuple(results)

  def _count_ipu_compilations(self):
    with self.session() as sess:
      with ops.device('cpu'):
        e = gen_ipu_ops.ipu_event_trace()
      events = sess.run(e)
    count = 0
    for evt_str in events:
      evt = IpuTraceEvent.FromString(evt_str)
      if evt.type == IpuTraceEvent.COMPILE_END:
        count += 1
    return count

  def setUp(self):
    super().setUp()
    with session.Session() as sess:
      sess.run(gen_ipu_ops.ipu_event_trace())

  @test_util.deprecated_graph_mode_only
  def test_with_infeed_and_outfeed_sequential(self):
    opts = ipu.config.IPUConfig()
    opts._profiling.enable_ipu_events = True  # pylint: disable=protected-access
    opts.configure_ipu_system()

    with self.session() as sess:

      def build_and_run_model():
        dataset = dataset_ops.Dataset.from_tensor_slices(
            np.ones(10, dtype=np.float32))
        infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
        outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

        def body(v, x):
          v = v + x
          outfed = outfeed_queue.enqueue(v)
          return v, outfed

        def my_net(v):
          return ipu.loops.repeat(10, body, v, infeed_queue)

        v = array_ops.placeholder(np.float32, shape=())
        with ipu.scopes.ipu_scope("/device:IPU:0"):
          result = ipu.ipu_compiler.compile(my_net, inputs=[v])

        dequeued = outfeed_queue.dequeue()

        sess.run(infeed_queue.initializer)
        res = sess.run(result, {v: 0.0})
        deq = sess.run(dequeued)
        return res, deq

      result0, dequeue0 = build_and_run_model()
      result1, dequeue1 = build_and_run_model()
    self.assertAllClose(result0, result1)
    self.assertAllClose(dequeue0, dequeue1)
    # Expect a single compilation when the model is built and run for the same
    # device in the same session but with different feed ids.
    self.assertEqual(1, self._count_ipu_compilations())

  @test_util.deprecated_graph_mode_only
  def test_with_infeed_and_outfeed_parallel(self):
    opts = ipu.config.IPUConfig()
    opts._profiling.enable_ipu_events = True  # pylint: disable=protected-access
    opts.configure_ipu_system()

    def build_and_run_model(barrier):
      with session.Session() as sess:
        barrier.wait()
        dataset = dataset_ops.Dataset.from_tensor_slices(
            np.ones(10, dtype=np.float32))
        infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
        outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

        def body(v, x):
          v = v + x
          outfed = outfeed_queue.enqueue(v)
          return v, outfed

        def my_net(v):
          return ipu.loops.repeat(10, body, v, infeed_queue)

        v = array_ops.placeholder(np.float32, shape=())
        with ipu.scopes.ipu_scope("/device:IPU:0"):
          result = ipu.ipu_compiler.compile(my_net, inputs=[v])

        dequeued = outfeed_queue.dequeue()

        sess.run(infeed_queue.initializer)
        res = sess.run(result, {v: 0.0})
        deq = sess.run(dequeued)
        # Block before destroying the session to make sure that parallel
        # compilation/executable reuse is performed.
        barrier.wait()
        return res, deq

    result0, result1 = self._run_in_threads([build_and_run_model] * 2)
    self.assertAllClose(result0, result1)
    # Expect a single compilation when the model is built and run in parallel for
    # the same device in different session with different feed ids.
    self.assertEqual(1, self._count_ipu_compilations())


if __name__ == "__main__":
  googletest.main()
