import numpy as np
import os

from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import test_util
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import loops
from tensorflow.python.ipu import scopes
from tensorflow.python.ipu import utils
from tensorflow.python.ipu.ops import rand_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest

# This test verifies that dropout works across replicas
#
# Each replica contains 2 dropouts, and there are 2 replicas
# and each dropout pattern should be different.
#
# We loop through the graph multiple times, and on each
# iteration the results should be different.
#
# We execute that loop twice, and on each execution the results
# should be different
#
# Total dropouts:
#  2 per replica x REPLICAS x REPEATS per run x EXECS runs
#  = 4 * REPEATS * EXECS number of dropout ops

SIZE = 4000
REPLICAS = 2
REPEATS = 10
EXECS = 2

os.environ['TF_POPLAR_FLAGS'] = "--force_replicated_mode"


class TestDropout(test_util.TensorFlowTestCase):
  def testResetSeed(self):
    # The dataset for feeding the graphs
    ds = dataset_ops.Dataset.from_tensors(
        array_ops.constant(1.0, shape=[SIZE]))
    ds = ds.map(lambda x: [x, x])
    ds = ds.repeat()

    # The host side queues
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(
        ds, feed_name="infeed", replication_factor=REPLICAS)
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(
        feed_name="outfeed", replication_factor=REPLICAS)

    # The device side
    def body(x1, x2):
      d1 = rand_ops.dropout(x1)
      d2 = rand_ops.dropout(x2)
      outfeed = outfeed_queue.enqueue({'d1': d1, 'd2': d2})
      return outfeed

    def my_net():
      r = loops.repeat(REPEATS, body, [], infeed_queue)
      return r

    with scopes.ipu_scope('/device:IPU:0'):
      res = ipu_compiler.compile(my_net, inputs=[])

    # The outfeed dequeue has to happen after the outfeed enqueue
    dequeue_outfeed = outfeed_queue.dequeue()

    # Configure the hardware
    config = utils.create_ipu_config(profiling=True)
    config = utils.auto_select_ipus(config, REPLICAS)
    config = utils.set_floating_point_behaviour_options(config)
    utils.configure_ipu_system(config)

    with session.Session() as sess:
      res_all = set()
      total = 0

      sess.run(infeed_queue.initializer)

      for _ in range(EXECS):
        result = sess.run(res)
        outfed_result = sess.run(dequeue_outfeed)
        for r in np.array(list(outfed_result.values())).reshape([-1, SIZE]):
          total += 1
          res_all.add(r.tostring())

      # 2 dropouts per replica * REPLICAS * REPEATS * EXECS
      expected = 2 * REPLICAS * REPEATS * EXECS
      self.assertEqual(total, expected)
      self.assertEqual(len(res_all), expected)


if __name__ == "__main__":
  googletest.main()
