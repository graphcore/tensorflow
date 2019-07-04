from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.platform import googletest


class ContribIpuOpsTest(test_util.TensorFlowTestCase):
  def testSortOp(self):
    with ops.device("/device:IPU:0"):
      with session.Session() as s:
        t1 = random_ops.random_uniform([1000], dtype=dtypes.float32)
        t2 = sort_ops.sort(t1, name="t2")
        h1, h2 = s.run([t1, t2])
        self.assertEqual(sorted(h1), list(h2))


if __name__ == "__main__":
  googletest.main()
