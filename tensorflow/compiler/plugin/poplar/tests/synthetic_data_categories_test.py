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
import multiprocessing
import os
import signal

import absl.testing
import numpy as np
import test_utils as tu

from tensorflow.python.framework import ops
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
# FIXME: This import isn't used but prevents an absl.flags UnrecognizedFlagError that otherwise gets thrown
from tensorflow.compiler.tests import xla_test  # pylint: disable=W0611
from tensorflow.python.ipu.optimizers import gradient_accumulation_optimizer
from tensorflow.python.ipu import functional_ops
from tensorflow.keras import layers
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import momentum
from tensorflow.python.ipu import embedding_ops
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import loops
from tensorflow.python.ipu import scopes
from tensorflow.python.client import session

from tensorflow.compat.v1 import disable_v2_behavior
disable_v2_behavior()

feed_counter = 0


def generate_feed_name():
  global feed_counter
  feed_counter += 1

  return "feed" + str(feed_counter)


def run_graph(worker, *args, **kwargs):
  # FIXME: We need to use the multiprocessing module to run the tests
  # because the flags which we're using in the tests are stored
  # statically and so can't be changed between test runs w/o a new process.
  def wrapper(result, *args, **kwargs):
    result.value = worker(*args, **kwargs)

  result = multiprocessing.Value('L', 0)
  args = (result,) + args
  process = multiprocessing.Process(target=wrapper, args=args, kwargs=kwargs)
  process.start()

  process.join()
  if process.exitcode < 0:
    signal_name = signal.Signals(abs(process.exitcode)).name
    raise multiprocessing.ProcessError(
        f"Process killed with signal {signal_name}. Check logs.")

  return result.value


def hostembedding_test_graph(flags):
  poplar_flags = os.environ.get("TF_POPLAR_FLAGS", "")
  poplar_flags += flags

  with test.mock.patch.dict("os.environ", {"TF_POPLAR_FLAGS": poplar_flags}):
    embedding = embedding_ops.create_host_embedding("test",
                                                    shape=[2, 2],
                                                    dtype=np.float32,
                                                    partition_strategy="TOKEN")

    # The device side main
    def body(x1, x2):
      d1 = embedding.lookup(constant_op.constant(0, shape=[]),
                            clip_indices=False)
      d2 = x1 + x2
      outfeed = outfeed_queue.enqueue({'d1': d1, 'd2': d2})
      return outfeed

    def my_net():
      r = loops.repeat(2, body, [], infeed_queue)
      return r

    with ops.device('cpu'):
      # The dataset for feeding the graphs
      ds = dataset_ops.Dataset.from_tensors(
          constant_op.constant(1.0, shape=[2]))
      ds = ds.map(lambda x: [x, x])
      ds = ds.repeat()

      # The host side queues
      infeed_queue = ipu_infeed_queue.IPUInfeedQueue(
          ds, feed_name=generate_feed_name())
      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(
          feed_name=generate_feed_name())

    with scopes.ipu_scope('/device:IPU:0'):
      run_loop = ipu_compiler.compile(my_net, inputs=[])

    # The outfeed dequeue has to happen after the outfeed enqueue
    dequeue_outfeed = outfeed_queue.dequeue()

    with session.Session() as sess:
      report = tu.ReportJSON(None, sess, minimum_remote_tensor_size=1)

      sess.run(variables.global_variables_initializer())
      sess.run(infeed_queue.initializer)

      report.reset()

      with embedding.register(sess):
        sess.run(run_loop)
        sess.run(dequeue_outfeed)

        report.parse_log()
        # TODO: Get the memory_test_graph working with embedding and multiprocessing.Pool.
        return report.get_total_tile_memory()


def memory_test_graph(flags=""):
  poplar_flags = os.environ.get("TF_POPLAR_FLAGS", "")
  poplar_flags += flags

  with test.mock.patch.dict("os.environ", {"TF_POPLAR_FLAGS": poplar_flags}):

    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(7, shape=[4, 4, 4])
      dataset = dataset.batch(batch_size=2, drop_remainder=True)

      def dataset_parser(value):
        img = value / 7
        label = math_ops.reduce_mean(img, axis=[1, 2, 3])
        return img, math_ops.cast(label, np.int32)

      return dataset.map(dataset_parser)

    num_batches_to_accumulate = 20
    repeat_count = 2
    optimizer = momentum.MomentumOptimizer(learning_rate=.001, momentum=0.9)

    def fwd_fn(c, img, label):  # pylint: disable=W0613
      y = layers.Conv2D(
          4,
          1,
          kernel_initializer=init_ops.constant_initializer(0.0125),
          bias_initializer=init_ops.constant_initializer(0.0125),
          name='conv1')(img)
      y = math_ops.reduce_mean(y, axis=[1, 2])
      y = layers.Dense(
          2,
          kernel_initializer=init_ops.constant_initializer(0.0125),
          bias_initializer=init_ops.constant_initializer(0.0125))(y)
      loss = math_ops.reduce_mean(
          nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=label))
      return loss

    def inputs_fn():
      with ops.device('cpu'):
        return [array_ops.placeholder(np.float32, shape=[])]

    num_iterations = repeat_count * num_batches_to_accumulate

    g = ops.Graph()
    with g.as_default(), session.Session(graph=g) as sess:
      dataset = dataset_fn()
      inputs = inputs_fn()

      infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset,
                                                     generate_feed_name())
      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(generate_feed_name())

      with variable_scope.variable_scope("ipu", use_resource=True,
                                         reuse=False):

        def model(*args):
          loss = fwd_fn(*functional_ops._convert_to_list(args))  # pylint: disable=W0212, E1120
          enqueue_op = outfeed_queue.enqueue(loss)
          opt = gradient_accumulation_optimizer.GradientAccumulationOptimizerV2(
              optimizer, num_batches_to_accumulate)
          outs = list(args[:len(args) - infeed_queue.number_of_tuple_elements])
          outs.append(enqueue_op)
          outs.append(opt.minimize(loss))
          return outs

        def my_net(*args):
          return loops.repeat(num_iterations,
                              model,
                              inputs=args,
                              infeed_queue=infeed_queue)

      with ops.device("/device:IPU:0"):
        loop_ret = ipu_compiler.compile(my_net, inputs=inputs)

      outfeed_op = outfeed_queue.dequeue()
      report = tu.ReportJSON(None, sess, minimum_remote_tensor_size=1)

      sess.run(variables.global_variables_initializer())
      sess.run(infeed_queue.initializer)
      report.reset()
      sess.run(loop_ret, feed_dict=dict(zip(inputs, [10.01])))
      sess.run(outfeed_op)

      report.parse_log()
      return report.get_total_tile_memory()


class SyntheticDataMemoryUsage(absl.testing.parameterized.TestCase):
  reference_usage = None
  categories = ['parameters', 'seed', 'hostembedding', 'infeed', 'outfeed']

  @classmethod
  def setUpClass(cls):
    cls.reference_usage = run_graph(memory_test_graph)

  @absl.testing.parameterized.parameters(
      *filter(lambda category: category != "hostembedding", categories))
  def testCategories(self, category):
    self.assertIsNotNone(SyntheticDataMemoryUsage.reference_usage)

    flag = f" --synthetic_data_categories='{category}'"
    new_usage = run_graph(memory_test_graph, flag)

    self.assertLess(new_usage, SyntheticDataMemoryUsage.reference_usage)

  def testHostEmbedding(self):
    # FIXME: memory_test_graph doesnt work with
    # embeddings as it introduces a pickling error with multiprocessing.Pool
    reference_usage = run_graph(hostembedding_test_graph, "")

    flags = " --synthetic_data_categories='hostembedding'"
    new_usage = run_graph(hostembedding_test_graph, flags)

    self.assertLess(new_usage, reference_usage)

  def testUseSyntheticDataFlagIsEquivalentToAllCategories(self):
    flags = " --use_synthetic_data=true"
    synthetic_data_usage = run_graph(memory_test_graph, flags)

    flag_separator = ","
    flags = f" --synthetic_data_categories='{flag_separator.join(self.categories)}'"  # pylint: disable=C0301
    categories_usage = run_graph(memory_test_graph, flags)

    self.assertEqual(categories_usage, synthetic_data_usage)

  def testUsingInvalidCategoryTerminates(self):
    flag = " --synthetic_data_categories=FOO"
    self.assertRaises(multiprocessing.ProcessError, run_graph,
                      memory_test_graph, flag)

  def testUsingFlagsWithWhitespace(self):
    flags = " --synthetic_data_categories=infeed,outfeed,seed"
    expected_output = run_graph(memory_test_graph, flags)

    flags = " --synthetic_data_categories='infeed,   outfeed , seed'"
    output = run_graph(memory_test_graph, flags)

    self.assertEqual(output, expected_output)

  def testFlagsCanBeUsedWithSyntheticDataInitializer(self):
    flags = " --synthetic_data_categories=infeed --synthetic_data_initializer=2"
    output = run_graph(memory_test_graph, flags)

    self.assertLess(output, SyntheticDataMemoryUsage.reference_usage)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
