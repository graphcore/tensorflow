# Copyright 2017 Graphcore Ltd
#


import numpy as np

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python import ipu
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ipu.optimizers import sharded_optimizer
from tensorflow.python.keras import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops as nn
from tensorflow.python.platform import googletest
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import gradient_descent as gd

logging.set_verbosity(logging.DEBUG)

allowed_op_types = [
    'NoOp', 'Identity', 'XlaClusterOutput', 'Enter', 'Exit', 'Switch', 'Merge',
    'NextIteration', 'LoopCond', 'VarHandleOp', 'Const'
]


def get_single_while_op_body(g):
  outer_ops = g.get_operations()
  while_ops = list(filter(lambda x: x.type == 'While', outer_ops))
  assert len(while_ops) == 1
  body = g._get_function(while_ops[0].get_attr('body').name)
  return body.graph


class AutoshardTest(test_util.TensorFlowTestCase):
  @test_util.deprecated_graph_mode_only
  def testFrozenInference(self):
    def my_model(inp):
      weight_ih = array_ops.constant(np.random.rand(1024, 512),
                                     dtype=np.float32,
                                     name="W_ih")
      hidden = math_ops.matmul(weight_ih, inp)
      weight_ho = array_ops.constant(np.random.rand(1000, 1024),
                                     dtype=np.float32,
                                     name="W_ho")
      output = math_ops.matmul(weight_ho, hidden)
      return [output]

    with ops.device("cpu"):
      inp = array_ops.placeholder(np.float32, [512, 1], name="a")

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      out = ipu.ipu_compiler.compile(my_model, inputs=[inp])

    ipu.autoshard.automatic_sharding(2, inp, out[0], frozen_inference=True)

    op_list = ops.get_default_graph().get_operations()
    for o in op_list:
      if o.device == '/device:IPU:0' and o.type != 'NoOp':
        self.assertTrue(o.get_attr('_XlaSharding') is not None)

  @test_util.deprecated_graph_mode_only
  def testSimpleXlaCompileInference(self):
    def my_model(inp):
      output = inp * inp
      return [output]

    with ops.device("cpu"):
      inp = array_ops.placeholder(np.float32, [], name="a")

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      out = ipu.ipu_compiler.compile(my_model, inputs=[inp])

    ipu.autoshard.automatic_sharding(2, inp, out[0])

    op_list = ops.get_default_graph().get_operations()
    for o in op_list:
      if o.device == '/device:IPU:0' and o.type != 'NoOp':
        self.assertTrue(o.get_attr('_XlaSharding') is not None)

  @test_util.deprecated_graph_mode_only
  def testSimpleXlaCompileTraining(self):
    def my_model(inp, lab):

      x = inp
      y = lab

      x = layers.Conv2D(8, 3, padding='same', name="conv1", use_bias=False)(x)
      x = layers.Conv2D(8, 3, padding='same', name="conv2", use_bias=False)(x)
      x = layers.Conv2D(8, 3, padding='same', name="conv3", use_bias=False)(x)
      x = math_ops.reduce_max(x, axis=[1, 2])

      cross_entropy = nn.softmax_cross_entropy_with_logits_v2(
          logits=x, labels=array_ops.stop_gradient(y))
      loss = math_ops.reduce_mean(cross_entropy)
      optim = sharded_optimizer.ShardedOptimizer(
          gd.GradientDescentOptimizer(0.01))
      train = optim.minimize(cross_entropy)

      ipu.autoshard.automatic_sharding(2, inp, loss)

      return [loss, train]

    with ops.device("cpu"):
      inp = array_ops.placeholder(np.float32, [1, 12, 12, 4], name="data")
      lab = array_ops.placeholder(np.float32, [1, 8], name="labl")

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      out = ipu.ipu_compiler.compile(my_model, inputs=[inp, lab])

    op_set = ipu.sharding.dependencies([out[0]])

    for o in op_set:
      if o.device == '/device:IPU:0' and o.type not in allowed_op_types:
        self.assertTrue(o.get_attr('_XlaSharding') is not None)

  @test_util.deprecated_graph_mode_only
  def testSimpleTraining(self):
    def my_model(x, y):
      x = layers.Conv2D(8, 3, padding='same', name="conv1", use_bias=False)(x)
      x = layers.Conv2D(8, 3, padding='same', name="conv2", use_bias=False)(x)
      x = layers.Conv2D(8, 3, padding='same', name="conv3", use_bias=False)(x)
      x = math_ops.reduce_max(x, axis=[1, 2])

      cross_entropy = nn.softmax_cross_entropy_with_logits_v2(
          logits=x, labels=array_ops.stop_gradient(y))
      loss = math_ops.reduce_mean(cross_entropy)
      optim = sharded_optimizer.ShardedOptimizer(
          gd.GradientDescentOptimizer(0.01))
      train = optim.minimize(cross_entropy)
      return [loss, train]

    with ops.device("cpu"):
      inp = array_ops.placeholder(np.float32, [1, 12, 12, 4], name="data")
      lab = array_ops.placeholder(np.float32, [1, 8], name="labl")

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      l, t = my_model(inp, lab)

    ipu.autoshard.automatic_sharding(2, inp, l)

    op_set = ipu.sharding.dependencies([l, t])

    for o in op_set:
      if o.device == '/device:IPU:0' and o.type not in allowed_op_types:
        self.assertTrue(o.get_attr('_XlaSharding') is not None)

  @test_util.deprecated_graph_mode_only
  def testSimpleTrainingWithEdgeFilter(self):
    def my_model(x, y):
      x = layers.Conv2D(8, 3, padding='same', name="conv1", use_bias=False)(x)
      x = layers.Conv2D(8, 3, padding='same', name="conv2", use_bias=False)(x)
      x = layers.Conv2D(8, 3, padding='same', name="conv3", use_bias=False)(x)
      x = math_ops.reduce_max(x, axis=[1, 2])

      cross_entropy = nn.softmax_cross_entropy_with_logits_v2(
          logits=x, labels=array_ops.stop_gradient(y))
      loss = math_ops.reduce_mean(cross_entropy)
      optim = sharded_optimizer.ShardedOptimizer(
          gd.GradientDescentOptimizer(0.01))
      train = optim.minimize(cross_entropy)
      return [loss, train]

    with ops.device("cpu"):
      inp = array_ops.placeholder(np.float32, [1, 12, 12, 4], name="data")
      lab = array_ops.placeholder(np.float32, [1, 8], name="labl")

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      l, t = my_model(inp, lab)

    filt = lambda e: not (e[0] != 'conv2/Conv2D' and e[1] != 'conv3/Conv2D')

    ipu.autoshard.automatic_sharding(2, inp, l, edge_filter=filt)

    op_set = ipu.sharding.dependencies([l, t])

    for o in op_set:
      if o.device == '/device:IPU:0' and o.type not in allowed_op_types:
        self.assertTrue(o.get_attr('_XlaSharding') is not None)

  @test_util.run_v1_only("while_v2 still experimental")
  def testSimpleXlaCompileTrainingInLoop(self):
    dataset = tu.create_dual_increasing_dataset(3)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset, "feed4")

    def my_net():
      def my_model(loss, x, y):
        with ipu.scopes.ipu_scope("/device:IPU:0"):
          inp = x

          x = layers.Conv2D(8, 3, padding='same', name="conv1",
                            use_bias=False)(x)
          x = layers.Conv2D(8, 3, padding='same', name="conv2",
                            use_bias=False)(x)
          x = layers.Conv2D(8, 3, padding='same', name="conv3",
                            use_bias=False)(x)
          x = math_ops.reduce_max(x, axis=[1, 2])

          cross_entropy = nn.softmax_cross_entropy_with_logits_v2(
              logits=x, labels=array_ops.stop_gradient(y))
          loss = math_ops.reduce_mean(cross_entropy)

          optim = sharded_optimizer.ShardedOptimizer(
              gd.GradientDescentOptimizer(0.01))
          train = optim.minimize(cross_entropy)

          ipu.autoshard.automatic_sharding(2, inp, loss)

          return [loss, train]

      loss = 0.0
      return ipu.loops.repeat(10,
                              my_model, [loss],
                              infeed_queue,
                              use_while_v1=False)

    ipu.ipu_compiler.compile(my_net, inputs=[])

    body = get_single_while_op_body(ops.get_default_graph())
    op_set = body.get_operations()
    op_types = set()
    for o in op_set:
      if o.device == '/device:IPU:0' and o.type not in allowed_op_types:
        op_types.add(o.type)
        self.assertTrue(o.get_attr('_XlaSharding') is not None)

    self.assertTrue(len(op_types) > 10)
    self.assertTrue('Conv2D' in op_types)
    self.assertTrue('Conv2DBackpropInput' in op_types)
    self.assertTrue('Conv2DBackpropFilter' in op_types)
    self.assertTrue('ResourceApplyGradientDescent' in op_types)

  @test_util.run_v1_only("while_v2 still experimental")
  def testPopnnLstmXlaCompileTrainingInLoop(self):
    dataset = tu.create_dual_increasing_dataset(3,
                                                data_shape=[16, 2, 8],
                                                label_shape=[16, 2, 256])

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset, "feed1")

    def my_net():
      def my_model(loss, x, y):
        with ipu.scopes.ipu_scope("/device:IPU:0"):
          inp = x

          lstm_cell = ipu.rnn_ops.PopnnLSTM(256, dtype=dtypes.float32)
          x, _ = lstm_cell(x, training=True)

          cross_entropy = nn.softmax_cross_entropy_with_logits_v2(
              logits=x, labels=array_ops.stop_gradient(y))
          loss = math_ops.reduce_mean(cross_entropy)

          optim = sharded_optimizer.ShardedOptimizer(
              gd.GradientDescentOptimizer(0.01))
          train = optim.minimize(cross_entropy)

          ipu.autoshard.automatic_sharding(2, inp, loss)

          return [loss, train]

      loss = 0.0
      return ipu.loops.repeat(10,
                              my_model, [loss],
                              infeed_queue,
                              use_while_v1=False)

    ipu.ipu_compiler.compile(my_net, inputs=[])

    body = get_single_while_op_body(ops.get_default_graph())
    op_set = body.get_operations()
    op_types = set()

    for o in op_set:
      if o.device == '/device:IPU:0' and o.type not in allowed_op_types:
        op_types.add(o.type)
        self.assertTrue(o.get_attr('_XlaSharding') is not None)

    self.assertTrue(len(op_types) > 10)
    self.assertTrue('PopnnLstmLayer' in op_types)
    self.assertTrue('PopnnLstmLayerBackprop' in op_types)
    self.assertTrue('LogSoftmax' in op_types)
    self.assertTrue('SoftmaxCrossEntropyWithLogits' in op_types)
    self.assertTrue('ResourceApplyGradientDescent' in op_types)

  @test_util.run_v1_only("while_v2 still experimental")
  def testSimpleXlaCompileTrainingInLoopWithParam(self):
    dataset = tu.create_dual_increasing_dataset(3)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset, "feed2")

    def my_net(lr):
      def my_model(loss, x, y):
        with ipu.scopes.ipu_scope("/device:IPU:0"):
          inp = x

          x = layers.Conv2D(8, 3, padding='same', name="conv1",
                            use_bias=False)(x)
          x = layers.Conv2D(8, 3, padding='same', name="conv2",
                            use_bias=False)(x)
          x = layers.Conv2D(8, 3, padding='same', name="conv3",
                            use_bias=False)(x)
          x = math_ops.reduce_max(x, axis=[1, 2])

          cross_entropy = nn.softmax_cross_entropy_with_logits_v2(
              logits=x, labels=array_ops.stop_gradient(y))
          loss = math_ops.reduce_mean(cross_entropy)

          optim = sharded_optimizer.ShardedOptimizer(
              gd.GradientDescentOptimizer(lr))
          train = optim.minimize(cross_entropy)

          ipu.autoshard.automatic_sharding(2, inp, loss)

          return [loss, train]

      loss = 0.0
      return ipu.loops.repeat(10,
                              my_model, [loss],
                              infeed_queue,
                              use_while_v1=False)

    lr = array_ops.placeholder(dtypes.float32, [])
    ipu.ipu_compiler.compile(my_net, inputs=[lr])

    body = get_single_while_op_body(ops.get_default_graph())
    op_set = body.get_operations()
    op_types = set()
    for o in op_set:
      if o.device == '/device:IPU:0' and o.type not in allowed_op_types:
        op_types.add(o.type)
        self.assertTrue(o.get_attr('_XlaSharding') is not None)

    self.assertTrue(len(op_types) > 10)
    self.assertTrue('Conv2D' in op_types)
    self.assertTrue('Conv2DBackpropInput' in op_types)
    self.assertTrue('Conv2DBackpropFilter' in op_types)
    self.assertTrue('ResourceApplyGradientDescent' in op_types)

  @test_util.run_v1_only("no v1 loops in v2")
  def testSimpleXlaCompileTrainingInLoopV1WithEarlySharding(self):

    dataset = tu.create_dual_increasing_dataset(3)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset, "feed3")

    def my_net():
      def my_model(loss, x, y):
        with ipu.scopes.ipu_scope("/device:IPU:0"):
          inp = x

          x = layers.Conv2D(8, 3, padding='same', name="conv1",
                            use_bias=False)(x)
          x = layers.Conv2D(8, 3, padding='same', name="conv2",
                            use_bias=False)(x)
          x = layers.Conv2D(8, 3, padding='same', name="conv3",
                            use_bias=False)(x)
          x = math_ops.reduce_max(x, axis=[1, 2])

          cross_entropy = nn.softmax_cross_entropy_with_logits_v2(
              logits=x, labels=array_ops.stop_gradient(y))
          loss = math_ops.reduce_mean(cross_entropy)

          optim = sharded_optimizer.ShardedOptimizer(
              gd.GradientDescentOptimizer(0.01))
          train = optim.minimize(cross_entropy)

          ipu.autoshard.automatic_sharding(2, inp, loss)

          return [loss, train]

      loss = 0.0
      return ipu.loops.repeat(10,
                              my_model, [loss],
                              infeed_queue,
                              use_while_v1=True)

    ipu.ipu_compiler.compile(my_net, inputs=[])

    op_set = ops.get_default_graph().get_operations()
    op_types = set()

    for o in op_set:
      if o.device == '/device:IPU:0' and o.type not in allowed_op_types:
        op_types.add(o.type)
        self.assertTrue(o.get_attr('_XlaSharding') is not None)

    self.assertTrue(len(op_types) > 10)
    self.assertTrue('Conv2D' in op_types)
    self.assertTrue('Conv2DBackpropInput' in op_types)
    self.assertTrue('Conv2DBackpropFilter' in op_types)
    self.assertTrue('ResourceApplyGradientDescent' in op_types)

  @test_util.deprecated_graph_mode_only
  def testMarkOpsWithAutoshardingContext(self):

    with ipu.scopes.ipu_scope("/device:IPU:0"):

      with ipu.autoshard.ipu_autoshard():
        x = array_ops.placeholder(dtypes.float32, shape=[1, 32, 32, 4])
        y = array_ops.placeholder(dtypes.float32, shape=[1, 8])

        inp = x

        with ops.name_scope('gradients'):

          x = layers.Conv2D(8, 3, padding='same', name="conv1",
                            use_bias=False)(x)
          x = layers.Conv2D(8, 3, padding='same', name="conv2",
                            use_bias=False)(x)
          x = layers.Conv2D(8, 3, padding='same', name="conv3",
                            use_bias=False)(x)
          x = math_ops.reduce_max(x, axis=[1, 2])

        cross_entropy = nn.softmax_cross_entropy_with_logits_v2(
            logits=x, labels=array_ops.stop_gradient(y))
        loss = math_ops.reduce_mean(cross_entropy)

      optim = sharded_optimizer.ShardedOptimizer(
          gd.GradientDescentOptimizer(0.01))
      optim.minimize(loss)

      ipu.autoshard.automatic_sharding(2, inp, loss)

    to_autoshard = ops.get_default_graph().get_collection(
        ipu.sharding._IPU_AUTOSHARD)

    fwd_ops = []
    bwd_ops = []

    all_ops = ops.get_default_graph().get_operations()
    for o in all_ops:
      if o in to_autoshard:
        fwd_ops.append(o)
      else:
        bwd_ops.append(o)

    self.assertTrue(len(fwd_ops) > 10)
    self.assertTrue(len(bwd_ops) > 10)

    self.assertEqual(len([o for o in fwd_ops if o.type == 'Conv2D']), 3)


if __name__ == "__main__":
  googletest.main()
