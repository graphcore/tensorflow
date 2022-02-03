# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

import numpy as np

from tensorflow.keras import layers
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import momentum
from tensorflow.python.ipu import embedding_ops
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import pipelining_ops
from tensorflow.python.ipu import internal_ops
from tensorflow.python.ipu import utils
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ipu.optimizers import gradient_accumulation_optimizer as ga
from tensorflow.python.ipu.tests import pipelining_test_util
from tensorflow.compat.v1 import disable_v2_behavior

disable_v2_behavior()


class PipeliningSeqTest(test_util.TensorFlowTestCase):
  @test_util.deprecated_graph_mode_only
  def testPipelineNoOutfeedInference(self):
    def stage1(x):
      with variable_scope.variable_scope("vs", use_resource=True):
        y = x + 1
        return y

    def stage2(x):
      loss = math_ops.reduce_sum(x)
      return loss

    def my_net(x):
      return pipelining_ops.pipeline(
          [stage1, stage2],
          10,
          inputs=[x],
          pipeline_schedule=pipelining_ops.PipelineSchedule.Sequential)

    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

    with ops.device("/device:IPU:0"):
      with self.assertRaisesRegex(
          ValueError, 'The last computational stage has tensor outputs'):
        ipu_compiler.compile(my_net, inputs=[x])

  @test_util.deprecated_graph_mode_only
  def testPipelineNoOutfeedWithOutputsTraining(self):
    def stage1(x):
      with variable_scope.variable_scope("vs", use_resource=True):
        y = x + 1
        return y

    def stage2(x):
      y = layers.Conv2D(2,
                        1,
                        use_bias=True,
                        bias_initializer=init_ops.ones_initializer(),
                        kernel_initializer=init_ops.ones_initializer())(x)
      loss = math_ops.reduce_sum(y)
      return loss

    def optimizer_function(loss):
      opt = gradient_descent.GradientDescentOptimizer(0.01)
      return pipelining_ops.OptimizerFunctionOutput(opt, loss)

    def my_net(x):
      return pipelining_ops.pipeline(
          [stage1, stage2],
          10,
          inputs=[x],
          optimizer_function=optimizer_function,
          pipeline_schedule=pipelining_ops.PipelineSchedule.Sequential)

    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

    with ops.device("/device:IPU:0"):
      with self.assertRaisesRegex(ValueError,
                                  'The last computational stage has tensor'):
        ipu_compiler.compile(my_net, inputs=[x])

  @test_util.deprecated_graph_mode_only
  def testPipelineInvalidDeviceMapping(self):
    dataset = tu.create_single_increasing_dataset(5, shape=[4, 4, 2])
    dataset = dataset.batch(batch_size=2, drop_remainder=True)

    def dataset_parser(value):
      a = value
      b = (value + 10.) / 2.0
      return {"a": a, "b": b}

    dataset = dataset.map(dataset_parser)
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    def stage1(c, **kwargs):
      with variable_scope.variable_scope("vs", use_resource=True):
        y = layers.Conv2D(2,
                          1,
                          use_bias=True,
                          kernel_initializer=init_ops.ones_initializer(),
                          name='conv1')(kwargs["a"])
        return y + kwargs["b"], c

    def stage2(x, c):
      return math_ops.reduce_sum(x) + c

    def stage3(x):
      return x

    with ops.device('cpu'):
      c = array_ops.placeholder(np.float32, shape=[])

    # Wrong type:
    with self.assertRaisesRegex(
        TypeError, 'device_mapping argument needs to be a list or a tuple'):
      pipelining_ops.pipeline(
          [stage1, stage2, stage3],
          3,
          inputs=[c],
          infeed_queue=infeed_queue,
          outfeed_queue=outfeed_queue,
          device_mapping=1,
          pipeline_schedule=pipelining_ops.PipelineSchedule.Sequential)

    # Too many values:
    with self.assertRaisesRegex(ValueError,
                                'Each stage must be mapped to an IPU'):
      pipelining_ops.pipeline(
          [stage1, stage2, stage3],
          3,
          inputs=[c],
          infeed_queue=infeed_queue,
          outfeed_queue=outfeed_queue,
          device_mapping=list(range(4)),
          pipeline_schedule=pipelining_ops.PipelineSchedule.Sequential)

    # Not enough values:
    with self.assertRaisesRegex(ValueError,
                                'Each stage must be mapped to an IPU'):
      pipelining_ops.pipeline(
          [stage1, stage2, stage3],
          3,
          inputs=[c],
          infeed_queue=infeed_queue,
          outfeed_queue=outfeed_queue,
          device_mapping=tuple(range(1)),
          pipeline_schedule=pipelining_ops.PipelineSchedule.Sequential)

  @test_util.deprecated_graph_mode_only
  def testPipelineWithDeviceMapping(self):
    with tu.ipu_session() as sess:
      dataset = tu.create_single_increasing_dataset(5, shape=[4, 4, 2])
      dataset = dataset.batch(batch_size=2, drop_remainder=True)

      def dataset_parser(value):
        a = value
        b = (value + 10.) / 2.0
        return {"a": a, "b": b}

      dataset = dataset.map(dataset_parser)
      infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()
      device_mapping = [2, 0, 1]

      def stage1(c, **kwargs):
        with variable_scope.variable_scope("vs", use_resource=True):
          y = layers.Conv2D(2,
                            1,
                            use_bias=True,
                            kernel_initializer=init_ops.ones_initializer(),
                            name='conv1')(kwargs["a"])
          return y + kwargs["b"], c

      def stage2(x, c):
        return math_ops.reduce_sum(x) + c

      def stage3(x):
        return x

      def my_net(c):
        return pipelining_ops.pipeline(
            [stage1, stage2, stage3],
            12,
            inputs=[c],
            infeed_queue=infeed_queue,
            outfeed_queue=outfeed_queue,
            device_mapping=device_mapping,
            pipeline_schedule=pipelining_ops.PipelineSchedule.Sequential,
            reduction_method=ga.GradientAccumulationReductionMethod.SUM)

      with ops.device('cpu'):
        c = array_ops.placeholder(np.float32, shape=[])

      with ops.device("/device:IPU:0"):
        r = ipu_compiler.compile(my_net, inputs=[c])

      cfg = IPUConfig()
      cfg.auto_select_ipus = 4
      cfg.ipu_model.tiles_per_ipu = 1472
      cfg.ipu_model.compile_ipu_code = False
      tu.enable_ipu_events(cfg)
      cfg.configure_ipu_system()
      utils.move_variable_initialization_to_cpu()

      outfeed_op = outfeed_queue.dequeue()

      report_json = tu.ReportJSON(self, sess)
      report_json.reset()
      sess.run(variables.global_variables_initializer())
      sess.run(infeed_queue.initializer)
      sess.run(r, {c: 10.01})
      losses_pipeline = sess.run(outfeed_op)
      self.assertAllClose(losses_pipeline, [
          410.01, 730.01, 650.01, 570.01, 890.01, 410.01, 730.01, 650.01,
          570.01, 890.01, 410.01, 730.01
      ])
      report_json.parse_log()
      report_json.assert_pipeline_stages_on_expected_ipu(
          device_mapping, cfg.ipu_model.tiles_per_ipu)

  @test_util.deprecated_graph_mode_only
  def testPipelineWithDeviceMappingSameIpu(self):
    with tu.ipu_session() as sess:
      dataset = tu.create_single_increasing_dataset(5, shape=[4, 4, 2])
      dataset = dataset.batch(batch_size=2, drop_remainder=True)

      def dataset_parser(value):
        a = value
        b = (value + 10.) / 2.0
        return {"a": a, "b": b}

      dataset = dataset.map(dataset_parser)
      infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()
      device_mapping = [2, 2, 2]

      def stage1(c, **kwargs):
        with variable_scope.variable_scope("vs", use_resource=True):
          y = layers.Conv2D(2,
                            1,
                            use_bias=True,
                            kernel_initializer=init_ops.ones_initializer(),
                            name='conv1')(kwargs["a"])
          return y + kwargs["b"], c

      def stage2(x, c):
        internal_ops.print_tensor(c, "stage2_c")
        return math_ops.reduce_sum(x) + c

      def stage3(x):
        internal_ops.print_tensor(x, "stage3_x")
        return x

      def my_net(c):
        return pipelining_ops.pipeline(
            [stage1, stage2, stage3],
            12,
            inputs=[c],
            infeed_queue=infeed_queue,
            outfeed_queue=outfeed_queue,
            device_mapping=device_mapping,
            pipeline_schedule=pipelining_ops.PipelineSchedule.Sequential,
            reduction_method=ga.GradientAccumulationReductionMethod.SUM)

      with ops.device('cpu'):
        c = array_ops.placeholder(np.float32, shape=[])

      with ops.device("/device:IPU:0"):
        r = ipu_compiler.compile(my_net, inputs=[c])

      cfg = IPUConfig()
      cfg.auto_select_ipus = 4
      cfg.ipu_model.tiles_per_ipu = 1472
      cfg.ipu_model.compile_ipu_code = False
      tu.enable_ipu_events(cfg)
      cfg.configure_ipu_system()
      utils.move_variable_initialization_to_cpu()

      outfeed_op = outfeed_queue.dequeue()

      report_json = tu.ReportJSON(self, sess)
      report_json.reset()
      sess.run(variables.global_variables_initializer())
      sess.run(infeed_queue.initializer)
      sess.run(r, {c: 10.01})
      losses_pipeline = sess.run(outfeed_op)
      self.assertAllClose(losses_pipeline, [
          410.01, 730.01, 650.01, 570.01, 890.01, 410.01, 730.01, 650.01,
          570.01, 890.01, 410.01, 730.01
      ])
      report_json.parse_log()
      report_json.assert_pipeline_stages_on_expected_ipu(
          device_mapping, cfg.ipu_model.tiles_per_ipu)

  @test_util.deprecated_graph_mode_only
  def testPipelineWithInfeedsKwargs(self):
    with tu.ipu_session() as sess:
      dataset = tu.create_single_increasing_dataset(5, shape=[4, 4, 2])
      dataset = dataset.batch(batch_size=2, drop_remainder=True)

      def dataset_parser(value):
        a = value
        b = (value + 10.) / 2.0
        return {"a": a, "b": b}

      dataset = dataset.map(dataset_parser)
      infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

      def stage1(c, **kwargs):
        with variable_scope.variable_scope("vs", use_resource=True):
          y = layers.Conv2D(2,
                            1,
                            use_bias=True,
                            kernel_initializer=init_ops.ones_initializer(),
                            name='conv1')(kwargs["a"])
          return y + kwargs["b"], c

      def stage2(x, c):
        return math_ops.reduce_sum(x) + c

      def stage3(x):
        return x

      def my_net(c):
        return pipelining_ops.pipeline(
            [stage1, stage2, stage3],
            12,
            inputs=[c],
            infeed_queue=infeed_queue,
            outfeed_queue=outfeed_queue,
            pipeline_schedule=pipelining_ops.PipelineSchedule.Sequential,
            reduction_method=ga.GradientAccumulationReductionMethod.SUM)

      with ops.device('cpu'):
        c = array_ops.placeholder(np.float32, shape=[])

      with ops.device("/device:IPU:0"):
        r = ipu_compiler.compile(my_net, inputs=[c])

      cfg = IPUConfig()
      cfg.auto_select_ipus = 4
      cfg.ipu_model.tiles_per_ipu = 1472
      tu.enable_ipu_events(cfg)
      cfg.configure_ipu_system()
      utils.move_variable_initialization_to_cpu()

      outfeed_op = outfeed_queue.dequeue()

      report_json = tu.ReportJSON(self, sess)
      report_json.reset()
      sess.run(variables.global_variables_initializer())
      sess.run(infeed_queue.initializer)
      sess.run(r, {c: 10.01})
      losses_pipeline = sess.run(outfeed_op)
      self.assertAllClose(losses_pipeline, [
          410.01, 730.01, 650.01, 570.01, 890.01, 410.01, 730.01, 650.01,
          570.01, 890.01, 410.01, 730.01
      ])
      report_json.parse_log()
      report_json.assert_pipeline_stages_on_expected_ipu(
          (0, 1, 3), cfg.ipu_model.tiles_per_ipu)

  @test_util.deprecated_graph_mode_only
  def testIllegalCapture(self):
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    with ops.device('cpu'):
      y = array_ops.placeholder(np.float32, shape=[])

    def stage1(x):
      return x * y

    def stage2(x):
      return x

    def model_pipeline(x):
      return pipelining_ops.pipeline(
          [stage1, stage2],
          10,
          inputs=[x],
          outfeed_queue=outfeed_queue,
          pipeline_schedule=pipelining_ops.PipelineSchedule.Sequential)

    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])
      y = array_ops.placeholder(np.float32, shape=[])

    with ops.device("/device:IPU:0"):
      with self.assertRaisesRegex(ValueError, 'Trying to capture the tensor'):
        ipu_compiler.compile(model_pipeline, inputs=[x])

  @test_util.deprecated_graph_mode_only
  def testPipelineOnlyOneStage(self):
    def stage1(x):
      return x

    def my_net(x):
      return pipelining_ops.pipeline(
          [stage1],
          10,
          inputs=[x],
          pipeline_schedule=pipelining_ops.PipelineSchedule.Sequential)

    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

    with ops.device("/device:IPU:0"):
      with self.assertRaisesRegex(ValueError,
                                  'Pipeline requires at least two'):
        ipu_compiler.compile(my_net, inputs=[x])

  @test_util.deprecated_graph_mode_only
  def testDuplicateInputsOutputs(self):
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    def stage1(x, y):
      return x, y, y, x

    # The above should be optimised to a single copy for each duplicate output.
    def stage2(x1, y1, y2, x2):
      return x1, y1, y2, x2

    # Same for this stage
    def stage3(_x1, _y1, y2, x2):
      return x2, y2

    def model_pipeline(x, y):
      return pipelining_ops.pipeline(
          [stage1, stage2, stage3],
          12,
          inputs=[x, y],
          outfeed_queue=outfeed_queue,
          pipeline_schedule=pipelining_ops.PipelineSchedule.Sequential)

    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])
      y = array_ops.placeholder(np.float32, shape=[1, 2])

    with ops.device("/device:IPU:0"):
      compiled_model_pipeline = ipu_compiler.compile(model_pipeline,
                                                     inputs=[x, y])

    cfg = IPUConfig()
    cfg.auto_select_ipus = 4
    cfg.configure_ipu_system()
    utils.move_variable_initialization_to_cpu()

    #TODO(T10784) test how many IPU copies are here once we insert IPU copies.
    outfeed_op = outfeed_queue.dequeue()
    with tu.ipu_session() as sess:
      sess.run(compiled_model_pipeline, {
          x: np.ones(x.shape),
          y: np.ones(y.shape)
      })
      output = sess.run(outfeed_op)
      for i in range(12):
        self.assertAllClose(output[0][i], np.ones(x.shape))
        self.assertAllClose(output[1][i], np.ones(y.shape))

  @test_util.deprecated_graph_mode_only
  def testPipelineCompare1(self):
    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(7, shape=[4, 4, 2])
      dataset = dataset.batch(batch_size=2, drop_remainder=True)

      def dataset_parser(value):
        img = value / 7
        label = value[0][0][0][0]
        return img, label

      return dataset.map(dataset_parser)

    gradient_accumulation_count = 24
    repeat_count = 2

    def optimizer_fn():
      return gradient_descent.GradientDescentOptimizer(0.01)

    def stage1(c, img, label):
      with variable_scope.variable_scope("stage1", use_resource=True):
        y = layers.Conv2D(
            2,
            1,
            use_bias=True,
            kernel_initializer=init_ops.constant_initializer(0.5),
            bias_initializer=init_ops.constant_initializer(0.5),
            name='conv1')(img)
        return y, c, label

    def stage2(x, c, label):
      with variable_scope.variable_scope("stage2", use_resource=True):
        return x * 20, c, label

    def stage3(x, c, label):
      with variable_scope.variable_scope("stage3", use_resource=True):
        return layers.Dense(
            2,
            kernel_initializer=init_ops.constant_initializer(0.5),
            bias_initializer=init_ops.constant_initializer(0.5))(x), c, label

    def stage4(x, c, label):
      with variable_scope.variable_scope("stage4", use_resource=True):
        return math_ops.reduce_sum(
            layers.Dense(2,
                         kernel_initializer=init_ops.constant_initializer(0.5),
                         bias_initializer=init_ops.constant_initializer(0.5))
            (x)) + c + label

    def inputs_fn():
      with ops.device('cpu'):
        return [array_ops.placeholder(np.float32, shape=[])]

    pipelining_test_util.PipelineTester.compare_pipeline_to_cpu(
        [stage1, stage2, stage3, stage4],
        inputs_fn, [10.01],
        repeat_count,
        gradient_accumulation_count,
        dataset_fn,
        optimizer_fn,
        self,
        14172,
        schedule=pipelining_ops.PipelineSchedule.Sequential)

  @test_util.deprecated_graph_mode_only
  def testPipelineCompare2(self):
    # Resnet like network.
    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(100, shape=[4])
      dataset = dataset.batch(batch_size=32, drop_remainder=True)
      dataset = dataset.batch(batch_size=32, drop_remainder=True)
      dataset = dataset.batch(batch_size=2, drop_remainder=True)

      def dataset_parser(value):
        img = value
        label = math_ops.reduce_mean(img, axis=[1, 2, 3])
        return img, math_ops.cast(label, np.int32)

      return dataset.map(dataset_parser)

    gradient_accumulation_count = 18
    repeat_count = 2

    def optimizer_fn():
      return gradient_descent.GradientDescentOptimizer(0.01)

    def fixed_padding(inputs, kernel_size):
      pad_total = kernel_size - 1
      pad_beg = pad_total // 2
      pad_end = pad_total - pad_beg
      padded_inputs = array_ops.pad(
          inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
      return padded_inputs

    def block(name, first_stride, out_filters, count, x):

      for i in range(count):
        shape_in = x.shape
        stride = first_stride if (i == 0) else 1
        if stride > 1:
          x = fixed_padding(x, 3)
        sc = x

        with variable_scope.variable_scope(name + "/" + str(i) + "/1"):
          x = conv(x, 3, stride, out_filters)
          x = nn.relu(x)

        with variable_scope.variable_scope(name + "/" + str(i) + "/2"):
          x = conv(x, 3, 1, out_filters)

          # shortcut
          if stride != 1:
            sc = array_ops.strided_slice(sc, [0, 0, 0, 0],
                                         sc.shape,
                                         strides=[1, stride, stride, 1])
          pad = int(x.shape[3] - shape_in[3])
          if pad != 0:
            sc = array_ops.pad(sc, paddings=[[0, 0], [0, 0], [0, 0], [0, pad]])

          x = nn.relu(x + sc)

      return x

    def fc(x, num_units_out):
      return layers.Dense(
          num_units_out,
          kernel_initializer=init_ops.constant_initializer(0.1),
          bias_initializer=init_ops.constant_initializer(0.0))(x)

    def max_pool(x, ksize=3, stride=2):
      return layers.MaxPooling2D(ksize, stride, padding='SAME')(x)

    def conv(x, ksize, stride, filters_out):
      return layers.Conv2D(
          filters_out,
          ksize,
          stride,
          'SAME',
          kernel_initializer=init_ops.constant_initializer(0.1),
          bias_initializer=init_ops.constant_initializer(0.0))(x)

    def stage1(img, label):
      with variable_scope.variable_scope("stage1", use_resource=True):
        x = conv(img, 3, 2, 4)
        x = nn.relu(x)
        x = max_pool(x, ksize=3, stride=2)
        return x, label

    def stage2(x, label):
      with variable_scope.variable_scope("stage2", use_resource=True):
        x = block("b", 2, 4, 1, x)
        return x, label

    def stage3(x, label):
      with variable_scope.variable_scope("stage3", use_resource=True):
        x = math_ops.reduce_mean(x, axis=[1, 2])
        x = fc(x, 50)
        loss = math_ops.reduce_mean(
            nn.sparse_softmax_cross_entropy_with_logits(logits=x,
                                                        labels=label))
        return loss

    pipelining_test_util.PipelineTester.compare_pipeline_to_sharding(
        [stage1, stage2, stage3],
        lambda: [], [],
        repeat_count,
        gradient_accumulation_count,
        dataset_fn,
        optimizer_fn,
        self,
        15214,
        schedule=pipelining_ops.PipelineSchedule.Sequential)

  @test_util.deprecated_graph_mode_only
  def testPipelineCompare3(self):
    if utils.running_on_ipu_model():
      self.skipTest("Replicated top level graphs are not supported on the "
                    "IPU_MODEL target")

    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(10, shape=[4])
      dataset = dataset.batch(batch_size=2, drop_remainder=True)

      def dataset_parser(value):
        label = math_ops.reduce_mean(value, axis=[1])
        return math_ops.cast(value,
                             np.int32), math_ops.cast(label / 10, np.int32)

      return dataset.map(dataset_parser)

    gradient_accumulation_count = 24
    repeat_count = 2

    def optimizer_fn():
      return gradient_descent.GradientDescentOptimizer(0.01)

    def stage1(idx, label):
      with variable_scope.variable_scope("stage1", use_resource=True):
        embedding = variable_scope.get_variable(
            "c",
            shape=[10, 12],
            dtype=np.float32,
            initializer=init_ops.constant_initializer(10.01),
            trainable=True)
        x = embedding_ops.embedding_lookup(embedding, idx)
        return x, label

    def stage2(x, label):
      with variable_scope.variable_scope("stage2", use_resource=True):
        return x, label

    def stage3(x, label):
      with variable_scope.variable_scope("stage3", use_resource=True):
        return x, label

    def stage4(x, label):
      with variable_scope.variable_scope("stage4", use_resource=True):
        logits = math_ops.reduce_sum(x, axis=[-1])
        loss = math_ops.reduce_mean(
            nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=label))
        return loss

    pipelining_test_util.PipelineTester.compare_pipeline_to_cpu(
        [stage1, stage2, stage3, stage4],
        lambda: [], [],
        repeat_count,
        gradient_accumulation_count,
        dataset_fn,
        optimizer_fn,
        self,
        13821,
        schedule=pipelining_ops.PipelineSchedule.Sequential)

  @test_util.deprecated_graph_mode_only
  def testPipelineCompareSharedWeights(self):
    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(7, shape=[4, 4])

      def dataset_parser(value):
        img = value
        label = value[0][0] % 4
        return img, math_ops.cast(label, np.int32)

      dataset = dataset.map(dataset_parser)

      return dataset.batch(batch_size=2, drop_remainder=True)

    gradient_accumulation_count = 20
    repeat_count = 2

    def optimizer_fn():
      return momentum.MomentumOptimizer(0.01, 0.98)

    def stage1(x, label):
      with variable_scope.variable_scope("vs", use_resource=True):
        weight = variable_scope.get_variable(
            "w0",
            shape=[4, 4],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
        x = math_ops.matmul(x, weight)
        return x, label

    def stage2(x, label):
      with variable_scope.variable_scope("vs", use_resource=True):
        weight = variable_scope.get_variable(
            "w1",
            shape=[4, 4],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
        x = math_ops.matmul(x, weight)
        return x, label

    def stage3(x, label):
      with variable_scope.variable_scope("vs", use_resource=True):
        weight = variable_scope.get_variable(
            "w2",
            shape=[4, 4],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
        x = math_ops.matmul(x, weight)
        return x, label

    def stage4(x, label):
      with variable_scope.variable_scope("vs", use_resource=True):
        weight = variable_scope.get_variable(
            "w3",
            shape=[4, 4],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
        x = math_ops.matmul(x, weight)
        return x, label

    def stage5(x, label):
      # Ruse the weight here.
      with variable_scope.variable_scope("vs", use_resource=True, reuse=True):
        weight = variable_scope.get_variable(
            "w0",
            shape=[4, 4],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
        x = math_ops.matmul(x, weight)
        logits = math_ops.reduce_mean(x, axis=[1])
        loss = math_ops.reduce_mean(
            nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=label))
        return loss

    def inputs_fn():
      with ops.device('cpu'):
        return []

    pipelining_test_util.PipelineTester.compare_pipeline_to_cpu(
        [stage1, stage2, stage3, stage4, stage5],
        inputs_fn, [10.01],
        repeat_count,
        gradient_accumulation_count,
        dataset_fn,
        optimizer_fn,
        self,
        21458,
        schedule=pipelining_ops.PipelineSchedule.Sequential,
        device_mapping=[0, 0, 2, 3, 0])

  @test_util.deprecated_graph_mode_only
  def testPipelineCompareSharedWeights2(self):
    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(7, shape=[4, 4])

      def dataset_parser(value):
        img = value
        label = value[0][0] % 4
        return img, math_ops.cast(label, np.int32)

      dataset = dataset.map(dataset_parser)

      return dataset.batch(batch_size=2, drop_remainder=True)

    gradient_accumulation_count = 20
    repeat_count = 2

    def optimizer_fn():
      return momentum.MomentumOptimizer(0.01, 0.98)

    def stage1(x, label):
      with variable_scope.variable_scope("vs", use_resource=True):
        weight = variable_scope.get_variable(
            "w0",
            shape=[4, 4],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
        x = math_ops.matmul(x, weight)
        return x, label

    def stage2(x, label):
      return nn.relu(x), label

    def stage3(x, label):
      return nn.relu(x), label

    def stage4(x, label):
      return nn.relu(x), label

    def stage5(x, label):
      # Ruse the weight here.
      with variable_scope.variable_scope("vs", use_resource=True, reuse=True):
        weight = variable_scope.get_variable(
            "w0",
            shape=[4, 4],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
        x = math_ops.matmul(x, weight)
        logits = math_ops.reduce_mean(x, axis=[1])
        loss = math_ops.reduce_mean(
            nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=label))
        return loss

    def inputs_fn():
      with ops.device('cpu'):
        return []

    pipelining_test_util.PipelineTester.compare_pipeline_to_cpu(
        [stage1, stage2, stage3, stage4, stage5],
        inputs_fn, [10.01],
        repeat_count,
        gradient_accumulation_count,
        dataset_fn,
        optimizer_fn,
        self,
        21458,
        schedule=pipelining_ops.PipelineSchedule.Sequential,
        device_mapping=[0, 1, 0, 2, 0],
        rtol=1e-5,
        atol=1e-5)

  @test_util.deprecated_graph_mode_only
  def testPipelineCompareSharedWeights3(self):
    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(7, shape=[4, 4])

      def dataset_parser(value):
        img = value
        label = value[0][0] % 4
        return img, math_ops.cast(label, np.int32)

      dataset = dataset.map(dataset_parser)

      return dataset.batch(batch_size=2, drop_remainder=True)

    gradient_accumulation_count = 20
    repeat_count = 2

    def optimizer_fn():
      return momentum.MomentumOptimizer(0.01, 0.98)

    def stage1(x, label):
      with variable_scope.variable_scope("vs", use_resource=True):
        weight = variable_scope.get_variable(
            "w0",
            shape=[4, 4],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
        x = math_ops.matmul(x, weight)
        return x, label

    def stage2(x, label):
      return nn.relu(x), label

    def stage3(x, label):
      # Ruse the weight here.
      with variable_scope.variable_scope("vs", use_resource=True, reuse=True):
        weight = variable_scope.get_variable(
            "w0",
            shape=[4, 4],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
        x = math_ops.matmul(x, weight)
      return nn.relu(x), label

    def stage4(x, label):
      return nn.relu(x), label

    def stage5(x, label):
      # Ruse the weight here.
      with variable_scope.variable_scope("vs", use_resource=True, reuse=True):
        weight = variable_scope.get_variable(
            "w0",
            shape=[4, 4],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
        x = math_ops.matmul(x, weight)
        logits = math_ops.reduce_mean(x, axis=[1])
        loss = math_ops.reduce_mean(
            nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=label))
        return loss

    def inputs_fn():
      with ops.device('cpu'):
        return []

    pipelining_test_util.PipelineTester.compare_pipeline_to_cpu(
        [stage1, stage2, stage3, stage4, stage5],
        inputs_fn, [10.01],
        repeat_count,
        gradient_accumulation_count,
        dataset_fn,
        optimizer_fn,
        self,
        21458,
        schedule=pipelining_ops.PipelineSchedule.Sequential,
        device_mapping=[0, 1, 0, 2, 0],
        rtol=1e-5,
        atol=1e-5)


if __name__ == "__main__":
  googletest.main()
