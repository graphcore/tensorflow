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

from absl.testing import parameterized
import numpy as np
import pva

from tensorflow.compat.v1.nn.rnn_cell import DropoutWrapper
from tensorflow.python.ipu import test_utils as tu
from tensorflow.python.keras.layers import SimpleRNN, Conv1D, MaxPooling1D, Dense, Dropout, Flatten
from tensorflow.python import keras
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import googletest
from tensorflow.python.training import training as train
from tensorflow.python.layers import layers
from tensorflow.python import data
from tensorflow.python import ipu
from tensorflow.python.ipu import pipelining_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.framework import dtypes

from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.engine import sequential


def build_cells(opts, cell_class, sizes):
  rnn_layers = []
  for idx, size in enumerate(sizes):
    cell = cell_class(size, name='RNNCell%d' % idx)
    dropout = DropoutWrapper(cell,
                             output_keep_prob=opts.output_keep_prob,
                             variational_recurrent=False,
                             dtype="float32")
    rnn_layers.append(dropout)
  return rnn_layers


def build_tf_rnn1(opts, inputs):
  rnn_layer = build_cells(opts, rnn_cell.BasicRNNCell, [opts.dim * 2])[0]
  _, rnn_layer = rnn.dynamic_rnn(cell=rnn_layer,
                                 inputs=inputs,
                                 dtype=inputs.dtype,
                                 time_major=False)
  return layers.dense(rnn_layer, opts.out_dim)


def build_tf_rnn2(opts, inputs):
  rnn_layers = build_cells(opts, rnn_cell.BasicRNNCell, [opts.dim, opts.dim])
  multi_rnn_cell = rnn_cell.MultiRNNCell(rnn_layers)
  _, final_state = rnn.dynamic_rnn(cell=multi_rnn_cell,
                                   inputs=inputs,
                                   dtype=inputs.dtype,
                                   time_major=False)
  return layers.dense(final_state[1], opts.out_dim)


def build_tf_lstm1(opts, inputs):
  rnn_layers = build_cells(opts, rnn_cell.LSTMCell, [opts.dim, opts.dim * 3])
  multi_rnn_cell = rnn_cell.MultiRNNCell(rnn_layers)
  _, final_state = rnn.dynamic_rnn(cell=multi_rnn_cell,
                                   inputs=inputs,
                                   dtype=inputs.dtype,
                                   time_major=False)
  return layers.dense(final_state[1].h, opts.out_dim)


def build_tf_gru1(opts, inputs):
  rnn_layers = build_cells(opts, rnn_cell.GRUCell, [opts.dim * 2, opts.dim])
  multi_rnn_cell = rnn_cell.MultiRNNCell(rnn_layers)
  _, final_state = rnn.dynamic_rnn(cell=multi_rnn_cell,
                                   inputs=inputs,
                                   dtype=inputs.dtype,
                                   time_major=False)
  return layers.dense(final_state[1], opts.out_dim)


def build_model_rnn1(opts, inputs):
  net = SimpleRNN(opts.dim * 2, input_shape=[opts.steps, opts.in_dim])(inputs)
  net = Dropout(1 - opts.output_keep_prob)(net)
  net = Dense(units=opts.out_dim)(net)
  return net


def build_model_rnn2(opts, inputs):
  net = SimpleRNN(opts.dim * 2,
                  input_shape=[opts.steps, opts.in_dim],
                  return_sequences=True)(inputs)
  net = Dropout(1 - opts.output_keep_prob)(net)
  net = SimpleRNN(opts.dim)(net)
  net = Dropout(1 - opts.output_keep_prob)(net)
  net = Dense(units=opts.out_dim)(net)
  return net


def build_model_cnn1(opts, inputs):
  size = opts.dim
  cnn = Conv1D(size,
               kernel_size=4,
               input_shape=[opts.steps, opts.in_dim],
               dtype=opts.dtype)(inputs)
  cnn = MaxPooling1D(pool_size=2, strides=1, dtype=opts.dtype)(cnn)
  cnn = Conv1D(size, kernel_size=4, dtype=opts.dtype)(cnn)
  cnn = MaxPooling1D(pool_size=2, strides=2, dtype=opts.dtype)(cnn)
  cnn = Conv1D(size, kernel_size=4, dtype=opts.dtype)(cnn)
  cnn = Conv1D(size, kernel_size=4, dtype=opts.dtype)(cnn)
  cnn = Flatten()(cnn)
  cnn = Dense(units=opts.out_dim)(cnn)
  return cnn


def build_trivial_while(_, inputs):
  x = array_ops.transpose(inputs, [1, 0, 2])
  output = tensor_array_ops.TensorArray(x.dtype,
                                        size=x.shape[0],
                                        element_shape=[x.shape[1], x.shape[2]],
                                        name="output")
  time_ = array_ops.constant(0, dtype=dtypes.int32, name="time")
  v = variables.Variable(initial_value=7.0, trainable=True)

  def body_(time, out_ta):
    in_slice = array_ops.slice(x, [time, 0, 0], [1, x.shape[1], x.shape[2]])
    in_slice = array_ops.squeeze(in_slice)
    mul = math_ops.multiply(in_slice, v)
    new_out = out_ta.write(time, mul)
    return (time + 1, new_out)

  _, output = control_flow_ops.while_loop(
      cond=lambda time_, *_: time_ < x.shape[0],
      body=body_,
      loop_vars=(time_, output),
      maximum_iterations=x.shape[0],
      name="easy-to-spot-while")
  return math_ops.reduce_sum(output.stack(), axis=0, name="reduce_dims")


def graph_builder(model_func, opts, x):
  preds = model_func(opts, x["inputs"])
  loss = math_ops.reduce_mean(losses.mean_squared_error(x["labels"], preds))
  optimizer = train.AdamOptimizer(learning_rate=0.01, epsilon=1e-3)
  return optimizer.minimize(loss)


class TestOptions:
  def __init__(self):
    self.batches_per_step = 100
    self.dtype = np.float32
    self.batch_size = 2
    self.steps = 5
    self.dim = 128
    self.in_dim = 16
    self.out_dim = 16
    self.output_keep_prob = 0.75
    # some tests are just for debugging/comparison and not useful
    # for CI or already tested in other files. For these just return
    self.skip = False

    self.assert_tolerance = 0.01

  def createDataset(self):
    dataset = data.Dataset \
        .range((self.steps + 2) * self.batches_per_step) \
        .map(lambda i: {
            #pylint: disable=line-too-long
            "inputs": array_ops.broadcast_to(math_ops.cast(i, self.dtype), [self.batch_size, self.steps, self.in_dim]),
            "labels": array_ops.broadcast_to(math_ops.cast(i, self.dtype), [self.batch_size, self.out_dim])
        }).prefetch(self.batches_per_step).cache()
    return dataset


class RNNModelTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  def _run_test(self, opts, model_func, cycles, total_mem, max_mem):
    dataset = opts.createDataset()
    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)

    with ipu.scopes.ipu_scope('/device:IPU:0'):

      def body(*body, **kwargs):
        # pylint: disable=unused-argument
        return graph_builder(model_func, opts, kwargs)

      out = ipu.ipu_compiler.compile(
          lambda: ipu.loops.repeat(opts.batches_per_step, body, [],
                                   infeed_queue), [])

    # Configure the IPU
    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg, output_execution_profile=True)
    cfg.ipu_model.compile_ipu_code = True
    cfg.ipu_model.tiles_per_ipu = 1472
    ipu.utils.configure_ipu_system(cfg)
    ipu.utils.move_variable_initialization_to_cpu()

    # Run the model
    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(infeed_queue.initializer)
      report_helper.clear_reports()

      sess.run(out)

    report = pva.openReport(report_helper.find_report())
    self.assert_number_of_executions(report, 1)

    if cycles is not None:
      with self.subTest("cycles check"):
        self.assert_execution_report_cycles(report,
                                            cycles,
                                            tolerance=opts.assert_tolerance)

    if total_mem is not None:
      with self.subTest("total tile memory check"):
        self.assert_total_tile_memory(report,
                                      total_mem,
                                      tolerance=opts.assert_tolerance)

    if max_mem is not None:
      with self.subTest("max tile memory check"):
        self.assert_max_tile_memory(report,
                                    max_mem,
                                    tolerance=opts.assert_tolerance)

  @parameterized.named_parameters(
      {
          'testcase_name': 'tf_rnn1',
          'build': build_tf_rnn1,
          'cycles': 10773225,
          'total_memory': 20259402,
          'max_memory': 21282
      }, {
          'testcase_name': 'tf_rnn2',
          'build': build_tf_rnn2,
          'cycles': 17340132,
          'total_memory': 19675908,
          'max_memory': 22038
      }, {
          'testcase_name': 'tf_lstm1',
          'build': build_tf_lstm1,
          'cycles': 24416546,
          'total_memory': 57301126,
          'max_memory': 47800,
          'options': {
              'dims': 64,
              'steps': 3
          }
      }, {
          'testcase_name': 'tf_gru1',
          'build': build_tf_gru1,
          'cycles': 47021629,
          'total_memory': 37378608,
          'max_memory': 35128
      }, {
          'testcase_name': 'model_rnn1',
          'build': build_model_rnn1,
          'cycles': 13779950,
          'total_memory': 20849887,
          'max_memory': 23069
      }, {
          'testcase_name': 'model_rnn2',
          'build': build_model_rnn2,
          'cycles': 26412619,
          'total_memory': 27357268,
          'max_memory': 29247
      }, {
          'testcase_name': 'model_cnn1',
          'build': build_model_cnn1,
          'cycles': 8213006,
          'total_memory': 29405932,
          'max_memory': 30289,
          'options': {
              'batch_size': 1,
              'steps': 32
          }
      }, {
          'testcase_name': 'trivial_multiply',
          'build': build_trivial_while,
          'cycles': 72865837,
          'total_memory': 18036292,
          'max_memory': 15367,
          'options': {
              'batch_size': 4,
              'steps': 32,
              'in_dim': 4056,
              'out_dim': 4056
          }
      })
  @test_util.deprecated_graph_mode_only
  def test_rnn(self, build, cycles, total_memory, max_memory, options=None):
    opts = TestOptions()
    if options:
      for key, value in options.items():
        setattr(opts, key, value)
      if opts.skip:
        self.skipTest("Doesn't add value to CI")
    self._run_test(opts, build, cycles, total_memory, max_memory)

  @test_util.deprecated_graph_mode_only
  def test_trivial_multiply_pipeline(self):
    opts = TestOptions()
    dataset = opts.createDataset()

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

    with ipu.scopes.ipu_scope('/device:IPU:0'):

      def body(*body, **kwargs):  # pylint: disable=unused-argument
        def stage1(**kwargs):
          inputs = kwargs["inputs"]
          labels = kwargs["labels"]
          x = array_ops.transpose(inputs, [1, 0, 2])
          output = tensor_array_ops.TensorArray(
              x.dtype,
              size=x.shape[0],
              element_shape=[x.shape[1], x.shape[2]],
              name="output")
          time_ = array_ops.constant(0, dtype=dtypes.int32, name="time")
          v = variables.Variable(initial_value=7.0, trainable=True)

          def body_(time, out_ta):
            in_slice = array_ops.slice(x, [time, 0, 0],
                                       [1, x.shape[1], x.shape[2]])
            in_slice = array_ops.squeeze(in_slice)
            mul = math_ops.multiply(in_slice, v)
            new_out = out_ta.write(time, mul)
            return (time + 1, new_out)

          _, output = control_flow_ops.while_loop(
              cond=lambda time_, *_: time_ < x.shape[0],
              body=body_,
              loop_vars=(time_, output),
              maximum_iterations=x.shape[0],
              name="easy-to-spot-while")
          preds = math_ops.reduce_sum(output.stack(),
                                      axis=0,
                                      name="reduce_dims")
          return preds, labels

        def stage2(preds, labels):
          loss = math_ops.reduce_mean(losses.mean_squared_error(labels, preds))
          return loss

        def optimizer_fn(loss):
          optimizer = train.AdamOptimizer(learning_rate=0.01, epsilon=1e-3)
          return pipelining_ops.OptimizerFunctionOutput(optimizer, loss)

        return pipelining_ops.pipeline([stage1, stage2],
                                       opts.batches_per_step,
                                       optimizer_function=optimizer_fn,
                                       infeed_queue=infeed_queue,
                                       outfeed_queue=outfeed,
                                       outfeed_loss=False)

      out = ipu.ipu_compiler.compile(body)

    # Configure the IPU with HW.
    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg, output_execution_profile=True)
    cfg.auto_select_ipus = 2
    ipu.utils.configure_ipu_system(cfg)
    ipu.utils.move_variable_initialization_to_cpu()

    # Run the model
    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(infeed_queue.initializer)
      report_helper.clear_reports()

      sess.run(out)

    report = pva.openReport(report_helper.find_report())
    self.assert_number_of_executions(report, 1)

    with self.subTest("cycles check"):
      self.assert_execution_report_cycles(report,
                                          3657722,
                                          tolerance=opts.assert_tolerance)

    with self.subTest("total tile memory check"):
      self.assert_total_tile_memory(report,
                                    115370,
                                    tolerance=opts.assert_tolerance)

    with self.subTest("max tile memory check"):
      self.assert_max_tile_memory(report,
                                  11585,
                                  tolerance=opts.assert_tolerance)


def createLSTMKerasModel(timesteps, features, hidden_size,
                         gradient_accumulation):  # pylint: disable=unused-argument
  inputs = keras.layers.Input(shape=(timesteps, features))
  rnns = keras.layers.LSTM(hidden_size)
  outputs = keras.layers.Dense(units=8)

  return sequential.Sequential([inputs, rnns, outputs])


def createPipelinedLSTMKerasModel(timesteps, features, hidden_size,
                                  gradient_accumulation):
  model = createLSTMKerasModel(timesteps, features, hidden_size,
                               gradient_accumulation)
  model.set_pipelining_options(
      gradient_accumulation_steps_per_replica=gradient_accumulation)
  model.set_pipeline_stage_assignment([0, 1])
  return model


class RNNKerasModelTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  def createDataset(self,
                    batch_size,
                    n_samples,
                    timesteps,
                    features,
                    model_output_shape,
                    dtype='float32'):
    input_shape = (n_samples, timesteps, features)
    output_shape = (n_samples, *model_output_shape[1:])

    X, y = np.random.random(input_shape).astype(dtype), np.random.random(
        output_shape).astype(dtype)

    ds = data.Dataset.from_tensor_slices((X, y))
    ds = ds.repeat()
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds

  @parameterized.named_parameters(
      {
          "testcase_name": "lstm",
          "model_fn": createLSTMKerasModel,
          "ipu_count": 1,
          "expected_cycles": 25305718,
          "expected_total_tile_memory": 21240386,
          "expected_max_tile_memory": 30110
      }, {
          "testcase_name": "pipelined_lstm",
          "model_fn": createPipelinedLSTMKerasModel,
          "ipu_count": 2,
          "expected_cycles": 23716090,
          "expected_total_tile_memory": 33659480,
          "expected_max_tile_memory": 26134
      })
  @test_util.run_v2_only
  def test_model(self, model_fn, ipu_count, expected_cycles,
                 expected_total_tile_memory, expected_max_tile_memory):
    timesteps = 20
    features = 16
    hidden_size = 64
    n_samples = 8192
    batch_size = 16
    gradient_accumulation = 16
    total_batch = batch_size * gradient_accumulation
    steps_per_epoch = n_samples // total_batch

    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg, output_execution_profile=True)
    cfg.auto_select_ipus = ipu_count
    cfg.ipu_model.tiles_per_ipu = 1472
    ipu.utils.configure_ipu_system(cfg)

    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      model = model_fn(timesteps, features, hidden_size, gradient_accumulation)
      model.compile(loss="mse",
                    optimizer=Adam(),
                    steps_per_execution=gradient_accumulation * 2)

      dataset = self.createDataset(batch_size, n_samples, timesteps, features,
                                   model.output_shape, np.float32)
      model.fit(dataset, steps_per_epoch=steps_per_epoch, epochs=1)

    report = pva.openReport(report_helper.find_report())
    self.assert_number_of_executions(report, 1)

    tolerance = 0.01
    with self.subTest("cycles check"):
      self.assert_execution_report_cycles(report,
                                          expected_cycles,
                                          tolerance=tolerance)

    with self.subTest("total tile memory check"):
      self.assert_total_tile_memory(report,
                                    expected_total_tile_memory,
                                    tolerance=tolerance)

    with self.subTest("max tile memory check"):
      self.assert_max_tile_memory(report,
                                  expected_max_tile_memory,
                                  tolerance=tolerance)


if __name__ == "__main__":
  googletest.main()
