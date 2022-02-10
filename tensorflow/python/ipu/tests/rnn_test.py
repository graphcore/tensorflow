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
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.keras.layers import SimpleRNN, Conv1D, MaxPooling1D, Dense, Dropout, Flatten
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
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.framework import dtypes
from tensorflow import __version__ as version

TF1 = version.split('.')[0] == '1'


def build_cells(opts, cell_class, sizes):
  rnn_layers = []
  for idx, size in enumerate(sizes):
    cell = cell_class(size, name='RNNCell%d' % idx)
    dropout = DropoutWrapper(cell, output_keep_prob=opts.output_keep_prob)
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


def build_custom_lstm(opts, inputs):
  x = array_ops.transpose(inputs, [1, 0, 2])
  x, _ = ipu.ops.rnn_ops.PopnnLSTM(128)(x)
  _, x = ipu.ops.rnn_ops.PopnnLSTM(384)(x)
  return layers.dense(x[1], opts.out_dim)


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


class RNNModelTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  def _run_test(self,
                opts,
                model_func,
                cycles,
                total_mem,
                max_mem,
                tolerance=0.01):
    dataset = data.Dataset \
        .range((opts.steps + 2) * opts.batches_per_step) \
        .map(lambda i: {
            #pylint: disable=line-too-long
            "inputs": array_ops.broadcast_to(math_ops.cast(i, opts.dtype), [opts.batch_size, opts.steps, opts.in_dim]),
            "labels": array_ops.broadcast_to(math_ops.cast(i, opts.dtype), [opts.batch_size, opts.out_dim])
        }).prefetch(opts.batches_per_step).cache()
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
    assertions = []
    try:
      if cycles is not None:
        self.assert_execution_report_cycles(report,
                                            cycles,
                                            tolerance=tolerance)
    except AssertionError as e:
      assertions.append("assert_execution_report_cycles: %s" % e)

    try:
      if total_mem is not None:
        self.assert_total_tile_memory(report, total_mem, tolerance=tolerance)
    except AssertionError as e:
      assertions.append("assert_total_tile_memory: %s" % e)

    try:
      if max_mem is not None:
        self.assert_max_tile_memory(report, max_mem, tolerance=tolerance)
    except AssertionError as e:
      assertions.append("assert_max_tile_memory: %s" % e)

    if assertions:
      raise AssertionError("\n".join(assertions))

  @parameterized.named_parameters(
      {
          'testcase_name': 'tf_rnn1',
          'build': build_tf_rnn1,
          'cycles': 11391266 if TF1 else 14855062,
          'total_memory': 32665178 if TF1 else 31937906,
          'max_memory': 29053 if TF1 else 30193
      },
      {
          'testcase_name': 'tf_rnn2',
          'build': build_tf_rnn2,
          'cycles': 20870123 if TF1 else 22976937,
          'total_memory': 37718509 if TF1 else 37896621,
          'max_memory': 36893 if TF1 else 37641
      },
      {
          'testcase_name': 'tf_lstm1',
          'build': build_tf_lstm1,
          'cycles': 25254169 if TF1 else 27211500,
          'total_memory': 84836344,
          'max_memory': 65399,
          'options': {
              'dims': 64,
              'steps': 3
          }
      },
      {
          # This is just included so that I have an easy comparison
          # against what the native RNNs should be performing at, and
          # an easy way to get a report for it
          'testcase_name': 'popnn_lstm1',
          'build': build_custom_lstm,
          # As this test is skipped by default we don't track these
          # values so skip the checks
          'cycles': None,
          'total_memory': None,
          'max_memory': None,
          'options': {
              'skip': True
          }
      },
      {
          'testcase_name': 'tf_gru1',
          'build': build_tf_gru1,
          'cycles': 51213564 if TF1 else 55012169,
          'total_memory': 84621193 if TF1 else 84968361,
          'max_memory': 72890 if TF1 else 73666
      },
      {
          'testcase_name': 'model_rnn1',
          'build': build_model_rnn1,
          'cycles': 16148604 if TF1 else 17910091,
          'total_memory': 37109561 if TF1 else 36887285,
          'max_memory': 35467 if TF1 else 35803
      },
      {
          'testcase_name': 'model_rnn2',
          'build': build_model_rnn2,
          'cycles': 33874539 if TF1 else 34376350,
          'total_memory': 59999918 if TF1 else 59689211,
          'max_memory': 54118 if TF1 else 54421
      },
      {
          'testcase_name': 'model_cnn1',
          'build': build_model_cnn1,
          'cycles': 10277400 if TF1 else 10517000,
          'total_memory': 28276555 if TF1 else 28643523,
          'max_memory': 29895 if TF1 else 29895,
          'options': {
              'batch_size': 1,
              'steps': 32
          }
      },
      {
          'testcase_name': 'trivial_multiply',
          'build': build_trivial_while,
          'cycles': 59433498 if TF1 else 69455700,
          'total_memory': 18138339 if TF1 else 20658385,
          'max_memory': 16493 if TF1 else 18771,
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


if __name__ == "__main__":
  googletest.main()
