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
  rnn_layer = build_cells(opts, rnn_cell.BasicRNNCell, [256])[0]
  _, rnn_layer = rnn.dynamic_rnn(cell=rnn_layer,
                                 inputs=inputs,
                                 dtype=inputs.dtype,
                                 time_major=False)
  return layers.dense(rnn_layer, opts.out_dim)


def build_tf_rnn2(opts, inputs):
  rnn_layers = build_cells(opts, rnn_cell.BasicRNNCell, [128, 128])
  multi_rnn_cell = rnn_cell.MultiRNNCell(rnn_layers)
  _, final_state = rnn.dynamic_rnn(cell=multi_rnn_cell,
                                   inputs=inputs,
                                   dtype=inputs.dtype,
                                   time_major=False)
  return layers.dense(final_state[1], opts.out_dim)


def build_tf_lstm1(opts, inputs):
  rnn_layers = build_cells(opts, rnn_cell.LSTMCell, [128, 384])
  multi_rnn_cell = rnn_cell.MultiRNNCell(rnn_layers)
  _, final_state = rnn.dynamic_rnn(cell=multi_rnn_cell,
                                   inputs=inputs,
                                   dtype=inputs.dtype,
                                   time_major=False)
  return layers.dense(final_state[1].h, opts.out_dim)


def build_tf_gru1(opts, inputs):
  rnn_layers = build_cells(opts, rnn_cell.GRUCell, [128, 64])
  multi_rnn_cell = rnn_cell.MultiRNNCell(rnn_layers)
  _, final_state = rnn.dynamic_rnn(cell=multi_rnn_cell,
                                   inputs=inputs,
                                   dtype=inputs.dtype,
                                   time_major=False)
  return layers.dense(final_state[1], opts.out_dim)


def build_model_rnn1(opts, inputs):
  net = SimpleRNN(256, input_shape=[opts.steps, opts.in_dim])(inputs)
  net = Dropout(1 - opts.output_keep_prob)(net)
  net = Dense(units=opts.out_dim)(net)
  return net


def build_model_rnn2(opts, inputs):
  net = SimpleRNN(256,
                  input_shape=[opts.steps, opts.in_dim],
                  return_sequences=True)(inputs)
  net = Dropout(1 - opts.output_keep_prob)(net)
  net = SimpleRNN(128)(net)
  net = Dropout(1 - opts.output_keep_prob)(net)
  net = Dense(units=opts.out_dim)(net)
  return net


def build_model_cnn1(opts, inputs):
  size = 64
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


def graph_builder(model_func, opts, x):
  preds = model_func(opts, x["inputs"])
  loss = math_ops.reduce_mean(losses.mean_squared_error(x["labels"], preds))
  optimizer = train.AdamOptimizer(learning_rate=0.01, epsilon=1e-3)
  return optimizer.minimize(loss)


class TestOptions:
  def __init__(self):
    self.batches_per_step = 12
    self.iterations = 6
    self.dtype = np.float32
    self.batch_size = 2
    self.steps = 5
    self.in_dim = 16
    self.out_dim = 16
    self.output_keep_prob = 0.75


class RNNModelTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  def _run_test(self, opts, model_func, cs_counters, total_mem, max_mem):
    dataset = data.Dataset \
        .range((opts.iterations + 2) * opts.batches_per_step) \
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
    report_helper.set_autoreport_options(cfg)
    ipu.utils.configure_ipu_system(cfg)
    ipu.utils.move_variable_initialization_to_cpu()

    # Run the model
    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(infeed_queue.initializer)
      report_helper.clear_reports()

      sess.run(out)

    report = pva.openReport(report_helper.find_report())

    self.assert_compute_sets_count(report, cs_counters)
    self.assert_total_tile_memory(report, total_mem, tolerance=0.1)
    self.assert_max_tile_memory(report, max_mem, tolerance=0.1)

  @parameterized.parameters(
      {
          'build': build_tf_rnn1,
          'counters': {
              'Copy': 23 if TF1 else 21,
              'host-exchange-local-copy': 1
          },
          'total_memory': 4232578 if TF1 else 3948575,
          'max_memory': 542547 if TF1 else 509134
      }, {
          'build': build_tf_rnn2,
          'counters': {
              'Copy': 34 if TF1 else 31,
              'host-exchange-local-copy': 1
          },
          'total_memory': 2908764 if TF1 else 2864563,
          'max_memory': 368781 if TF1 else 363662
      }, {
          'build': build_tf_lstm1,
          'counters': {
              'Copy': 48 if TF1 else 41,
              'host-exchange-local-copy': 1
          },
          'total_memory': 47341002,
          'max_memory': 5923927
      }, {
          'build': build_tf_gru1,
          'counters': {
              'Copy': 51 if TF1 else 58,
              'host-exchange-local-copy': 1
          },
          'total_memory': 4744597 if TF1 else 4725345,
          'max_memory': 616981 if TF1 else 594266
      }, {
          'build': build_model_rnn1,
          'counters': {
              'Copy': 35 if TF1 else 32,
              'host-exchange-local-copy': 2
          },
          'total_memory': 4544890 if TF1 else 3893674,
          'max_memory': 572439 if TF1 else 490372
      }, {
          'build': build_model_rnn2,
          'counters': {
              'Copy': 57 if TF1 else 51,
              'host-exchange-local-copy': 2
          },
          'total_memory': 6082439 if TF1 else 6148616,
          'max_memory': 777772 if TF1 else 786597
      }, {
          'build': build_model_cnn1,
          'counters': {
              'Copy': 14
          },
          'total_memory': 1277668,
          'max_memory': 160393,
          'options': {
              'batch_size': 1,
              'steps': 32
          }
      })
  @test_util.deprecated_graph_mode_only
  def test_rnn(self, build, counters, total_memory, max_memory, options=None):
    opts = TestOptions()
    if options:
      for key, value in options.items():
        setattr(opts, key, value)
    self._run_test(opts, build, counters, total_memory, max_memory)


if __name__ == "__main__":
  googletest.main()
