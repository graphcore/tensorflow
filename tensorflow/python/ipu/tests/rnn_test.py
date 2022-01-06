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
    self.dim = 256
    self.in_dim = 16
    self.out_dim = 16
    self.output_keep_prob = 0.75


class RNNModelTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  def _run_test(self, opts, model_func, cycles, total_mem, max_mem):
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
    report_helper.set_autoreport_options(cfg, output_execution_profile=True)
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 4
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
    self.assert_execution_report_cycles(report, 0, cycles, tolerance=0.01)
    self.assert_total_tile_memory(report, total_mem, tolerance=0.01)
    self.assert_max_tile_memory(report, max_mem, tolerance=0.01)

  @parameterized.parameters(
      {
          'build': build_tf_rnn1,
          'cycles': 87610341 if TF1 else 87860863,
          'total_memory': 12378522 if TF1 else 13129963,
          'max_memory': 3117577 if TF1 else 3359730
      }, {
          'build': build_tf_rnn2,
          'cycles': 62667061 if TF1 else 118460479,
          'total_memory': 8634783 if TF1 else 8909938,
          'max_memory': 2177166 if TF1 else 2292513
      }, {
          'build': build_tf_lstm1,
          'cycles': 1086942603 if TF1 else 1102156072,
          'total_memory': 151061717,
          'max_memory': 37776657
      }, {
          'build': build_tf_gru1,
          'cycles': 469429671 if TF1 else 586828276,
          'total_memory': 58114499 if TF1 else 57145333,
          'max_memory': 14563678 if TF1 else 14289524
      }, {
          'build': build_model_rnn1,
          'cycles': 86929788 if TF1 else 78661615,
          'total_memory': 12141874 if TF1 else 12359011,
          'max_memory': 3044815 if TF1 else 3106806
      }, {
          'build': build_model_rnn2,
          'cycles': 145986926 if TF1 else 133548944,
          'total_memory': 18177930 if TF1 else 17409220,
          'max_memory': 4550619 if TF1 else 4362290
      }, {
          'build': build_model_cnn1,
          'cycles': 144497392,
          'total_memory': 13088427 if TF1 else 17369471,
          'max_memory': 3275543 if TF1 else 4345119,
          'options': {
              'batch_size': 1,
              'steps': 32
          }
      })
  @test_util.deprecated_graph_mode_only
  def test_rnn(self, build, cycles, total_memory, max_memory, options=None):
    opts = TestOptions()
    if options:
      for key, value in options.items():
        setattr(opts, key, value)
    self._run_test(opts, build, cycles, total_memory, max_memory)


if __name__ == "__main__":
  googletest.main()
