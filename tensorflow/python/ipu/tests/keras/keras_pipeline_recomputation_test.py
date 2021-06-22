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

import pva

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python import keras
from tensorflow.python import ipu
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent


class KerasPipelineRecomputationTest(test.TestCase):
  @test_util.run_v2_only
  def testRecomputationCheckpoint(self):
    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.auto_select_ipus = 2
    cfg.allow_recompute = True
    cfg.configure_ipu_system()

    IM_SIZE = [64, 64, 3]
    BATCH_SIZE = 1
    NUM_FILTERS = 4

    constant_d = constant_op.constant(1.0, shape=IM_SIZE)
    constant_l = constant_op.constant(0.2, shape=[2])
    ds = dataset_ops.Dataset.from_tensors((constant_d, constant_l))
    ds = ds.repeat(64)
    ds = ds.batch(BATCH_SIZE, drop_remainder=True)

    def make_model(checkpoints=False):
      input_layer = keras.layers.Input(shape=IM_SIZE)
      with ipu.keras.PipelineStage(0):
        x = keras.layers.Conv2D(NUM_FILTERS, 3, activation="relu")(input_layer)
        x = keras.layers.Conv2D(NUM_FILTERS, 3, activation="relu")(x)
        x = keras.layers.Conv2D(NUM_FILTERS, 3, activation="relu")(x)
        if checkpoints:
          x = ipu.keras.layers.RecomputationCheckpoint()(x)
        x = keras.layers.Conv2D(NUM_FILTERS, 3, activation="relu")(x)
        x = keras.layers.Conv2D(NUM_FILTERS, 3, activation="relu")(x)
        x = keras.layers.Conv2D(NUM_FILTERS, 3, activation="relu")(x)
      with ipu.keras.PipelineStage(1):
        x = keras.layers.Dense(1,
                               kernel_initializer='glorot_uniform',
                               activation=keras.activations.relu)(x)

      # Checkpoints require Grouped/Seq and RecomputeAndBackpropagateInterleaved
      m = ipu.keras.PipelineModel(
          inputs=input_layer,
          outputs=x,
          gradient_accumulation_count=4,
          pipeline_schedule=ipu.ops.pipelining_ops.PipelineSchedule.Grouped,
          recomputation_mode=ipu.ops.pipelining_ops.RecomputationMode.
          RecomputeAndBackpropagateInterleaved)
      opt = gradient_descent.GradientDescentOptimizer(0.001)
      m.compile(opt, loss='mse')
      return m

    # Make sure the checkpoints reduce peak liveness.
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = make_model(checkpoints=False)
      m.fit(ds)
      report = pva.openReport(report_helper.find_report())
      recomp_always_live = sum(tile.memory.alwaysLiveBytes
                               for tile in report.compilation.tiles)
      recomp_peak_liveness = recomp_always_live + max(
          step.notAlwaysLiveMemory.bytes
          for step in report.compilation.livenessProgramSteps)

      report_helper.clear_reports()

      m = make_model(checkpoints=True)
      m.fit(ds)

      report = pva.openReport(report_helper.find_report())
      ckpt_always_live = sum(tile.memory.alwaysLiveBytes
                             for tile in report.compilation.tiles)
      ckpt_peak_liveness = ckpt_always_live + max(
          step.notAlwaysLiveMemory.bytes
          for step in report.compilation.livenessProgramSteps)
      self.assertGreater(recomp_peak_liveness, ckpt_peak_liveness)


if __name__ == '__main__':
  test.main()
