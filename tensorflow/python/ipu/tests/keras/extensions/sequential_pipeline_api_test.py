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
"""Tests for Sequential Pipelining API interface."""
import tempfile
import os

from tensorflow.python.ipu import config
from tensorflow.python.ipu import ipu_strategy
from tensorflow.python.ipu.keras import extensions
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


def get_simple_model():
  l = [layers.Dense(16) for _ in range(10)]
  return sequential.Sequential(l)


def check_assignments(instance, model, assignments):
  instance.assertTrue(
      all(
          isinstance(
              assignment, extensions.sequential_extensions.
              SequentialLayerPipelineStageAssignment)
          for assignment in assignments))
  instance.assertTrue(
      all(assignment.layer_index == i
          for i, assignment in enumerate(assignments)))
  instance.assertTrue(
      all(assignment.layer == model.layers[i]
          for i, assignment in enumerate(assignments)))


def create_default_assignment(model):
  return [
      extensions.sequential_extensions.SequentialLayerPipelineStageAssignment(
          model, i, pipeline_stage=i) for i in range(len(model.layers))
  ]


class SequentialPipelineApiTest(test.TestCase):
  @test_util.run_v2_only
  def testGetSetReset(self):
    cfg = config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = get_simple_model()

      # Test default assignment.
      assignments = m.get_pipeline_stage_assignment()
      self.assertFalse(m._pipeline_stage_assignment_valid)  # pylint: disable=protected-access
      check_assignments(self, m, assignments)
      self.assertTrue(
          all(assignment.pipeline_stage is None for assignment in assignments))

      # Test setting valid (ints).
      m.set_pipeline_stage_assignment(list(range(10)))
      self.assertTrue(m._pipeline_stage_assignment_valid)  # pylint: disable=protected-access
      assignments = m.get_pipeline_stage_assignment()
      check_assignments(self, m, assignments)
      self.assertEqual(
          [assignment.pipeline_stage for assignment in assignments],
          list(range(10)))

      # Test setting valid (SequentialLayerPipelineStageAssignment).
      for i in range(10):
        assignments[i].pipeline_stage = 0 if i < 5 else 1

      m.set_pipeline_stage_assignment(assignments)
      self.assertTrue(m._pipeline_stage_assignment_valid)  # pylint: disable=protected-access
      assignments = m.get_pipeline_stage_assignment()
      check_assignments(self, m, assignments)
      self.assertEqual(
          [assignment.pipeline_stage for assignment in assignments],
          [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

      # Test that reset removes pipelining.
      m.reset_pipeline_stage_assignment()
      self.assertFalse(m._pipeline_stage_assignment_valid)  # pylint: disable=protected-access
      assignments = m.get_pipeline_stage_assignment()
      check_assignments(self, m, assignments)
      self.assertTrue(
          all(assignment.pipeline_stage is None for assignment in assignments))

      m.set_pipeline_stage_assignment(list(range(10)))

      # Check adding a layer invalidates pipelining.
      m.add(layers.Dense(1))
      self.assertFalse(m._pipeline_stage_assignment_valid)  # pylint: disable=protected-access

      m.set_pipeline_stage_assignment(list(range(11)))
      self.assertTrue(m._pipeline_stage_assignment_valid)  # pylint: disable=protected-access

      # Check removing a layer invalidates pipelining.
      m.pop()
      self.assertFalse(m._pipeline_stage_assignment_valid)  # pylint: disable=protected-access

      m.set_pipeline_stage_assignment(list(range(10)))
      self.assertTrue(m._pipeline_stage_assignment_valid)  # pylint: disable=protected-access

      # Test invalid object.
      with self.assertRaisesRegex(
          ValueError, "All elements of `pipeline_stage_assignment`"):
        m.set_pipeline_stage_assignment([0.0] * 10)

      # Test wrong length.
      with self.assertRaisesRegex(
          ValueError,
          r"The size of the provided `pipeline_stage_assignment` \(8\) does "
          r"not match the number of layers in the model \(currently 10\)"):
        m.set_pipeline_stage_assignment(list(range(8)))

      # Test duplicate assignment.
      assignments = create_default_assignment(m)
      assignments[-1]._layer_index = 1  # pylint: disable=protected-access
      with self.assertRaisesRegex(
          ValueError,
          r"The provided `pipeline_stage_assignment` contains a duplicate "
          r"assignment for layer dense.* at sequential index 1."):
        m.set_pipeline_stage_assignment(assignments)

      # Test indexes match.
      m = get_simple_model()
      assignments = create_default_assignment(m)
      assignments[0], assignments[1] = assignments[1], assignments[0]
      with self.assertRaisesRegex(
          ValueError,
          "The provided assignment at index 0 `pipeline_stage_assignment` does "
          "not match with the layer index 1"):
        m.set_pipeline_stage_assignment(assignments)

      # Test first layer is on pipeline stage 0.
      m = get_simple_model()
      assignments = create_default_assignment(m)
      assignments[0].pipeline_stage = 1
      with self.assertRaisesRegex(
          ValueError,
          "The first layer in a pipelined sequential model needs to be "
          "assigned to the 0th pipeline stage, however it was assigned to 1."):
        m.set_pipeline_stage_assignment(assignments)

      # Test layers on consecutive pipeline stages.
      m = get_simple_model()
      assignments = create_default_assignment(m)
      assignments[-1].pipeline_stage = 0
      with self.assertRaisesRegex(
          ValueError,
          r"Layer dense.* at sequential index 9 has been assigned to pipeline "
          r"stage 0, however the previous layer in the Sequential model was "
          r"assigned to pipeline stage 8. A layer in a Sequential model can "
          r"only be assigned to the same pipeline stage as the previous layer "
          r"or to the next pipeline stage."):
        m.set_pipeline_stage_assignment(assignments)

  @test_util.run_v2_only
  def testSaveRestore(self):
    cfg = config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = get_simple_model()

      with tempfile.TemporaryDirectory() as tmp:
        save_path = os.path.join(tmp, "model")
        m.build((4, 1))
        m.set_pipeline_stage_assignment(list(range(10)))
        m.save(save_path)
        m = models.load_model(save_path)
        self.assertTrue(m._pipeline_stage_assignment_valid)  # pylint: disable=protected-access
        assignments = m.get_pipeline_stage_assignment()
        check_assignments(self, m, assignments)
        self.assertEqual(
            [assignment.pipeline_stage for assignment in assignments],
            list(range(10)))


if __name__ == '__main__':
  test.main()
