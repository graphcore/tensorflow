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
"""Tests for Functional Pipelining API interface."""
import tempfile
import os

from tensorflow.python.ipu import config
from tensorflow.python.ipu import ipu_strategy
from tensorflow.python.ipu.keras import extensions
from tensorflow.python.keras.engine import training as training_module
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


def get_simple_model():
  d1 = layers.Input(32)
  d2 = layers.Input(32)
  f1 = layers.Flatten()(d1)
  f2 = layers.Flatten()(d2)
  x1 = layers.Dense(4)(f1)
  x2 = layers.Dense(4)(f2)
  l = layers.Dense(8)
  o1 = l(x1)
  o2 = l(x2)
  return training_module.Model((d1, d2), (o1, o2))


def check_assignments(instance, assignments):
  instance.assertTrue(
      all(
          isinstance(
              assignment, extensions.functional_extensions.
              FunctionalLayerPipelineStageAssignment)
          for assignment in assignments))


class FunctionalPipelineApiTest(test.TestCase):
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
      # 5 layers, but one layer is called twice.
      self.assertEqual(len(assignments), 6)
      check_assignments(self, assignments)
      self.assertTrue(
          all(assignment.pipeline_stage is None for assignment in assignments))

      # Test setting valid.
      nodes_to_stage = {}
      for i, assignment in enumerate(assignments):
        assignment.pipeline_stage = i
        nodes_to_stage[str(
            id(assignment.layer._inbound_nodes[assignment.node_index]))] = i  # pylint: disable=protected-access
      m.set_pipeline_stage_assignment(assignments)
      assignments = m.get_pipeline_stage_assignment()
      check_assignments(self, assignments)
      for assignment in assignments:
        self.assertEqual(assignment.pipeline_stage, nodes_to_stage[str(
            id(assignment.layer._inbound_nodes[assignment.node_index]))])  # pylint: disable=protected-access

      # Test that reset removes pipelining.
      m.reset_pipeline_stage_assignment()
      assignments = m.get_pipeline_stage_assignment()
      check_assignments(self, assignments)
      self.assertTrue(
          all(assignment.pipeline_stage is None for assignment in assignments))

      # Test number of assignments.
      m.reset_pipeline_stage_assignment()
      assignments = m.get_pipeline_stage_assignment()
      assignments.pop()
      with self.assertRaisesRegex(
          ValueError,
          r"The size of the provided `pipeline_stage_assignment` \(5\) does "
          r"not match the total number of invocations of layers in the model "
          r"\(currently 6\)."):
        m.set_pipeline_stage_assignment(assignments)

      # Test type of elements.
      m.reset_pipeline_stage_assignment()
      assignments = m.get_pipeline_stage_assignment()
      assignments[-1] = None
      with self.assertRaisesRegex(
          ValueError,
          "All elements of `pipeline_stage_assignment` need to be instances of "
          "`FunctionalLayerPipelineStageAssignment`."):
        m.set_pipeline_stage_assignment(assignments)

      # Test all nodes assigned a pipeline stage.
      m.reset_pipeline_stage_assignment()
      assignments = m.get_pipeline_stage_assignment()
      for i, assignment in enumerate(assignments[:-1]):
        assignment.pipeline_stage = i
      with self.assertRaisesRegex(
          ValueError,
          r"Layer dense.* with node index 0 has not been assigned a pipeline "
          r"stage"):
        m.set_pipeline_stage_assignment(assignments)

      # Test all nodes are unique.
      m.reset_pipeline_stage_assignment()
      assignments = m.get_pipeline_stage_assignment()
      for i, assignment in enumerate(assignments):
        assignment.pipeline_stage = i
      assignments[-1] = assignments[0]
      with self.assertRaisesRegex(
          ValueError,
          r"Duplicate assignment for layer flatten.* with node index 0"):
        m.set_pipeline_stage_assignment(assignments)

      # Test not all stages have been assigned to.
      m.reset_pipeline_stage_assignment()
      assignments = m.get_pipeline_stage_assignment()
      for i, assignment in enumerate(assignments):
        assignment.pipeline_stage = i * 2
      with self.assertRaisesRegex(
          ValueError,
          "Pipeline stages in the graph need to be strictly increasing, found "
          "pipeline stages 0, 2, 4, 6, 8, 10, however the following pipeline "
          "stages are missing 1, 3, 5, 7, 9"):
        m.set_pipeline_stage_assignment(assignments)

      # Test can make pipeline stage post order.
      m.reset_pipeline_stage_assignment()
      assignments = m.get_pipeline_stage_assignment()
      for i, assignment in enumerate(assignments):
        assignment.pipeline_stage = i - 1
      assignments[0].pipeline_stage = assignments[-1].pipeline_stage
      with self.assertRaisesRegex(
          ValueError,
          r"Layer dense.* in pipeline stage 1 has a dependency from a pipeline "
          r"stage"):
        m.set_pipeline_stage_assignment(assignments)

      # Test assignment with scopes.
      d1 = layers.Input(32)
      d2 = layers.Input(32)
      with extensions.functional_extensions.PipelineStage(0):
        f1 = layers.Flatten()(d1)

      with extensions.functional_extensions.PipelineStage(1):
        f2 = layers.Flatten()(d2)

      with extensions.functional_extensions.PipelineStage(2):
        x1 = layers.Dense(4)(f1)

      with extensions.functional_extensions.PipelineStage(3):
        x2 = layers.Dense(4)(f2)

      l = layers.Dense(8)
      with extensions.functional_extensions.PipelineStage(4):
        o1 = l(x1)

      with extensions.functional_extensions.PipelineStage(5):
        o2 = l(x2)

      m = training_module.Model((d1, d2), (o1, o2))
      assignments = m.get_pipeline_stage_assignment()
      for assignment in assignments:
        if assignment.layer is f1:
          self.assertEqual(assignment.node_index, 0)
          self.assertEqual(assignment.pipeline_stage, 0)
        if assignment.layer is f2:
          self.assertEqual(assignment.node_index, 0)
          self.assertEqual(assignment.pipeline_stage, 1)
        if assignment.layer is x1:
          self.assertEqual(assignment.node_index, 0)
          self.assertEqual(assignment.pipeline_stage, 2)
        if assignment.layer is x2:
          self.assertEqual(assignment.node_index, 0)
          self.assertEqual(assignment.pipeline_stage, 3)
        if assignment.layer is l:
          if assignment.node_index == 0:
            self.assertEqual(assignment.pipeline_stage, 4)
          else:
            self.assertEqual(assignment.node_index, 1)
            self.assertEqual(assignment.pipeline_stage, 5)

      # Test assignment with scope - missing assignment.
      d = layers.Input(32)
      with extensions.functional_extensions.PipelineStage(0):
        f = layers.Flatten()(d)
      x = layers.Dense(4)(f)
      with self.assertRaisesRegex(
          ValueError,
          r"All layers of a pipelined model must have an associated pipeline "
          f"stage. However, dense.* has not been assigned to one."):
        m = training_module.Model(d, x)

  @test_util.run_v2_only
  def testSaveRestore(self):
    cfg = config.IPUConfig()

    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = get_simple_model()

      def check_assignments_are(model, assignments):
        for assignment in assignments:
          if assignment.layer is model.layers[2]:
            self.assertEqual(assignment.node_index, 0)
            self.assertEqual(assignment.pipeline_stage, 0)
          if assignment.layer is model.layers[3]:
            self.assertEqual(assignment.node_index, 0)
            self.assertEqual(assignment.pipeline_stage, 1)
          if assignment.layer is model.layers[4]:
            self.assertEqual(assignment.node_index, 0)
            self.assertEqual(assignment.pipeline_stage, 2)
          if assignment.layer is model.layers[5]:
            self.assertEqual(assignment.node_index, 0)
            self.assertEqual(assignment.pipeline_stage, 3)
          if assignment.layer is model.layers[6]:
            if assignment.node_index == 0:
              self.assertEqual(assignment.pipeline_stage, 4)
            else:
              self.assertEqual(assignment.node_index, 1)
              self.assertEqual(assignment.pipeline_stage, 5)

      # Test default assignment.
      assignments = m.get_pipeline_stage_assignment()
      for assignment in assignments:
        if assignment.layer is m.layers[2]:
          self.assertEqual(assignment.node_index, 0)
          assignment.pipeline_stage = 0
        if assignment.layer is m.layers[3]:
          self.assertEqual(assignment.node_index, 0)
          assignment.pipeline_stage = 1
        if assignment.layer is m.layers[4]:
          self.assertEqual(assignment.node_index, 0)
          assignment.pipeline_stage = 2
        if assignment.layer is m.layers[5]:
          self.assertEqual(assignment.node_index, 0)
          assignment.pipeline_stage = 3
        if assignment.layer is m.layers[6]:
          if assignment.node_index == 0:
            assignment.pipeline_stage = 4
          else:
            self.assertEqual(assignment.node_index, 1)
            assignment.pipeline_stage = 5

      m.set_pipeline_stage_assignment(assignments)
      check_assignments_are(m, assignments)

      with tempfile.TemporaryDirectory() as tmp:
        save_path = os.path.join(tmp, "model")
        m.save(save_path)
        m = models.load_model(save_path)
        self.assertTrue(m._is_pipelined)  # pylint: disable=protected-access
        assignments = m.get_pipeline_stage_assignment()
        check_assignments_are(m, assignments)

  @test_util.run_v2_only
  def testSetPipeliningOptions(self):
    cfg = config.IPUConfig()

    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = get_simple_model()
      assignments = m.get_pipeline_stage_assignment()
      for i, assignment in enumerate(assignments):
        assignment.pipeline_stage = i

      with self.assertRaisesRegex(
          ValueError,
          "Expected `gradient_accumulation_steps` to be a positive integer, "
          "but got -1 instead"):
        m.set_pipelining_options(gradient_accumulation_steps=-1)

      with self.assertRaisesRegex(
          ValueError, "Expected `device_mapping` to be a list of integers"):
        m.set_pipelining_options(device_mapping=[0.0] * 10)

      with self.assertRaisesRegex(
          ValueError,
          "Found `gradient_accumulation_count` key in `pipelining_kwargs`. Set "
          "the `gradient_accumulation_steps` argument to "
          "`set_pipelining_options` instead."):
        m.set_pipelining_options(gradient_accumulation_count=10)

      with self.assertRaisesRegex(
          ValueError,
          "Found `repeat_count` key in `pipelining_kwargs`. This argument is "
          "automatically set by Keras"):
        m.set_pipelining_options(repeat_count=10)

      with self.assertRaisesRegex(
          ValueError,
          "Found `repeat_count` key in `pipelining_kwargs`. This argument is "
          "automatically set by Keras"):
        m.set_pipelining_options(repeat_count=10)

      with self.assertRaisesRegex(
          ValueError,
          "Found `batch_serialization_iterations` key in `pipelining_kwargs`. "
          "This argument is not compatible with Keras"):
        m.set_pipelining_options(batch_serialization_iterations=10)

      m.set_pipelining_options(gradient_accumulation_steps=10,
                               device_mapping=[4, 3, 2, 1, 0])

      with tempfile.TemporaryDirectory() as tmp:
        save_path = os.path.join(tmp, "model")
        m.save(save_path)
        m = models.load_model(save_path)
        self.assertEqual(m._pipelining_gradient_accumulation_steps, 10)  # pylint: disable=protected-access
        self.assertEqual(m._pipelining_device_mapping, [4, 3, 2, 1, 0])  # pylint: disable=protected-access


if __name__ == '__main__':
  test.main()
