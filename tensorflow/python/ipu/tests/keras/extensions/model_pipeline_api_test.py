# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Model subclass Pipelining API interface."""
import tempfile
import os
import numpy as np

from tensorflow.python.framework import test_util
from tensorflow.python.ipu import ipu_strategy
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ipu.keras import extensions
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras.optimizer_v2 import rmsprop
from tensorflow.python.keras.engine import training as training_module
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


# Basic subclassed model.
class SimpleModel(training_module.Model):  # pylint: disable=abstract-method
  def __init__(self):
    super(SimpleModel, self).__init__()
    self.flatten_layer = layers.Flatten()
    self.dense_layers = [layers.Dense(4), layers.Dense(8)]

  def call(self, inputs):  # pylint: disable=arguments-differ
    x1 = self.flatten_layer(inputs[0])
    x2 = self.flatten_layer(inputs[1])

    def net(x):
      for layer in self.dense_layers:
        x = layer(x)
      return x

    return net(x1), net(x2)


# Subclassed model with get_config and from_config overridden.
class RestorableModel(SimpleModel):  # pylint: disable=abstract-method
  def get_config(self):
    config = super(RestorableModel, self).get_config()
    config['flatten_layer'] = self.flatten_layer
    config['dense_layer_0'] = self.dense_layers[0]
    config['dense_layer_1'] = self.dense_layers[1]
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    instance = cls()
    instance.flatten_layer = config['flatten_layer']
    instance.dense_layers = [config['dense_layer_0'], config['dense_layer_1']]
    return instance


# Model using PipelineStage scope to assign pipeline stages.
class ModelWithPipelineStageAssignments(training_module.Model):  # pylint: disable=abstract-method
  def __init__(self):
    super(ModelWithPipelineStageAssignments, self).__init__()
    self.flatten_layer = layers.Flatten()

    # Apply stage to layer, this can be overridden by stages assigned to
    # specific nodes.
    with extensions.functional_extensions.PipelineStage(2):
      self.dense_layer_1 = layers.Dense(4)

    self.dense_layer_2 = layers.Dense(8)

  def call(self, inputs):  # pylint: disable=arguments-differ
    x1, x2 = inputs
    with extensions.functional_extensions.PipelineStage(0):
      x1 = self.flatten_layer(x1)
    with extensions.functional_extensions.PipelineStage(1):
      x2 = self.flatten_layer(x2)

    # Already has stage 2 assigned to the layer.
    x1 = self.dense_layer_1(x1)

    # Overrides layer assignment (stage 2).
    with extensions.functional_extensions.PipelineStage(3):
      x2 = self.dense_layer_1(x2)

    with extensions.functional_extensions.PipelineStage(4):
      x1 = self.dense_layer_2(x1)

    with extensions.functional_extensions.PipelineStage(5):
      x2 = self.dense_layer_2(x2)

    return x1, x2


# Model using PipelineStage scope to assign stages for some but not all nodes.
class ModelWithMissingPipelineStage(training_module.Model):  # pylint: disable=abstract-method
  def __init__(self):
    super(ModelWithMissingPipelineStage, self).__init__()
    self.flatten_layer = layers.Flatten()
    self.dense_layer = layers.Dense(4)

  def call(self, inputs):  # pylint: disable=arguments-differ
    x = inputs
    with extensions.functional_extensions.PipelineStage(0):
      x = self.flatten_layer(x)

    # Create dense layer node outside of PipelineStage scope.
    return self.dense_layer(x)


# A model which expects a dict in its call method.
# Based off a model which already has pipeline stage assignments.
class ModelWithDictInputs(ModelWithPipelineStageAssignments):  # pylint: disable=abstract-method
  def call(self, inputs):
    x1 = inputs["x1"]
    x2 = inputs["x2"]
    return super(ModelWithDictInputs, self).call([x1, x2])


def check_assignments(instance, assignments):
  instance.assertTrue(
      all(
          isinstance(
              assignment,
              extensions.model_extensions.ModelLayerPipelineStageAssignment)
          for assignment in assignments))


class ModelPipelineApiTest(test.TestCase):
  @test_util.run_v2_only
  def testGetPipelistStageAssignmentOnUninitializedModel(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = SimpleModel()

      with self.assertRaisesRegex(
          RuntimeError,
          r"Cannot get pipeline stage assignments for model `.*` which has "
          r"not been built yet"):
        m.get_pipeline_stage_assignment()

  @test_util.run_v2_only
  def testGetPipelineStageAssignmentDefault(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = SimpleModel()
      m.build([(1, 32), (1, 32)])

      # Test default assignment.
      assignments = m.get_pipeline_stage_assignment()

      # 3 layers, but each layer is called twice.
      self.assertEqual(len(assignments), 6)
      check_assignments(self, assignments)
      self.assertTrue(
          all(assignment.pipeline_stage is None for assignment in assignments))

  @test_util.run_v2_only
  def testSetPipelineStageAssignment(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = SimpleModel()
      m.build([(1, 32), (1, 32)])

      # Set assignments.
      nodes_to_stage = {}
      assignments = m.get_pipeline_stage_assignment()
      for i, assignment in enumerate(assignments):
        assignment.pipeline_stage = i
        nodes_to_stage[str(
            id(assignment.layer._inbound_nodes[assignment.node_index]))] = i  # pylint: disable=protected-access
      m.set_pipeline_stage_assignment(assignments)

      # Get assignments, and verify they are the same as the ones we set.
      assignments = m.get_pipeline_stage_assignment()
      check_assignments(self, assignments)
      for assignment in assignments:
        self.assertEqual(assignment.pipeline_stage, nodes_to_stage[str(
            id(assignment.layer._inbound_nodes[assignment.node_index]))])  # pylint: disable=protected-access

  @test_util.run_v2_only
  def testResetPipelineStageAssignment(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = SimpleModel()
      m.build([(1, 32), (1, 32)])

      # Set assignments.
      nodes_to_stage = {}
      assignments = m.get_pipeline_stage_assignment()
      for i, assignment in enumerate(assignments):
        assignment.pipeline_stage = i
        nodes_to_stage[str(
            id(assignment.layer._inbound_nodes[assignment.node_index]))] = i  # pylint: disable=protected-access
      m.set_pipeline_stage_assignment(assignments)
      self.assertTrue(m._is_pipelined())  # pylint: disable=protected-access

      # Reset assignments, and check all nodes have no assignment.
      m.reset_pipeline_stage_assignment()
      assignments = m.get_pipeline_stage_assignment()
      check_assignments(self, assignments)
      self.assertTrue(
          all(assignment.pipeline_stage is None for assignment in assignments))

  @test_util.run_v2_only
  def testSetPipelineStageAssignmentWithInvalidNumberOfAssignments(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = SimpleModel()
      m.build([(1, 32), (1, 32)])

      # Test number of assignments.
      assignments = m.get_pipeline_stage_assignment()
      assignments.pop()
      with self.assertRaisesRegex(
          ValueError,
          r"The length of the provided `pipeline_stage_assignment` \(5\) does "
          r"not match the number of layers in the graph \(6\)."):
        m.set_pipeline_stage_assignment(assignments)

  @test_util.run_v2_only
  def testSetPipelineStageAssignmentWithInvalidAssignmentClass(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = SimpleModel()
      m.build([(1, 32), (1, 32)])

      # Test type of elements.
      assignments = m.get_pipeline_stage_assignment()
      assignments[0] = None
      with self.assertRaisesRegex(
          ValueError,
          "All elements of `pipeline_stage_assignment` must be instances of "
          "`ModelLayerPipelineStageAssignment`."):
        m.set_pipeline_stage_assignment(assignments)

  @test_util.run_v2_only
  def testSetPipelineStageAssignmentWithMissingAssignment(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = SimpleModel()
      m.build([(1, 32), (1, 32)])

      # Test all nodes assigned a pipeline stage.
      assignments = m.get_pipeline_stage_assignment()
      for i, assignment in enumerate(assignments[:-1]):
        assignment.pipeline_stage = i
      with self.assertRaisesRegex(
          ValueError,
          r"Layer dense.* with node_index 0 has not been assigned a pipeline "
          r"stage in `pipeline_stage_assignment`."):
        m.set_pipeline_stage_assignment(assignments)

  @test_util.run_v2_only
  def testSetPipelineStageAssignmentWithAssignmentForNonExistantNode(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = SimpleModel()
      m.build([(1, 32), (1, 32)])

      # Test node exists.
      assignments = m.get_pipeline_stage_assignment()
      for i, assignment in enumerate(assignments):
        assignment.pipeline_stage = i
      temp = assignments[0]
      assignments[0] = assignments[1]
      assignments[1] = temp
      with self.assertRaisesRegex(
          ValueError,
          "The order of `pipeline_stage_assignment` does not match the "
          "post-order generated from the graph"):
        m.set_pipeline_stage_assignment(assignments)

  @test_util.run_v2_only
  def testSetPipelineStageAssignmentWithEmptyStages(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = SimpleModel()
      m.build([(1, 32), (1, 32)])

      # Test not all stages have been assigned to.
      assignments = m.get_pipeline_stage_assignment()
      for i, assignment in enumerate(assignments):
        assignment.pipeline_stage = i * 2
      with self.assertRaisesRegex(
          ValueError,
          r"Pipeline stage assignments must start at 0 and be strictly "
          r"increasing. The highest assignment found in "
          r"`pipeline_stage_assignment` was stage 10, however the "
          r"preceeding stages \[1, 3, 5, 7, 9\] had no assignments."):
        m.set_pipeline_stage_assignment(assignments)

  @test_util.run_v2_only
  def testSetPipelineStageAssignmentWithDependencyOnLaterStage(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = SimpleModel()
      m.build([(1, 32), (1, 32)])

      # Test can make pipeline stage post order.
      assignments = m.get_pipeline_stage_assignment()
      for i, assignment in enumerate(assignments):
        assignment.pipeline_stage = i - 1
      assignments[0].pipeline_stage = assignments[-1].pipeline_stage
      with self.assertRaisesRegex(
          ValueError,
          r"Layer dense.* in pipeline stage 1 has a dependency from a pipeline "
          r"stage"):
        m.set_pipeline_stage_assignment(assignments)

  @test_util.run_v2_only
  def testPipelineStageAssignmentWithScopes(self):
    cfg = IPUConfig()
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = ModelWithPipelineStageAssignments()
      # Should not need to build the model. That should be done automatically,
      # since we are passing in data.
      inputs = [
          np.ones(shape=(6, 32), dtype=np.int32),
          np.ones(shape=(6, 32), dtype=np.int32)
      ]
      m.compile(steps_per_execution=6)
      m.set_pipelining_options(device_mapping=[0] * 6)
      m.predict(inputs, batch_size=1)

      self.assertTrue(m._is_pipelined())  # pylint: disable=protected-access

      # Check assignments from scopes are applied.
      for assignment in m.get_pipeline_stage_assignment():
        if assignment.layer == m.flatten_layer:
          if assignment.node_index == 0:
            self.assertEqual(assignment.pipeline_stage, 0)
          else:
            self.assertEqual(assignment.node_index, 1)
            self.assertEqual(assignment.pipeline_stage, 1)
        if assignment.layer == m.dense_layer_1:
          if assignment.node_index == 0:
            self.assertEqual(assignment.pipeline_stage, 2)
          else:
            self.assertEqual(assignment.node_index, 1)
            self.assertEqual(assignment.pipeline_stage, 3)
        if assignment.layer == m.dense_layer_2:
          if assignment.node_index == 0:
            self.assertEqual(assignment.pipeline_stage, 4)
          else:
            self.assertEqual(assignment.node_index, 1)
            self.assertEqual(assignment.pipeline_stage, 5)

  @test_util.run_v2_only
  def testRunModelWithPartialPipelineStageAssignments(self):
    cfg = IPUConfig()
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():

      m = ModelWithMissingPipelineStage()
      inputs = np.ones(shape=(6, 32), dtype=np.int32)
      m.compile(steps_per_execution=6)
      m.set_pipelining_options(device_mapping=[0] * 6)

      with self.assertRaisesRegex(
          ValueError,
          r"All layers of a pipelined model must have an associated pipeline "
          r"stage. However, dense.* has not been assigned to one."):
        # Calls build internally.
        m.predict(inputs, batch_size=1)

      with self.assertRaisesRegex(
          ValueError,
          r"All layers of a pipelined model must have an associated pipeline "
          r"stage. However, dense.* has not been assigned to one."):
        # Explicit call to build should also fail.
        m.build((6, 32))

  @test_util.run_v2_only
  def testSaveRestore(self):
    cfg = IPUConfig()
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():

      # A model with get_config and from_config overridden such that the layers
      # are restored rather than being created new when a saved model is loaded.
      m = RestorableModel()
      m.build([(1, 32), (1, 32)])

      def check_assignments_are(model, assignments):
        for assignment in assignments:
          if assignment.layer is model.flatten_layer:
            if assignment.node_index == 0:
              self.assertEqual(assignment.pipeline_stage, 0)
            else:
              self.assertEqual(assignment.node_index, 1)
              self.assertEqual(assignment.pipeline_stage, 1)
          if assignment.layer is model.dense_layers[0]:
            if assignment.node_index == 0:
              self.assertEqual(assignment.pipeline_stage, 2)
            else:
              self.assertEqual(assignment.node_index, 1)
              self.assertEqual(assignment.pipeline_stage, 3)
          if assignment.layer is model.dense_layers[1]:
            if assignment.node_index == 0:
              self.assertEqual(assignment.pipeline_stage, 4)
            else:
              self.assertEqual(assignment.node_index, 1)
              self.assertEqual(assignment.pipeline_stage, 5)

      # Create and apply pipeline stage assignments.
      assignments = m.get_pipeline_stage_assignment()
      for assignment in assignments:
        if assignment.layer is m.flatten_layer:
          if assignment.node_index == 0:
            assignment.pipeline_stage = 0
          else:
            self.assertEqual(assignment.node_index, 1)
            assignment.pipeline_stage = 1
        if assignment.layer is m.dense_layers[0]:
          if assignment.node_index == 0:
            assignment.pipeline_stage = 2
          else:
            self.assertEqual(assignment.node_index, 1)
            assignment.pipeline_stage = 3
        if assignment.layer is m.dense_layers[1]:
          if assignment.node_index == 0:
            assignment.pipeline_stage = 4
          else:
            self.assertEqual(assignment.node_index, 1)
            assignment.pipeline_stage = 5

      m.set_pipeline_stage_assignment(assignments)
      check_assignments_are(m, assignments)

      # Check that the model can successfully be executed.
      m.set_pipelining_options(device_mapping=[0] * 6)
      inputs = [
          np.ones(shape=(6, 32), dtype=np.int32),
          np.ones(shape=(6, 32), dtype=np.int32)
      ]
      m.compile(steps_per_execution=6)
      m.predict(inputs, batch_size=1)

      with tempfile.TemporaryDirectory() as tmp:
        # Save and reload the model.
        save_path = os.path.join(tmp, "model")
        m.save(save_path)
        m = models.load_model(
            save_path, custom_objects={"RestorableModel": RestorableModel})

        # Check that the pipeline stages have been correctly restored.
        self.assertTrue(m._is_pipelined)  # pylint: disable=protected-access
        assignments = m.get_pipeline_stage_assignment()
        check_assignments_are(m, assignments)

        # Check that the restored model can successfully be executed.
        m.compile(steps_per_execution=6)
        m.predict(inputs, batch_size=1)

        # Check we can succesfully update the graph network and predict again.
        m.build([(1, 32), (1, 32)])
        m.predict(inputs, batch_size=1)

  @test_util.run_v2_only
  def testModelWithDictInputs(self):
    cfg = IPUConfig()
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = ModelWithDictInputs()
      m.set_pipelining_options(device_mapping=[0] * 6,
                               gradient_accumulation_steps_per_replica=12)
      m.compile(loss=losses.SparseCategoricalCrossentropy(),
                optimizer=rmsprop.RMSprop(),
                metrics=["accuracy"],
                steps_per_execution=12)

      inputs = {
          "x1": np.ones(shape=(12, 32), dtype=np.int32),
          "x2": np.ones(shape=(12, 32), dtype=np.int32),
      }
      labels = [
          np.ones(shape=(12), dtype=np.int32),
          np.ones(shape=(12), dtype=np.int32),
      ]
      m.predict(inputs, batch_size=1)
      m.fit(inputs, labels, batch_size=1)
      m.evaluate(inputs, labels, batch_size=1)

  @test_util.run_v2_only
  def testSetPipeliningOptionsWithNegativeSteps(self):
    cfg = IPUConfig()

    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = SimpleModel()

      with self.assertRaisesRegex(
          ValueError,
          "Expected `gradient_accumulation_steps_per_replica` to be a positive "
          "integer, but got -1 instead"):
        m.set_pipelining_options(gradient_accumulation_steps_per_replica=-1)

  @test_util.run_v2_only
  def testSetPipeliningOptionsWithNonIntegerTypeDeviceMapping(self):
    cfg = IPUConfig()

    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = SimpleModel()

      with self.assertRaisesRegex(
          ValueError, "Expected `device_mapping` to be a list of integers"):
        m.set_pipelining_options(device_mapping=[0.0] * 10)

  @test_util.run_v2_only
  def testSetPipeliningOptionsWithInvalidKeys(self):
    cfg = IPUConfig()

    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = SimpleModel()

      with self.assertRaisesRegex(
          ValueError,
          "Found `gradient_accumulation_count` key in `pipelining_kwargs`. Set "
          "the `gradient_accumulation_steps_per_replica` argument to "
          "`set_pipelining_options` instead."):
        m.set_pipelining_options(gradient_accumulation_count=10)

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

  @test_util.run_v2_only
  def testSaveAndRestorePipeliningOptions(self):
    cfg = IPUConfig()

    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = SimpleModel()
      m.build([(1, 1), (1, 1)])

      assignments = m.get_pipeline_stage_assignment()
      for i, assignment in enumerate(assignments):
        assignment.pipeline_stage = i

      m.set_pipelining_options(gradient_accumulation_steps_per_replica=10,
                               device_mapping=[4, 3, 2, 1, 0],
                               accumulate_outfeed=True,
                               experimental_normalize_gradients=True)

      with tempfile.TemporaryDirectory() as tmp:
        save_path = os.path.join(tmp, "model")
        m.save(save_path)
        m = models.load_model(save_path,
                              custom_objects={"SimpleModel": SimpleModel})
        self.assertEqual(
            m._pipelining_gradient_accumulation_steps_per_replica,  # pylint: disable=protected-access
            10)
        self.assertEqual(m._pipelining_device_mapping, [4, 3, 2, 1, 0])  # pylint: disable=protected-access
        self.assertEqual(m._pipelining_accumulate_outfeed, True)  # pylint: disable=protected-access
        self.assertEqual(m._experimental_pipelining_normalize_gradients, True)  # pylint: disable=protected-access

  def testPrintPipelineStageSummary(self):
    cfg = IPUConfig()

    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():

      class SimplePrintableModel(training_module.Model):  # pylint: disable=abstract-method
        def __init__(self):
          super(SimplePrintableModel, self).__init__()

          self.flatten_layer = layers.Flatten()
          self.concat_layer = layers.Concatenate()
          self.dense_layer = layers.Dense(4)

        def call(self, inputs):  # pylint: disable=arguments-differ
          d1, d2 = inputs
          f1 = self.flatten_layer(d1)
          f2 = self.flatten_layer(d2)
          c1 = self.concat_layer([f1, f2])
          x1 = self.dense_layer(c1)
          return math_ops.multiply(1.0, x1)

      m = SimplePrintableModel()
      m.build([(1, 1), (1, 1)])

      strings = []

      def print_fn(x):
        strings.append(x)

      m.print_pipeline_stage_assignment_summary(line_length=85,
                                                print_fn=print_fn)

      # pylint: disable=line-too-long
      self.assertEqual(strings[0], 'Model: "simple_printable_model"')
      self.assertEqual(strings[1], '_' * 85)
      self.assertEqual(
          strings[2],
          'Layer (type) (node index)         Input Layers                      Pipeline Stage   '
      )
      self.assertEqual(strings[3], '=' * 85)
      self.assertRegex(
          strings[4],
          r'flatten \(Flatten\) \(0\)             input_[0-9]+                           None             '
      )
      self.assertEqual(strings[5], '_' * 85)
      self.assertRegex(
          strings[6],
          r'flatten \(Flatten\) \(1\)             input_[0-9]+                           None             '
      )
      self.assertEqual(strings[7], '_' * 85)
      self.assertEqual(
          strings[8],
          'concatenate (Concatenate) (0)     flatten                           None             '
      )
      self.assertEqual(
          strings[9],
          '                                  flatten                                            '
      )
      self.assertEqual(strings[10], '_' * 85)
      self.assertEqual(
          strings[11],
          'dense (Dense) (0)                 concatenate                       None             '
      )
      self.assertEqual(strings[12], '_' * 85)
      self.assertEqual(
          strings[13],
          'tf.math.multiply (TFOpLambda) (0) dense                             None             '
      )
      self.assertEqual(strings[14], '=' * 85)
      # pylint: enable=line-too-long


if __name__ == '__main__':
  test.main()
