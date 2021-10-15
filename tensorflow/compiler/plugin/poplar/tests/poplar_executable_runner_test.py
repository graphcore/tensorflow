# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import tempfile
import glob
import os
import subprocess
import sys

import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.platform import googletest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.compat.v1.train import Saver
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ipu import utils
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.platform import test
from tensorflow.python.ops.variables import global_variables_initializer
from tensorflow.python.saved_model import saved_model


def filesInFolder(folder):
  return [
      name for name in os.listdir(folder)
      if os.path.isfile(os.path.join(folder, name))
  ]


class MyInitializer:
  def __init__(self, value):
    self.value = value

  def __call__(self, shape, dtype=None):
    assert dtype in [None, dtypes.float32]

    def generator(*args):
      return self.value + sum([10 * idx + v for idx, v in enumerate(args)])

    return np.fromfunction(generator, shape)


def instantiate_lenet():
  from tensorflow.python import keras
  from tensorflow.python.keras import layers
  model = keras.Sequential()

  model.add(
      layers.Conv2D(filters=6,
                    kernel_size=(3, 3),
                    activation='relu',
                    input_shape=(32, 32, 1)))
  model.add(layers.AveragePooling2D())
  model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
  model.add(layers.AveragePooling2D())
  model.add(layers.Flatten())
  model.add(layers.Dense(units=120, activation='relu'))
  model.add(layers.Dense(units=84, activation='relu'))
  model.add(layers.Dense(units=10, activation='softmax'))

  inp = keras.Input(shape=(32, 32, 1), dtype=np.float32)
  out = model(inp)
  return out, inp, model


def instantiate_lenet_fix_weights():
  from tensorflow.python import keras
  from tensorflow.python.keras import layers
  model = keras.Sequential()

  model.add(
      layers.Conv2D(filters=6,
                    kernel_size=(3, 3),
                    activation='relu',
                    input_shape=(32, 32, 1),
                    kernel_initializer=MyInitializer(10.0)))
  model.add(layers.AveragePooling2D())
  model.add(
      layers.Conv2D(filters=16,
                    kernel_size=(3, 3),
                    activation='relu',
                    kernel_initializer=MyInitializer(20.0)))
  model.add(layers.AveragePooling2D())
  model.add(layers.Flatten())
  model.add(
      layers.Dense(units=120,
                   activation='relu',
                   kernel_initializer=MyInitializer(30.0)))
  model.add(
      layers.Dense(units=84,
                   activation='relu',
                   kernel_initializer=MyInitializer(0.4)))
  model.add(
      layers.Dense(units=10,
                   activation='softmax',
                   kernel_initializer=MyInitializer(5.5)))

  inp = keras.Input(shape=(32, 32, 1), dtype=np.float32)
  out = model(inp)
  return out, inp, model


class PoplarExecutableRunnerTest(xla_test.XLATestCase):
  # Overriding abstract method.
  def cached_session(self):
    return 0

  # Overriding abstract method.
  def test_session(self):
    return 0

  def configureIPU(self, serialization_folder=None, offline_compilation=True):
    opts = IPUConfig()
    if offline_compilation:
      opts.device_connection.version = 'ipu1'
      opts.device_connection.type = utils.DeviceConnectionType.NEVER
    if serialization_folder:
      opts.serialization_output_folder = serialization_folder
    opts.configure_ipu_system()

  def runCommand(self, cmd):
    logging.info("Running: %s", " ".join(cmd))
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    self.assertTrue(out.returncode == 0, out.stdout.decode("utf-8"))
    logging.info(out.stdout.decode("utf-8"))

  def runPythonCommand(self, cmd):
    python_cmd = cmd
    python_cmd.insert(0, sys.executable)
    self.runCommand(python_cmd)

  def getSingleFileWithExt(self, folder, extension):
    all_files = glob.glob("%s/*.%s" % (folder, extension))
    logging.info("%s files in %s: %s", extension, folder, all_files)
    self.assertEqual(
        len(all_files), 1,
        "There should be exactly one file with the extension %s in %s: %s" %
        (extension, folder, all_files))
    return all_files[0]

  @test_util.deprecated_graph_mode_only
  def testKerasLenet(self):
    """Check that the output of PoplarExecutableRunner produces the same output as the original Graph execution.
    """
    if utils.running_on_ipu_model():
      self.skipTest("PoplarExecutableRunner only works with physical IPUs")

    with tempfile.TemporaryDirectory() as tmp:
      poplar_binaries_folder = os.path.join(tmp, "poplar")
      model_path = os.path.join(tmp, "model")
      weights_file = os.path.join(tmp, "weights.bin")
      output_path = os.path.join(tmp, "output")
      input_values = np.random.uniform(size=(1, 32, 32, 1))
      input_file = "%s/input.bin" % tmp

      with self.session() as sess:

        self.configureIPU(poplar_binaries_folder, False)
        with ops.device("/device:IPU:0"):
          out, inp, model = instantiate_lenet()

        utils.move_variable_initialization_to_cpu()
        sess.run(global_variables_initializer())

        utils.export_inputs_to_file([inp], input_file, {inp: input_values})

        # Run the model once to generate the poplar binaries.
        reference_values = sess.run(out, {inp: input_values})

        # Export the model & weights.
        saved_model.save(model, model_path)

      metadata_file = self.getSingleFileWithExt(poplar_binaries_folder, "json")
      executable_file = self.getSingleFileWithExt(poplar_binaries_folder,
                                                  "ipu_bin")

      self.runPythonCommand(
          (("./tensorflow/compiler/plugin/poplar/tools/"
            "tensorflow_weights_extractor.py -o %s -s %s -m %s") %
           (weights_file, model_path, metadata_file)).split())

      self.runCommand((("./third_party/ipus/tools/PoplarExecutableRunner"
                        " --binaries %s,%s,%s "
                        "--output_folder=%s --strict") % (
                            executable_file,
                            weights_file,
                            input_file,
                            output_path,
                        )).split())

      output_file = self.getSingleFileWithExt(output_path, "data")
      with open(output_file, 'r') as f:
        runner_values = np.array(json.load(f))
        logging.info("Reference %s\nRunner: %s", reference_values,
                     runner_values)
        self.assertAllClose(reference_values, runner_values)

  @test_util.deprecated_graph_mode_only
  def testWeightsExportersNoMetadata(self):
    """ Check that the weights extractor produces the same output with
     TF v1 and v2 models."""
    # Disable the IPU model
    poplar_flags = os.environ.get("TF_POPLAR_FLAGS",
                                  "").replace("--use_ipu_model", "")
    with test.mock.patch.dict("os.environ",
                              {"TF_POPLAR_FLAGS": poplar_flags
                               }), tempfile.TemporaryDirectory() as tmp:
      model_path_keras = os.path.join(tmp, "model_keras")
      model_path_session = os.path.join(tmp, "model_session")
      weights_keras = os.path.join(tmp, "weights_keras.bin")
      weights_session = os.path.join(tmp, "weights_session.bin")

      with self.session() as sess:
        self.configureIPU()
        with ops.device("/device:IPU:0"):
          _, _, model = instantiate_lenet()
        utils.move_variable_initialization_to_cpu()
        sess.run(global_variables_initializer())

        # Export the model & weights.
        saved_model.save(model, model_path_keras)
        Saver().save(sess, model_path_session)

      self.runPythonCommand((("./tensorflow/compiler/plugin/poplar/tools/"
                              "tensorflow_weights_extractor.py -o %s -s %s") %
                             (weights_keras, model_path_keras)).split())

      self.runPythonCommand((("./tensorflow/compiler/plugin/poplar/tools/"
                              "tensorflow_weights_extractor.py -o %s -s %s") %
                             (weights_session, model_path_session)).split())

      with open(weights_session, 'rb') as s, open(weights_keras, 'rb') as k:
        self.assertEqual(s.read(), k.read())

  @test_util.deprecated_graph_mode_only
  def testWeightsExportersMetadataLive(self):
    """Export weights directly from a live model.
    """
    poplar_flags = os.environ.get("TF_POPLAR_FLAGS",
                                  "").replace("--use_ipu_model", "")
    with test.mock.patch.dict("os.environ",
                              {"TF_POPLAR_FLAGS": poplar_flags
                               }), tempfile.TemporaryDirectory() as tmp:
      poplar_binaries_folder = os.path.join(tmp, "poplar")
      weights_keras = os.path.join(tmp, "weights_keras.bin")
      weights_session = os.path.join(tmp, "weights_session.bin")

      with self.session() as sess:
        self.configureIPU(poplar_binaries_folder)
        with ops.device("/device:IPU:0"):
          out, inp, model = instantiate_lenet_fix_weights()

        utils.move_variable_initialization_to_cpu()
        sess.run(global_variables_initializer())

        # Run the model once to generate the poplar binaries.
        try:
          sess.run(out, {inp: np.ones((1, 32, 32, 1))})
        except errors.InvalidArgumentError:
          pass

      metadata_file = self.getSingleFileWithExt(poplar_binaries_folder, "json")

      with self.session() as sess:
        self.configureIPU()
        with ops.device("/device:IPU:0"):
          _, _, _ = instantiate_lenet_fix_weights()

        utils.move_variable_initialization_to_cpu()
        sess.run(global_variables_initializer())

        utils.export_variables_from_live_session(sess, weights_session,
                                                 metadata_file)

      with self.session() as sess:
        self.configureIPU()
        with ops.device("/device:IPU:0"):
          _, _, model = instantiate_lenet_fix_weights()

        utils.move_variable_initialization_to_cpu()
        sess.run(global_variables_initializer())
        utils.export_variables_from_live_model(model, weights_keras,
                                               metadata_file)

      with open(weights_session, 'rb') as s, open(weights_keras, 'rb') as k:
        self.assertEqual(s.read(), k.read())


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
