#!/usr/bin/env python3
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

import abc
import argparse
import logging
import json
import os
import shutil

import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.compat.v1 import Session
from tensorflow.compat.v1 import global_variables
from tensorflow.compat.v1 import train
from tensorflow.python.ops.variables import global_variables_initializer
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.ipu.dataset_extractor import export_variables, import_variables, get_variable_handles
from tensorflow.python.saved_model import saved_model


# Check type and shape match based on name
class Model(abc.ABC):
  @abc.abstractmethod
  def variables(self):
    pass

  @abc.abstractmethod
  def save(self, path):
    pass

  @abc.abstractmethod
  def session(self):
    pass

  def import_variables(self, input_files):  #pylint: disable=unused-argument
    with self.session().graph.as_default():
      self.session().run(
          import_variables(self.variables(),
                           input_files,
                           is_input=False,
                           strict=False))

  def export_variables(self, output_file, metadata):
    dirname = os.path.dirname(output_file)
    variables = self.variables()
    if dirname:
      os.makedirs(dirname, exist_ok=True)
    handle_names = self.session().run(get_variable_handles(variables))
    handle_names = [x[0] for x in handle_names]
    self.session().run(
        export_variables(variables,
                         names=handle_names,
                         filename=output_file,
                         metadata=metadata))


class LiveModel(Model):
  def __init__(self, session, model):
    self.session_ = session
    self.model_ = model

  def variables(self):
    return self.model_.variables

  def save(self, path):
    saved_model.save(self.model_, path)

  def session(self):
    return self.session_


class LiveSession(Model):
  def __init__(self, session):
    self.session_ = session
    with self.session_.graph.as_default():
      self.variables_ = global_variables()

  def variables(self):
    return self.variables_

  def save(self, path):
    train.Saver().save(self.session_, path)

  def session(self):
    return self.session_


class SavedModel(LiveModel):
  def __init__(self, session, save_path):
    with session.graph.as_default():
      model = saved_model.load(save_path)
      session.run(global_variables_initializer())
      super().__init__(session, model)


class MetaGraph(LiveSession):
  def __init__(self, session, save_path):
    self.saver_ = train.import_meta_graph(save_path + ".meta")
    # pylint: disable=protected-access
    # Move all the Ops to the CPU
    for op in session.graph.get_operations():
      op._set_device('/device:CPU:0')
      op._set_attr('_XlaCompile', attr_value_pb2.AttrValue(b=False))
      op._set_attr('_XlaScope', attr_value_pb2.AttrValue(s=b''))
    # pylint: enable=protected-access

    # Load the variables from file.
    self.saver_.restore(session, save_path)
    super().__init__(session)


def export_variables_from_live_session(session, output_file, gc_metadata=None):
  """Export the variables from a given tf.Session to a given output folder and
  optionally validate them against a given json metadata file.
  """
  LiveSession(session).export_variables(output_file, gc_metadata)


def export_variables_from_live_model(model, output_file, gc_metadata=None):
  """Export the variables from a given Keras model to a given output folder and
  optionally validate them against a given json metadata file.
  """
  LiveModel(backend.get_session(),
            model).export_variables(output_file, gc_metadata)


def export_model(save_path, output_file, gc_metadata=None):
  """ Export the variables from a TF v1 or v2 model to a given output folder and
  optionally validate them against a given json metadata file.
  """
  meta_file = save_path + ".meta"
  pb_file = os.path.join(save_path, "saved_model.pb")
  pbtxt_file = os.path.join(save_path, "saved_model.pbtxt")
  with Session(graph=tf.Graph()) as session:
    if os.path.isfile(meta_file):
      logging.info("Loading v1 saved model from folder %s", save_path)
      MetaGraph(session, save_path).export_variables(output_file, gc_metadata)
    elif os.path.isfile(pb_file) or os.path.isfile(pbtxt_file):
      logging.info("Loading v2 Keras SavedModel from folder %s", save_path)
      SavedModel(session, save_path).export_variables(output_file, gc_metadata)
    else:
      logging.fatal(("Could not find any TensorFlow v1 (%s) or v2 (%s or %s)"
                     " models"), meta_file, pb_file, pbtxt_file)
      exit(1)


def import_data_in_live_session(session, data_files, output_folder=None):
  """Import the data stored in the given data_files in the variables of
  the passed session.
  If output_folder is provided then the live session is saved to that folder.
  """
  live = LiveSession(session)
  live.import_variables(data_files)
  if output_folder:
    live.save(output_folder)


def import_data_in_live_model(model, data_files, output_folder=None):
  """Import the data stored in the given data_files in the variables of
  the passed model.
  If output_folder is provided then the live model is saved to that folder.
  """
  live = LiveModel(backend.get_session(), model)
  live.import_variables(data_files)
  if output_folder:
    live.save(output_folder)


def import_data_in_model(import_path, save_path, data_files):
  """Import the data stored in the given data_files in the variables of a
  TF v1 or v2 model and save the updated model to save_path.
  """
  meta_file = import_path + ".meta"
  pb_file = os.path.join(import_path, "saved_model.pb")
  pbtxt_file = os.path.join(import_path, "saved_model.pbtxt")
  with Session(graph=tf.Graph()) as session:
    if os.path.isfile(meta_file):
      logging.info("Loading v1 saved model from folder %s", import_path)
      graph = MetaGraph(session, import_path)
      graph.import_variables(data_files)
      graph.save(save_path)
    elif os.path.isfile(pb_file) or os.path.isfile(pbtxt_file):
      logging.info("Loading v2 Keras SavedModel from folder %s", import_path)
      model = SavedModel(session, import_path)
      model.import_variables(data_files)
      model.save(save_path)
    else:
      logging.fatal(("Could not find any TensorFlow v1 (%s) or v2 (%s or %s)"
                     " models"), meta_file, pb_file, pbtxt_file)
      exit(1)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument("-m",
                      "--metadata",
                      type=str,
                      help="Metadata file to validate the weights against")
  parser.add_argument(
      "-o",
      "--output",
      type=str,
      default=".",
      help=
      "Directory or file where to write the extracted weights / updated model."
  )
  parser.add_argument("-f",
                      "--force",
                      action='store_true',
                      help="Force delete and re-create output folder")
  parser.add_argument("-s",
                      "--save-path",
                      type=str,
                      default=".",
                      help="Path to the SavedModel or meta graph to export.")
  parser.add_argument("-D",
                      "--debug",
                      action='store_true',
                      help="Enable debug printing")
  parser.add_argument("-i",
                      "--import-data",
                      type=str,
                      help="Folder or file to import data from")

  args = parser.parse_args()
  logging_level = logging.DEBUG if args.debug else logging.INFO
  logging.basicConfig(level=logging_level)

  if args.force:
    assert args.output != ".", "You cannot delete the current directory"
    if os.path.isdir(args.output):
      shutil.rmtree(args.output)

  if args.import_data:
    if os.path.isdir(args.import_data):
      files = glob.glob("%s/*.bin" % args.import_data)
    else:
      files = [args.import_data]
    import_data_in_model(args.save_path, args.output, files)
  else:
    output_file = args.output
    # Check if --output is a folder or a file
    if args.output.endswith(".bin"):
      # If it's a file use it  as is.
      output_file = args.output
    else:
      # If it's a folder append a file name.
      output_file = os.path.join(args.output, "weights.bin")
    export_model(args.save_path, output_file, args.metadata)
