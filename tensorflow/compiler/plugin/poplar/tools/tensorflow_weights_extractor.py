#!/usr/bin/env python3

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
from tensorflow.python.ipu.dataset_extractor import export_variables
from tensorflow.python.saved_model import saved_model


class _Variable:
  type_map = {"F32": tf.float32, "F16": tf.float16}

  def __init__(self, shape, type_str):
    self.type = _Variable.type_map[type_str]
    self.shape = shape
    self.validated = False

  def validate(self, graph_shape, graph_type):
    assert self.type == graph_type
    assert self.shape == graph_shape
    self.validated = True


class _Metadata:
  def __init__(self, metadata_file):
    assert os.path.isfile(metadata_file)
    with open(metadata_file, 'r') as f:
      content = json.load(f)
      variables = [
          inp for inp in content.get("inputs", [])
          if inp["type"] == "parameter"
      ]

      self.variables = {}
      for var in variables:
        assert var["name"] not in self.variables, (
            "There is more than one variable "
            "with the name %s") % var["name"]
        self.variables[var["name"]] = _Variable(var["shape"], var["data_type"])

  def validate(self, name, shape, dtype):
    # Remove :0 suffix from TF
    if name.endswith(":0"):
      name = name[:-2]

    assert name in self.variables, "%s not found in list of variables %s" % (
        name, list(self.variables.keys()))
    var = self.variables[name]
    assert not var.validated, ("There is more than one variable named %s in "
                               "the graph") % name
    var.validate(shape, dtype)

  def assert_all_validated(self):
    not_found = []
    for var_name, var in self.variables.items():
      if not var.validated:
        not_found.append("Variable %s was not found in the graph" % var_name)

    assert not not_found, "\n".join(not_found)


def _export_variables(sess, variables, meta, output_file):
  dirname = os.path.dirname(output_file)
  if dirname:
    os.makedirs(dirname, exist_ok=True)
  if meta:
    for v in variables:
      logging.debug("Validating Variable name = %s , dtype = %s, shape = %s",
                    v.name, v.dtype, v.shape)
      meta.validate(v.name, v.shape, v.dtype)
  sess.run(export_variables(variables, output_file, is_input=False))


def _export_v2_SavedModel(save_path, output_file, meta):
  with Session(graph=tf.Graph()) as sess:
    model = saved_model.load(save_path)
    sess.run(global_variables_initializer())
    _export_variables(sess, model.variables, meta, output_file)


def _export_v1_meta_graph(save_path, output_file, meta):
  graph = tf.Graph()
  with Session(graph=graph) as sess:
    saver = train.import_meta_graph(save_path + ".meta")
    # pylint: disable=protected-access
    # Move all the Ops to the CPU
    for op in graph.get_operations():
      op._set_device('/device:CPU:0')
      op._set_attr('_XlaCompile', attr_value_pb2.AttrValue(b=False))
      op._set_attr('_XlaScope', attr_value_pb2.AttrValue(s=b''))
    # pylint: enable=protected-access

    # Load the variables from file.
    saver.restore(sess, save_path)
    # Export the variables to the output_file
    _export_variables(sess, global_variables(), meta, output_file)


def export_variables_from_live_session(sess, output_file, gc_metadata=None):
  """Export the variables from a given tf.Session to a given output folder and
  optionally validate them against a given json metadata file.
  """
  meta = _Metadata(gc_metadata) if gc_metadata else None
  with sess.graph.as_default():
    _export_variables(sess, global_variables(), meta, output_file)


def export_variables_from_live_model(model, output_file, gc_metadata=None):
  """Export the variables from a given Keras model to a given output folder and
  optionally validate them against a given json metadata file.
  """
  sess = backend.get_session()
  meta = _Metadata(gc_metadata) if gc_metadata else None
  _export_variables(sess, model.variables, meta, output_file)


def export_model(save_path, output_file, gc_metadata=None):
  """ Export the variables from a TF v1 or v2 model to a given output folder and
  optionally validate them against a given json metadata file.
  """
  meta = _Metadata(gc_metadata) if gc_metadata else None

  meta_file = save_path + ".meta"
  pb_file = os.path.join(save_path, "saved_model.pb")
  pbtxt_file = os.path.join(save_path, "saved_model.pbtxt")
  if os.path.isfile(meta_file):
    logging.info("Loading v1 saved model from folder %s", save_path)
    _export_v1_meta_graph(save_path, output_file, meta)
  elif os.path.isfile(pb_file) or os.path.isfile(pbtxt_file):
    logging.info("Loading v2 Keras SavedModel from folder %s", save_path)
    _export_v2_SavedModel(save_path, output_file, meta)
  else:
    logging.fatal(("Could not find any Tensorflow v1 (%s) or v2 (%s or %s)"
                   " models"), meta_file, pb_file, pbtxt_file)
    exit(1)
  if meta:
    meta.assert_all_validated()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument("-m",
                      "--metadata",
                      type=str,
                      help="Metadata file to validate the weights against")
  parser.add_argument("-o",
                      "--output",
                      type=str,
                      default=".",
                      help="File where to write the extracted weights.")
  parser.add_argument("-f",
                      "--force",
                      action='store_true',
                      help="Force delete and re-create output folder")
  parser.add_argument("-s",
                      "--save-path",
                      type=str,
                      default=".",
                      help="Path to the SavedModel to export.")
  parser.add_argument("-D",
                      "--debug",
                      action='store_true',
                      help="Enable debug printing")

  args = parser.parse_args()
  logging_level = logging.DEBUG if args.debug else logging.INFO
  logging.basicConfig(level=logging_level)

  if args.force:
    assert args.output != ".", "You cannot delete the current directory"
    if os.path.isdir(args.output):
      shutil.rmtree(args.output)

  export_model(args.save_path, args.output, args.metadata)
