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
# =============================================================================
"""
Serving utilities
~~~~~~~~~~~~~~~~~
"""

import collections
import inspect
import os
import tempfile
import uuid

from tensorflow.python.client import session as session_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ipu import application_compile_op
from tensorflow.python.ipu import embedded_runtime
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import loops
from tensorflow.python.ipu.ops import pipelining_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils


def _validate_signature(defunc,
                        input_signature,
                        input_dataset,
                        non_feed_inputs=None):
  """Validate and update `input_signature` if necessary to match the arguments
  of `defunc`.

  Args:
    defunc (Callable or tf.function): Function whose signature
      is analyzed.
    input_signature (list or tuple): A sequence of `tf.TensorSpec` objects
      that describe the input arguments of `defunc`. If `defunc` is a
      `tf.function` and `input_signature` was specified during `tf.function`
      creation then this argument can be None.
    input_dataset (tf.Dataset): Dataset from which `input_signature` will be
      inferred.
    non_feed_inputs (list, optional): List of inputs that will be provided
      to the graph without usage of infeed queue.

  Returns:
    list: List of `tf.TensorSpec` objects with types, shapes and names.

  Raises:
    TypeError: If `input_signature` is not a `tf.Dataset`, tuple, list
      or `NoneType`.
    ValueError: If `input_signature` is not provided and `defunc` is
      not a `tf.function`.
    ValueError: If the number of passed/inferred signatures of inputs that
      are passed to the graph using infeed queue is different than the number
      of arguments of `defunc`.
  """
  if input_dataset is not None:
    input_signature = input_dataset.element_spec
    if isinstance(input_signature, tensor_spec.TensorSpec):
      input_signature = [input_signature]
  elif input_signature is None:
    if isinstance(defunc, def_function.Function):
      input_signature = defunc.input_signature

  if input_signature is None:
    raise ValueError(f'Missing input_signature for {defunc.__name__}')

  if not isinstance(input_signature, (tuple, list)):
    raise TypeError('input_signature must be either a tuple or a '
                    'list, received ' + str(type(input_signature)))

  names = list(inspect.signature(defunc).parameters.keys())
  if non_feed_inputs:
    names = names[len(non_feed_inputs):]
    if len(input_signature) > len(names):
      input_signature = input_signature[len(non_feed_inputs):]

  if len(input_signature) != len(names):
    raise ValueError('Length of input_signature does not match the number of '
                     f'{defunc.__name__} arguments')

  # Store argument names in the input_signature
  input_signature = [
      tensor_spec.TensorSpec.from_spec(spec, name=name)
      for spec, name in zip(input_signature, names)
  ]

  return input_signature


def _create_feeds(input_signature, input_dataset=None):
  """Create infeed and outfeed queues for the given signature.

  Args:
    input_signature (list): List of signatures describing types and shapes of
      the dataset elements.
    input_dataset (tf.Dataset, optional): Dataset to be used for creating feeds.

  Returns:
    tuple(IPUInfeedQueue, IPUOutfeedQueue): Infeed and outfeed queues created
      based on the `input_signature`.
  """
  if input_dataset is None:
    inputs = [array_ops.zeros(s.shape, s.dtype) for s in input_signature]
    input_dataset = dataset_ops.Dataset.from_tensors(tuple(inputs))
    input_dataset = input_dataset.repeat()

  infeed = ipu_infeed_queue.IPUInfeedQueue(input_dataset)
  outfeed = ipu_outfeed_queue.IPUOutfeedQueue()
  return (infeed, outfeed)


def _export_saved_model(defunc, export_dir, variable_initializer,
                        input_signature, output_names):
  """Compile Poplar executable and export saved model.

  Args:
    defunc (Callable or tf.function): Function that runs inference step.
    export_dir (str): Path to the directory where the SavedModel will be
      written.
    variable_initializer (function): A function that initializes variables.
      Takes a `tf.Session` as the only argument.
    input_signature (list): List of signatures of inputs that will be provided
      to the graph using infeed queue.
    output_names (str or list): Output name or list of output names for the
      outputs in the SavedModel's SignatureDef. If ``output_names`` is ``None``,
      outputs will be named: ``output_0``, ``output_1`` and so on.

  Returns:
    function: A reference to the same predict function that was exported
    using the SavedModel format. This function uses the embedded runtime op to
    run the executable that was included in the SavedModel's `assets` subfolder.

  Raises:
    TypeError: If ``output_names`` is neither a string nor a list.
    ValueError: If length of ``output_names`` does not match the number of
      results returned by ``defunc``.
  """
  with tempfile.TemporaryDirectory() as tmp_folder:
    unique_name = str(uuid.uuid4())
    # Compile the Poplar executable
    exec_filename = f'application_{unique_name}.popef'
    poplar_exec_filepath = os.path.join(tmp_folder, exec_filename)
    g = ops.Graph()
    with g.as_default(), session_lib.Session(graph=g) as sess:
      compile_op = application_compile_op.experimental_application_compile_op(
          defunc, output_path=poplar_exec_filepath, freeze_variables=True)
      if variable_initializer:
        variable_initializer(sess)
      sess.run(compile_op)

    # Create SavedModel with Embedded Runtime
    g = ops.Graph()
    with g.as_default(), session_lib.Session(graph=g).as_default() as sess:
      engine_name = f'engine_{unique_name}'

      def predict_func(*args, asset_path=None):
        if asset_path is None:
          asset_path = constant_op.constant(poplar_exec_filepath,
                                            dtype=dtypes.string)
          asset_path_var = variables.Variable(asset_path,
                                              dtype=dtypes.string,
                                              trainable=False,
                                              name="asset_path_var")
          sess.run(variables.variables_initializer([asset_path_var]))
          g.add_to_collection(ops.GraphKeys.ASSET_FILEPATHS, asset_path)
          final_path = asset_path_var.read_value()
        else:
          final_path = asset_path

        ctx = embedded_runtime.embedded_runtime_start(final_path, [],
                                                      engine_name)
        return embedded_runtime.embedded_runtime_call(args, ctx)

      input_phs = collections.OrderedDict()
      for s in input_signature:
        input_phs[s.name] = array_ops.placeholder(dtype=s.dtype,
                                                  shape=s.shape,
                                                  name=s.name)

      outputs = predict_func(*tuple(input_phs.values()))

      if output_names:
        if isinstance(output_names, str):
          output_names = [output_names]

        if not isinstance(output_names, list):
          raise TypeError('output_names must be either a string or a '
                          'list, received ' + str(type(output_names)))

        if len(outputs) != len(output_names):
          raise ValueError(
              'Length of output_names does not match the number of results '
              'returned by the predict function.')
      else:
        output_names = [f"output_{i}" for i, _ in enumerate(outputs)]

      outputs = dict(zip(output_names, outputs))

      saved_model_builder = builder.SavedModelBuilder(export_dir)
      sig_def = signature_def_utils.predict_signature_def(inputs=input_phs,
                                                          outputs=outputs)
      init_op = variables.global_variables_initializer()
      sess.run(init_op)
      saved_model_builder.add_meta_graph_and_variables(
          sess,
          tags=[tag_constants.SERVING],
          signature_def_map={
              signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: sig_def
          },
          assets_collection=g.get_collection(ops.GraphKeys.ASSET_FILEPATHS),
          main_op=init_op)
      saved_model_builder.save()

  # Adjust asset_path to the proper location in the SavedModel
  asset_path = os.path.join(export_dir, "assets", exec_filename)

  def runtime_func(*args):
    return predict_func(*args, asset_path=asset_path)

  return runtime_func


def export_single_step(predict_step,
                       export_dir,
                       iterations,
                       input_signature=None,
                       input_dataset=None,
                       variable_initializer=None,
                       output_names=None):
  """Create a SavedModel in `export_dir` for TensorFlow Serving.

  Wrap `predict_step` inside a while loop, add an infeed for the inputs and
  an outfeed for the outputs, freeze any variables into constants and write
  a SavedModel containing an IPU runtime function and Poplar executable.

  Args:
    predict_step (Callable or tf.function): Function to export.
    export_dir (str): Path to the directory where the SavedModel will be
      written.
    iterations (int): Number of loop iterations.
    input_signature (list or tuple, optional): A sequence of `tf.TensorSpec`
      objects that describe the input arguments of the `predict_step` function.
      If `input_dataset` is provided, this argument should be None.
      If `input_dataset` is not provided and `predict_step` is a `tf.function`
      and `input_signature` was specified during `tf.function` creation then
      this argument can be None and the signature will be captured directly from
      `predict_step`.
    input_dataset (tf.Dataset, optional): Dataset from which `input_signature`
      will be inferred. If `input_signature` is provided, this argument should
      be None.
    variable_initializer (Callable, optional): A function that initializes
      variables. Takes a `tf.Session` as the only argument.
      For example, this function allows restoring model's variables from a
      checkpoint:

      .. code-block:: python

        def variable_initializer(session):
          saver = tf.train.Saver()
          ipu.utils.move_variable_initialization_to_cpu()
          init = tf.global_variables_initializer()
          session.run(init)
          saver.restore(session, 'path/to/checkpoint')

    output_names (str or list, optional): Output name or list of output names
      for the outputs in the SavedModel's SignatureDef. If not provided, outputs
      will be named: ``output_0``, ``output_1`` and so on.

  Returns:
    function: A reference to the same predict function that was exported using
    the SavedModel format. This function uses the embedded runtime op to run
    the executable that was included in the SavedModel's `assets` subfolder.

  Raises:
    ValueError: If both `input_signature` and `input_dataset` are provided.
    ValueError: If ``export_dir`` is not an empty directory.
    TypeError: If `input_dataset` was provided and is not an instance of
      `tf.Dataset`
  """
  if os.path.isdir(export_dir) and os.listdir(export_dir):
    raise ValueError(
        "Directory is not empty. Please specify an empty directory.")

  if input_signature is not None and input_dataset is not None:
    raise ValueError('Both input_signature and input_dataset cannot be '
                     'provided to export_single_step. Please pass only '
                     'one of them.')

  if input_dataset is not None and not isinstance(input_dataset,
                                                  dataset_ops.Dataset):
    raise TypeError('If input_dataset is provided, it should be an instance '
                    'of tf.Dataset.')

  input_signature = _validate_signature(predict_step, input_signature,
                                        input_dataset)
  infeed, outfeed = _create_feeds(input_signature, input_dataset)

  # Add outfeed queue
  def predict_step_with_outfeed(*args):
    output_enqueue = outfeed.enqueue(predict_step(*args))
    return output_enqueue

  # Wrap in a loop
  def predict_loop():
    r = loops.repeat(iterations, predict_step_with_outfeed, [], infeed)
    return r

  return _export_saved_model(predict_loop, export_dir, variable_initializer,
                             input_signature, output_names)


def export_pipeline(computational_stages,
                    export_dir,
                    iterations,
                    inputs=None,
                    device_mapping=None,
                    pipeline_schedule=None,
                    poplar_options=None,
                    name=None,
                    input_signature=None,
                    input_dataset=None,
                    variable_initializer=None,
                    output_names=None):
  """Create a pipelined SavedModel in `export_dir` for TensorFlow Serving.

  Create a pipeline op using `computational_stages`, add an infeed for
  the inputs and an outfeed for the outputs, freeze any variables into constants
  and write a SavedModel containing an IPU runtime function and Poplar
  executable.

  Args:
    computational_stages (list): A list of python functions, where each function
      represents a computational pipeline stage. The function takes the
      outputs of the previous pipeline stage as its inputs.
    export_dir (str): Path to the directory where the SavedModel will be
      written.
    iterations (int): The number of times each computational stage
      will be executed during the execution of the pipeline. It can also be
      considered as the pipeline depth.
    inputs (list, optional): Arguments passed to the first computational stage.
    device_mapping (list, optional): If provided, a list of length equal to the
      number of computational stages. An element at index `i` in the list
      represents which IPU the `computational_stages[i]` should reside on. This
      can be used to make sure computational stages which share `tf.Variable`
      objects are resident on the same IPU.
    pipeline_schedule (PipelineSchedule, optional): Which scheduling algorithm
      to use for pipeline lowering. Defaults to `PipelineSchedule.Grouped`.
    poplar_options (list, optional): If provided, a list of length equal to the
      number of computational stages. Each element is a `PipelineStageOptions`
      object which allows for fine grain control of the Poplar options for a
      given forward propagation computational stage.
    name (str, optional): Name of this pipeline.
    input_signature (list or tuple, optional): A sequence of `tf.TensorSpec`
      objects that describe the input arguments of the first computational
      stage. If `input_dataset` is provided, this argument should be None.
      If `input_dataset` is not provided and the first computational stage is a
      `tf.function` and `input_signature` was specified during `tf.function`
      creation then this argument can be None and the signature will be captured
      directly from the first computational stage.
    input_dataset (tf.Dataset, optional): Dataset from which `input_signature`
      will be inferred. If `input_signature` is provided, this argument should
      be None.
    variable_initializer (Callable, optional): A function that initializes
      variables. Takes a `tf.Session` as the only argument.
      For example, this function allows restoring model's variables from a
      checkpoint:

      .. code-block:: python

        def variable_initializer(session):
          saver = tf.train.Saver()
          ipu.utils.move_variable_initialization_to_cpu()
          init = tf.global_variables_initializer()
          session.run(init)
          saver.restore(session, 'path/to/checkpoint')

    output_names (str or list, optional): Output name or list of output names
      for the outputs in the SavedModel's SignatureDef. If not provided, outputs
      will be named: ``output_0``, ``output_1`` and so on.

  Returns:
    function: A reference to the same predict function that was exported using
    the SavedModel format. This function uses the embedded runtime op to run
    the executable that was included in the SavedModel's `assets` subfolder.

  Raises:
    ValueError: If both `input_signature` and `input_dataset` are provided.
    ValueError: If ``export_dir`` is not an empty directory.
    TypeError: If `input_dataset` was provided and is not an instance of
      `tf.Dataset`.
  """
  if os.path.isdir(export_dir) and os.listdir(export_dir):
    raise ValueError(
        "Directory is not empty. Please specify an empty directory.")

  if input_signature is not None and input_dataset is not None:
    raise ValueError('Both input_signature and input_dataset cannot be '
                     'provided to export_pipeline. Please pass only '
                     'one of them.')

  if input_dataset is not None and not isinstance(input_dataset,
                                                  dataset_ops.Dataset):
    raise TypeError('If input_dataset is provided, it should be an instance '
                    'of tf.Dataset.')

  first_stage = computational_stages[0]
  input_signature = _validate_signature(first_stage, input_signature,
                                        input_dataset, inputs)
  infeed, outfeed = _create_feeds(input_signature, input_dataset)

  def defunc():
    return pipelining_ops.pipeline(
        computational_stages=computational_stages,
        gradient_accumulation_count=iterations,
        inputs=inputs,
        infeed_queue=infeed,
        outfeed_queue=outfeed,
        device_mapping=device_mapping,
        pipeline_schedule=pipeline_schedule,
        forward_propagation_stages_poplar_options=poplar_options,
        name=name)

  return _export_saved_model(defunc, export_dir, variable_initializer,
                             input_signature, output_names)
