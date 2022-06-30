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
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.ipu import application_compile_op
from tensorflow.python.ipu import embedded_runtime
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import loops
from tensorflow.python.ipu.ops import pipelining_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.training import saver


def _validate_dir(directory, param_name):
  """Verifies that `directory` exists and is not empty.
  Args:
    directory (str): Path to the directory to be checked if exists and is not
      empty.
  Raises:
    ValueError: If `directory` is a non-existing or empty directory.
  """
  if os.path.isdir(directory) and not os.listdir(directory):
    raise ValueError(f'Directory "{directory}" passed as "{param_name}" param '
                     'is empty. Please specify a correct directory.')


def _validate_export_dir(export_dir, purge_export_dir=False):
  """Verify that `export_dir` exists and is empty. Remove the content of it
      if `purge_export_dir` is True.
  Args:
    export_dir (str): Path to the directory where the SavedModel will be
      written.
    purge_export_dir (Boolean): If True, before starting the export, the target
      directory is emptied. Otherwise no cleaning is performed and if target dir
      is not empty, the function fails with an error. Default is False.
  Raises:
    ValueError: If `export_dir` is not an empty directory.
  """

  if os.path.isdir(export_dir) and os.listdir(export_dir):
    if purge_export_dir:
      shutil.rmtree(export_dir)
    else:
      raise ValueError(f'Directory "{export_dir}" is not empty. '
                       'Please specify an empty directory.')


def _validate_signatures(predict_step,
                         predict_step_signature,
                         input_dataset=None,
                         preprocessing_step=None,
                         preprocessing_step_signature=None):
  """Verify that input signatures can be deduced from the given arguments of the
     exported model for preprocessing and inference parts.

  Args:
    predict_step (Callable or tf.function): Function that runs an inference
      step.
    predict_step_signature (list or tuple): A sequence of `tf.TensorSpec`
      objects that describe the input arguments of `predict_step`. If
      `predict_step` is a `tf.function` and `input_signature` was specified
      during `tf.function` creation then this argument can be None as it is
      derived from the tf.function directly.
    input_dataset (tf.Dataset): Dataset from which the exported model's
      `predict_step_signature` will be inferred.
    preprocessing_step (Callable or tf.function): Function that runs
      the preprocessing step.
    preprocessing_step_signature (list or tuple, optional): A sequence of
      `tf.TensorSpec` objects that describe the input arguments of
      `preprocessing_step`. If `preprocessing_step` is a `tf.function` and
      `input_signature` was specified during `tf.function` creation then
      this argument can be None as it is derived from the tf.function directly.
    non_feed_inputs (list, optional): List of inputs that will be provided
      to the graph without usage of infeed queue.

  Raises:
    TypeError: If `input_dataset` is not a `tf.Dataset` nor `NoneType`.
    TypeError: If `predict_step_signature` is not a tuple, list of
      `tf.TensorSpec` objects or `NoneType`.
    TypeError: If `preprocessing_step_signature` is not a tuple, list of
      `tf.TensorSpec` objects or `NoneType`.
    ValueError: If `predict_step_signature` is an empty tuple or list.
    ValueError: If `preprocessing_step_signature` is an empty tuple or list.
    ValueError: If `preprocessing_step` is not provided and both
      `predict_step_signature` and `input_dataset` are provided.
    ValueError: If `preprocessing_step`, `predict_step_signature`,
      `input_dataset` are not provided and `predict_step` is not a
      `tf.function` or `predict_step` is a `tf.function` but no
      `predict_step_signature` is provided.
    ValueError: If `preprocessing_step`, `preprocessing_step_signature`,
      `input_dataset` are provided at the same time.
    ValueError: If `preprocessing_step` is provided and both
      `preprocessing_step_signature` and `input_dataset` are not provided
      and `preprocessing_step` is not a `tf.function` or `preprocessing_step` is
      a `tf.function` but with no `predict_step_signature` provided.
    ValueError: If `preprocessing_step`, `predict_step_signature` are not
      provided and `predict_step` is not a `tf.function` or `predict_step` is a
      `tf.function` but with no `input_signature` provided.
  """

  is_predict_step_signature_set = predict_step_signature is not None
  is_input_dataset_set = input_dataset is not None
  is_preprocessing_step_signature_set = preprocessing_step_signature is not None
  is_preprocessing_step_set = preprocessing_step is not None

  def validate_tf_function(fn, fn_name, deduction_from_datset_possible=True):
    if not isinstance(fn, def_function.Function):
      raise ValueError(
          f'`input_signature` deduction from given `{fn_name}` is not '
          'possible. Please mark it as tf.function with `input_signature` '
          f'parameter set or provide `{fn_name}_signature`' +
          (' or `input_dataset`.' if deduction_from_datset_possible else "."))
    if fn.input_signature is None:
      raise ValueError(
          'Empty `input_signature` in the provided tf.function `predict_step`. '
          'Please specify it or provide `predict_step_signature`' +
          (' or `input_dataset`.' if deduction_from_datset_possible else "."))

  def validate_single_signature(signature_name, signature):
    if not isinstance(signature, (tuple, list)):
      raise TypeError(f'`{signature_name}` has to be an instance of tuple or '
                      f'list. Received {str(type(signature))}')
    if not signature:
      raise ValueError(f'`{signature_name}` cannot be empty.')

    for idx, value in enumerate(signature):
      if not isinstance(value, tensor_spec.TensorSpec):
        raise TypeError(f'`{signature_name}[{idx}]` is not an instance of '
                        'TensorSpec')

  if not is_preprocessing_step_set:
    if is_input_dataset_set and is_predict_step_signature_set:
      raise ValueError(
          'Both `predict_step_signature` and `input_dataset` cannot '
          'be provided at the same time. Please pass only one of them.')
    if not is_input_dataset_set and not is_predict_step_signature_set:
      validate_tf_function(predict_step, "predict_step")
  else:
    if is_preprocessing_step_signature_set and is_input_dataset_set:
      raise ValueError('Both `preprocessing_step_signature` and '
                       '`input_dataset` cannot be provided at the same '
                       'time. Please pass only one of them.')
    if not is_predict_step_signature_set and not is_input_dataset_set:
      validate_tf_function(preprocessing_step, "preprocessing_step")

    if not is_predict_step_signature_set:
      validate_tf_function(predict_step,
                           "predict_step",
                           deduction_from_datset_possible=False)

  if is_input_dataset_set and not isinstance(input_dataset,
                                             dataset_ops.Dataset):
    raise TypeError(
        'If `input_dataset` is provided, it should be an instance of '
        'tf.Dataset.')

  if is_preprocessing_step_signature_set:
    validate_single_signature('preprocessing_step_signature',
                              preprocessing_step_signature)

  if is_predict_step_signature_set:
    validate_single_signature('predict_step_signature', predict_step_signature)


def _prepare_input_signature(defunc,
                             defunc_signature=None,
                             input_dataset=None,
                             non_feed_inputs=None):
  """Prepare `input_signature` for `defunc` from given arguments.

  Args:
    defunc (Callable or tf.function): Function whose signature is analyzed.
    defunc_signature (list or tuple, optional): A sequence of `tf.TensorSpec`
      objects that describe the input arguments of `defunc`. If `defunc` is a
      `tf.function` and `input_signature` was specified during `tf.function`
      creation then this argument can be None.
    input_dataset (tf.Dataset, optional): Dataset from which `input_signature`
      will be inferred.
    non_feed_inputs (list, optional): List of inputs that will be provided
      to the graph without usage of infeed queue.

  Returns:
    list: List of `tf.TensorSpec` objects with types, shapes and names.

  Raises:
    ValueError: If it is not possible to create `input_signature` from given
      arguments.
    ValueError: If created `input_signature` is not an instance of list or
      tuple.
    ValueError: If the number of passed/inferred signatures of inputs that
      are passed to the graph using infeed queue is different than the number
      of arguments of `defunc`.
  """

  input_signature = None

  if defunc_signature is not None:
    input_signature = defunc_signature
  elif input_dataset is not None and isinstance(input_dataset,
                                                dataset_ops.Dataset):
    input_signature = input_dataset.element_spec
    if isinstance(input_signature, tensor_spec.TensorSpec):
      input_signature = (input_signature,)
  elif defunc_signature is None:
    if isinstance(defunc, def_function.Function):
      input_signature = defunc.input_signature

  if input_signature is None:
    raise ValueError(f'Missing input_signature for {defunc.__name__}')

  if not isinstance(input_signature, (tuple, list)):
    raise TypeError('input_signature must be either a tuple or a '
                    f'list, received {str(type(input_signature))}')

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


def _export_saved_model(predict_step,
                        export_dir,
                        variable_initializer,
                        input_signature,
                        output_names=None,
                        predict_step_signature=None,
                        preprocessing_step=None,
                        checkpoint_restore_dir=None):
  """Compile Poplar executable and export saved model.

  Args:
    predict_step (Callable or tf.function): Function that runs an inference
      step.
    export_dir (str): Path to the directory where the SavedModel will be
      written.
    variable_initializer (function): A function that initializes variables.
      Takes a `tf.Session` as the only argument.
    input_signature (list): List of signatures of inputs that will be provided
      to the exported model graph. If `preprocessing_step` (optional) is set,
      inputs will be processed by the `preprocessing_step` function on the CPU
      and the function's output will get passed further to `predict_step`
      through infeed queue. Otherwise inputs will be provided directly to the
      `predict_step` infeed queue.
    predict_step_signature (list): List of signatures of inputs that will be
      provided to `predict_step` using an infeed queue. If `preprocessing_step`
      is set, `predict_step_signature` is used for validation of compatibility
      between `preprocessing_step` outputs and `predict_step` inputs.
    output_names (str or list): Output name or list of output names for the
      outputs in the SavedModel's SignatureDef. If `output_names` is `None`,
      outputs will be named: `output_0`, `output_1` and so on.
    preprocessing_step (Callable or tf.function, optional): Function that runs
      preprocessing step on the CPU device.
    checkpoint_restore_dir (str): Path to saved checkpoint, where the model
      Variables are to be restored. To be used with preprocessing only.

  Returns:
    function: A reference to the same predict function that was exported
    using the SavedModel format. This function uses the embedded runtime op to
    run the executable that was included in the SavedModel's `assets`
    subfolder.

  Raises:
    TypeError: If `output_names` is neither a string nor a list.
    ValueError: If length of `output_names` does not match the number of
      results returned by `predict_step`.
  """
  if output_names:
    if isinstance(output_names, str):
      output_names = [output_names]

    if not isinstance(output_names, list):
      raise TypeError('output_names must be either a string or a '
                      'list, received ' + str(type(output_names)))

  with tempfile.TemporaryDirectory() as tmp_folder:
    unique_name = str(uuid.uuid4())
    # Compile the Poplar executable
    exec_filename = f'application_{unique_name}.popef'
    poplar_exec_filepath = os.path.join(tmp_folder, exec_filename)
    g = ops.Graph()
    with g.as_default(), session_lib.Session(graph=g) as sess:
      compile_op = application_compile_op.experimental_application_compile_op(
          predict_step,
          output_path=poplar_exec_filepath,
          freeze_variables=True)
      if variable_initializer:
        variable_initializer(sess)
      sess.run(compile_op)

    # Create SavedModel with Embedded Runtime
    asset_path_var_str = "asset_path_var"
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
                                              name=asset_path_var_str)
          sess.run(variables.variables_initializer([asset_path_var]))
          g.add_to_collection(ops.GraphKeys.ASSET_FILEPATHS, asset_path)
          final_path = asset_path_var.read_value()
        else:
          final_path = asset_path

        # Execute user provided preprocessing_step function (if it exists).
        if preprocessing_step:
          args = preprocessing_step(*args)
          args = args if isinstance(args, (tuple, list)) else (args,)

          if len(args) != len(predict_step_signature):
            raise ValueError('Length of preprocessing_step output does not '
                             'match the number of TensorSpec in the embedded '
                             'runtime call predict_step signature.')

          for idx, (tensor, tensor_specification) in enumerate(
              zip(args, predict_step_signature)):
            if not isinstance(tensor, Tensor):
              raise ValueError('`preprocessing_step` returned value at '
                               f'position {idx} it is not an instance '
                               'tf.Tensor')
            if tensor.shape != tensor_specification.shape or \
                tensor.dtype != tensor_specification.dtype:
              raise ValueError(
                  '`preprocessing_step` returned Tensor at position '
                  f'{idx} does not match required TensorSpec.\n'
                  f'Tensor shape{str(tensor.shape)}, dtype '
                  f'{str(tensor.dtype)}\n'
                  f'Expected TensorSpec shape '
                  f'{str(tensor_specification.shape)}, dtype '
                  f'{str(tensor_specification.dtype)}\n')

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
      init_op = control_flow_ops.group(
          variables.global_variables_initializer(),
          lookup_ops.tables_initializer())
      sess.run(init_op)

      if preprocessing_step and checkpoint_restore_dir:
        var_list = [
            v for v in ops.get_collection_ref(ops.GraphKeys.GLOBAL_VARIABLES)
            if asset_path_var_str not in v.name
        ]
        saver_op = saver.Saver(var_list=var_list, allow_empty=True)
        saver_op.restore(sess, saver.latest_checkpoint(checkpoint_restore_dir))

        # Create a list of unrestored variables
        unrestored_var_list = [
            v for v in ops.get_collection_ref(ops.GraphKeys.GLOBAL_VARIABLES)
            if v not in var_list
        ]
        # Add an initializer to initialize only the unrestored variables.
        init_op = control_flow_ops.group(
            variables.variables_initializer(unrestored_var_list),
            lookup_ops.tables_initializer())

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


def _loop_wrapper(predict_step, input_signature, input_dataset, iterations):
  infeed, outfeed = _create_feeds(input_signature, input_dataset)

  # Add outfeed queue
  def predict_step_with_outfeed(*args):
    output_enqueue = outfeed.enqueue(predict_step(*args))
    return output_enqueue

  # Wrap in a loop
  def predict_loop():
    r = loops.repeat(iterations, predict_step_with_outfeed, [], infeed)
    return r

  return predict_loop


def export_single_step(predict_step,
                       export_dir,
                       iterations,
                       predict_step_signature=None,
                       input_dataset=None,
                       variable_initializer=None,
                       output_names=None,
                       preprocessing_step=None,
                       preprocessing_step_signature=None,
                       purge_export_dir=False,
                       checkpoint_restore_dir=None):
  """Create a SavedModel in `export_dir` for TensorFlow Serving.

  Wrap `predict_step` inside a while loop, add an infeed for the inputs and
  an outfeed for the outputs, freeze any variables into constants and write
  a SavedModel containing an IPU runtime function and Poplar executable.

  Args:
    predict_step (Callable or tf.function): Function to compile for the IPU
      platform and export.
    export_dir (str): Path to the directory where the SavedModel will be
      written.
    iterations (int): Number of loop iterations.
    predict_step_signature (list or tuple, optional): A sequence of
      `tf.TensorSpec` objects that describe the input arguments of the
      `predict_step` function. If `preprocessing_step` is not provided and
      `input_dataset` is provided, this argument should be None.
      If `preprocessing_step` is provided or `preprocessing_step` and
      `input_dataset` are not provided and `predict_step` is a `tf.function`
      and `input_signature` was specified during `tf.function` creation then
      this argument can be None and the signature will be captured directly
      from `predict_step`.
    input_dataset (tf.Dataset', optional): Dataset from which SavedModel's
      `input_signature` will be inferred.
      If `preprocessing_step` is not provided and `predict_step_signature` is
      provided,this argument should be None.
      If `preprocessing_step` and `preprocessing_step_signature` are provided
      this argument should be None.
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

    output_names (str or list, optional): Output name or list of names
      for the outputs in the SavedModel's SignatureDef. If not provided,
      outputs will be named: `output_0`, `output_1` and so on.
    preprocessing_step (Callable or tf.function, optional): Function that runs
      preprocessing step on the CPU device. Function is called just before
      `predict_step`. `preprocessing_step` and `predict_step` are exported
      together. `preprocessing_step` output will be directly passed to the
      `predict_step` input queue.
    preprocessing_step_signature (list or tuple, optional): A sequence of
      `tf.TensorSpec` objects that describe the input arguments of the
      `preprocessing_step` function. If `preprocessing_step` and `input_dataset`
      are provided, this argument should be None.
      If `preprocessing_step` is provided and `input_dataset` is not provided
      and `preprocessing_step` is a `tf.function` and `input_signature` was
      specified during `tf.function` creation then this argument can be None and
      the signature will be captured directly from `preprocessing_step`.
    checkpoint_restore_dir (str): Path to saved checkpoint, for which the model
      Variables are to be restored.

  Returns:
    function: A reference to the same predict function that was exported using
    the SavedModel format. This function uses the embedded runtime op to run
    the executable that was included in the SavedModel's `assets` subfolder.

  Raises:
    ValueError: If both `input_signature` and `input_dataset` are provided.
    ValueError: If `export_dir` is not an empty directory.
    TypeError: If `input_dataset` was provided and is not an instance of
      `tf.Dataset`
  """
  _validate_export_dir(export_dir, purge_export_dir)
  _validate_signatures(predict_step, predict_step_signature, input_dataset,
                       preprocessing_step, preprocessing_step_signature)
  if preprocessing_step and checkpoint_restore_dir:
    _validate_dir(checkpoint_restore_dir, "checkpoint_restore_dir")

  if preprocessing_step is not None:
    input_signature = _prepare_input_signature(preprocessing_step,
                                               preprocessing_step_signature,
                                               input_dataset)
    predict_step_signature = _prepare_input_signature(predict_step,
                                                      predict_step_signature)
  else:
    input_signature = _prepare_input_signature(predict_step,
                                               predict_step_signature,
                                               input_dataset)
    predict_step_signature = input_signature

  predict_loop = _loop_wrapper(predict_step, predict_step_signature,
                               input_dataset, iterations)
  return _export_saved_model(predict_loop, export_dir, variable_initializer,
                             input_signature, output_names,
                             predict_step_signature, preprocessing_step,
                             checkpoint_restore_dir)


def export_pipeline(computational_stages,
                    export_dir,
                    iterations,
                    inputs=None,
                    device_mapping=None,
                    pipeline_schedule=None,
                    poplar_options=None,
                    name=None,
                    predict_step_signature=None,
                    input_dataset=None,
                    variable_initializer=None,
                    output_names=None,
                    preprocessing_step=None,
                    preprocessing_step_signature=None,
                    purge_export_dir=False,
                    checkpoint_restore_dir=None):
  """Create a pipelined SavedModel in `export_dir` for TensorFlow Serving.

  Create a pipeline op using `computational_stages`, add an infeed for
  the inputs and an outfeed for the outputs, freeze any variables into constants
  and write a SavedModel containing an IPU runtime function (preceded by an
  optional preprocessing step) and Poplar executable.

  SavedModel flow:
  predict_step = computational_stages[0]
  `preprocessing_step` (optional, CPU) -> predict_step (IPU) -> result

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
    predict_step_signature (list or tuple, optional): A sequence of
      `tf.TensorSpec` objects that describe the input arguments of the first
      computational stage.
      If `preprocessing_step` is not provided and `input_dataset` is provided,
      this argument should be None.
      If `preprocessing_step` is provided or `preprocessing_step` and
      `input_dataset` are not provided and first computational stage is a
      `tf.function` and `input_signature` was specified during `tf.function`
      creation then this argument can be None and the signature will be captured
      directly from the first computational stage.
    input_dataset (tf.Dataset, optional): Dataset from which SavedModel
      `input_signature` will be inferred.
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
      for the outputs in the SavedModel's SignatureDef. If not provided,
      outputs will be named: `output_0`, `output_1`, ... `output_n`.
    preprocessing_step (Callable or tf.function, optional): Function that runs
      the preprocessing step on the CPU. Function is called just before
      the first computational stage. `preprocessing_step` and compiled pipelined
      computational stages are exported together. `preprocessing_step` output
      will be directly passed to the input queue of the first computational
      stage.
    preprocessing_step_signature (list or tuple, optional): A sequence of
      `tf.TensorSpec` objects that describe the input arguments of the
      `preprocessing_step` function.
      If `preprocessing_step` and `input_dataset` are provided, this argument
      should be None.
      If `preprocessing_step` is provided and `input_dataset` is not provided
      and `preprocessing_step` is a `tf.function` and `input_signature` was
      specified during `tf.function` creation then this argument can be None and
      the signature will be captured directly from `preprocessing_step`.
    checkpoint_restore_dir (str): Path to saved checkpoint, where the model
      Variables are to be restored. To be used with preprocessing only.

  Returns:
    function: A reference to the same predict function that was exported using
    the SavedModel format. This function uses the embedded runtime op to run
    the executable that was included in the SavedModel's `assets` subfolder.

  Raises:
    ValueError: If both `input_signature` and `input_dataset` are provided.
    ValueError: If `export_dir` is not an empty directory.
    TypeError: If `input_dataset` was provided and is not an instance of
      `tf.Dataset`.
  """
  _validate_export_dir(export_dir, purge_export_dir)

  predict_step = computational_stages[0]

  _validate_signatures(predict_step, predict_step_signature, input_dataset,
                       preprocessing_step, preprocessing_step_signature)
  if preprocessing_step and checkpoint_restore_dir:
    _validate_dir(checkpoint_restore_dir, "checkpoint_restore_dir")

  if preprocessing_step is not None:
    input_signature = _prepare_input_signature(preprocessing_step,
                                               preprocessing_step_signature,
                                               input_dataset,
                                               non_feed_inputs=inputs)
    predict_step_signature = _prepare_input_signature(predict_step,
                                                      predict_step_signature,
                                                      non_feed_inputs=inputs)
  else:
    input_signature = _prepare_input_signature(predict_step,
                                               predict_step_signature,
                                               input_dataset,
                                               non_feed_inputs=inputs)
    predict_step_signature = input_signature

  infeed, outfeed = _create_feeds(predict_step_signature, input_dataset)

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
                             input_signature, output_names,
                             predict_step_signature, preprocessing_step,
                             checkpoint_restore_dir)
