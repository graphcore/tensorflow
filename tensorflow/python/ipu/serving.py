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

import inspect
import os
import shutil
import tempfile
import uuid

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework.ops import convert_to_tensor
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.ipu import application_compile_op
from tensorflow.python.ipu import embedded_runtime
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import loops
from tensorflow.python.ipu.ops import pipelining_ops
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.training.tracking import tracking
from tensorflow.python.saved_model import save


def _validate_export_dir(export_dir, purge_export_dir):
  """ Validate if `export dir` exists. Remove the content of it
      if `purge_export_dir` is set.
  Args:
    export_dir (str): Path to the directory where the SavedModel will be
      written.
    purge_export_dir (Boolean): If True, before starting the export, the target
      directory is emptied. Otherwise no cleaning is performed and if target dir
      is not empty, the function fails with an error.
  Raises:
    ValueError: If ``export_dir`` is not an empty directory.
  """

  if os.path.isdir(export_dir) and os.listdir(export_dir):
    if purge_export_dir:
      shutil.rmtree(export_dir)
    else:
      raise ValueError(f'Directory "{export_dir}" is not empty. '
                       'Please specify an empty directory.')


def _validate_signatures(predict_step,
                         predict_step_signature=None,
                         input_dataset=None,
                         preprocessing_step=None,
                         preprocessing_step_signature=None,
                         postprocessing_step=None,
                         postprocessing_step_signature=None):
  """Validate if input signatures can be deduced from given arguments for the
     exported model for preprocessing and inference parts.

  Args:
    predict_step (Callable or tf.function): Function that runs inference step.
    predict_step_signature (list or tuple, optional): A sequence of
      `tf.TensorSpec` objects that describe the input arguments of
      `predict_step`. If `predict_step` is a `tf.function` and `input_signature`
      was specified during `tf.function` creation then this argument can be
      None.
    input_dataset (tf.Dataset): Dataset from which exported model's
      input_signature will be inferred.
    preprocessing_step (Callable or tf.function, optional): Function that runs
      the preprocessing step.
    preprocessing_step_signature (list or tuple, optional): A sequence of
      `tf.TensorSpec` objects that describe the input arguments of
      `preprocessing_step`. If `preprocessing_step` is a `tf.function` and
      `input_signature` was specified during `tf.function` creation then this
      argument can be None.
    postprocessing_step (Callable or tf.function, optional): Function that runs
      the postprocessing step.
    postprocessing_step_signature (list or tuple, optional): A sequence of
      `tf.TensorSpec` objects that describe the input arguments of
      `postprocessing_step`. If `postprocessing_step` is a `tf.function` and
      `input_signature` was specified during `tf.function` creation then this
      argument can be None.

  Raises:
    TypeError: If `input_dataset` is not a `tf.Dataset` or `NoneType`.
    TypeError: If `predict_step_signature` is neither a tuple, list of
      `tf.TensorSpec` objects nor a `NoneType`.
    TypeError: If `preprocessing_step_signature` is neither a tuple, list of
      `tf.TensorSpec` objects nor a `NoneType`.
    TypeError: If `postprocessing_step_signature` is neither a tuple, list of
      `tf.TensorSpec` objects nor a `NoneType`.
    ValueError: If `predict_step_signature` is an empty tuple or list.
    ValueError: If `preprocessing_step_signature` is an empty tuple or list.
    ValueError: If `postprocessing_step_signature` is an empty tuple or list.
    ValueError: If `preprocessing_step` is not provided and both
      `predict_step_signature` and `input_dataset` are provided.
    ValueError: If `preprocessing_step`, `predict_step_signature`,
      `input_dataset` are not provided and `predict_step` is not a `tf.function`
      or is a `tf.function` but `input_signature` is not provided.
    ValueError:  If `preprocessing_step`, `preprocessing_step_signature`,
      `input_dataset` are provided.
    ValueError: If `preprocessing_step` is provided and both
      `preprocessing_step_signature`, `input_dataset` are not provided and
      `preprocessing_step` is not a `tf.function` or is a `tf.function` but no
      `input_signature` is provided.
    ValueError: If `preprocessing_step`, `predict_step_signature` are not
      provided and `predict_step` is not a `tf.function` or is a `tf.function`
      but no `input_signature` is provided.
    ValueError: If `postprocessing_step` is provided and
      `postprocessing_step_signature` is not provided and
      `postprocessing_step` is not a `tf.function` or is a `tf.function` but no
      `input_signature` is provided.
  """

  is_predict_step_signature_set = predict_step_signature is not None
  is_input_dataset_set = input_dataset is not None
  is_preprocessing_step_signature_set = preprocessing_step_signature is not None
  is_preprocessing_set = preprocessing_step is not None
  is_postprocessing_set = postprocessing_step is not None
  is_postprocessing_step_signature_set = postprocessing_step_signature \
                                         is not None

  def validate_tf_function(fn, fn_name, deduction_from_datset_possible=True):
    if not isinstance(fn, def_function.Function):
      raise ValueError(
          f'`input_signature` deduction from given `{fn_name}` is not '
          'possible. Please mark it as tf.function with `input_signature` '
          f'parameter set or provide `{fn_name}_signature`' +
          (' or `input_dataset`.' if deduction_from_datset_possible else "."))
    elif fn.input_signature is None:
      raise ValueError(
          'Empty `input_signature` inside provided '
          'tf.function `predict_step`. Please specify it or '
          'provide `predict_step_signature`' +
          (' or `input_dataset`.' if deduction_from_datset_possible else "."))

  def validate_single_signature(signature_name, signature):
    if not isinstance(signature, (tuple, list)):
      raise TypeError(f'`{signature_name}` must be an instance of tuple or '
                      f'list. Received {str(type(signature))}')
    elif not len(signature):
      raise ValueError(f'`{signature_name}` must be not empty.')
    else:
      for idx, value in enumerate(signature):
        if not isinstance(value, tensor_spec.TensorSpec):
          raise TypeError(f'`{signature_name}[{idx}]` is not an instance of '
                          'TensorSpec')

  if not is_preprocessing_set:
    if is_input_dataset_set and is_predict_step_signature_set:
      raise ValueError(
          'Both `predict_step_signature` and `input_dataset` cannot '
          'be provided. Please pass only one of them.')
    elif not is_input_dataset_set and not is_predict_step_signature_set:
      validate_tf_function(predict_step, "predict_step")
  else:
    if is_preprocessing_step_signature_set and is_input_dataset_set:
      raise ValueError(
          'Both `preprocessing_step_signature` and `input_dataset` '
          'cannot be provided. Please pass only one of them.')
    elif not is_predict_step_signature_set and not is_input_dataset_set:
      validate_tf_function(preprocessing_step, "preprocessing_step")

    if not is_predict_step_signature_set:
      validate_tf_function(predict_step,
                           "predict_step",
                           deduction_from_datset_possible=False)

  if is_input_dataset_set and not isinstance(input_dataset,
                                             dataset_ops.Dataset):
    raise TypeError('If `input_dataset` is provided, it should be an '
                    'instance of tf.Dataset.')

  if is_postprocessing_set and not is_postprocessing_step_signature_set:
    validate_tf_function(postprocessing_step,
                         "postprocessing_step",
                         deduction_from_datset_possible=False)

  if is_preprocessing_step_signature_set:
    validate_single_signature('preprocessing_step_signature',
                              preprocessing_step_signature)

  if is_postprocessing_step_signature_set:
    validate_single_signature('postprocessing_step_signature',
                              postprocessing_step_signature)

  if is_predict_step_signature_set:
    validate_single_signature('predict_step_signature', predict_step_signature)


def _prepare_input_signature(defunc,
                             defunc_signature=None,
                             input_dataset=None,
                             non_feed_inputs=None,
                             remove_non_feed_inputs_from_signature=True):
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
    remove_non_feed_inputs_from_signature (bool, optional): If True passed
      non_feed_inputs will be removed from created input signature.

  Returns:
    list: List of `tf.TensorSpec` objects with types, shapes and names.

  Raises:
    ValueError: If not possible to create `input_signature` from given
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
      input_signature = input_signature,
    input_signature = tuple(
        _get_signature_from_tensors(non_feed_inputs)) + input_signature
  elif defunc_signature is None:
    if isinstance(defunc, def_function.Function):
      input_signature = defunc.input_signature

  if input_signature is None:
    raise ValueError(f'Missing input_signature for {defunc.__name__}')

  if not isinstance(input_signature, (tuple, list)):
    raise TypeError('input_signature must be either a tuple or a '
                    f'list, received {str(type(input_signature))}')

  names = list(inspect.signature(defunc).parameters.keys())
  if non_feed_inputs is not None and remove_non_feed_inputs_from_signature:
    names = names[len(non_feed_inputs):]
    if len(input_signature) > len(names):
      input_signature = input_signature[len(non_feed_inputs):]

  if len(input_signature) != len(names):
    raise ValueError(
        'Length of input_signature does not match the number of '
        f'{defunc.__name__} arguments, input_signature : {input_signature}, '
        f'names : {names}')

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


def _get_signature_from_tensors(tensors):
  if tensors is None:
    return []

  def to_tensor(data):
    if not isinstance(data, Tensor):
      return convert_to_tensor(data)

    return data

  return [
      tensor_spec.TensorSpec.from_tensor(to_tensor(tensor))
      for tensor in tensors
  ]


def _freeze_defunc(defunc, input_signature):
  @def_function.function(input_signature=input_signature)
  def defunc_wrapper(*args):
    return defunc(*args)

  concrete_defunc = convert_to_constants.convert_variables_to_constants_v2(
      defunc_wrapper.get_concrete_function(*input_signature))

  @def_function.function(input_signature=input_signature)
  def transformed_defunc_wrapper(*args):
    return concrete_defunc(*args)

  return transformed_defunc_wrapper, _get_signature_from_tensors(
      concrete_defunc.outputs)


def _freeze_single_step(defunc, input_signature):
  return _freeze_defunc(defunc, input_signature)[0]


def _freeze_computational_stages(computational_stages, input_signature):
  def transform(stage):
    nonlocal input_signature
    transformed_stage, output_signature = _freeze_defunc(
        stage, input_signature)
    input_signature = output_signature
    return transformed_stage

  return [transform(stage) for stage in computational_stages]


def _export_saved_model(predict_step,
                        export_dir,
                        input_signature,
                        output_names=None,
                        predict_step_signature=None,
                        preprocessing_step=None,
                        postprocessing_step_signature=None,
                        postprocessing_step=None):
  """Compile Poplar executable and export saved model.

  Args:
    predict_step (Callable or tf.function): Function that runs inference step.
    export_dir (str): Path to the directory where the SavedModel will be
      written.
    input_signature (list): List of signatures of inputs that will be provided
      to the exported model graph. If `preprocessing_step` (optional) is set,
      inputs will be processed by `preprocessing_step` function on the CPU and
      passed further to `predict_step` using infeed queue. Otherwise inputs will
      be provided directly to the `predict_step` infeed queue.
    output_names (str or list, optional): Output name or list of output names
      for the outputs in the SavedModel's SignatureDef. If not provided, outputs
      will be named: ``output_0``, ``output_1`` and so on.
    predict_step_signature (list, optional): List of signatures of inputs that
      will be provided to `predict_step` using infeed queue. If
      `preprocessing_step` is set, `predict_step_signature` is used for
      validation of compatibility between `preprocessing_step` outputs and
      `predict_step` inputs.
    preprocessing_step (Callable or tf.function, optional): Function that runs
      the preprocessing step on the CPU device.
    postprocessing_step_signature (list or tuple, optional): A sequence of
      `tf.TensorSpec` objects that describe the input arguments of
      `postprocessing_step`. If `postprocessing_step` is a `tf.function` and
      `input_signature` was specified during `tf.function` creation then this
      argument can be None.
    postprocessing_step (Callable or tf.function, optional): Function that runs
      the postprocessing step on the CPU device.


  Returns:
    tf.function: A reference to the same predict function that was exported
    using the SavedModel format. This function uses the embedded runtime op to
    run the executable that was included in the SavedModel's `assets` subfolder.

  Raises:
    TypeError: If ``output_names`` is neither a string nor a list.
    ValueError: If `preprocessing_step` is set and `preprocessing_step` outputs
      are incompatible with `predict_step` inputs - different shapes,
      data types, `predict_step` inputs length does not match the number of the
      `preprocessing_step` outputs.
    ValueError: If length of ``output_names`` does not match the number of
      results returned by ``predict_step``.
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
    application_compile_op.experimental_application_compile_op(
        predict_step, output_path=poplar_exec_filepath, freeze_variables=True)
    with_preprocessing = preprocessing_step is not None
    with_postprocessing = postprocessing_step is not None

    if with_preprocessing:
      preprocessing_step = _freeze_single_step(preprocessing_step,
                                               input_signature)

    if with_postprocessing:
      postprocessing_step = _freeze_single_step(postprocessing_step,
                                                postprocessing_step_signature)

    def validate_io_matching(src_return_tensors, dst_input_signature,
                             src_step_name, dst_step_name):
      if len(src_return_tensors) != len(dst_input_signature):
        raise ValueError(f'The number of the `{src_step_name}` outputs does '
                         'not match the number of `tf.TensorSpec` '
                         f'objects in the signature of `{dst_step_name}`.')

      for idx, (tensor, tensor_spec) in enumerate(
          zip(src_return_tensors, dst_input_signature)):
        if not isinstance(tensor, Tensor):
          raise ValueError(f'`{src_step_name}` returned value at '
                           f'position {idx} it is not an instance of '
                           'tf.Tensor')
        if tensor.shape != tensor_spec.shape or \
          tensor.dtype != tensor_spec.dtype:
          raise ValueError(f'`{src_step_name}` returned Tensor at '
                           f'postion {idx} does not match required '
                           f'`{dst_step_name}` TensorSpec.\n'
                           f'Tensor shape{str(tensor.shape)}, dtype '
                           f'{str(tensor.dtype)}\n'
                           f'Expected TensorSpec shape '
                           f'{str(tensor_spec.shape)}, dtype '
                           f'{str(tensor_spec.dtype)}\n')

    class EmbeddedModel(module.Module):
      def __init__(self, filepath):
        super(EmbeddedModel, self).__init__()
        self.filename = tracking.Asset(filepath)
        self.engine_name = f'engine_{unique_name}'
        self.predict = self.predict_wrapper()

      def predict_wrapper(self):
        @def_function.function(input_signature=input_signature)
        def predict(*args):
          asset_path = self.filename.asset_path
          ctx = embedded_runtime.embedded_runtime_start(
              asset_path, [], self.engine_name)

          if with_preprocessing:
            args = preprocessing_step(*args)
            args = args if isinstance(args, (tuple, list)) else (args,)

            if predict_step_signature is not None:
              validate_io_matching(args, predict_step_signature,
                                   "preprocessing_step", "predict_step")

          ret = embedded_runtime.embedded_runtime_call(args, ctx)

          if with_postprocessing:
            ret = ret if isinstance(ret, (tuple, list)) else (ret,)
            if postprocessing_step_signature is not None:
              # application_compile_op always returns tensors with
              # shape=<unknown>
              ret = [
                  array_ops.ensure_shape(tensor, tensor_spec.shape)
                  for tensor, tensor_spec in zip(
                      ret, postprocessing_step_signature)
                  if postprocessing_step_signature is not None
              ]

              validate_io_matching(ret, postprocessing_step_signature,
                                   "predict_step",
                                   "postprocessing_step_signature")
            ret = postprocessing_step(*ret)

          if output_names:
            if len(ret) != len(output_names):
              raise ValueError(
                  'Length of output_names does not match the number of results '
                  'returned by the predict function.')
            else:
              return dict(zip(output_names, ret))
          else:
            return ret

        return predict

    model_to_export = EmbeddedModel(poplar_exec_filepath)
    save.save(model_to_export, export_dir)

  # Adjust the executable filepath to the proper location in the SavedModel
  model_to_export.filename = tracking.Asset(
      os.path.join(export_dir, "assets", exec_filename))
  return model_to_export.predict


def _wrap_in_loop(predict_step, input_signature, input_dataset, iterations):
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
                       output_names=None,
                       preprocessing_step=None,
                       preprocessing_step_signature=None,
                       postprocessing_step=None,
                       postprocessing_step_signature=None,
                       purge_export_dir=False):
  """Create a SavedModel in `export_dir` for TensorFlow Serving.

  Wrap `predict_step` inside a while loop, add an infeed for the inputs and
  an outfeed for the outputs, freeze any variables into constants and write
  a SavedModel containing a compiled IPU runtime function (preceded by
  optional preprocessing step) and Poplar executable.

  SavedModel flow:
  `preprocessing_step` (optional, CPU) -> `predict_step` (IPU) ->
  `postprocessing_step` (optional, CPU) -> result

  Args:
    predict_step (Callable or tf.function): Function to compile into the IPU
      platform and export.
    export_dir (str): Path to the directory where the SavedModel will be
      written.
    iterations (int): Number of loop iterations.
    predict_step_signature (list or tuple, optional): A sequence of
      `tf.TensorSpec` objects that describe the input arguments of the
      `predict_step` function.
      If `preprocessing_step` is not provided and `input_dataset` is provided,
      this argument should be None.
      If `preprocessing_step` is provided or `preprocessing_step` and
      `input_dataset`are not provided and `predict_step` is a `tf.function`
      and `input_signature` was specified during `tf.function` creation then
      this argument can be None and the signature will be captured directly from
      `predict_step`.
    input_dataset (tf.Dataset, optional): Dataset from which SavedModel
      `input_signature` will be inferred.
      If `preprocessing_step` is not provided and `predict_step_signature` is
      provided,this argument should be None.
      If `preprocessing_step` and `preprocessing_step_signature` are provided
      this argument should be None.
    output_names (str or list, optional): Output name or list of output names
      for the outputs in the SavedModel's SignatureDef. If not provided, outputs
      will be named: ``output_0``, ``output_1`` and so on.
    preprocessing_step (Callable or tf.function, optional): Function that runs
      the preprocessing step on the CPU device. Function is called just before
      `predict_step`. `preprocessing_step` and `predict_step` are exported
      together.
      `preprocessing_step` output is directly passed to the `predict_step`
      input queue.
    preprocessing_step_signature (list or tuple, optional): A sequence of
      `tf.TensorSpec` objects that describe the input arguments of the
      `preprocessing_step` function.
      If `preprocessing_step` and `input_dataset` are provided, this argument
      should be None.
      If `preprocessing_step` is provided and `input_dataset` is not provided
      and `preprocessing_step` is a `tf.function` and `input_signature` was
      specified during `tf.function` creation then this argument can be None and
      the signature will be captured directly from `preprocessing_step`.
    postprocessing_step (Callable or tf.function, optional): Function that runs
      the postprocessing step on the CPU. Function is called after
      `predict_step`. `postprocessing_step` and `predict_step` are exported
      together.
      Tensors from the `predict_step` output queue are `postprocessing_step`
      inputs.
    postprocessing_step_signature (list or tuple, optional): A sequence of
      `tf.TensorSpec` objects that describe the input arguments of the
      `postprocessing_step` function.
      If `postprocessing_step` is a `tf.function` and `input_signature` was
      specified during `tf.function` creation then this argument can be None and
      the signature will be captured directly from `postprocessing_step`.
    purge_export_dir (Boolean, optional): If True, before starting the export,
      the target directory is emptied. Otherwise no cleaning is performed and if
      target dir is not empty, the function fails with an error.

  Returns:
    tf.function: A reference to the same predict function that was exported
      using the SavedModel format. This function uses the embedded runtime op to
      run the executable that was included in the SavedModel's `assets`
      subfolder.

  Raises:
    ValueError: If ``export_dir`` is not an empty directory.
    TypeError: If `input_dataset` is not a `tf.Dataset` or `NoneType`.
    TypeError: If `predict_step_signature` is neither a tuple, list of
      `tf.TensorSpec` objects nor a `NoneType`.
    TypeError: If `preprocessing_step_signature` is neither a tuple, list of
      `tf.TensorSpec` objects nor a `NoneType`.
    TypeError: If `postprocessing_step_signature` is neither a tuple, list of
      `tf.TensorSpec` objects nor a `NoneType`.
    ValueError: If `predict_step_signature` is an empty tuple or list.
    ValueError: If `preprocessing_step_signature` is an empty tuple or list.
    ValueError: If `postprocessing_step_signature` is an empty tuple or list.
    ValueError: If `preprocessing_step` is not provided and both
      `predict_step_signature` and `input_dataset` are provided.
    ValueError: If `preprocessing_step`, `predict_step_signature`,
      `input_dataset` are not provided and `predict_step` is not a `tf.function`
      or is a `tf.function` with not provided `input_signature`.
    ValueError:  If `preprocessing_step`, `preprocessing_step_signature`,
      `input_dataset` are provided.
    ValueError: If `preprocessing_step` is provided and both
      `preprocessing_step_signature`, `input_dataset` are not provided and
      `preprocessing_step` is not a `tf.function` or is a `tf.function` but no
      `input_signature` is provided.
    ValueError: If `preprocessing_step`, `predict_step_signature` are not
      provided and `predict_step` is not a `tf.function` or is a `tf.function`
      but no `input_signature` is provided.
    ValueError: If `postprocessing_step` is provided and
      `postprocessing_step_signature` is not provided and
      `postprocessing_step` is not a `tf.function` or is a `tf.function` but no
      `input_signature` is provided.
  """

  _validate_export_dir(export_dir, purge_export_dir)
  _validate_signatures(predict_step, predict_step_signature, input_dataset,
                       preprocessing_step, preprocessing_step_signature,
                       postprocessing_step, postprocessing_step_signature)

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

  if postprocessing_step is not None:
    postprocessing_step_signature = _prepare_input_signature(
        postprocessing_step, postprocessing_step_signature)

  predict_step = _freeze_single_step(predict_step, predict_step_signature)
  predict_loop = _wrap_in_loop(predict_step, predict_step_signature,
                               input_dataset, iterations)
  return _export_saved_model(predict_loop, export_dir, input_signature,
                             output_names, predict_step_signature,
                             preprocessing_step, postprocessing_step_signature,
                             postprocessing_step)


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
                    output_names=None,
                    preprocessing_step=None,
                    preprocessing_step_signature=None,
                    postprocessing_step=None,
                    postprocessing_step_signature=None,
                    purge_export_dir=False):
  """Create a pipelined SavedModel in `export_dir` for TensorFlow Serving.

  Create a pipeline op using `computational_stages`, add an infeed for
  the inputs and an outfeed for the outputs, freeze any variables into constants
  and write a SavedModel containing an IPU runtime function (preceded by
  optional preprocessing step) and Poplar executable.

  SavedModel flow:
  predict_step = computational_stages[0]
  `preprocessing_step` (optional, CPU) -> predict_step (IPU) ->
  `postprocessing_step` (optional, CPU) -> result

  Args:
    computational_stages (list): A list of Python functions or TensorFlow
      functions, where each function represents a computational stage in the
      pipeline. The function takes the outputs of the previous pipeline stage as
      its inputs.
    export_dir (str): Path to the directory where the SavedModel will be
      written.
    iterations (int): The number of times each computational stage
      will be executed during the execution of the pipeline. It can also be
      considered as the pipeline depth.
    inputs (list, optional): Arguments passed to the first computational stage
      without usage of infeed queue.
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
    input_dataset (tf.Dataset, optional): Dataset from which SavedModel's
      `input_signature` will be inferred.
    output_names (str or list, optional): Output name or list of output names
      for the outputs in the SavedModel's SignatureDef. If not provided, outputs
      will be named: ``output_0``, ``output_1`` and so on.
    preprocessing_step (Callable or tf.function, optional): Function that runs
      preprocessing step on the CPU device. Function is called just before
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
    postprocessing_step (Callable or tf.function, optional): Function that runs
      the postprocessing step on the CPU. Function is called after
      `predict_step`. `postprocessing_step` and `predict_step` are exported
      together.
      Tensors from the `predict_step` output queue are `postprocessing_step`
      inputs.
    postprocessing_step_signature (list or tuple, optional): A sequence of
      `tf.TensorSpec` objects that describe the input arguments of the
      `postprocessing_step` function.
      If `postprocessing_step` is a `tf.function` and `input_signature` was
      specified during `tf.function` creation then this argument can be None and
      the signature will be captured directly from `postprocessing_step`.
    purge_export_dir (Boolean, optional): If True, before starting the export,
      the target directory is emptied. Otherwise no cleaning is performed and if
      target dir is not empty, the function fails with an error.

  Returns:
    tf.function: A reference to the same predict function that was exported
      using the SavedModel format. This function uses the embedded runtime op to
      run the executable that was included in the SavedModel's `assets`
      subfolder.

  Raises:
    ValueError: If ``export_dir`` is not an empty directory.
    TypeError: If `input_dataset` is not a `tf.Dataset` or `NoneType`.
    TypeError: If `predict_step_signature` is neither a tuple, list of
      `tf.TensorSpec` objects nor a `NoneType`.
    TypeError: If `preprocessing_step_signature` is neither a tuple, list of
      `tf.TensorSpec` objects nor a `NoneType`.
    TypeError: If `postprocessing_step_signature` is neither a tuple, list of
      `tf.TensorSpec` objects nor a `NoneType`.
    ValueError: If `predict_step_signature` is an empty tuple or list.
    ValueError: If `preprocessing_step_signature` is an empty tuple or list.
    ValueError: If `postprocessing_step_signature` is an empty tuple or list.
    ValueError: If `preprocessing_step` is not provided and both
      `predict_step_signature` and `input_dataset` are provided.
    ValueError: If `preprocessing_step`, `predict_step_signature`,
      `input_dataset` are not provided and `predict_step` is not a `tf.function`
      or is a `tf.function` with not provided `input_signature`.
    ValueError:  If `preprocessing_step`, `preprocessing_step_signature`,
      `input_dataset` are provided.
    ValueError: If `preprocessing_step` is provided and both
      `preprocessing_step_signature`, `input_dataset` are not provided and
      `preprocessing_step` is not a `tf.function` or is a `tf.function` but no
      `input_signature` is provided.
    ValueError: If `preprocessing_step`, `predict_step_signature` are not
      provided and `predict_step` is not a `tf.function` or is a `tf.function`
      but no `input_signature` is provided.
    ValueError: If `postprocessing_step` is provided and
      `postprocessing_step_signature` is not provided and
      `postprocessing_step` is not a `tf.function` or is a `tf.function` but no
      `input_signature` is provided.
  """
  _validate_export_dir(export_dir, purge_export_dir)

  predict_step = computational_stages[0]

  _validate_signatures(predict_step, predict_step_signature, input_dataset,
                       preprocessing_step, preprocessing_step_signature,
                       postprocessing_step, postprocessing_step_signature)

  computational_stages = _freeze_computational_stages(
      computational_stages,
      _prepare_input_signature(predict_step,
                               predict_step_signature,
                               input_dataset,
                               non_feed_inputs=inputs,
                               remove_non_feed_inputs_from_signature=False))
  if preprocessing_step is not None:
    input_signature = _prepare_input_signature(preprocessing_step,
                                               preprocessing_step_signature,
                                               input_dataset)
    predict_step_signature = _prepare_input_signature(predict_step,
                                                      predict_step_signature,
                                                      non_feed_inputs=inputs)
  else:
    input_signature = _prepare_input_signature(predict_step,
                                               predict_step_signature,
                                               input_dataset,
                                               non_feed_inputs=inputs)
    predict_step_signature = input_signature

  if postprocessing_step is not None:
    postprocessing_step_signature = _prepare_input_signature(
        postprocessing_step, postprocessing_step_signature)

  infeed, outfeed = _create_feeds(predict_step_signature, input_dataset)

  @def_function.function
  def defunc():
    pipelining_ops.pipeline(
        computational_stages=computational_stages,
        gradient_accumulation_count=iterations,
        inputs=inputs,
        infeed_queue=infeed,
        outfeed_queue=outfeed,
        device_mapping=device_mapping,
        pipeline_schedule=pipeline_schedule,
        forward_propagation_stages_poplar_options=poplar_options,
        name=name)

  return _export_saved_model(defunc, export_dir, input_signature, output_names,
                             predict_step_signature, preprocessing_step,
                             postprocessing_step_signature,
                             postprocessing_step)


def export_keras(model,
                 export_dir,
                 batch_size=None,
                 output_names=None,
                 preprocessing_step=None,
                 preprocessing_step_signature=None,
                 postprocessing_step=None,
                 postprocessing_step_signature=None,
                 purge_export_dir=False):
  """Export Keras model using the SavedModel format for TensorFlow serving.

  Wrap model's ``call`` function inside a ``while`` loop, add an infeed for the
  inputs and an outfeed for the outputs, convert any variables into constants
  and write a SavedModel containing an IPU runtime function and Poplar
  executable.

  Args:
    model (tf.keras.Model): The Keras model to export.
    export_dir (str): The path to the directory where the SavedModel will be
      written.
    batch_size (int, optional): The batch size value to be used in the exported
      model. If not specified and the model was built with a specified batch
      size (different than None), the exported model will use the currently set
      batch size. This argument must be specified if the model's batch size is
      `None`.
    output_names (str or list, optional): Output name or list of output names
      for the outputs in the SavedModel's SignatureDef. If not provided, outputs
      will be named: ``output_0``, ``output_1`` and so on.
     preprocessing_step (Callable or tf.function, optional): Function that runs
      the preprocessing step on the CPU device. This function is called just
      before the Keras model. `preprocessing_step` and the Keras model are
      exported together.
      The `preprocessing_step` output is passed directly to the Keras modelel
      input queue.
    preprocessing_step_signature (list or tuple, optional): A sequence of
      `tf.TensorSpec` objects that describe the input arguments of the
      `preprocessing_step` function.
      If `preprocessing_step` is a `tf.function` and `input_signature` was
      specified during `tf.function` creation then this argument can be None
      and the signature will be captured directly from `preprocessing_step`.
    postprocessing_step (Callable or tf.function, optional): Function that
      runs the postprocessing step on the CPU. This function is called after
      the Keras model. `postprocessing_step` and the Keras model are exported
      together.
      Tensors from the Keras model output queue are inputs to
      `postprocessing_step`.
    postprocessing_step_signature (list or tuple, optional): A sequence of
      `tf.TensorSpec` objects that describe the input arguments of the
      `postprocessing_step` function.
      If `postprocessing_step` is a `tf.function` and `input_signature` was
      specified during `tf.function` creation then this argument can be None
      and the signature will be captured directly from `postprocessing_step`.
    purge_export_dir (Boolean, optional): If True, before starting the export,
      the target directory is emptied. Otherwise no cleaning is performed and
      if the target directory is not empty, the function fails with an error.
  Returns:
    tf.function: A reference to the same predict function that was exported
      using the SavedModel format. This function uses the embedded runtime op
      to run the executable that was included in the SavedModel's ``assets``
      subfolder.

  Raises:
    ValueError: If `model` does not have the `export_for_ipu_serving` method.
    ValueError: If `export_dir` is not an empty directory and
      `purge_export_dir` is not set to True.
    TypeError: If `preprocessing_step_signature` is neither a tuple, a list of
      `tf.TensorSpec` objects nor a `NoneType`.
    TypeError: If `postprocessing_step_signature` is neither a tuple, a list of
      `tf.TensorSpec` objects nor a `NoneType`.
    ValueError: If `preprocessing_step_signature` is an empty tuple or a list.
    ValueError: If `postprocessing_step_signature` is an empty tuple or a list.
    ValueError: If `preprocessing_step` is provided and
      `preprocessing_step_signature` is not provided and `preprocessing_step`
      is not a `tf.function` or is a `tf.function` but no `input_signature` is
      provided.
    ValueError: If `postprocessing_step` is provided and
      `postprocessing_step_signature` is not provided and
      `postprocessing_step` is not a `tf.function` or is a `tf.function` but
      no `input_signature` is provided.
  """

  if not hasattr(model, 'export_for_ipu_serving'):
    raise ValueError(
        "Provided model was not created inside an IPU strategy, so it "
        "does not contain IPU-specific functions. Please wrap its "
        "creation inside an IPU strategy.")

  return model.export_for_ipu_serving(export_dir, batch_size, output_names,
                                      preprocessing_step,
                                      preprocessing_step_signature,
                                      postprocessing_step,
                                      postprocessing_step_signature,
                                      purge_export_dir)
