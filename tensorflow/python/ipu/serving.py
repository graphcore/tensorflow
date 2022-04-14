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
import tempfile
import uuid

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import tensor_spec
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


def _validate_signature(defunc,
                        input_signature,
                        input_dataset,
                        non_feed_inputs=None):
  """Validate and update `input_signature` if necessary to match the arguments
  of `defunc`.

  Args:
    defunc (Callable or tf.function): Function whose signature is analyzed.
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


def _export_saved_model(defunc, export_dir, input_signature):
  """Compile Poplar executable and export saved model.

  Args:
    defunc (Callable or tf.function): Function that runs inference step.
    export_dir (str): Path to the directory where the SavedModel will be
      written.
    input_signature (list): List of signatures of inputs that will be provided
      to the graph using infeed queue.

  Returns:
    tf.function: A reference to the same predict function that was exported
    using the SavedModel format. This function uses the embedded runtime op to
    run the executable that was included in the SavedModel's `assets` subfolder.
  """
  with tempfile.TemporaryDirectory() as tmp_folder:
    unique_name = str(uuid.uuid4())
    # Compile poplar_exec
    exec_filename = f'application_{unique_name}.poplar_exec'
    poplar_exec_filepath = os.path.join(tmp_folder, exec_filename)
    application_compile_op.experimental_application_compile_op(
        defunc, output_path=poplar_exec_filepath, freeze_variables=True)

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
          ret = embedded_runtime.embedded_runtime_call(args, ctx)
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
                       input_signature=None,
                       input_dataset=None):
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

  Returns:
    tf.function: A reference to the same predict function that was exported
    using the SavedModel format. This function uses the embedded runtime op to
    run the executable that was included in the SavedModel's `assets` subfolder.

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
                     'provided to export_single_step. Please pass only '
                     'one of them.')

  if input_dataset is not None and not isinstance(input_dataset,
                                                  dataset_ops.Dataset):
    raise TypeError('If input_dataset is provided, it should be an instance '
                    'of tf.Dataset.')

  input_signature = _validate_signature(predict_step, input_signature,
                                        input_dataset)
  predict_loop = _wrap_in_loop(predict_step, input_signature, input_dataset,
                               iterations)
  return _export_saved_model(predict_loop, export_dir, input_signature)


def export_pipeline(computational_stages,
                    export_dir,
                    pipeline_depth,
                    iterations,
                    inputs=None,
                    device_mapping=None,
                    pipeline_schedule=None,
                    poplar_options=None,
                    name=None,
                    input_signature=None,
                    input_dataset=None):
  """Create a pipelined SavedModel in `export_dir` for TensorFlow Serving.

  Create a pipeline op using `computational_stages`, add an infeed for
  the inputs and an outfeed for the outputs, freeze any variables into constants
  and write a SavedModel containing an IPU runtime function and Poplar
  executable.

  Args:
    computational_stages (list): A list of Python functions or TensorFlow
      functions, where each function represents a computational stage in the
      pipeline. The function takes the outputs of the previous pipeline stage as
      its inputs.
    export_dir (str): Path to the directory where the SavedModel will be
      written.
    pipeline_depth (int): The number of times each computational stage
      will be executed. It should be a multiple of the number of computational
      stages.
    iterations (int): The number of times the pipeline will be executed.
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

  Returns:
    tf.function: A reference to the same predict function that was exported
    using the SavedModel format. This function uses the embedded runtime op to
    run the executable that was included in the SavedModel's `assets` subfolder.

  Raises:
    ValueError: If both `input_signature` and `input_dataset` are provided.
    ValueError: If `pipeline_depth` is not a multiple of the number of
      computational stages.
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

  if pipeline_depth % len(computational_stages) != 0:
    raise ValueError(f'pipeline_depth ({pipeline_depth}) should be a multiple '
                     f'of the number of computational stages '
                     f'({len(computational_stages)}).')

  first_stage = computational_stages[0]
  input_signature = _validate_signature(first_stage, input_signature,
                                        input_dataset, inputs)
  infeed, outfeed = _create_feeds(input_signature, input_dataset)

  @def_function.function
  def defunc():
    pipelining_ops.pipeline(
        computational_stages=computational_stages,
        gradient_accumulation_count=pipeline_depth,
        repeat_count=iterations,
        inputs=inputs,
        infeed_queue=infeed,
        outfeed_queue=outfeed,
        device_mapping=device_mapping,
        pipeline_schedule=pipeline_schedule,
        forward_propagation_stages_poplar_options=poplar_options,
        name=name)

  return _export_saved_model(defunc, export_dir, input_signature)


def export_keras(model, export_dir):
  """Export Keras model using the SavedModel format for TensorFlow serving.

  Wrap model's ``call`` function inside a ``while`` loop, add an infeed for the
  inputs and an outfeed for the outputs, convert any variables into constants
  and write a SavedModel containing an IPU runtime function and Poplar
  executable.

  Args:
    model (tf.keras.Model): The Keras model to export.
    export_dir (str): The path to the directory where the SavedModel will be
      written.

  Returns:
    tf.function: A reference to the same predict function that was exported
    using the SavedModel format. This function uses the embedded runtime op to
    run the executable that was included in the SavedModel's ``assets``
    subfolder.

  Raises:
    ValueError: If model was not created inside IPU strategy.
    ValueError: If ``export_dir`` is not an empty directory.
  """
  if os.path.isdir(export_dir) and os.listdir(export_dir):
    raise ValueError(
        "Directory is not empty. Please specify an empty directory.")

  if not hasattr(model, 'export_for_ipu_serving'):
    raise ValueError(
        "Provided model was not created inside an IPU strategy, so it "
        "does not contain IPU-specific functions. Please wrap its "
        "creation inside an IPU strategy.")
  return model.export_for_ipu_serving(export_dir)
