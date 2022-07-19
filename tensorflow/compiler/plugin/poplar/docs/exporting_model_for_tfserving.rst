Exporting precompiled models for TensorFlow Serving
---------------------------------------------------
TensorFlow applications compiled for the IPU can be exported to the standard TensorFlow SavedModel format
and deployed to a TensorFlow Serving instance. The exported SavedModel contains the executable compiled for the IPU,
and a TensorFlow graph with :ref:`embedded_application_runtime` operations which allow you to run the executable as
part of the TensorFlow graph. The exported graph may contain optional preprocessing and postprocessing parts that are executed on the CPU.

The Graphcore TensorFlow API for exporting models for TensorFlow Serving supports three different use cases:

1. Models defined inside a function without using pipelining can be exported using the :py:func:`tensorflow.python.ipu.serving.export_single_step` function.
2. Pipelined models defined as a list of functions can be exported using the :py:func:`tensorflow.python.ipu.serving.export_pipeline` function.
3. Keras models can be exported using :py:func:`tensorflow.python.ipu.serving.export_keras` or the model's :py:func:`export_for_ipu_serving` method. Both ways are functionally identical and support both pipelined and non-pipelined models.

General notes about using the Graphcore TensorFlow API for exporting models with TensorFlow Serving:

1. Since the exported SavedModel contains custom :ref:`embedded_application_runtime` operations, it can be used only with the Graphcore distribution of TensorFlow Serving.
2. The exported SavedModel cannot be loaded back into a TensorFlow script and used as a regular model because, in the export stage, the model is compiled into an IPU executable.
   The exported TensorFlow graph contains only :ref:`embedded_application_runtime` operations and has no information about specific layers, and so on.
3. TensorFlow and TensorFlow Serving versions must always match. It means that you have to use the same version of TensorFlow Serving as the version of TensorFlow that was used to export the model.
   Moreover, Poplar versions must also match between the TensorFlow Serving instance and the version of TensorFlow that was used to export the model.

.. _exporting-non-pipelined-model:

Exporting non-pipelined models defined inside a function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Exporting the forward pass of a non-pipelined model can be done with the
:py:func:`tensorflow.python.ipu.serving.export_single_step` function.
A function that defines the forward pass of the model is required as a first argument.
Under the hood, :py:func:`export_single_step` wraps that function into a while loop optimized for the IPU,
with the `iterations` parameter denoting the number of loop iterations.
You can use this parameter to tweak the model's latency; its optimal value is use-case specific.
Additionally, the function adds the infeed and outfeed queues, so you do not have to take care of it.
Then the model is compiled into an executable and included as an asset in the SavedModel
stored at the `export_dir` location.
The :py:func:`export_single_step` function adds the possibility of passing the `preprocessing_step` and `postprocessing_step`
functions which will be included into the SavedModel graph and executed on the CPU on the server side. If all preprocessing
and postprocessing operations are available on the IPU, `preprocessing_step` and `postprocessing_step` functions should
be called inside the `predict_step` function. Then function bodies will be compiled together with the inference model.


To export such a model, the `predict_step` function's input signature has to be defined. This can be accomplished in one of three ways:

* You can decorate the function with the `@tf.function` decorator which takes the `input_signature` argument;
* You can pass the `predict_step` function signature (`predict_step_signature`) directly to the :py:func:`tensorflow.python.ipu.serving.export_single_step` function;
* You can pass the input dataset (`input_dataset`) to the :py:func:`tensorflow.python.ipu.serving.export_single_step` function and the exported model's input signature will be inferred from it.

All of the above methods are functionally equivalent and can be used interchangeably based on what you find more convenient.

Example of exporting non-pipelined model defined inside a function
__________________________________________________________________

This example exports a very simple model with embedded IPU program that doubles the input tensor.

.. literalinclude:: exporting_model_example.py
  :language: python
  :linenos:

Example of exporting non-pipelined model defined inside a function with additional preprocessing and postprocessing steps
_________________________________________________________________________________________________________________________

This example exports a very simple model with an embedded IPU program, which doubles the input tensor. The model also
performs a preprocessing step (on the IPU) to compute the absolute value of the input and a postprocessing step
(on the IPU) to reduce the output.

.. literalinclude:: exporting_model_preprocessing_postprocessing_example.py
  :language: python
  :linenos:

This example exports a very simple model with an embedded IPU program, which doubles the input tensor. The model also
performs a preprocessing step (on the CPU) to convert string tensors to floats and a postprocessing step (on the CPU) to
compute the absolute value of the outputs.

.. literalinclude:: exporting_model_preprocessing_postprocessing_cpu_example.py
  :language: python
  :linenos:

.. _exporting-pipelined-model:

Exporting pipelined models defined as a list of functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Exporting the forward pass of a pipelined models can be accomplished using :py:func:`tensorflow.python.ipu.serving.export_pipeline` function.

The use of that function is very similar to the creation of a pipeline op
using the :py:func:`tensorflow.python.ipu.pipelining_ops.pipeline` function.
You have to provide a list of functions that represent the pipeline's computational stages.

Function :py:func:`tensorflow.python.ipu.serving.export_pipeline` also has an `iteration` argument.
It denotes the number of times each pipeline stage is executed before the pipeline is restarted.
Again, you can use it to tweak the model's latency. This argument is sometimes called `steps_per_execution`, especially for Keras models.

Similarly to :ref:`exporting non-pipelined models<exporting-non-pipelined-model>`,
to export a pipelined model the signature of the first computational stage has to be known.
You can do this in the same three ways as non-pipelined models.
It's worth noting that for the first option---passing the input signature to the
`@tf.function` decorator---you only need to do that for the first computational stage.


Pipeline example
________________

This example exports a simple pipelined IPU program that performs `2x+3` function on the input.

.. literalinclude:: exporting_pipelined_model_example.py
  :language: python
  :linenos:

Pipeline example with preprocessing and postprocessing steps
____________________________________________________________

This example exports a simple pipelined IPU program that computes the function ``2x+3`` on the input. The model
includes a preprocessing computational stage which computes the absolute value of the input and an IPU postprocessing
step to reduce the output.

.. literalinclude:: exporting_pipelined_model_preprocessing_postprocessing_example.py
  :language: python
  :linenos:


This example exports a simple pipelined IPU program that computes the function ``2x+3`` on the input tensor. The model
also performs a preprocessing step (on the CPU) to convert string tensors to floats and a postprocessing step
(on the CPU) to compute the absolute value of the outputs.

.. literalinclude:: exporting_pipelined_model_preprocessing_postprocessing_cpu_example.py
  :language: python
  :linenos:



Exporting Keras models
~~~~~~~~~~~~~~~~~~~~~~
There are two ways of exporting Keras models for TensorFlow Serving, independent of whether they're pipelined or not.
Keras models can be exported using the :py:func:`tensorflow.python.ipu.serving.export_keras` function or the model's :py:func:`export_for_ipu_serving` method.

See the :numref:`keras-with-ipus` section for details and examples of exporting precompiled Keras models for TensorFlow Serving.


Running the model in TensorFlow Serving
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To test the exported SavedModel you can just start a TensorFlow Serving instance and point it to the model's location.
Graphcore's distribution of TensorFlow Serving can be run directly in the host system:

.. code-block:: console

  $ tensorflow_model_server --rest_api_port=8501 --model_name=my_model \
        --model_base_path="$(pwd)/my_saved_model_ipu"

And then you can start sending inference requests, for example:

.. code-block:: console

  $ curl -d '{"instances": [1.0, 2.0, 5.0, 7.0]}'   \
      -X POST http://localhost:8501/v1/models/my_model:predict

Graphcore does not distribute the TensorFlow Serving API package. If you want to use it, you need to install it from the official distribution using `pip`.
