Introduction
------------

The purpose of this document is to introduce the TensorFlow framework from the
perspective of developing and training models for the IPU. It assumes you have
some knowledge of machine learning and TensorFlow.

See the "Getting Started" guide for your IPU system in the
`Graphcore documentation <https://docs.graphcore.ai>`_
for information on installing the Poplar SDK and TensorFlow.

TensorFlow is a powerful graph-modelling framework that can be used for the
development, training and deployment of deep learning models. In the Graphcore
software stack, TensorFlow sits at the highest level of abstraction. Poplar
and PopLibs provide a software interface to operations running on the IPU.

.. figure:: figures/Tensorflow_Poplar.png
    :width: 100%
    :alt: TensorFlow abstraction
    :align: center

    TensorFlow abstraction in relation to Poplar and the IPU

There are two main differences in the Graphcore implementation of TensorFlow:

* Some machine-learning ops are optimised for the IPU
  hardware. For example, our custom dropout op is designed to use less memory
  by not storing the dropout mask between forward and backward passes.
* It provides extra IPU-specific functions, such as those for selecting and
  configuring IPUs.

Implementing programs at the framework level is relatively independent of
the underlying hardware. The process of defining a graph and
its components (for example, how a convolutional layer is defined) is largely the
same when targeting the IPU.

There are a small number of changes that you need to make to your TensorFlow
code, when constructing and executing a model, in order to target the IPU
efficiently. These include IPU-specific API configurations, model parallelism,
error logging and report generation, as well as strategies for dealing with
out-of-memory (OOM) issues.

These will be described in the appropriate sections of this document.

Document overview
~~~~~~~~~~~~~~~~~

* The first section provides information about :any:`selecting specific IPU
  <device_selection>` hardware and configuring the code to run on it.
* After this, there is an overview of the support for TensorFlow 2 and Keras.
* The following sections provide information on various methods related to
  training a model on the IPU.
* The next few sections provide information on IPU-specific features.
* Finally, there are reference chapters describing the API and supported operators.

You can find further information on porting a TensorFlow program to the IPU and
parallelising it, in our `TensorFlow technical notes
<https://docs.graphcore.ai/en/latest/#tensorflow>`_.