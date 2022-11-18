IPU-optimised operations
------------------------

Several custom versions of operators are provided to target functions
available in PopLibs. See the :ref:`api-section` for more details.

Image operations
~~~~~~~~~~~~~~~~

Our architecture is well-suited to efficiently handle convolutions over four-channel tensors, however it is common
for images to be represented with three channels.
In order to obtain better IPU performance, both from a latency and memory standpoint, we advise that when
dealing with three-channel inputs, you pad the fourth channel dimension.

See :py:func:`tensorflow.python.ipu.image_ops.normalise_image`
for the op that can perform this padding, in addition to normalising and casting if needed. Note that this padding will be
performed on-device, after the data has been transferred to the IPU.

An example of its use can be found in the ``fused_normalise_image()`` function in the `CNN training application
example <https://github.com/graphcore/examples/blob/v3.0.0/vision/cnns/tensorflow1/training/Datasets/imagenet_preprocessing.py>`_
in Graphcore's examples repository on GitHub.

.. note:: This example is for TensorFlow 1.
    From Poplar SDK 3.1, TensorFlow 1 will only be supported in CentOS 7. In addition, `Examples <https://github.com/graphcore/examples/tree/v3.0.0>`__ and `Tutorials <https://github.com/graphcore/tutorials/tree/sdk-release-3.0>`__ for TensorFlow 1 are only available up to version 3.0 of the SDK. There has been limited testing of the 3.0 versions of the TensorFlow 1 tutorials and examples with Poplar SDK 3.1.


Matmul serialisation
~~~~~~~~~~~~~~~~~~~~

You have the option to serialise matrix multiplications along a particular dimension, in order to reduce
the code size of the multiplication and the temporary memory requirements of the matmul, at the expense of extra computation.

See :py:func:`tensorflow.python.ipu.math_ops.serialized_matmul` for details of the op.

An example of its use can be found in the ``mlm_head()`` function in the `BERT application example <https://github.com/graphcore/examples/blob/v3.0.0/nlp/bert/tensorflow1/modeling.py>`_
in Graphcore's examples repository on GitHub.

.. note:: This example is for TensorFlow 1.
    From Poplar SDK 3.1, TensorFlow 1 will only be supported in CentOS 7. In addition, `Examples <https://github.com/graphcore/examples/tree/v3.0.0>`__ and `Tutorials <https://github.com/graphcore/tutorials/tree/sdk-release-3.0>`__ for TensorFlow 1 are only available up to version 3.0 of the SDK. There has been limited testing of the 3.0 versions of the TensorFlow 1 tutorials and examples with Poplar SDK 3.1.


Dropout
~~~~~~~

The PopLibs version of dropout does not need to store the dropout mask
between the forward and backward parts of the graph, saving memory.

See :py:func:`tensorflow.python.ipu.rand_ops.dropout`.

Embedding lookup
~~~~~~~~~~~~~~~~

This is a version of embedding lookup that has been optimised for the IPU.
It allows the embedding lookup to be serialised into smaller lookups, which can
reduce the maximum memory at the cost of extra computation when the embedding
tensors are used by multiple operations.

See :py:func:`tensorflow.python.ipu.embedding_ops.embedding_lookup`.

Group normalisation
~~~~~~~~~~~~~~~~~~~

Group normalisation is an alternative to batch normalisation, and produces
smaller and more optimised graphs.

The original paper on group normalisation is
`"Group Normalization", Yuxin Wu, Kaiming He <https://arxiv.org/abs/1803.08494>`_.

See :py:func:`tensorflow.python.ipu.normalization_ops.group_norm`.

Instance normalisation
~~~~~~~~~~~~~~~~~~~~~~

Instance normalisation is another alternative to batch normalisation.

The original paper on instance normalisation is
`"Instance Normalization: The Missing Ingredient for Fast Stylization"
Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky
<https://arxiv.org/abs/1607.08022>`_.

See :py:func:`tensorflow.python.ipu.normalization_ops.instance_norm`.

Layer normalisation
~~~~~~~~~~~~~~~~~~~

Layer normalisation is another alternative to batch normalisation.

The original paper on layer normalisation is
`"Layer Normalization" Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
<https://arxiv.org/abs/1607.06450>`_.

See :py:func:`tensorflow.python.ipu.normalization_ops.layer_norm`.

GeLU activation
~~~~~~~~~~~~~~~

Gaussian error linear units (GeLU) is an alternative to the ReLU non-linearity.
This is described in `"Gaussian Error Linear Units (GELUs)" Dan Hendrycks, Kevin
Gimpel <https://arxiv.org/abs/1606.08415>`_.

See :py:func:`tensorflow.python.ipu.nn_ops.gelu`.

Sequence slice
~~~~~~~~~~~~~~

The set of sequence slicing ops provided for the IPU.

See :py:func:`tensorflow.python.ipu.slicing_ops.sequence_slice`,
:py:func:`tensorflow.python.ipu.slicing_ops.sequence_slice_unpack` and
:py:func:`tensorflow.python.ipu.slicing_ops.sequence_slice_pack`.

Histogram
~~~~~~~~~~~~~~

The set of histogram ops provided for the IPU.

See :py:func:`tensorflow.python.ipu.statistics_ops.histogram`,
:py:func:`tensorflow.python.ipu.statistics_ops.histogram_update`,
:py:func:`tensorflow.python.ipu.statistics_ops.fixed_width_bins` and
:py:func:`tensorflow.python.ipu.statistics_ops.histogram_normalize`.
