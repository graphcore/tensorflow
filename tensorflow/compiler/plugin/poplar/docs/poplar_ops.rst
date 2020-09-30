IPU-optimised operations
------------------------

Several custom versions of operators are provided to target functions
available in PopLibs. See the :ref:`api-section` for more details.

LSTM and GRU
~~~~~~~~~~~~

For recurrent neural networks, the custom LSTM
(:py:class:`~tensorflow.python.ipu.rnn_ops.PopnnLSTM`) and GRU
(:py:class:`~tensorflow.python.ipu.rnn_ops.PopnnGRU`) ops need to be used
because they will use significantly less memory on the IPU.

These are also available as :ref:`keras-layers-api`.

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
