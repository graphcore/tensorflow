Using IPU optimized operations
------------------------------

Several custom versions of operators are provided to target operators
available in Poplibs.  See the :ref:`api-section` for more details.

LSTM
~~~~

See :py:class:`tensorflow.python.ipu.ops.rnn_ops.PopnnLSTM`.

Dropout
~~~~~~~

The Poplibs version of dropout does not need to store the dropout mask
between the forward and backward parts of the graph, saving memory.

See :py:func:`tensorflow.python.ipu.ops.rand_ops.dropout`.

Embedding lookup
~~~~~~~~~~~~~~~~

This is a version of embedding lookup which will produce a smaller memory
footprint for small lookups. Instead of using dynamic lookup into the main
embedding dictionary, it uses a one hot operator and a multiply.

See :py:func:`tensorflow.python.ipu.embedding_ops.embedding_lookup`.

Group normalization
~~~~~~~~~~~~~~~~~~~

Group normalization is an alternative to batch normalization, and produces
smaller and more optimized graphs.

The original paper on group normalization:
`"Group Normalization", Yuxin Wu, Kaiming He <https://arxiv.org/abs/1803.08494>`_.

See :py:func:`tensorflow.python.ipu.normalization_ops.group_norm`.

Instance normalization
~~~~~~~~~~~~~~~~~~~~~~

Instance normalization is another alternative to batch normalization.

The original paper on instance normalization:
`"Instance Normalization: The Missing Ingredient for Fast Stylization"
Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky
<https://arxiv.org/abs/1607.08022>`_.

See :py:func:`tensorflow.python.ipu.normalization_ops.group_norm`.

Layer normalization
~~~~~~~~~~~~~~~~~~~

Layer normalization is another alternative to batch normalization.

The original paper on layer normalization:
`"Layer Normalization" Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
<https://arxiv.org/abs/1607.06450>`_.

See :py:func:`tensorflow.python.ipu.normalization_ops.layer_norm`.

