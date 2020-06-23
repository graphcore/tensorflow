IPU Host Embeddings
-------------
An IPU host embedding is a mechanism for using large embeddings in both
inference and training, when the embedding would not otherwise fit in device
memory. This is achieved in one of two ways:
# Using a sequence of Poplar callbacks to exchange indices and values.
# Storing the embedding in Poplar remote buffer (off-chip) memory, which may be
   host memory or memory associated with the IPU system.
Applications access this functionality through the
:py:class:`tensorflow.python.ipu.embedding_ops.HostEmbedding` class and the
:py:func:`tensorflow.python.ipu.embedding_ops.create_host_embedding` helper
function. Optimisation of the host embedding is described in the
:py:class:`tensorflow.python.ipu.embedding_ops.HostEmbeddingOptimizerSpec`
class, which currently only supports SGD with a constant learning rate.

Note that IPU host embeddings are not recommended for use in pipelines and will
likely decrease the pipeline's parallel efficiency.

Usage
~~~~~
IPU host embeddings rely on instances of the ``HostEmbedding`` class to
coordinate the communication between the host and device. This object is created
with a call to
:py:func:`tensorflow.python.ipu.embedding_ops.create_host_embedding`. This
created object is then passed to the user model where the
:py:meth:`tensorflow.python.ipu.embedding_ops.HostEmbedding.lookup` method can
be called with a similar API to ``tf.nn.embedding_lookup``.

Once the IPU host embedding has been created and used within the model, the
object must be "executed" with its call operator
(:py:meth:`tensorflow.python.ipu.embedding_ops.HostEmbedding.__call__`) in the
``session.run``. This is so that the callbacks or remote buffers, depending on
the chosen implementation, can be registered with the Poplar engine before
execution. The potentially modified embedding value will be returned from the
session.

Enabling Remote Buffers
-----------------------
In :py:func:`tensorflow.python.ipu.utils.create_ipu_config` there is an option
``enable_experimental_remote_buffer_embedding``. When this option is set to
``True`` (defaults to ``False``), the IPU host embedding implementation will be
globally set to use remote buffers.

Note This options is experimental, and may be changed or removed in future
releases.

Partitioning Strategies
-----------------------
When using IPU host embeddings, the experimental remote buffer implementation,
and replication, it becomes necessary to decide how to partition the embedding.
This is because Poplar remote buffers are replica-unique, so there's no sharing.

To overcome this constraint we offer two partitioning strategies. Each has
tradeoffs which can be chosen depending on the application.

Token Strategy
^^^^^^^^^^^^^^
The token strategy unsurprisingly chooses to partition the embedding on the
token axis. This means that there will be ``ceil(t/r)`` whole tokens on each
replica, where ``t`` is the token count and ``r`` is the replica count.

.. figure:: figures/host_emb_token_strategy.png

Given that each replica's portion of the whole embedding table is private, we
must introduce cross-replica operations to allow lookups and updates on the
whole table. Below is the psuedo-code, with explicit types and static shapes,
for how this is implemented:

.. code-block:: cpp
  // Psuedo-code assuming we have table size `t`, and replica count `r`.
  f16[14, 64] global_lookup(
    local_table : f16[ceil(t/r), 64]
    global_indices : i32[14]
  ):
    // The unique replica ID for "this" replica.
    replica_id = i32[] get-replica-id

    // Distribute the indices to all devices.
    indices = all-gather(indices) : i32[r, 14]

    // Scale the indices down by the replication factor. Indices not meant for
    // this replica will map to a valid, but incorrect index.
    local_indices = indices / r : i32[r, 14]

    // Gather on the local embedding region.
    result = lookup(embedding, indices) : f16[r, 14, 64]

    // The mask of which indices are valid.
    mask = (indices % r) == replica_id : bool[r, 14]

    // Zero out the invalid regions of the result
    result = select(result, 0, mask) : f16[r, 14, 64]

    // Reduce scatter sum the masked result tensor. The zeroed regions of the
    // result tensor ensure that invalid values are ignore and each replica has
    // the correct result.
    result = reduce-scatter-sum(result) : f16[1, 14, 64]

    // Reshape to the expected shape
    return reshape(result), shape=[14, 64] : f16[14, 64]

Encoding Strategy
^^^^^^^^^^^^^^^^^
The encoding strategy, in contrast to the token strategy, chooses to partition
the embedding on the encoding axis. This means that there will be ``ceil(1/r)``
of every tokens on each replica, where ``r`` is the replica count. This means
for a given token every replica will store ``ceil(e/r)`` elements, where ``e``
is the element count for a single token.

.. figure:: figures/host_emb_enc_strategy.png

Similarly to the token strategy, each replica's portion of the whole embedding
table is private, so we must introduce cross-replica operations to allow lookups
and updates on the whole table. Below is the psuedo-code, with explicit types
and static shapes, for how this is implemented:

.. code-block:: cpp
  // Psuedo-code assuming we have table size `t`, replica count `r`, and
  // encoding size `e`.
  f16[14, e] global_lookup(
    local_table : f16[t, ceil(e/r)]
    global_indices : i32[14]
  ):
    // Distribute the indices to all devices
    indices = all-gather(global_indices) : i32[r, 14]

    // Gather on the local embedding
    result = lookup(local_embedding, indices) : f16[r, 14, ceil(e/r)]

    // Communicate the relevant parts of the embedding to their respective
    // replicas. This distributes the ith slice in the outermost dimension to
    // ith replica.
    result = all-to-all(result, slice_dim=2, concat_dim=3) : f16[r, 14, ceil(e/r)]

    // Transpose the dimensions back into the correct order.
    result = transpose(result), permutation=[1, 0, 2] : f16[14, r, ceil(e/r)]

    // Flatten the innermost dimensions
    result = flatten(result), begin=1, end=2 : f16[14, r*ceil(e/r)]

    // Slice off the excess padding on the encoding
    return slice(result), dim=1, begin=0, end=e : f16[14, e]

Summary
^^^^^^^
Although it is application dependant, generally the token strategy is used when
the encoding is much smaller than the token count. An example application for
this would be language models where the vocabulary size is much larger than the
encoding.

The encoding strategy is used when the token count is small, the encoding is
large enough to be split. This avoids a large amount of very small
communication. An example application for this would be game playing models,
where a small numbers of available actions are encoded in an embedding.

Profiling will ultimately decide which splitting strategy works best for any
given application.

Example
~~~~~~~~

.. literalinclude:: host_embedding_example.py
