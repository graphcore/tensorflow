Distributed training
--------------------

Distributed training is supported for IPU Pod systems. Here, the IPUs in a
rack are interconnected by IPU-Links, and IPUs in different racks are interconnected
by GW-Links. Distributed training uses these links to perform
collective operations without host involvement. When using multiple
instances (host processes), there may however still be a need for
communication over the host network, for example for broadcasting the
initial values of variables from the first instance to the others.

To perform distributed training on Pod systems, use
:class:`~tensorflow.python.ipu.distributed.popdist_strategy.PopDistStrategy`,
which performs data-parallel synchronous training using multiple host processes.
In this sense it is similar to
`MultiWorkerMirroredStrategy <https://www.tensorflow.org/api_docs/python/tf/distribute/MultiWorkerMirroredStrategy>`_
provided in standard TensorFlow. Initial values are broadcast over the host
network using Horovod.

Collective operations (explicitly through a member function like ``reduce()`` or
implicitly by using an optimizer under the strategy scope) will be performed
directly on the IPU by using compiled communications with the GCL library
over the IPU-Links and GW-Links. The
``PopDistStrategy`` is designed for use with PopDist and PopRun.
Refer to the `PopDist and PopRun User Guide
<https://docs.graphcore.ai/projects/poprun-user-guide/>`_ for more details.

A distinction should be made between the ``PopDistStrategy`` and
the ``IPUStrategy`` provided in TensorFlow 2. The ``IPUStrategy`` targets
a single system with one or more IPUs attached, whereas ``PopDistStrategy``
targets distributed Pod systems.
Note that the use of ``ipu_compiler.compile()`` is still required to ensure a single
XLA graph is compiled, except when using ``IPUEstimator`` or ``IPUPipelineEstimator``
which already use it internally.

PopDistStrategy examples
########################

There are examples for `PopDistStrategy` in the Graphcore :tutorials-repo:`feature examples on GitHub <feature_examples/tensorflow/popdist>`.
