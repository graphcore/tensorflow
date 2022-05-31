Distributed training
--------------------

We support distributed training for two different types of systems:

1. Pod systems: An IPU-Machine-based system where IPUs in a rack are
   interconnected by IPU-Links, and IPUs in different racks are interconnected
   by GW-Links. Distributed training on Pod systems use these links to perform
   collective operations without host involvement. When using multiple
   instances (host processes), there may however still be a need for
   communication over the host network, for example for broadcasting the
   initial values of variables from the first instance to the others.

2. IPU-Server systems: A Mk1 PCIe card-based system with IPUs
   interconnected by IPU-Links. IPUs in distinct IPU-Servers are not directly
   interconnected. Distributed training on IPU-Servers therefore uses the host
   network for communication. A collective operation is typically performed in
   a hierarchical fashion where the IPU-Links are used first for intra-server
   communication, and then the host network is used for inter-server
   communication.

We provide distribution strategies that are designed for these two types of
systems, and with different implementations of the host communication:

* :class:`~tensorflow.python.ipu.horovod.popdist_strategy.PopDistStrategy`
* :class:`~tensorflow.python.ipu.horovod.IPUHorovodStrategy`
* :class:`~tensorflow.python.ipu.ipu_multi_worker_strategy.IPUMultiWorkerStrategy`

Their main differences can be summarized like this:

+-----------------------------+------------+--------------------+
| Distribution strategy       | System     | Host communication |
+=============================+============+====================+
| ``PopDistStrategy``         | Pod        | Horovod (OpenMPI)  |
+-----------------------------+------------+--------------------+
| ``IPUHorovodStrategy``      | IPU-Server | Horovod (OpenMPI)  |
+-----------------------------+------------+--------------------+
| ``IPUMultiWorkerStrategy``  | IPU-Server | gRPC               |
+-----------------------------+------------+--------------------+

There are some things they have in common:

* They all perform data-parallel synchronous training using multiple host processes.
  In this sense they are all similar to the
  `MultiWorkerMirroredStrategy <https://www.tensorflow.org/api_docs/python/tf/distribute/MultiWorkerMirroredStrategy>`_
  provided in standard TensorFlow.
* They all broadcast the initial values of variables over the host
  network (using either Horovod or gRPC as described above).

And these are the main differences:

* With the ``PopDistStrategy`` designed for Pod systems, a
  collective operation (performed either explicitly by calling a member function
  like ``reduce()`` or implicitly by using an optimizer under the strategy
  scope) will be performed directly on the IPU by using compiled communications
  with the GCL library over the IPU-Links and GW-Links. The
  ``PopDistStrategy`` is designed for use with PopDist and PopRun.
  Refer to the `PopDist and PopRun User Guide
  <https://docs.graphcore.ai/projects/poprun-user-guide/>`_ for more details.

* With the two distribution strategies designed for IPU-Server systems, an
  equivalent collective operation will involve a transfer of the tensor from
  the IPU to the host for performing the collective communication over the host
  network (using either Horovod with OpenMPI or gRPC). A local (cross-replica)
  collective operation can be performed by using the
  :mod:`~tensorflow.python.ipu.cross_replica_ops`.

A distinction should be made between these distribution strategies and
the ``IPUStrategy`` provided in TensorFlow 2. The ``IPUStrategy`` targets
a single system with one or more IPUs attached, while the distribution
strategies we discuss here target distributed systems like those described
above (Pod systems or multiple IPU-Servers). Also, unlike the ``IPUStrategy``,
these distribution strategies do not currently support the Keras
``Model.fit()`` family of APIs, and the use of ``ipu_compiler.compile()``
is still required to ensure a single XLA graph is compiled, except when
using ``IPUEstimator`` or ``IPUPipelineEstimator`` which already use it
internally.

Example using IPUMultiWorkerStrategy
####################################

This example shows how to use the ``IPUEstimator`` with the
``IPUMultiWorkerStrategyV1`` to perform distributed training of
a model on the MNIST dataset.

The example is based on the  official `Multi-worker training with Estimator
<https://www.tensorflow.org/tutorials/distribute/multi_worker_with_estimator>`__ tutorial
with some modifications for use with the IPU.

We highlight the changes needed to convert code using ``IPUEstimator``
to support distributed training below.

The input function
******************

In multi-worker training, it is necessary to shard the dataset such
that each worker processes distinct portions of the dataset.

When used in a distributed context, the input function is passed an
additional argument ``input_context`` that can be used to get the
current worker index and the total number of workers. We pass this
information to the ``Dataset.shard()`` function to perform the
sharding.

Note that the batch size provided by the input function is the
per-worker batch size. The global batch size will be this
multiplied by the number of workers.

The model function
******************

The optimiser will automatically divide the loss by the number of workers,
so in the model function we should only divide the loss by the local
batch size.

We will do some changes to how we update the weights of the model.
Instead of using the high-level ``Optimizer.minimize()`` function,
we will use the ``Optimizer.compute_gradients()`` and
``Optimizer.apply_gradients()`` separately in order to control
their placement. The ``Optimizer.compute_gradients()`` call (the
backward pass) is placed on the IPU, while the
``Optimizer.apply_gradients()`` call (the allreduce of gradients and
weight updates) is placed on the host. This is done by using the
``host_call`` parameter in ``IPUEstimatorSpec``.

In practice this means that the gradients will be streamed from the
IPU to the host as soon as they are computed. The workers will
then start reducing the gradients amongst themselves, allowing overlap
between the backward pass on the IPUs with the reductions on the hosts.
After a gradient is reduced across the workers, the corresponding
weight update is also done on the host.

The reduction is done using a ring-based collectives implementation
with gRPC as the cross-host communication layer.

One benefit of this approach is that any additional optimiser
state (such as momentum) is only needed in host memory, so there
is no additional IPU memory consumption when using stateful
optimisers with this approach.

Cluster definition
******************

We use the ``TFConfigClusterResolver`` which reads the ``TF_CONFIG``
environment variable to determine the cluster definition.

There are two components of ``TF_CONFIG``: ``cluster`` and ``task``.

* ``cluster`` provides information about the entire cluster, namely
  the workers and parameter servers in the cluster.

* ``task`` provides information about the current task.

In this example, the task ``type`` is ``worker`` and the task ``index`` is 0.
You could run this example with two workers on the same machine
(in different terminals) like this:

.. code-block:: bash

    $ TF_CONFIG='{"cluster":{"worker":["localhost:3737","localhost:3738"]},"task":{"type":"worker","index":0}}' python distributed_training_example.py
    $ TF_CONFIG='{"cluster":{"worker":["localhost:3737","localhost:3738"]},"task":{"type":"worker","index":1}}' python distributed_training_example.py

Complete example
****************

.. literalinclude:: distributed_training_example.py
  :language: python
  :linenos:

Download :download:`distributed_training_example.py`

Distributed training with Horovod
#################################

Distributed training can also be performed using
`Horovod <https://github.com/horovod/horovod/>`_ which is included in the
TensorFlow wheel provided by Graphcore. Please refer to the section on installing Horovod for TensorFlow in the `PopDist and PopRun User Guide <https://docs.graphcore.ai/projects/poprun-user-guide/html/configuration.html#tensorflow-1-and-tensorflow-2>`__ for more details.

The class
:class:`~tensorflow.python.ipu.horovod.ipu_horovod_strategy.IPUHorovodStrategyV1`
can be used in the same manner as the
:class:`~tensorflow.python.ipu.ipu_multi_worker_strategy.IPUMultiWorkerStrategyV1`.

While the ``IPUMultiWorkerStrategyV1`` uses collective operations over gRPC, the
``IPUHorovodStrategyV1`` uses the collective operations provided by Horovod, based on
MPI. Horovod also has built-in cluster discovery, so there is no cluster resolver
argument that must be provided like there is for the ``IPUMultiWorkerStrategyV1``,
and there is no need for starting a ``tf.distribute.Server``.

Apart from these differences, the API and semantics should be the same for the
``IPUHorovodStrategyV1`` and ``IPUMultiWorkerStrategyV1``. In other words, they
both provide data parallel distributed training that keeps the variables in sync
on the different workers. During variable initialisation the values are broadcast
from the root rank to the other ranks, and during training the gradients are
all-reduced as a part of the ``Optimizer.apply_gradients`` call.

Launching Horovod training
##########################

The ``mpirun`` tool can be used to run the distributed training across a cluster.
For instance, running distributed training across two processes on the same machine
can be done with the following command:

.. code-block:: bash

    $ mpirun -np 2 -H localhost:2 python distributed_training_horovod_example.py

Complete Horovod example
########################

Below is a complete example using Horovod, adapted from the example above.

.. literalinclude:: distributed_training_horovod_example.py
  :language: python
  :linenos:

Download :download:`distributed_training_horovod_example.py`
