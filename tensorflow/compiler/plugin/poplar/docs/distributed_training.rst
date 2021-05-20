Distributed training
--------------------

This example shows how to use the ``IPUEstimator`` with the
``IPUMultiWorkerStrategyV1`` to perform distributed training of
a model on the MNIST dataset.

The example is based on the following official tutorial
with some modifications for use with the IPU:
https://www.tensorflow.org/tutorials/distribute/multi_worker_with_estimator

We highlight the changes needed to convert code using ``IPUEstimator``
to support distributed training below.

The input function
##################

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
##################

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
##################

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
################

.. literalinclude:: distributed_training_example.py
  :language: python
  :linenos:

Distributed training with Horovod
#################################

Distributed training can also be performed using
`Horovod <https://github.com/horovod/horovod/>`_ which is included in the
TensorFlow wheel provided by Graphcore.

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

Horovod Open MPI dependency
###########################

Horovod depends on Open MPI being installed on the system. The Open MPI library is
dynamically loaded when the module ``tensorflow.python.ipu.horovod`` is
imported, and this will fail if Open MPI is not installed. It is recommended to
install the Open MPI version that is provided by your operating system package manager.

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
