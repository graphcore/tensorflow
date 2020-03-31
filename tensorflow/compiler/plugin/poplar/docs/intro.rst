Introduction
------------

The purpose of this document is to introduce the TensorFlow framework from the
perspective of developing and training models for the IPU. It assumes you have
some knowledge of TensorFlow and machine learning.

See the "Getting Started" guide for your IPU system on the
`Graphcore support portal <https://support.graphcore.ai>`_
for installation instructions.

To some extent, implementing at the framework level is relatively independent of
the underlying hardware as it relates to the specifics of defining a graph and
its components (for example, how a convolutional layer is defined).

However, there are critical elements of targeting the IPU from TensorFlow that
need to be understood to successfully use it as a training and inference
engine. These include IPU-specific API configurations, model parallelism, error
logging and report generation, as well as strategies for dealing with
out-of-memory (OOM) issues.

Requirements
............

The Graphcore TensorFlow implementation requires Ubuntu 18.04 and Python 3.6.
It will only run on a processor that supports the Intel AVX-512 extension to
the instructions set.
