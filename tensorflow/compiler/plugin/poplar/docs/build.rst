
How to build TensorFlow from source
-----------------------------------
Build tool prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~

    To build TensorFlow you will need to install bazel, pip, curl, and various Python packages.

    - Curl

      .. code-block:: shell

        sudo apt install curl

    - Bazel requires Java (depending on the version of linux, you may need the Oracle Java, not the openjdk one)

      .. code-block:: shell

        sudo apt install openjdk-8-jdk

    - Bazel is the build tool - use version 0.24.1 or later versions

      .. code-block:: shell

        export BAZEL_VERSION=0.24.1
        wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
        chmod +x bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
        ./bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh --user

      This will install it in ``~/bin`` which you can add to your ``PATH`` if it isn't already.

    - Python:

      .. code-block:: shell

        sudo apt install python-numpy python-dev python-pip python-wheel virtualenv

    - libc-ares-dev (maybe): If you get errors about missing ``ares.h``, try this

      .. code-block:: shell

        sudo apt install libc-ares-dev

    You will also need a Poplar installation. The Poplar SDK can be downloaded from: https://downloads.graphcore.ai/

Build instructions
~~~~~~~~~~~~~~~~~~

    Create your workspace

    .. code-block:: console

        mkdir -p tf_build/install
        cd tf_build

    Git clone the repositories of interest

    .. code-block:: console

        git clone https://placeholder@github.com/graphcore/tensorflow_packaging.git
        git clone https://placeholder@github.com/graphcore/tensorflow.git
    Check bazel version (make sure it is >= 0.21.0)

    .. code-block:: console

        bazel version

    To build against a release, set an environment variable to point to the base of a built Poplar installation

    .. code-block:: console

        export TF_POPLAR_BASE=/path/to/poplar_sdk/poplar-ubuntu_18_04-x.x.x

    (Optional) To build against a custom built PopLibs version which is not the one shipped with the SDK, set an environment variable to point to the base of a built PopLibs installation

    .. code-block:: console

        export TF_POPLIBS_BASE=/path/to/poplibs/

    To set up the Python build environment and configure TensorFlow

    .. code-block:: console

        bash tensorflow_packaging/configure python3

    Using the pip wheel package generator as the final target, build TensorFlow

    .. code-block:: console

        bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package

    Once the build has completed, make the pip wheel using the package generator

    .. code-block:: console

        ./bazel-bin/tensorflow/tools/pip_package/build_pip_package ../install

    To run the suit of unit tests

    .. code-block:: console

        bash ../tensorflow_packaging/run_ci_tests.sh

    Adding  ``--test_env TF_CPP_MIN_VLOG_LEVEL=1`` to the command line will dump out more debug information, including the work done by the XLA driver turning the XLA graph into a Poplar graph.

    To repeat a test multiple times, add ``--runs_per_test N``.

    To ensure a test is run, even when it ran successfully and is cached, add ``--no_cache_test_results``.
