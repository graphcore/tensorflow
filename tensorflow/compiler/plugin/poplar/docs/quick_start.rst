.. _setup_quick_start:

Setup quick start
------------------

This section describes how to set up your system to start using TensorFlow 1 for the IPU.

.. note:: From Poplar SDK 3.1, TensorFlow 1 will only be supported in CentOS 7. In addition, `Examples <https://github.com/graphcore/examples/tree/v3.0.0>`__ and `Tutorials <https://github.com/graphcore/tutorials/tree/sdk-release-3.0>`__ for TensorFlow 1 are only available up to version 3.0 of the SDK. There has been limited testing of the 3.0 versions of the TensorFlow 1 tutorials and examples with later versions of the Poplar SDK.


Ensure you have completed the steps described in the `getting started guide for your system <https://docs.graphcore.ai/en/latest/getting-started.html>`__ before completing the steps in this section.

Enable Poplar SDK
~~~~~~~~~~~~~~~~~

You need to enable the Poplar SDK before you can use TensorFlow 1.

.. code-block::

    $ source [path-to-sdk]/enable

where ``[path-to-sdk]`` is the path to the Poplar SDK.

You can verify that Poplar has been successfully set up by running:

.. code-block:: console

  $ popc --version

This will display the version of the installed software.

Create and enable a Python virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is recommended that you work in a Python virtual environment. You can create and activate a virtual environment as follows:

.. code-block::

    $ python[py_ver] -m venv ~/[base_dir]/[venv_name]
    $ source ~/[base_dir]/[venv_name]/bin/activate

where ``[base_dir]`` is a location of your choice and ``[venv_name]`` is the name of the directory that will be created for the virtual environment. ``[py_ver]`` is the version of Python you are using and it depends on your OS.

You can get more information about the versions of Python and other tools supported in the Poplar SDK for different operating systems in the :doc:`release-notes:index`.  You can check which OS you are running with ``lsb_release -a``.

Install the TensorFlow 1 wheels and validate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two TensorFlow 1 wheels included in the Poplar SDK, one for AMD processors and one for Intel processors. Check which processor is used on your system by running:

.. code-block:: console

   $ lscpu | grep name

Install the wheel files needed to run TensorFlow 1 on the IPU.

.. tabs::

   .. group-tab:: AMD

        .. code-block::

            $ python -m pip install ${POPLAR_SDK_ENABLED?}/../tensorflow-1.*+amd_*.whl
            $ python -m pip install ${POPLAR_SDK_ENABLED?}/../ipu_tensorflow_addons-1.*.whl
            $ python3 -c "from tensorflow.python import ipu"


   .. group-tab:: Intel

        .. code-block::

            $ python -m pip install ${POPLAR_SDK_ENABLED?}/../tensorflow-1.*+intel_*.whl
            $ python -m pip install ${POPLAR_SDK_ENABLED?}/../ipu_tensorflow_addons-1.*.whl
            $ python3 -c "from tensorflow.python import ipu"
