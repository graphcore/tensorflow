.. _setup_quick_start:

Setup quick start
------------------

.. note:: There are two ways you can run TensorFlow 1 applications:

    #. Directly on the system using the setup described in this chapter.
    #. In a TensorFlow 1 Docker container that has already been setup. Refer to :doc:`poplar-docker:index` for more information.

This section describes how to set up your system to start using TensorFlow 1 for the IPU.

Ensure you have completed the steps described in the `getting started guide for your system <https://docs.graphcore.ai/en/latest/getting-started.html>`__ before completing the steps in this section.

The setup for TensorFlow 1 depends on whether your system is running :ref:`ubuntu-18-04` or :ref:`ubuntu-20-04`.

You can check which OS you are running with:

.. code-block:: console

  $ lsb_release -a

.. _ubuntu-18-04:

Ubuntu 18.04
------------

.. _sec_quick_enable_sdk:

Enable Poplar SDK
~~~~~~~~~~~~~~~~~~

.. code-block::

    $ source [path-to-sdk]/enable
    $ popc --version

where ``[path-to-sdk]`` is the path to the Poplar SDK.

Create and enable a Python virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block::

    $ virtualenv -p python[py_ver] ~/[base_dir]/[venv_name]
    $ source ~/[base_dir]/[venv_name]/bin/activate

where ``[base_dir]`` is a location of your choice and ``[venv_name]`` is the name of the directory that will be created for the virtual environment. ``[py_ver]`` is the version of Python you are using and it depends on your OS.

On Ubuntu 18 systems we support Python 3.6, and on Ubuntu 20 systems we support Python 3.8. You can get more information about the versions of tools supported in the Poplar SDK for different operating systems in the :doc:`release-notes:index`.  You can check which OS you are running with ``lsb_release -a``.

Install the TensorFlow 1 wheels and validate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tabs::

   .. group-tab:: AMD

        .. code-block::

            $ python -m pip install ${POPLAR_SDK_ENABLED?}/../tensorflow-1.*+amd_*.whl
            $ python -m pip install ${POPLAR_SDK_ENABLED?}/../ipu_tensorflow_addons-1.*.whl
            $ python3 -c "from tensorflow.python import ipu"
            $ python3 -c "ipu_tensorflow_addons.keras import layers"

   .. group-tab:: Intel

        .. code-block::

            $ python -m pip install ${POPLAR_SDK_ENABLED?}/../tensorflow-1.*+intel_*.whl
            $ python -m pip install ${POPLAR_SDK_ENABLED?}/../ipu_tensorflow_addons-1.*.whl
            $ python3 -c "from tensorflow.python import ipu"
            $ python3 -c "ipu_tensorflow_addons.keras import layers"


.. _ubuntu-20-04:

Ubuntu 20.04
------------

Ubuntu 20.04 does not natively support TensorFlow 1.
This means that you need to run TensorFlow 1 applications in an Ubuntu 18.04 Docker container. Refer to :doc:`poplar-docker:index` for more information.

The following commands provide an example of how to pull the latest TensorFlow 1 image from Docker Hub, and then instantiate the container(:numref:`code-tf1-docker`):

.. code-block:: console
    :name: code-tf1-docker
    :caption: Creating a TensorFlow 1 Docker container

    $ docker pull graphcore/tensorflow:1-intel
    $ gc-docker -- -ti -v /home/ubuntu/graphcore:/graphcore -e IPUOF_VIPU_API_HOST -e IPUOF_VIPU_API_PARTITION_ID graphcore/tensorflow:1-intel
