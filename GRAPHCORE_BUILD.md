# Build TensorFlow from source for the IPU

Building TensorFlow for the IPU is very similar to the process for building
TensorFlow described here: <https://www.tensorflow.org/install/source>.

Building TensorFlow for the IPU is supported for Ubuntu 18.04, CentOS 7.6 and CentOS 8.3.

The differences from the standard build process are:

1. Install the Poplar SDK
2. Use the Graphcore TensorFlow source repository


## Install the Poplar SDK

In addition to the dependencies documented for TensorFlow, you will also need to
install the Poplar SDK. The Poplar SDK can be downloaded from:
<https://downloads.graphcore.ai/>.

Set the following environment variable to point to the Poplar installation in
the Poplar SDK:

``` {.shell}
$ export TF_POPLAR_BASE=/path/to/poplar_sdk/poplar-ubuntu_18_04-x.x.x
```

TensorFlow must be built against a compatible version of the SDK. For example,
the "r2.1/sdk-release-1.3" branch of TensorFlow 2 for the IPU must be built
against Poplar SDK 1.3.


## Download the TensorFlow source

Instead of downloading the TensorFlow source from the TensorFlow repository,
clone Graphcore's TensorFlow GitHub repository:

``` {.shell}
$ git clone https://github.com/graphcore/tensorflow.git
$ cd tensorflow
```

## Build

You can now build TensorFlow by following the instructions here:
<https://www.tensorflow.org/install/source>.


### Build options

The default compilation option `-march=native` optimizes the generated code for
your machine's CPU type. However, if building TensorFlow for a different CPU
type, you can use a more specific optimization option. For example, the
Graphcore TensorFlow packages distributed in the Poplar SDK are built with:
* `-march=skylake-avx512` for the Intel package,
* `-march=znver1` for the AMD package.
See the GCC manual for more information.

### Building against custom PopLibs

If you have built PopLibs from source and want your build of TensorFlow to use
that, you can set the following environment variable:

``` {.shell}
$ export TF_POPLIBS_BASE=/path/to/poplibs/
```
