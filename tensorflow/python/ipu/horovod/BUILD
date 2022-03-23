"""
Horovod support.

We build Horovod as a shared library that depends on MPI, and ship this library
as a part of the TensorFlow wheel file. This Horovod library conforms to the
TensorFlow plugin library conventions, and registers its Horovod ops and kernels
at load-time. It also includes some C functions for basic functionality
(init/shutdown/rank/size etc.) that are called from Python using ctypes.

We lazily load the shared library when the horovod Python module is imported,
such that MPI is not loaded/required before this. We do not want MPI as a
required dependency for all our users, only for those that actually use Horovod.
"""

package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow:tensorflow.bzl", "tf_cc_shared_object", "tf_custom_op_py_library", "tf_gen_op_wrapper_py")
load("@local_config_ipu_horovod//:build_defs_horovod.bzl", "horovod_py_test", "poprun_py_test")

cc_library(
    name = "lbfgspp_headers",
    hdrs = [
        "@lbfgspp_repo//:lbfgspp_headers",
    ],
)

cc_library(
    name = "boost_headers",
    hdrs = [
        "@horovod_boost//:boost_headers",
    ],
)

cc_library(
    name = "horovod_ops",
    srcs = [
        "@horovod_repo//:horovod_common_headers",
        "@horovod_repo//:horovod_common_sources",
        "@horovod_repo//:horovod_tensorflow_sources",
    ],
    copts = [
        "-DHAVE_MPI=1",
        "-Iexternal/lbfgspp_repo/include",
        "-Iexternal/local_config_poplar/poplar/openmpi/include/openmpi",  # poplar sandbox
        "-Iexternal/local_config_poplar/poplar/poplar/include/openmpi",  # poplar release
        "-Iexternal/horovod_boost",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":boost_headers",
        ":lbfgspp_headers",
        "//tensorflow/core:framework",
        "@flatbuffers",
        "@local_config_poplar//poplar:poplar_headers",
    ],
    alwayslink = True,
)

tf_gen_op_wrapper_py(
    name = "horovod_ops_wrapper_py",
    out = "gen_horovod_ops.py",
    visibility = ["//tensorflow:internal"],
    deps = [":horovod_ops"],
)

tf_custom_op_py_library(
    name = "horovod_ops_py",
    kernels = [":horovod_ops"],
    srcs_version = "PY2AND3",
    deps = [":horovod_ops_wrapper_py"],
)

tf_cc_shared_object(
    name = "horovod_plugin.so",
    deps = [
        ":horovod_ops",
        "@local_config_poplar//poplar:mpi_lib",
    ],
)

py_library(
    name = "horovod_lib",
    srcs = [
        "__init__.py",
        "basics.py",
        "ipu_horovod_strategy.py",
        "popdist_strategy.py",
    ],
    data = [
        ":horovod_plugin.so",
    ],
    deps = [
        ":horovod_ops_py",
    ],
)

horovod_py_test(
    name = "horovod_test",
    size = "medium",
    srcs = ["horovod_test.py"],
    main = "horovod_test.py",
    num_processes = 2,
    deps = [
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

horovod_py_test(
    name = "popdist_strategy_test",
    size = "medium",
    srcs = ["popdist_strategy_test.py"],
    main = "popdist_strategy_test.py",
    num_processes = 2,
    deps = [
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

poprun_py_test(
    name = "poprun_popdist_strategy_test",
    size = "large",
    srcs = ["poprun_popdist_strategy_test.py"],
    main = "poprun_popdist_strategy_test.py",
    num_instances = 2,
    num_replicas = 2,
    deps = [
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

test_suite(
    name = "horovod_test_suite",
    tests = [
        "horovod_test",
        "popdist_strategy_test",
    ],
)
