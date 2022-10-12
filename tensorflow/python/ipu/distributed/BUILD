"""
PopDist support.
"""

package(default_visibility = ["//tensorflow:internal"])

load("@local_config_popdist//:build_defs_popdist.bzl", "poprun_py_test")

py_library(
    name = "host_collective_ops",
    srcs = ["host_collective_ops.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/compiler/plugin/poplar/ops:popdist_ops_py",
    ],
)

py_library(
    name = "distributed_lib",
    srcs = [
        "__init__.py",
        "popdist_strategy.py",
    ],
    deps = [
        ":host_collective_ops",
    ],
)

poprun_py_test(
    name = "popdist_strategy_test",
    size = "medium",
    srcs = ["popdist_strategy_test.py"],
    main = "popdist_strategy_test.py",
    num_instances = 2,
    num_replicas = 2,
    deps = [
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

poprun_py_test(
    name = "host_collective_ops_test",
    size = "medium",
    srcs = ["host_collective_ops_test.py"],
    main = "host_collective_ops_test.py",
    num_instances = 2,
    num_replicas = 2,
    deps = [
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

test_suite(
    name = "popdist_test_suite",
)
