# Description:
#   Contains the Keras API (internal TensorFlow version).

load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

tf_py_test(
    name = "callbacks_test",
    size = "large",
    srcs = ["callbacks_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:tf2",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:combinations",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "correctness_test",
    size = "large",
    srcs = ["correctness_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:tf2",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:combinations",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "deferred_sequential_test",
    size = "large",
    srcs = ["deferred_sequential_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:tf2",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:combinations",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "functional_test",
    size = "large",
    srcs = ["functional_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:tf2",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:combinations",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "metrics_correctness_test",
    size = "large",
    srcs = ["metrics_correctness_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:tf2",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:combinations",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "metrics_test",
    size = "large",
    srcs = ["metrics_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:tf2",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:combinations",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "sequential_test",
    size = "large",
    srcs = ["sequential_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:tf2",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:combinations",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "training_test",
    size = "large",
    srcs = ["training_test.py"],
    python_version = "PY3",
    shard_count = 20,
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:tf2",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:combinations",
        "//third_party/py/numpy",
    ],
)

test_suite(
    name = "all_tests",
    tests = [],
)
