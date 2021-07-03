load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    licenses = ["notice"],  # Apache 2.0
)

tf_py_test(
    name = "extensions_test",
    size = "large",
    srcs = ["extensions_test.py"],
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
    name = "extensions_save_load_test",
    size = "large",
    srcs = ["extensions_save_load_test.py"],
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
    name = "data_adapter_test",
    size = "large",
    srcs = ["data_adapter_test.py"],
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

test_suite(
    name = "tests",
    tests = [],
)

test_suite(
    name = "extensions_test_suite",
    tests = [
        ":tests",
    ],
)
