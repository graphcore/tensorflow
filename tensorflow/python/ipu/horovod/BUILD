package(default_visibility = ["//tensorflow:internal"])

py_library(
    name = "horovod_lib",
    srcs = [
        "__init__.py",
        "basics.py",
        "ipu_horovod_strategy.py",
        "popdist_strategy.py",
    ],
    deps = [
        "//tensorflow/python/ipu/distributed:horovod_lib",
    ],
)
