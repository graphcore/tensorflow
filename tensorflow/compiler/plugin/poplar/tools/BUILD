load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

tf_cc_binary(
    name = "dump_graphs",
    srcs = ["dump_graphs.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
    ],
)

tf_cc_binary(
    name = "DataSetRunner",
    srcs = ["dataset_runner.cc"],
    deps = [
        "//tensorflow/core/common_runtime/data:standalone",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:session_options",
    ],
)
