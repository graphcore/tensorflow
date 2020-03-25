load("//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_copts")

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
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:session_options",
        "//tensorflow/core/common_runtime/data:standalone",
    ],
)

cc_library(
    name = "poplar_executable_runner",
    srcs = [
        "poplar_executable_runner.cc",
    ],
    hdrs = [
        "poplar_executable_runner.h",
    ],
    copts = ["-fexceptions"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/eigen3",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@jsoncpp_git//:jsoncpp",
        "@local_config_poplar//poplar:poplar_libs",
    ],
)

cc_binary(
    name = "PoplarBinaryReader",
    srcs = [
        "poplar_binary_reader.cc",
    ],
    copts = tf_copts() + [
        "-fexceptions",
    ],
    linkopts = ["-ldl"],
    visibility = ["//visibility:public"],
    deps = [
        ":poplar_executable_runner",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)

cc_binary(
    name = "PoplarExecutableRunner",
    srcs = [
        "poplar_executable_runner_main.cc",
    ],
    copts = tf_copts() + [
        "-fexceptions",
    ],
    linkopts = ["-ldl"],
    visibility = ["//visibility:public"],
    deps = [
        ":poplar_executable_runner",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)

py_binary(
    name = "tensorflow_weights_extractor",
    srcs = ["tensorflow_weights_extractor.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "tensorflow_weights_extractor_lib",
    srcs = ["tensorflow_weights_extractor.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)
