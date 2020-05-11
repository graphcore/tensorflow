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
    name = "poplar_executable_data",
    srcs = [
        "poplar_executable_data.cc",
    ],
    hdrs = [
        "poplar_executable_data.h",
    ],
    copts = ["-fexceptions"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/eigen3",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@jsoncpp_git//:jsoncpp",
    ],
)

genrule(
    name = "libpoplar_executable_runner",
    srcs = [
        ":libpoplar_executable_runner.so",
        ":PoplarExecutableRunner",
        ":PoplarBinaryReader",
        "@local_config_poplar//poplar:popsec_libs",
        "@local_config_poplar//poplar:popsec_headers",
        "@jsoncpp_git//:jsoncpp",
        "@jsoncpp_git//:headers",
    ] + glob(["poplar_*.h"]),
    outs = ["libpoplar_executable_runner.tar.gz"],
    cmd = """
        tmp_root=`mktemp -d`
        tmp_folder=$$tmp_root/libpoplar_executable_runner
        dst_headers=$$tmp_folder/include
        dst_lib=$$tmp_folder/lib
        dst_bin=$$tmp_folder/bin
        mkdir -p $$dst_headers
        mkdir -p $$dst_lib
        mkdir -p $$dst_bin
        for f in $(SRCS); do
            dst=$$f
            case $$dst in
                *.so|*.a|*.so.*)
                    dst=$$dst_lib
                    ;;
                *.h|*.hpp)
                    dst=$${dst/*jsoncpp_git/$$dst_headers}
                    dst=$${dst/*popsec\/include/$$dst_headers}
                    dst=$${dst/tensorflow/$$dst_headers\/tensorflow}
                    mkdir -p `dirname $$dst`
                    ;;
                *)
                    dst=$$dst_bin
                    ;;
            esac
            cp $$f $$dst
        done
        tar czf $@ -C $$tmp_root libpoplar_executable_runner
        rm -rf $$tmp_root
        """,
)

cc_binary(
    name = "libpoplar_executable_runner.so",
    linkshared = 1,
    linkstatic = 1,
    deps = [
        ":poplar_executable_runner",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@jsoncpp_git//:jsoncpp",
        "@local_config_poplar//poplar:poplar_libs",
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
        ":poplar_executable_data",
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
