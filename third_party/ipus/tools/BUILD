licenses(["restricted"])

load("//tensorflow:tensorflow.bzl", "tf_copts")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "poplar_executable_data",
    srcs = [
        "poplar_executable_data.cc",
    ],
    hdrs = [
        "include/ipu/poplar_executable_data.h",
    ],
    copts = ["-fexceptions"],
    includes = ["include"],
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
        ":libpoplar_executable_data.so",
        ":PoplarExecutableRunner",
        ":PoplarBinaryReader",
        "@local_config_poplar//poplar:popsec_libs",
        "@local_config_poplar//poplar:popsec_headers",
        "@jsoncpp_git//:jsoncpp",
        "@jsoncpp_git//:headers",
    ] + glob(["include/ipu/*.h"]),
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
                    dst=$${dst/*popsec\\/include/$$dst_headers}
                    dst=$${dst/*include\\/ipu/$$dst_headers\\/ipu}
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
    name = "libpoplar_executable_data.so",
    linkshared = 1,
    linkstatic = 1,
    deps = [
        ":poplar_executable_data",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@jsoncpp_git//:jsoncpp",
    ],
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
        "include/ipu/poplar_executable_runner.h",
    ],
    copts = ["-fexceptions"],
    includes = ["include"],
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

cc_library(
    name = "poplar_command_line_utils",
    srcs = [
        "poplar_command_line_utils.cc",
    ],
    hdrs = [
        "include/ipu/poplar_command_line_utils.h",
    ],
    copts = ["-fexceptions"],
    includes = ["include"],
    linkstatic = 1,
    deps = [
        ":poplar_executable_data",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@jsoncpp_git//:jsoncpp",
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
        ":poplar_command_line_utils",
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
        ":poplar_command_line_utils",
        ":poplar_executable_runner",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)
