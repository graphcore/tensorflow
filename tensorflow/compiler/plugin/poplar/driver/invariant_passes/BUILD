licenses(["restricted"])

load("//tensorflow/compiler/plugin/poplar:poplar.bzl", "poplar_cc_library")

package(default_visibility = ["//tensorflow/compiler/plugin/poplar:__subpackages__"])

poplar_cc_library(
    name = "no_control_deps_checker",
    srcs = [
        "no_control_deps_checker.cc",
    ],
    hdrs = [
        "no_control_deps_checker.h",
    ],
    deps = [
        "//tensorflow/compiler/xla:xla_headers_lib",
        "//third_party/eigen3",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf_headers",
    ],
)

poplar_cc_library(
    name = "resource_update_checker",
    srcs = [
        "resource_update_checker.cc",
    ],
    hdrs = [
        "resource_update_checker.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar:config_protos_cc_impl",
        "//tensorflow/compiler/plugin/poplar:tools_headers",
        "//tensorflow/compiler/plugin/poplar:xla_backend_protos",
        "//tensorflow/compiler/xla:xla_headers_lib",
        "//third_party/eigen3",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf_headers",
    ],
)
