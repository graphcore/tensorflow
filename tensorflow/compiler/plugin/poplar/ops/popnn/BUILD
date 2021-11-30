load("//tensorflow/compiler/plugin/poplar:poplar.bzl", "poplar_cc_library")

licenses(["restricted"])

package(default_visibility = ["//tensorflow/compiler/plugin/poplar:__subpackages__"])

poplar_cc_library(
    name = "ctc_loss",
    srcs = [
        "ctc_loss.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "gelu",
    srcs = [
        "gelu.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "gru",
    srcs = [
        "gru.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "hard_sigmoid",
    srcs = [
        "hard_sigmoid.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "lstm",
    srcs = [
        "lstm.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "norm",
    srcs = [
        "norm.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "swish",
    srcs = [
        "swish.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "ops",
    srcs = [],
    deps = [
        ":ctc_loss",
        ":gelu",
        ":gru",
        ":hard_sigmoid",
        ":lstm",
        ":norm",
        ":swish",
    ],
)
