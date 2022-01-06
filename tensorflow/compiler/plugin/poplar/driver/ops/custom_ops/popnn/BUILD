load("//tensorflow/compiler/plugin/poplar:poplar.bzl", "poplar_cc_library")

licenses(["restricted"])

package(default_visibility = ["//tensorflow/compiler/plugin/poplar:__subpackages__"])

poplar_cc_library(
    name = "arg_min_max",
    srcs = [
        "arg_min_max.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:arg_min_max",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "ctc_loss",
    srcs = [
        "ctc_loss.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:ctc_loss",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "gru",
    srcs = [
        "gru.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools:rnn_util",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "lstm",
    srcs = [
        "lstm.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools:rnn_util",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "non_linearity",
    srcs = [
        "non_linearity.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:non_linearity",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "norm",
    srcs = [
        "norm.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:norm",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "onehot",
    srcs = [
        "onehot.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:onehot",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "pooling",
    srcs = [
        "pooling.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:pooling",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "topk",
    srcs = [
        "topk.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:topk",
    ],
    alwayslink = True,
)
