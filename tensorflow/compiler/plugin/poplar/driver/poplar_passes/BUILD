load("//tensorflow/compiler/plugin/poplar:poplar.bzl", "poplar_cc_library")

licenses(["restricted"])

package(default_visibility = ["//tensorflow/compiler/plugin/poplar:__subpackages__"])

poplar_cc_library(
    name = "convolution_preplanning",
    srcs = [
        "convolution_preplanning.cc",
    ],
    hdrs = [
        "convolution_preplanning.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:conv_poplar_util",
        "//tensorflow/compiler/plugin/poplar/driver/tools:matcher_predicates",
        "//tensorflow/compiler/plugin/poplar/driver/tools:ml_type_helper",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "@local_config_poplar//poplar:poplar_libs",
    ],
)

poplar_cc_library(
    name = "ctc_preplanning",
    srcs = [
        "ctc_preplanning.cc",
    ],
    hdrs = [
        "ctc_preplanning.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:ctc_loss",
    ],
)

poplar_cc_library(
    name = "embedding_plans_preplanning",
    srcs = [
        "embedding_plans_preplanning.cc",
    ],
    hdrs = [
        "embedding_plans_preplanning.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
    ],
)

poplar_cc_library(
    name = "map_hlo_instruction_to_debug_id",
    srcs = [
        "map_hlo_instruction_to_debug_id.cc",
    ],
    hdrs = [
        "map_hlo_instruction_to_debug_id.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:debug_info",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "matmul_preplanning",
    srcs = [
        "matmul_preplanning.cc",
    ],
    hdrs = [
        "matmul_preplanning.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:matcher_predicates",
        "//tensorflow/compiler/plugin/poplar/driver/tools:matmul_util",
        "//tensorflow/compiler/plugin/poplar/driver/tools:ml_type_helper",
        "//tensorflow/compiler/plugin/poplar/driver/tools:poplar_util",
        "//tensorflow/compiler/plugin/poplar/driver/tools:rnn_util",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "@local_config_poplar//poplar:poplar_libs",
    ],
)

poplar_cc_library(
    name = "poplar_passes",
    deps = [
        ":convolution_preplanning",
        ":ctc_preplanning",
        ":embedding_plans_preplanning",
        ":map_hlo_instruction_to_debug_id",
        ":matmul_preplanning",
    ],
)
