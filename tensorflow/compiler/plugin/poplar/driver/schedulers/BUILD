load("//tensorflow/compiler/plugin/poplar:poplar.bzl", "poplar_cc_library")

licenses(["restricted"])

package(default_visibility = ["//tensorflow/compiler/plugin/poplar:__subpackages__"])

poplar_cc_library(
    name = "clustering_scheduler",
    srcs = [
        "clustering_scheduler.cc",
    ],
    hdrs = [
        "clustering_scheduler.h",
    ],
    deps = [
        ":ipu_scheduler",
        "//tensorflow/compiler/plugin/poplar/driver:compiler_information",
        "//tensorflow/compiler/plugin/poplar/driver/tools:flags",
        "//tensorflow/compiler/plugin/poplar/driver/tools:instruction_colocator_helper",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:inter_ipu_copy",
    ],
)

poplar_cc_library(
    name = "ipu_scheduler",
    srcs = [
        "ipu_scheduler.cc",
    ],
    hdrs = [
        "ipu_scheduler.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_poplar_dataflow_analysis",
        "//tensorflow/compiler/xla/service:buffer_value",
        "//tensorflow/compiler/xla/service:heap_simulator",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_buffer",
        "//tensorflow/compiler/xla/service:hlo_memory_scheduler",
        "//tensorflow/compiler/xla/service:hlo_reachability",
        "//tensorflow/compiler/xla/service:hlo_value",
    ],
)

poplar_cc_library(
    name = "liveness_look_ahead_scheduler",
    srcs = [
        "liveness_look_ahead_scheduler.cc",
    ],
    hdrs = [
        "liveness_look_ahead_scheduler.h",
    ],
    deps = [
        ":ipu_scheduler",
        ":schedule_tree",
        ":schedule_utils",
        "//tensorflow/compiler/plugin/poplar/driver:compiler_information",
        "//tensorflow/compiler/plugin/poplar/driver/tools:flags",
    ],
)

poplar_cc_library(
    name = "post_order_scheduler",
    srcs = [
        "post_order_scheduler.cc",
    ],
    hdrs = [
        "post_order_scheduler.h",
    ],
    deps = [
        ":ipu_scheduler",
    ],
)

poplar_cc_library(
    name = "schedule_tree",
    hdrs = [
        "schedule_tree.h",
    ],
)

poplar_cc_library(
    name = "schedule_utils",
    hdrs = [
        "schedule_utils.h",
    ],
)

poplar_cc_library(
    name = "shortest_path_scheduler",
    srcs = [
        "shortest_path_scheduler.cc",
    ],
    hdrs = [
        "shortest_path_scheduler.h",
    ],
    deps = [
        ":ipu_scheduler",
    ],
)

poplar_cc_library(
    name = "sync_list_scheduler",
    srcs = [
        "sync_list_scheduler.cc",
    ],
    hdrs = [
        "sync_list_scheduler.h",
    ],
    deps = [
        ":ipu_scheduler",
    ],
)

poplar_cc_library(
    name = "schedulers",
    deps = [
        ":clustering_scheduler",
        ":ipu_scheduler",
        ":liveness_look_ahead_scheduler",
        ":post_order_scheduler",
        ":schedule_tree",
        ":schedule_utils",
        ":shortest_path_scheduler",
        ":sync_list_scheduler",
    ],
)
