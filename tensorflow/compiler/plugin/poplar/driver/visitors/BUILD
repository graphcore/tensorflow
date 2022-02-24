load("//tensorflow/compiler/plugin/poplar:poplar.bzl", "poplar_cc_library")

licenses(["restricted"])

package(default_visibility = ["//tensorflow/compiler/plugin/poplar:__subpackages__"])

poplar_cc_library(
    name = "entry_visitor",
    srcs = [
        "entry_visitor.cc",
    ],
    hdrs = [
        "entry_visitor.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:backend_config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:poplar_feed_config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:threestate_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:trace_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver/tools:alias_info_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver/tools:data_initializer",
        "//tensorflow/compiler/plugin/poplar/driver/tools:poplar_util",
        "//tensorflow/compiler/plugin/poplar/kernels:ops_cc_impl",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/ipus/tools:poplar_executable_data",
        "@local_config_poplar//poplar:poplar_libs",
    ],
)

poplar_cc_library(
    name = "visitor_base",
    srcs = [
        "visitor_base.cc",
    ],
    hdrs = [
        "visitor_base.h",
        "//tensorflow/compiler/plugin/poplar/driver:tensor.h",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops:poplar_ops.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:backend_config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:option_flag_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:poplar_feed_config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:threestate_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver/tools:alias_info_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver/tools:matcher_predicates",
        "//tensorflow/compiler/plugin/poplar/driver/tools:poplar_util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:host_embedding",
        "//tensorflow/compiler/plugin/poplar/kernels:ops_cc_impl",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/stream_executor:dnn_proto_cc_impl",
        "//third_party/ipus/tools:poplar_executable_data",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_protobuf//:protobuf_headers",
        "@local_config_poplar//poplar:poplar_libs",
    ],
)

poplar_cc_library(
    name = "visitor_full",
    srcs = [
        "visitor_full.cc",
    ],
    hdrs = [
        "visitor_full.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/ops",
        "//tensorflow/compiler/plugin/poplar/driver/tools:poplar_util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:rnn",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/ipus/tools:poplar_executable_data",
        "@local_config_poplar//poplar:poplar_libs",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "deferred_visitor",
    srcs = [
        "deferred_visitor.cc",
    ],
    hdrs = [
        "deferred_visitor.h",
        "//tensorflow/compiler/plugin/poplar/driver:poplar_executor.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:trace_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver/tools:data_initializer",
        "//tensorflow/compiler/plugin/poplar/driver/tools:poplar_util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:hlo_remote_buffer_info",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:remote_parameter",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:stateful_gradient_accumulate",
        "//third_party/ipus/tools:poplar_executable_data",
        "@local_config_poplar//poplar:poplar_libs",
    ],
)

poplar_cc_library(
    name = "pipeline_visitor",
    srcs = [
        "pipeline_visitor.cc",
    ],
    hdrs = [
        "pipeline_visitor.h",
        "pipeline_visitor_utils.h",
    ],
    deps = [
        ":pipeline_stage_visitor",
        "//tensorflow/compiler/plugin/poplar/driver/tools:make_visitor",
        "//tensorflow/compiler/plugin/poplar/driver/tools:pipeline_util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:inter_tileset_copy",
    ],
)

poplar_cc_library(
    name = "pipeline_stage_visitor",
    srcs = [
        "pipeline_stage_visitor.cc",
    ],
    hdrs = [
        "pipeline_stage_visitor.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:option_flag_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:poplar_feed_config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:threestate_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver/tools:alias_info_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver/tools:execution_counter_util",
        "//tensorflow/compiler/plugin/poplar/driver/tools:poplar_util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:hlo_poplar_instruction",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:host_embedding",
        "//tensorflow/compiler/plugin/poplar/kernels:ops_cc_impl",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/stream_executor:dnn_proto_cc_impl",
        "//third_party/ipus/tools:poplar_executable_data",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_protobuf//:protobuf_headers",
        "@local_config_poplar//poplar:poplar_libs",
    ],
)

poplar_cc_library(
    name = "pipeline_visitor_creator",
    srcs = [
        "pipeline_visitor_creator.cc",
    ],
    hdrs = [
        "pipeline_visitor_creator.h",
    ],
    deps = [
        ":deferred_visitor",
        ":grouped_overlap_pipeline_visitor",
        ":pipeline_visitor",
    ],
)

poplar_cc_library(
    name = "grouped_overlap_pipeline_visitor",
    srcs = [
        "grouped_overlap_pipeline_visitor.cc",
    ],
    hdrs = [
        "grouped_overlap_pipeline_visitor.h",
    ],
    deps = [
        ":deferred_visitor",
        ":pipeline_visitor",
    ],
)

poplar_cc_library(
    name = "repeat_loop_overlap_io_visitor",
    srcs = [
        "repeat_loop_overlap_io_visitor.cc",
    ],
    hdrs = [
        "repeat_loop_overlap_io_visitor.h",
    ],
    deps = [
        ":repeat_loop_visitor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:inter_tileset_copy",
    ],
)

poplar_cc_library(
    name = "repeat_loop_visitor",
    srcs = [
        "repeat_loop_visitor.cc",
    ],
    hdrs = [
        "repeat_loop_visitor.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:poplar_util",
    ],
)

poplar_cc_library(
    name = "visitor_arithmetic_expr",
    srcs = [
        "visitor_arithmetic_expr.cc",
    ],
    hdrs = [
        "visitor_arithmetic_expr.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:poplar_util",
    ],
)

poplar_cc_library(
    name = "visitor_map",
    srcs = [
        "visitor_map.cc",
    ],
    hdrs = [
        "visitor_map.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:poplar_util",
    ],
)

poplar_cc_library(
    name = "partitioned_elementwise_cluster_visitor",
    srcs = [
        "partitioned_elementwise_cluster_visitor.cc",
    ],
    hdrs = [
        "partitioned_elementwise_cluster_visitor.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:poplar_util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:collective_reorder",
    ],
)

poplar_cc_library(
    name = "visitors",
    deps = [
        ":deferred_visitor",
        ":entry_visitor",
        ":grouped_overlap_pipeline_visitor",
        ":partitioned_elementwise_cluster_visitor",
        ":pipeline_stage_visitor",
        ":pipeline_visitor",
        ":pipeline_visitor_creator",
        ":repeat_loop_overlap_io_visitor",
        ":repeat_loop_visitor",
        ":visitor_arithmetic_expr",
        ":visitor_base",
        ":visitor_full",
        ":visitor_map",
    ],
)
