load("//tensorflow/compiler/plugin/poplar:poplar.bzl", "poplar_cc_library")

licenses(["restricted"])

package(default_visibility = ["//tensorflow/compiler/plugin/poplar:__subpackages__"])

poplar_cc_library(
    name = "ops_helper",
    srcs = [
        "ops_helper.cc",
    ],
    hdrs = [
        "ops_helper.h",
        "//tensorflow/compiler/plugin/poplar/driver:tensor.h",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops:poplar_ops.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:backend_config_proto_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:option_flag_proto_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:threestate_proto_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:vertex_templates",
        "//tensorflow/compiler/plugin/poplar/driver/tools:alias_info_proto_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver/tools:debug_info",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc_impl",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_proto_cc_impl",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/ipus/tools:poplar_executable_data",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_protobuf//:protobuf_headers",
        "@local_config_poplar//poplar:poplar_libs",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "ops",
    srcs = [
        "cast_to_gfloat_ops.cc",
        "feed_ops.cc",
        "map_ops.cc",
        "maths_ops.cc",
        "reduction_ops.cc",
        "tensor_ops.cc",
    ],
    hdrs = [
        "ops.h",
        "//tensorflow/compiler/plugin/poplar/driver:tensor.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:compiler_resources",
        "//tensorflow/compiler/plugin/poplar/driver:vertex_templates",
        "//tensorflow/compiler/plugin/poplar/driver/tools:matmul_util",
        "//tensorflow/compiler/plugin/poplar/driver/tools:poplar_util",
        "//tensorflow/compiler/plugin/poplar/driver/tools:reduction_util",
        "//tensorflow/compiler/plugin/poplar/driver/tools:slice_util",
        "//tensorflow/compiler/plugin/poplar/driver/tools:subcomputation_graph_caching",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:cast_to_gfloat_hlo",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:slice_apply",
        "//tensorflow/compiler/plugin/poplar/driver/visitors:partitioned_elementwise_cluster_visitor",
        "//tensorflow/compiler/plugin/poplar/driver/visitors:pipeline_visitor",
        "//tensorflow/compiler/plugin/poplar/driver/visitors:pipeline_visitor_creator",
        "//tensorflow/compiler/plugin/poplar/driver/visitors:repeat_loop_overlap_io_visitor",
        "//tensorflow/compiler/plugin/poplar/driver/visitors:visitor_arithmetic_expr",
        "//tensorflow/compiler/plugin/poplar/driver/visitors:visitor_map",
        "//tensorflow/compiler/plugin/poplar/kernels:popfloat_config_protos_cc_impl",
        "//third_party/eigen3",
        "//third_party/ipus/tools:poplar_executable_data",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf_headers",
        "@local_config_poplar//poplar:poplar_libs",
    ],
    alwayslink = True,
)
