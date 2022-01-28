load("//tensorflow/compiler/plugin/poplar:poplar.bzl", "poplar_cc_library")
load("//tensorflow/core/platform:default/build_config.bzl", "tf_proto_library")

licenses(["restricted"])

package(default_visibility = ["//tensorflow/compiler/plugin/poplar:__subpackages__"])

tf_proto_library(
    name = "alias_info",
    srcs = [
        "alias_info.proto",
    ],
    cc_api_version = 2,
    j2objc_api_version = 1,
    make_default_target_header_only = True,
)

poplar_cc_library(
    name = "flags",
    srcs = [
        "flags.cc",
    ],
    hdrs = [
        "flags.h",
    ],
    deps = [
        ":hash",
        "//tensorflow/compiler/plugin/poplar/driver:config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:option_flag_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:threestate_cc_impl",
        "//tensorflow/compiler/xla:parse_flags_from_env",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_google_protobuf//:protobuf_headers",
    ],
)

poplar_cc_library(
    name = "hash",
    hdrs = [
        "hash.h",
    ],
)

poplar_cc_library(
    name = "tensor_location",
    hdrs = [
        "tensor_location.h",
    ],
    deps = [
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core:framework_headers_lib",
        "@com_google_absl//absl/types:optional",
    ],
)

poplar_cc_library(
    name = "poplar_util",
    srcs = [
        "poplar_util.cc",
    ],
    hdrs = [
        "debug_info.h",
        "execution_counter_util.h",
        "generic_graph_caching.h",
        "offloading_util.h",
        "poplar_util.h",
        "spsc_outfeed_queue.h",
        "subcomputation_graph_caching.h",
        "//tensorflow/compiler/plugin/poplar/driver:poplar_executor.h",
        "//tensorflow/compiler/plugin/poplar/driver:poplar_transfer_manager.h",
        "//tensorflow/compiler/plugin/poplar/driver:poplar_version.h",
        "//tensorflow/compiler/plugin/poplar/driver:tensor.h",
        "//tensorflow/compiler/plugin/poplar/driver/ops:ops.h",
        "//tensorflow/compiler/plugin/poplar/driver/ops:ops_helper.h",
        "//tensorflow/compiler/plugin/poplar/driver/passes:allocation_finder.h",
        "//tensorflow/compiler/plugin/poplar/driver/passes:convolution_classifier.h",
        "//tensorflow/compiler/plugin/poplar/driver/visitors:deferred_visitor.h",
        "//tensorflow/compiler/plugin/poplar/driver/visitors:visitor_base.h",
        "//tensorflow/compiler/plugin/poplar/driver/visitors:visitor_full.h",
    ],
    deps = [
        ":feed_info",
        ":flags",
        ":infeed_allocator",
        ":infeed_iterator",
        ":input_output_aliasing_map",
        ":io_thread",
        ":mapping_helper",
        ":ml_type_helper",
        ":poplar_replica_groups",
        ":progress_bar",
        ":seed_generator",
        ":tensor_map",
        ":util",
        "//tensorflow/compiler/plugin/poplar:poplar_version",
        "//tensorflow/compiler/plugin/poplar/driver:backend_config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:compiler_annotations",
        "//tensorflow/compiler/plugin/poplar/driver:compiler_information",
        "//tensorflow/compiler/plugin/poplar/driver:compiler_resources",
        "//tensorflow/compiler/plugin/poplar/driver:config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:poplar_feed_config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:prng_seed_state",
        "//tensorflow/compiler/plugin/poplar/driver:trace_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:host_embedding",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:multi_slice",
        "//tensorflow/compiler/plugin/poplar/kernels:ops_cc_impl",
        "//tensorflow/compiler/xla:array",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/service:call_graph",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:generic_transfer_manager",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_execution_profile",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:hlo_reachability",
        "//tensorflow/compiler/xla/service:transfer_manager",
        "//tensorflow/compiler/xla/service/cpu:cpu_runtime",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core/kernels/data:unbounded_thread_pool",
        "//tensorflow/stream_executor:dnn",
        "//tensorflow/stream_executor:dnn_proto_cc_impl",
        "//third_party/ipus/tools:poplar_executable_data",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:any",
        "@com_google_absl//absl/types:bad_any_cast",
        "@com_google_absl//absl/types:optional",
        "@com_google_protobuf//:protobuf_headers",
        "@local_config_poplar//poplar:poplar_libs",
    ],
)

poplar_cc_library(
    name = "hlo_poplar_buffer",
    srcs = [
        "hlo_poplar_buffer.cc",
    ],
    hdrs = [
        "hlo_poplar_buffer.h",
    ],
    deps = [
        ":alias_info_cc_impl",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/service:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

poplar_cc_library(
    name = "hlo_poplar_buffer_util",
    srcs = [
        "hlo_poplar_buffer_util.cc",
    ],
    hdrs = [
        "hlo_poplar_buffer_util.h",
    ],
    deps = [
        ":hlo_poplar_buffer",
        "@com_google_absl//absl/types:span",
    ],
)

poplar_cc_library(
    name = "poplar_replica_groups",
    srcs = [
        "poplar_replica_groups.cc",
    ],
    hdrs = [
        "poplar_replica_groups.h",
    ],
    deps = [
        ":hash",
        "//tensorflow/compiler/xla:util",
        "@com_google_absl//absl/types:optional",
    ],
)

poplar_cc_library(
    name = "util",
    srcs = [
        "util.cc",
    ],
    hdrs = [
        "util.h",
    ],
    deps = [
        ":flags",
        ":hash",
        ":hlo_poplar_buffer",
        ":tensor_location",
        "//tensorflow/compiler/plugin/poplar/driver:backend_config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:hlo_poplar_instruction",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:inter_ipu_copy",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:within_replicas",
        "//tensorflow/compiler/plugin/poplar/kernels:ops_cc_impl",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla/service:hlo_evaluator",
        "//tensorflow/compiler/xla/service:hlo_reachability",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_protobuf//:protobuf_headers",
    ],
)

poplar_cc_library(
    name = "slice_util",
    srcs = [
        "slice_util.cc",
    ],
    hdrs = [
        "slice_util.h",
    ],
    deps = [
        ":util",
    ],
)

poplar_cc_library(
    name = "inplace_util",
    srcs = [
        "inplace_util.cc",
    ],
    hdrs = [
        "inplace_util.h",
        "util.h",
    ],
    deps = [
        ":alias_info_cc_impl",
        ":extension_registry",
        ":flags",
        ":hash",
        ":hlo_instruction_extensions",
        ":hlo_poplar_buffer",
        ":tensor_location",
        "//tensorflow/compiler/plugin/poplar/driver:backend_config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:threestate_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:hlo_poplar_instruction",
        "//tensorflow/compiler/plugin/poplar/kernels:custom_kernels_util",
        "//tensorflow/compiler/plugin/poplar/kernels:ops_cc_impl",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_reachability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:any",
        "@com_google_absl//absl/types:bad_any_cast",
        "@com_google_absl//absl/types:optional",
        "@com_google_protobuf//:protobuf_headers",
        "@jsoncpp_git//:jsoncpp",
    ],
)

poplar_cc_library(
    name = "infeed_allocator",
    srcs = [
        "infeed_allocator.cc",
    ],
    hdrs = [
        "infeed_allocator.h",
    ],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@com_google_absl//absl/strings",
    ],
)

poplar_cc_library(
    name = "infeed_iterator",
    srcs = [
        "infeed_iterator.cc",
    ],
    hdrs = [
        "infeed_iterator.h",
        "spsc_queue.h",
    ],
    deps = [
        ":flags",
        ":infeed_allocator",
        "//tensorflow/compiler/plugin/poplar/driver:config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:option_flag_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:poplar_feed_config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:threestate_cc_impl",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core/kernels/data:unbounded_thread_pool",
    ],
)

poplar_cc_library(
    name = "extension_registry",
    hdrs = [
        "extension_registry.h",
    ],
    deps = [
        "//tensorflow/compiler/xla/service:hlo",
    ],
)

poplar_cc_library(
    name = "pipeline_util",
    srcs = [
        "pipeline_util.cc",
    ],
    hdrs = [
        "pipeline_util.h",
    ],
    deps = [
        ":util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:execution_counter",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:fifo",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:stateful_noop",
        "//tensorflow/compiler/xla/service:hlo",
    ],
)

poplar_cc_library(
    name = "matcher_predicates",
    srcs = [
        "matcher_predicates.cc",
    ],
    hdrs = [
        "matcher_predicates.h",
    ],
    deps = [
        ":alias_info_cc_impl",
        ":hash",
        ":hlo_poplar_buffer",
        ":inplace_util",
        ":util",
        "//tensorflow/compiler/plugin/poplar/driver:backend_config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:option_flag_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:threestate_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:norm",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:pooling",
        "//tensorflow/compiler/plugin/poplar/kernels:ops_cc_impl",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_group",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:hlo_reachability",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//third_party/eigen3",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:any",
        "@com_google_absl//absl/types:bad_any_cast",
        "@com_google_protobuf//:protobuf_headers",
        "@jsoncpp_git//:jsoncpp",
    ],
)

poplar_cc_library(
    name = "hlo_instruction_extensions",
    srcs = [
        "hlo_instruction_extensions.cc",
    ],
    hdrs = [
        "hlo_instruction_extensions.h",
        "inplace_util.h",
    ],
    deps = [
        ":extension_registry",
        ":hash",
        ":hlo_poplar_buffer",
        ":tensor_location",
        "//tensorflow/compiler/plugin/poplar/driver:backend_config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:option_flag_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:threestate_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:hlo_poplar_instruction",
        "//tensorflow/compiler/plugin/poplar/kernels:ops_cc_impl",
        "//tensorflow/compiler/xla/service:hlo_reachability",
    ],
)

poplar_cc_library(
    name = "debug_info",
    srcs = [
        "debug_info.cc",
    ],
    hdrs = [
        "debug_info.h",
        "execution_counter_util.h",
    ],
    deps = [
        ":feed_info",
        ":input_output_aliasing_map",
        ":mapping_helper",
        ":ml_type_helper",
        ":poplar_util",
        ":progress_bar",
        ":tensor_map",
        "//tensorflow/compiler/plugin/poplar/driver:backend_config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:compiler_annotations",
        "//tensorflow/compiler/plugin/poplar/driver:compiler_resources",
        "//tensorflow/compiler/plugin/poplar/driver:config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:poplar_feed_config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:prng_seed_state",
        "//tensorflow/compiler/plugin/poplar/kernels:ops_cc_impl",
        "//tensorflow/compiler/xla/service:call_graph",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_group",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//third_party/ipus/tools:poplar_executable_data",
        "@com_google_absl//absl/types:any",
        "@com_google_absl//absl/types:bad_any_cast",
        "@local_config_poplar//poplar:poplar_libs",
    ],
)

poplar_cc_library(
    name = "execution_counter_util",
    srcs = [
        "execution_counter_util.cc",
    ],
    hdrs = [
        "execution_counter_util.h",
    ],
    deps = [
        ":alias_info_cc_impl",
        ":debug_info",
        ":progress_bar",
        ":tensor_location",
        "//tensorflow/compiler/plugin/poplar/driver:backend_config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:compiler_resources",
        "//tensorflow/compiler/plugin/poplar/driver:config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:option_flag_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:poplar_feed_config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:threestate_cc_impl",
        "//tensorflow/compiler/plugin/poplar/kernels:ops_cc_impl",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/ipus/tools:poplar_executable_data",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:any",
        "@com_google_absl//absl/types:bad_any_cast",
        "@local_config_poplar//poplar:poplar_libs",
    ],
)

poplar_cc_library(
    name = "tensor_map",
    srcs = [
        "tensor_map.cc",
    ],
    hdrs = [
        "tensor_map.h",
    ],
    deps = [
        ":alias_info_cc_impl",
        ":flags",
        ":tensor_location",
        ":util",
        "//tensorflow/compiler/plugin/poplar/driver:backend_config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:option_flag_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:threestate_cc_impl",
        "@com_google_absl//absl/types:any",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:variant",
        "@jsoncpp_git//:jsoncpp",
        "@local_config_poplar//poplar:poplar_libs",
    ],
)

poplar_cc_library(
    name = "feed_info",
    srcs = [
        "feed_info.cc",
    ],
    hdrs = [
        "feed_info.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:poplar_feed_config_cc_impl",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/stream_executor:dnn_proto_cc_impl",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

poplar_cc_library(
    name = "progress_bar",
    srcs = [
        "progress_bar.cc",
    ],
    hdrs = [
        "progress_bar.h",
    ],
    deps = [
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/stream_executor:dnn_proto_cc_impl",
        "//third_party/eigen3",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_protobuf//:protobuf_headers",
    ],
)

poplar_cc_library(
    name = "io_thread",
    srcs = [
        "io_thread.cc",
    ],
    hdrs = [
        "io_thread.h",
    ],
    deps = [
        "//tensorflow/compiler/xla:status",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf_headers",
    ],
)

poplar_cc_library(
    name = "mapping_helper",
    srcs = [
        "mapping_helper.cc",
    ],
    hdrs = [
        "mapping_helper.h",
    ],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@local_config_poplar//poplar:poplar_libs",
    ],
)

poplar_cc_library(
    name = "input_output_aliasing_map",
    srcs = [
        "input_output_aliasing_map.cc",
    ],
    hdrs = [
        "input_output_aliasing_map.h",
    ],
    deps = [
        ":alias_info_cc_impl",
        ":flags",
        ":util",
        "//tensorflow/compiler/plugin/poplar/driver:backend_config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:option_flag_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:threestate_cc_impl",
        "//tensorflow/compiler/xla/service:hlo",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf_headers",
    ],
)

poplar_cc_library(
    name = "ml_type_helper",
    srcs = [
        "ml_type_helper.cc",
    ],
    hdrs = [
        "ml_type_helper.h",
    ],
    deps = [
        ":util",
        "//tensorflow/compiler/plugin/poplar/driver:backend_config_cc_impl",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/service:hlo",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf_headers",
    ],
)

poplar_cc_library(
    name = "subcomputation_graph_caching",
    srcs = [
        "subcomputation_graph_caching.cc",
    ],
    hdrs = [
        "subcomputation_graph_caching.h",
        "//tensorflow/compiler/plugin/poplar/driver/ops:ops.h",
        "//tensorflow/compiler/plugin/poplar/driver/visitors:visitor_full.h",
    ],
    deps = [
        ":util",
        "//tensorflow/compiler/plugin/poplar/driver:poplar_feed_config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:trace_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver/ops:ops_helper",
        "//tensorflow/compiler/plugin/poplar/driver/passes:allocation_finder",
        "//tensorflow/compiler/plugin/poplar/driver/passes:convolution_classifier",
        "//tensorflow/compiler/plugin/poplar/driver/visitors:deferred_visitor",
        "//tensorflow/compiler/plugin/poplar/driver/visitors:partitioned_elementwise_cluster_visitor",
        "//tensorflow/compiler/plugin/poplar/driver/visitors:visitor_base",
        "//tensorflow/compiler/xla:status",
        "//third_party/ipus/tools:poplar_executable_data",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_protobuf//:protobuf_headers",
        "@local_config_poplar//poplar:poplar_libs",
    ],
)

poplar_cc_library(
    name = "seed_generator",
    srcs = [
        "seed_generator.cc",
    ],
    hdrs = [
        "seed_generator.h",
    ],
    deps = [
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@com_google_absl//absl/strings",
    ],
)

poplar_cc_library(
    name = "conv_util",
    srcs = [
        "conv_util.cc",
    ],
    hdrs = [
        "conv_util.h",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:conv_with_reverse.h",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:weights_transpose_chans_flip_xy.h",
    ],
    deps = [
        ":hlo_poplar_buffer_util",
        "//tensorflow/compiler/plugin/poplar/driver:backend_config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:option_flag_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:poplar_feed_config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:threestate_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver/tools:alias_info_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:hlo_poplar_instruction",
        "//tensorflow/compiler/plugin/poplar/kernels:custom_kernels_util",
        "//tensorflow/compiler/plugin/poplar/kernels:ops_cc_impl",
        "//tensorflow/stream_executor:dnn_proto_cc_impl",
        "//third_party/eigen3",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:any",
        "@com_google_absl//absl/types:bad_any_cast",
        "@com_google_protobuf//:protobuf_headers",
        "@jsoncpp_git//:jsoncpp",
    ],
)

poplar_cc_library(
    name = "conv_poplar_util",
    srcs = [
        "conv_poplar_util.cc",
    ],
    hdrs = [
        "conv_poplar_util.h",
    ],
    deps = [
        ":conv_util",
        ":poplar_replica_groups",
        ":poplar_util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:multi_conv",
    ],
)

poplar_cc_library(
    name = "generic_graph_caching",
    srcs = [
        "generic_graph_caching.cc",
    ],
    hdrs = [
        "generic_graph_caching.h",
        "//tensorflow/compiler/plugin/poplar/driver:tensor.h",
    ],
    deps = [
        ":execution_counter_util",
        ":feed_info",
        ":input_output_aliasing_map",
        ":mapping_helper",
        ":ml_type_helper",
        ":poplar_util",
        ":progress_bar",
        ":subcomputation_graph_caching",
        ":tensor_map",
        "//tensorflow/compiler/plugin/poplar/driver:backend_config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:compiler_annotations",
        "//tensorflow/compiler/plugin/poplar/driver:compiler_information",
        "//tensorflow/compiler/plugin/poplar/driver:compiler_resources",
        "//tensorflow/compiler/plugin/poplar/driver:config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:option_flag_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:poplar_feed_config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:prng_seed_state",
        "//tensorflow/compiler/plugin/poplar/driver:threestate_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:trace_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver/tools:alias_info_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:host_embedding",
        "//tensorflow/compiler/plugin/poplar/kernels:ops_cc_impl",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla/service:hlo",
        "//third_party/ipus/tools:poplar_executable_data",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@local_config_poplar//poplar:poplar_libs",
    ],
)

poplar_cc_library(
    name = "matmul_util",
    srcs = [
        "matmul_util.cc",
    ],
    hdrs = [
        "matmul_util.h",
    ],
    deps = [
        ":util",
    ],
)

poplar_cc_library(
    name = "conversions",
    srcs = [
        "conversions.cc",
    ],
    hdrs = [
        "conversions.h",
        "spsc_outfeed_queue.h",
        "spsc_queue.h",
        "//tensorflow/compiler/plugin/poplar/driver:poplar_executor.h",
        "//tensorflow/compiler/plugin/poplar/driver:poplar_transfer_manager.h",
        "//tensorflow/compiler/plugin/poplar/driver/passes:allocation_finder.h",
    ],
    deps = [
        ":alias_info_cc_impl",
        ":feed_info",
        ":flags",
        ":hlo_poplar_buffer",
        ":infeed_allocator",
        ":infeed_iterator",
        ":input_output_aliasing_map",
        ":io_thread",
        ":seed_generator",
        ":tensor_location",
        ":util",
        "//tensorflow/compiler/plugin/poplar/driver:backend_config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:compiler_annotations",
        "//tensorflow/compiler/plugin/poplar/driver:config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:option_flag_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:poplar_feed_config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:threestate_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:trace_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:host_embedding",
        "//tensorflow/compiler/plugin/poplar/kernels:ops_cc_impl",
        "//tensorflow/compiler/xla/service:generic_transfer_manager",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service/cpu:cpu_runtime",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core/kernels/data:unbounded_thread_pool",
        "//third_party/eigen3",
        "@com_google_absl//absl/types:optional",
        "@com_google_protobuf//:protobuf_headers",
        "@local_config_poplar//poplar:poplar_libs",
    ],
)

poplar_cc_library(
    name = "tracepoint",
    srcs = [
        "tracepoint.cc",
    ],
    hdrs = [
        "tracepoint.h",
    ],
    deps = [
        "@com_google_absl//absl/strings",
        "@local_config_poplar//poplar:poplar_libs",
    ],
)

poplar_cc_library(
    name = "poplar_executable_binary_file",
    srcs = [
        "poplar_executable_binary_file.cc",
    ],
    hdrs = [
        "poplar_executable_binary_file.h",
    ],
    deps = [
        ":popef_util",
        ":poplar_util",
        "//tensorflow/compiler/plugin/poplar/driver:poplar_executable",
    ],
)

poplar_cc_library(
    name = "hlo_hash",
    srcs = [
        "hlo_hash.cc",
    ],
    hdrs = [
        "hlo_hash.h",
    ],
    deps = [
        ":hash",
        "//tensorflow/compiler/xla/service/cpu:cpu_runtime",
        "//third_party/eigen3",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf_headers",
    ],
)

poplar_cc_library(
    name = "hlo_matcher",
    srcs = [
        "hlo_matcher.cc",
    ],
    hdrs = [
        "hlo_matcher.h",
    ],
    deps = [
        ":meta_graph",
        "//tensorflow/compiler/plugin/poplar/driver/passes:allocation_finder",
    ],
)

poplar_cc_library(
    name = "meta_graph",
    hdrs = [
        "meta_graph.h",
    ],
    deps = [
    ],
)

poplar_cc_library(
    name = "offloading_util",
    srcs = [
        "offloading_util.cc",
    ],
    hdrs = [
        "offloading_util.h",
    ],
    deps = [
        ":util",
    ],
)

poplar_cc_library(
    name = "instruction_colocator_helper",
    srcs = [
        "instruction_colocator_helper.cc",
    ],
    hdrs = [
        "instruction_colocator_helper.h",
    ],
    deps = [
        ":poplar_replica_groups",
        ":util",
        "//tensorflow/compiler/plugin/poplar/driver:compiler_information",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:all_gather",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:hlo_remote_buffer_info",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:recv_from_host",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:reduce_many",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:reduce_scatter",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:send_to_host",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:stateful_gradient_accumulate",
    ],
)

poplar_cc_library(
    name = "reduction_util",
    srcs = [
        "reduction_util.cc",
    ],
    hdrs = [
        "poplar_util.h",
        "reduction_util.h",
        "//tensorflow/compiler/plugin/poplar/driver:tensor.h",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops:poplar_ops.h",
    ],
    deps = [
        ":feed_info",
        ":flags",
        ":hash",
        ":input_output_aliasing_map",
        ":ml_type_helper",
        ":poplar_replica_groups",
        ":tensor_map",
        ":util",
        "//tensorflow/compiler/plugin/poplar/driver:backend_config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:compiler_annotations",
        "//tensorflow/compiler/plugin/poplar/driver:config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:option_flag_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:poplar_feed_config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:threestate_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver/passes:allocation_finder",
        "//tensorflow/compiler/plugin/poplar/driver/tools:alias_info_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:host_embedding",
        "//tensorflow/compiler/plugin/poplar/kernels:ops_cc_impl",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/stream_executor:dnn_proto_cc_impl",
        "//third_party/eigen3",
        "//third_party/ipus/tools:poplar_executable_data",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_protobuf//:protobuf_headers",
        "@local_config_poplar//poplar:poplar_libs",
    ],
)

poplar_cc_library(
    name = "replica_identical_dataflow_analysis",
    srcs = [
        "replica_identical_dataflow_analysis.cc",
    ],
    hdrs = [
        "replica_identical_dataflow_analysis.h",
    ],
    deps = [
        ":matcher_predicates",
        ":pipeline_util",
        ":util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:all_gather",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:user_op_hlo",
    ],
)

poplar_cc_library(
    name = "single_hlo_matcher",
    srcs = [
        "single_hlo_matcher.cc",
    ],
    hdrs = [
        "single_hlo_matcher.h",
    ],
    deps = [
        ":feed_info",
        ":hlo_matcher",
        ":inplace_util",
        ":input_output_aliasing_map",
        "//tensorflow/compiler/plugin/poplar/driver:poplar_feed_config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:host_embedding",
    ],
)

poplar_cc_library(
    name = "hlo_poplar_dataflow_analysis",
    srcs = [
        "hlo_poplar_dataflow_analysis.cc",
    ],
    hdrs = [
        "hlo_poplar_dataflow_analysis.h",
    ],
    deps = [
        ":feed_info",
        ":hlo_poplar_buffer",
        ":input_output_aliasing_map",
        ":pipeline_util",
        ":util",
        "//tensorflow/compiler/plugin/poplar/driver:compiler_annotations",
        "//tensorflow/compiler/plugin/poplar/driver/passes:allocation_finder",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:host_embedding",
    ],
)

poplar_cc_library(
    name = "find_all_users",
    srcs = [
        "find_all_users.cc",
    ],
    hdrs = [
        "find_all_users.h",
    ],
    deps = [
        ":util",
    ],
)

poplar_cc_library(
    name = "isomorphic_functions_map",
    hdrs = [
        "isomorphic_functions_map.h",
    ],
)

poplar_cc_library(
    name = "elementwise_cluster",
    srcs = [
        "elementwise_cluster.cc",
    ],
    hdrs = [
        "elementwise_cluster.h",
    ],
    deps = [
        ":matcher_predicates",
        ":offloading_util",
        ":pipeline_util",
        ":util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:all_gather",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:reduce_scatter",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:remote_parameter",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:replication_factor",
    ],
)

poplar_cc_library(
    name = "while_loop_util",
    srcs = [
        "while_loop_util.cc",
    ],
    hdrs = [
        "while_loop_util.h",
    ],
    deps = [
        ":matcher_predicates",
        ":util",
    ],
)

poplar_cc_library(
    name = "rnn_util",
    srcs = [
        "rnn_util.cc",
    ],
    hdrs = [
        "rnn_util.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:gru",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:lstm",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:rnn",
    ],
)

poplar_cc_library(
    name = "data_initializer",
    srcs = [
        "data_initializer.cc",
    ],
    hdrs = [
        "data_initializer.h",
    ],
    deps = [
        ":flags",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
    ],
)

poplar_cc_library(
    name = "xla_util",
    srcs = [
        "xla_util.cc",
    ],
    hdrs = [
        "xla_util.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/kernels:ops_cc_impl",
        "//tensorflow/compiler/xla/client:xla_builder",
    ],
)

poplar_cc_library(
    name = "popef_util",
    srcs = [
        "popef_util.cc",
    ],
    hdrs = [
        "popef_util.h",
    ],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/stream_executor/lib",
        "//third_party/eigen3",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf_headers",
        "@local_config_poplar//poplar:poplar_libs",
    ],
)

poplar_cc_library(
    name = "make_visitor",
    hdrs = [
        "make_visitor.h",
    ],
)

poplar_cc_library(
    name = "hlo_poplar_test_util",
    testonly = True,
    srcs = [
        "hlo_poplar_test_util.cc",
    ],
    hdrs = [
        "hlo_poplar_test_util.h",
    ],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@com_google_absl//absl/strings",
    ],
)

poplar_cc_library(
    name = "hlo_poplar_test_base",
    testonly = True,
    srcs = [
        "hlo_poplar_test_base.cc",
    ],
    hdrs = [
        "hlo_poplar_test_base.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:poplar_executor",
        "//tensorflow/compiler/plugin/poplar/driver/visitors:entry_visitor",
        "//tensorflow/compiler/xla/service:hlo_memory_scheduler",
        "//tensorflow/compiler/xla/service:hlo_runner",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
    ],
)

poplar_cc_library(
    name = "attributes_utils",
    hdrs = [
        "attributes_utils.h",
    ],
)

poplar_cc_library(
    name = "poplar_range_sampler",
    srcs = [
        "poplar_range_sampler.cc",
    ],
    hdrs = [
        "poplar_range_sampler.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
    ],
)

poplar_cc_library(
    name = "tools",
    deps = [
        ":alias_info_cc_impl",
        ":attributes_utils",
        ":conv_poplar_util",
        ":conv_util",
        ":conversions",
        ":data_initializer",
        ":debug_info",
        ":elementwise_cluster",
        ":execution_counter_util",
        ":extension_registry",
        ":feed_info",
        ":find_all_users",
        ":flags",
        ":generic_graph_caching",
        ":hash",
        ":hlo_hash",
        ":hlo_instruction_extensions",
        ":hlo_matcher",
        ":hlo_poplar_buffer",
        ":hlo_poplar_buffer_util",
        ":hlo_poplar_dataflow_analysis",
        ":infeed_allocator",
        ":infeed_iterator",
        ":inplace_util",
        ":input_output_aliasing_map",
        ":instruction_colocator_helper",
        ":io_thread",
        ":isomorphic_functions_map",
        ":make_visitor",
        ":mapping_helper",
        ":matcher_predicates",
        ":matmul_util",
        ":meta_graph",
        ":ml_type_helper",
        ":offloading_util",
        ":pipeline_util",
        ":popef_util",
        ":poplar_executable_binary_file",
        ":poplar_range_sampler",
        ":poplar_replica_groups",
        ":poplar_util",
        ":progress_bar",
        ":reduction_util",
        ":replica_identical_dataflow_analysis",
        ":rnn_util",
        ":seed_generator",
        ":single_hlo_matcher",
        ":subcomputation_graph_caching",
        ":tensor_location",
        ":tensor_map",
        ":tracepoint",
        ":util",
        ":while_loop_util",
        ":xla_util",
    ],
)
