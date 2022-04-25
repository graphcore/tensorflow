load("//tensorflow/compiler/plugin/poplar:poplar.bzl", "poplar_cc_library")

licenses(["restricted"])

package(default_visibility = ["//tensorflow/compiler/plugin/poplar:__subpackages__"])

poplar_cc_library(
    name = "add_block_recompute",
    srcs = [
        "add_block_recompute.cc",
    ],
    hdrs = [
        "add_block_recompute.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_matcher",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:recompute",
    ],
)

poplar_cc_library(
    name = "add_stochastic_rounding_options",
    srcs = [
        "add_stochastic_rounding_options.cc",
    ],
    hdrs = [
        "add_stochastic_rounding_options.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_matcher",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:recompute",
    ],
)

poplar_cc_library(
    name = "all_reduce_simplifier",
    srcs = [
        "all_reduce_simplifier.cc",
    ],
    hdrs = [
        "all_reduce_simplifier.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_matcher",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:all_gather",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:multi_slice",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:replication_factor",
    ],
)

poplar_cc_library(
    name = "all_to_all_finder",
    srcs = [
        "all_to_all_finder.cc",
    ],
    hdrs = [
        "all_to_all_finder.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_matcher",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:all_gather",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:multi_slice",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:replication_factor",
    ],
)

poplar_cc_library(
    name = "allocation_analysis",
    srcs = [
        "allocation_analysis.cc",
    ],
    hdrs = [
        "allocation_analysis.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:print_util",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "allocation_finder",
    srcs = [
        "allocation_finder.cc",
    ],
    hdrs = [
        "allocation_finder.h",
        "//tensorflow/compiler/plugin/poplar/driver:compiler_annotations.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:feed_info",
        "//tensorflow/compiler/plugin/poplar/driver/tools:input_output_aliasing_map",
        "//tensorflow/compiler/plugin/poplar/driver/tools:ml_type_helper",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:hlo_remote_buffer_info",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:host_embedding",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:remap_deduce",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:remote_parameter",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:user_op_hlo",
    ],
)

poplar_cc_library(
    name = "apply_recompute_suggestion",
    srcs = [
        "apply_recompute_suggestion.cc",
    ],
    hdrs = [
        "apply_recompute_suggestion.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_matcher",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:recompute",
    ],
)

poplar_cc_library(
    name = "call_optimizer",
    srcs = [
        "call_optimizer.cc",
    ],
    hdrs = [
        "call_optimizer.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:pipeline_util",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "casts_elimination",
    srcs = [
        "casts_elimination.cc",
    ],
    hdrs = [
        "casts_elimination.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_matcher",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "combine_instructions",
    srcs = [
        "combine_instructions.cc",
    ],
    hdrs = [
        "combine_instructions.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:instruction_colocator_helper",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "commutative_instruction_reorder_operands",
    srcs = [
        "commutative_instruction_reorder_operands.cc",
    ],
    hdrs = [
        "commutative_instruction_reorder_operands.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:matcher_predicates",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:scaled_inplace",
    ],
)

poplar_cc_library(
    name = "computation_flattener",
    srcs = [
        "computation_flattener.cc",
    ],
    hdrs = [
        "computation_flattener.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:pipeline_util",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "constant_nan",
    srcs = [
        "constant_nan.cc",
    ],
    hdrs = [
        "constant_nan.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "constant_slice_folding",
    srcs = [
        "constant_slice_folding.cc",
    ],
    hdrs = [
        "constant_slice_folding.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "conv_bwd_input_to_fwd_weights_transpose",
    srcs = [
        "conv_bwd_input_to_fwd_weights_transpose.cc",
    ],
    hdrs = [
        "conv_bwd_input_to_fwd_weights_transpose.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_matcher",
        "//tensorflow/compiler/plugin/poplar/driver/tools:ml_type_helper",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:multi_slice",
    ],
)

poplar_cc_library(
    name = "convolution_classifier",
    srcs = [
        "convolution_classifier.cc",
    ],
    hdrs = [
        "convolution_classifier.h",
    ],
    deps = [
        ":allocation_finder",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:fifo",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:multi_conv",
    ],
)

poplar_cc_library(
    name = "copy_inserter",
    srcs = [
        "copy_inserter.cc",
    ],
    hdrs = [
        "copy_inserter.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "custom_op_replacer",
    srcs = [
        "custom_op_replacer.cc",
    ],
    hdrs = [
        "custom_op_replacer.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_matcher",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:hlo_poplar_instruction",
    ],
)

poplar_cc_library(
    name = "dead_control_dependencies_elimination",
    srcs = [
        "dead_control_dependencies_elimination.cc",
    ],
    hdrs = [
        "dead_control_dependencies_elimination.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "dependency_replacer",
    srcs = [
        "dependency_replacer.cc",
    ],
    hdrs = [
        "dependency_replacer.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "distributed_batch_norm_decomposer",
    srcs = [
        "distributed_batch_norm_decomposer.cc",
        # "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:batch_norm_stats.cc",
    ],
    hdrs = [
        "distributed_batch_norm_decomposer.h",
    ],
    deps = [
        ":pipeline_recomputation",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:norm",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:recompute",
    ],
)

poplar_cc_library(
    name = "dynamic_slice_replacer",
    srcs = [
        "dynamic_slice_replacer.cc",
    ],
    hdrs = [
        "dynamic_slice_replacer.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:slice_util",
    ],
)

poplar_cc_library(
    name = "elementwise_broadcast_converter",
    srcs = [
        "elementwise_broadcast_converter.cc",
    ],
    hdrs = [
        "elementwise_broadcast_converter.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "elementwise_simplifier",
    srcs = [
        "elementwise_simplifier.cc",
    ],
    hdrs = [
        "elementwise_simplifier.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:elementwise",
    ],
)

poplar_cc_library(
    name = "embeddings_gradient_optimizer",
    srcs = [
        "embeddings_gradient_optimizer.cc",
    ],
    hdrs = [
        "embeddings_gradient_optimizer.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:execution_counter",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:hlo_remote_buffer_info",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:multi_slice",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:stateful_gradient_accumulate",
    ],
)

poplar_cc_library(
    name = "expression_outliner",
    srcs = [
        "expression_outliner.cc",
    ],
    hdrs = [
        "expression_outliner.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "f16_constant_folding",
    srcs = [
        "f16_constant_folding.cc",
    ],
    hdrs = [
        "f16_constant_folding.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "fix_root_instruction",
    srcs = [
        "fix_root_instruction.cc",
    ],
    hdrs = [
        "fix_root_instruction.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_matcher",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "forward_allocation",
    srcs = [
        "forward_allocation.cc",
    ],
    hdrs = [
        "forward_allocation.h",
    ],
    deps = [
        ":allocation_finder",
        "//tensorflow/compiler/plugin/poplar/driver:compiler_annotations",
        "//tensorflow/compiler/plugin/poplar/driver/tools:feed_info",
        "//tensorflow/compiler/plugin/poplar/driver/tools:input_output_aliasing_map",
        "//tensorflow/compiler/plugin/poplar/driver/tools:meta_graph",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:hlo_remote_buffer_info",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:host_embedding",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:remap_deduce",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:remote_parameter",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:user_op_hlo",
    ],
)

poplar_cc_library(
    name = "function_combiner",
    srcs = [
        "function_combiner.cc",
    ],
    hdrs = [
        "function_combiner.h",
    ],
    deps = [
        ":outline_remote_buffers",
        "//tensorflow/compiler/plugin/poplar/driver/tools:isomorphic_functions_map",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:hlo_remote_buffer_info",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:remote_parameter",
    ],
)

poplar_cc_library(
    name = "function_optimizer",
    srcs = [
        "function_optimizer.cc",
    ],
    hdrs = [
        "function_optimizer.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "fuse_ops_early",
    srcs = [
        "fuse_ops_early.cc",
    ],
    hdrs = [
        "fuse_ops_early.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_matcher",
        "//tensorflow/compiler/plugin/poplar/driver/tools:single_hlo_matcher",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "fuse_ops_into_poplar_ops",
    srcs = [
        "fuse_ops_into_poplar_ops.cc",
    ],
    hdrs = [
        "fuse_ops_into_poplar_ops.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_matcher",
        "//tensorflow/compiler/plugin/poplar/driver/tools:single_hlo_matcher",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:elementwise",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:non_linearity",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:scaled_inplace",
    ],
)

poplar_cc_library(
    name = "fuse_ops_late",
    srcs = [
        "fuse_ops_late.cc",
    ],
    hdrs = [
        "fuse_ops_late.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_matcher",
        "//tensorflow/compiler/plugin/poplar/driver/tools:single_hlo_matcher",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:multi_slice",
    ],
)

poplar_cc_library(
    name = "fuse_wide_const",
    srcs = [
        "fuse_wide_const.cc",
    ],
    hdrs = [
        "fuse_wide_const.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_matcher",
        "//tensorflow/compiler/plugin/poplar/driver/tools:single_hlo_matcher",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "fusion_inliner",
    srcs = [
        "fusion_inliner.cc",
    ],
    hdrs = [
        "fusion_inliner.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "gather_simplifier",
    srcs = [
        "gather_simplifier.cc",
    ],
    hdrs = [
        "gather_simplifier.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_matcher",
        "//tensorflow/compiler/plugin/poplar/driver/tools:slice_util",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:multi_slice",
    ],
)

poplar_cc_library(
    name = "gradient_accumulation_buffers_offload",
    srcs = [
        "gradient_accumulation_buffers_offload.cc",
    ],
    hdrs = [
        "gradient_accumulation_buffers_offload.h",
    ],
    deps = [
        ":allocation_finder",
        ":pipeline_optimizer",
        "//tensorflow/compiler/plugin/poplar/driver/tools:feed_info",
        "//tensorflow/compiler/plugin/poplar/driver/tools:input_output_aliasing_map",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:hlo_remote_buffer_info",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:host_embedding",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:remote_parameter",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:stateful_gradient_accumulate",
    ],
)

poplar_cc_library(
    name = "gradient_accumulation_fuser",
    srcs = [
        "gradient_accumulation_fuser.cc",
    ],
    hdrs = [
        "gradient_accumulation_fuser.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_matcher",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:hlo_remote_buffer_info",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:replication_factor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:stateful_gradient_accumulate",
    ],
)

poplar_cc_library(
    name = "gradient_accumulation_verifier",
    srcs = [
        "gradient_accumulation_verifier.cc",
    ],
    hdrs = [
        "gradient_accumulation_verifier.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:offloading_util",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:hlo_remote_buffer_info",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:stateful_gradient_accumulate",
    ],
)

poplar_cc_library(
    name = "hlo_computation_name_uniquify",
    srcs = [
        "hlo_computation_name_uniquify.cc",
    ],
    hdrs = [
        "hlo_computation_name_uniquify.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "host_compute_barrier_inserter",
    srcs = [
        "host_compute_barrier_inserter.cc",
    ],
    hdrs = [
        "host_compute_barrier_inserter.h",
    ],
    deps = [
        ":host_compute_util",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:send_recv_barrier",
    ],
)

poplar_cc_library(
    name = "host_compute_schedule_optimizer",
    srcs = [
        "host_compute_schedule_optimizer.cc",
    ],
    hdrs = [
        "host_compute_schedule_optimizer.h",
    ],
    deps = [
        ":host_compute_util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:recv_from_host",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:send_to_host",
    ],
)

poplar_cc_library(
    name = "host_compute_util",
    srcs = [
        "host_compute_util.cc",
    ],
    hdrs = [
        "host_compute_util.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:send_recv_barrier",
    ],
)

poplar_cc_library(
    name = "host_embedding_notification",
    srcs = [
        "host_embedding_notification.cc",
    ],
    hdrs = [
        "host_embedding_notification.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:host_embedding",
    ],
)

poplar_cc_library(
    name = "inplace_finder",
    srcs = [
        "inplace_finder.cc",
    ],
    hdrs = [
        "inplace_finder.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_poplar_dataflow_analysis",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "inter_ipu_copy_inserter",
    srcs = [
        "inter_ipu_copy_inserter.cc",
    ],
    hdrs = [
        "inter_ipu_copy_inserter.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:find_all_users",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "inter_tileset_copy_inserter",
    srcs = [
        "inter_tileset_copy_inserter.cc",
    ],
    hdrs = [
        "inter_tileset_copy_inserter.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:hlo_remote_buffer_info",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:inter_tileset_copy",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:remote_parameter",
    ],
)

poplar_cc_library(
    name = "io_tiles_placer",
    srcs = [
        "io_tiles_placer.cc",
    ],
    hdrs = [
        "io_tiles_placer.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:hlo_remote_buffer_info",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:remote_parameter",
    ],
)

poplar_cc_library(
    name = "lift_recompute_suggestion",
    srcs = [
        "lift_recompute_suggestion.cc",
    ],
    hdrs = [
        "lift_recompute_suggestion.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_matcher",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:recompute",
    ],
)

poplar_cc_library(
    name = "lower_frontend_attributes",
    srcs = [
        "lower_frontend_attributes.cc",
    ],
    hdrs = [
        "lower_frontend_attributes.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:pipeline_config_proto_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "mark_replica_identical_instructions",
    srcs = [
        "mark_replica_identical_instructions.cc",
    ],
    hdrs = [
        "mark_replica_identical_instructions.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:replica_identical_dataflow_analysis",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "matmul_combiner",
    srcs = [
        "matmul_combiner.cc",
    ],
    hdrs = [
        "matmul_combiner.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_matcher",
        "//tensorflow/compiler/plugin/poplar/driver/tools:matmul_util",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "module_flatten",
    srcs = [
        "module_flatten.cc",
    ],
    hdrs = [
        "module_flatten.h",
    ],
    deps = [
        ":allocation_finder",
        "//tensorflow/compiler/plugin/poplar/driver:compiler_annotations",
        "//tensorflow/compiler/plugin/poplar/driver/tools:feed_info",
        "//tensorflow/compiler/plugin/poplar/driver/tools:input_output_aliasing_map",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:host_embedding",
    ],
)

poplar_cc_library(
    name = "multi_conv_fixer",
    srcs = [
        "multi_conv_fixer.cc",
    ],
    hdrs = [
        "multi_conv_fixer.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:multi_conv",
    ],
)

poplar_cc_library(
    name = "multi_slice_combiner",
    srcs = [
        "multi_slice_combiner.cc",
    ],
    hdrs = [
        "multi_slice_combiner.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_matcher",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:multi_slice",
    ],
)

poplar_cc_library(
    name = "multi_slice_simplifier",
    srcs = [
        "multi_slice_simplifier.cc",
    ],
    hdrs = [
        "multi_slice_simplifier.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_matcher",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:multi_slice",
    ],
)

poplar_cc_library(
    name = "multi_update_apply",
    srcs = [
        "multi_update_apply.cc",
    ],
    hdrs = [
        "multi_update_apply.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_matcher",
        "//tensorflow/compiler/plugin/poplar/driver/tools:poplar_replica_groups",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:all_gather",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:multi_slice",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:replication_factor",
    ],
)

poplar_cc_library(
    name = "multi_update_combiner",
    srcs = [
        "multi_update_combiner.cc",
    ],
    hdrs = [
        "multi_update_combiner.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_matcher",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:multi_slice",
    ],
)

poplar_cc_library(
    name = "multi_update_scale_apply",
    srcs = [
        "multi_update_scale_apply.cc",
    ],
    hdrs = [
        "multi_update_scale_apply.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_matcher",
        "//tensorflow/compiler/plugin/poplar/driver/tools:poplar_replica_groups",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:all_gather",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:multi_slice",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:replication_factor",
    ],
)

poplar_cc_library(
    name = "multi_use_feeds_finder",
    srcs = [
        "multi_use_feeds_finder.cc",
    ],
    hdrs = [
        "multi_use_feeds_finder.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "not_supported_gather_expander",
    srcs = [
        "not_supported_gather_expander.cc",
    ],
    hdrs = [
        "not_supported_gather_expander.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/xla/service:gather_expander",
    ],
)

poplar_cc_library(
    name = "not_supported_scatter_expander",
    srcs = [
        "not_supported_scatter_expander.cc",
    ],
    hdrs = [
        "not_supported_scatter_expander.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/xla/service:scatter_expander",
    ],
)

poplar_cc_library(
    name = "outline_remote_buffers",
    srcs = [
        "outline_remote_buffers.cc",
    ],
    hdrs = [
        "outline_remote_buffers.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:isomorphic_functions_map",
        "//tensorflow/compiler/plugin/poplar/driver/tools:offloading_util",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:hlo_remote_buffer_info",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:remote_parameter",
    ],
)

poplar_cc_library(
    name = "parse_poplar_backend_config",
    srcs = [
        "parse_poplar_backend_config.cc",
    ],
    hdrs = [
        "parse_poplar_backend_config.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "pipeline_batch_serialization_buffer_inserter",
    srcs = [
        "pipeline_batch_serialization_buffer_inserter.cc",
    ],
    hdrs = [
        "pipeline_batch_serialization_buffer_inserter.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:execution_counter",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:hlo_remote_buffer_info",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:remote_parameter",
    ],
)

poplar_cc_library(
    name = "pipeline_batch_serialization_loop_inserter",
    srcs = [
        "pipeline_batch_serialization_loop_inserter.cc",
    ],
    hdrs = [
        "pipeline_batch_serialization_loop_inserter.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:offloading_util",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "pipeline_communication_optimizer",
    srcs = [
        "pipeline_communication_optimizer.cc",
    ],
    hdrs = [
        "pipeline_communication_optimizer.h",
    ],
    deps = [
        ":pipeline_fifo_inserter",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "pipeline_control_dependency_inserter",
    srcs = [
        "pipeline_control_dependency_inserter.cc",
    ],
    hdrs = [
        "pipeline_control_dependency_inserter.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:hlo_remote_buffer_info",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:stateful_gradient_accumulate",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:stateful_noop",
        "//tensorflow/compiler/xla/service:tuple_simplifier",
    ],
)

poplar_cc_library(
    name = "pipeline_copy_inserter",
    srcs = [
        "pipeline_copy_inserter.cc",
    ],
    hdrs = [
        "pipeline_copy_inserter.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "pipeline_feed_hoisting",
    srcs = [
        "pipeline_feed_hoisting.cc",
    ],
    hdrs = [
        "pipeline_feed_hoisting.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:fifo",
    ],
)

poplar_cc_library(
    name = "pipeline_fifo_inserter",
    srcs = [
        "pipeline_fifo_inserter.cc",
    ],
    hdrs = [
        "pipeline_fifo_inserter.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:fifo",
    ],
)

poplar_cc_library(
    name = "pipeline_fixer",
    srcs = [
        "pipeline_fixer.cc",
    ],
    hdrs = [
        "pipeline_fixer.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:hlo_remote_buffer_info",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:stateful_gradient_accumulate",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:stateful_noop",
        "//tensorflow/compiler/xla/service:tuple_simplifier",
    ],
)

poplar_cc_library(
    name = "pipeline_gradient_accumulation_optimizer",
    srcs = [
        "pipeline_gradient_accumulation_optimizer.cc",
    ],
    hdrs = [
        "pipeline_gradient_accumulation_optimizer.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:hlo_remote_buffer_info",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:stateful_gradient_accumulate",
    ],
)

poplar_cc_library(
    name = "pipeline_optimizer",
    srcs = [
        "pipeline_optimizer.cc",
    ],
    hdrs = [
        "pipeline_optimizer.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "pipeline_recomputation",
    srcs = [
        "pipeline_recomputation.cc",
    ],
    hdrs = [
        "pipeline_recomputation.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:recompute",
    ],
)

poplar_cc_library(
    name = "pipeline_recomputation_stage_inserter",
    srcs = [
        "pipeline_recomputation_stage_inserter.cc",
    ],
    hdrs = [
        "pipeline_recomputation_stage_inserter.h",
    ],
    deps = [
        ":pipeline_fifo_inserter",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:fifo",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:stateful_noop",
    ],
)

poplar_cc_library(
    name = "pipeline_stage_merger",
    srcs = [
        "pipeline_stage_merger.cc",
    ],
    hdrs = [
        "pipeline_stage_merger.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:stateful_noop",
        "//tensorflow/compiler/xla/service:tuple_simplifier",
    ],
)

poplar_cc_library(
    name = "pipeline_tuple_remover",
    srcs = [
        "pipeline_tuple_remover.cc",
    ],
    hdrs = [
        "pipeline_tuple_remover.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/xla/service:tuple_simplifier",
    ],
)

poplar_cc_library(
    name = "pipeline_verifier",
    srcs = [
        "pipeline_verifier.cc",
    ],
    hdrs = [
        "pipeline_verifier.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "poplar_algebraic_simplifier",
    srcs = [
        "poplar_algebraic_simplifier.cc",
        "poplar_algebraic_simplifier/poplar_algebraic_simplifier_convolution.cc",
        "poplar_algebraic_simplifier/poplar_algebraic_simplifier_dot.cc",
        "poplar_algebraic_simplifier/poplar_algebraic_simplifier_utils.cc",
    ],
    hdrs = [
        "poplar_algebraic_simplifier.h",
        "poplar_algebraic_simplifier/poplar_algebraic_simplifier_convolution.h",
        "poplar_algebraic_simplifier/poplar_algebraic_simplifier_dot.h",
        "poplar_algebraic_simplifier/poplar_algebraic_simplifier_utils.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:arg_min_max",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:hlo_remote_buffer_info",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:stateful_gradient_accumulate",
    ],
)

poplar_cc_library(
    name = "post_serialize_gradient_accumulation",
    srcs = [
        "post_serialize_gradient_accumulation.cc",
    ],
    hdrs = [
        "post_serialize_gradient_accumulation.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "recomputation_checkpoint_remover",
    srcs = [
        "recomputation_checkpoint_remover.cc",
    ],
    hdrs = [
        "recomputation_checkpoint_remover.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "recomputation_input_remover",
    srcs = [
        "recomputation_input_remover.cc",
    ],
    hdrs = [
        "recomputation_input_remover.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "recompute_casts",
    srcs = [
        "recompute_casts.cc",
    ],
    hdrs = [
        "recompute_casts.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "recompute_instructions",
    srcs = [
        "recompute_instructions.cc",
    ],
    hdrs = [
        "recompute_instructions.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:meta_graph",
        "//tensorflow/compiler/plugin/poplar/driver/tools:ml_type_helper",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "redundant_triangular_mask_remover",
    srcs = [
        "redundant_triangular_mask_remover.cc",
    ],
    hdrs = [
        "redundant_triangular_mask_remover.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_matcher",
    ],
)

poplar_cc_library(
    name = "remote_buffer_canonicalizer",
    srcs = [
        "remote_buffer_canonicalizer.cc",
    ],
    hdrs = [
        "remote_buffer_canonicalizer.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_matcher",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "remote_buffer_merger",
    srcs = [
        "remote_buffer_merger.cc",
    ],
    hdrs = [
        "remote_buffer_merger.h",
    ],
    deps = [
        ":allocation_finder",
        "//tensorflow/compiler/plugin/poplar/driver/tools:feed_info",
        "//tensorflow/compiler/plugin/poplar/driver/tools:input_output_aliasing_map",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:hlo_remote_buffer_info",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:host_embedding",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:remote_parameter",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:stateful_gradient_accumulate",
    ],
)

poplar_cc_library(
    name = "remote_parameter_parallel_combiner",
    srcs = [
        "remote_parameter_parallel_combiner.cc",
    ],
    hdrs = [
        "remote_parameter_parallel_combiner.h",
    ],
    deps = [
        ":allocation_finder",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:hlo_remote_buffer_info",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:remote_parameter",
    ],
)

poplar_cc_library(
    name = "remove_blocked_recompute_suggestions",
    srcs = [
        "remove_blocked_recompute_suggestions.cc",
    ],
    hdrs = [
        "remove_blocked_recompute_suggestions.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_matcher",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "remove_recompute_suggestions",
    srcs = [
        "remove_recompute_suggestions.cc",
    ],
    hdrs = [
        "remove_recompute_suggestions.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_matcher",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "repeat_loop_copy_inserter",
    srcs = [
        "repeat_loop_copy_inserter.cc",
    ],
    hdrs = [
        "repeat_loop_copy_inserter.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "replicated_resource_update_elementwise_clustering",
    srcs = [
        "replicated_resource_update_elementwise_clustering.cc",
    ],
    hdrs = [
        "replicated_resource_update_elementwise_clustering.h",
    ],
    deps = [
        ":allocation_finder",
        ":resource_update_elementwise_clustering",
        "//tensorflow/compiler/plugin/poplar/driver:compiler_annotations",
        "//tensorflow/compiler/plugin/poplar/driver/tools:elementwise_cluster",
        "//tensorflow/compiler/plugin/poplar/driver/tools:feed_info",
        "//tensorflow/compiler/plugin/poplar/driver/tools:input_output_aliasing_map",
        "//tensorflow/compiler/plugin/poplar/driver/tools:replica_identical_dataflow_analysis",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:collective_reorder",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:host_embedding",
    ],
)

poplar_cc_library(
    name = "replication_factor_to_constant",
    srcs = [
        "replication_factor_to_constant.cc",
    ],
    hdrs = [
        "replication_factor_to_constant.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:replication_factor",
    ],
)

poplar_cc_library(
    name = "resource_update_copy_inserter",
    srcs = [
        "resource_update_copy_inserter.cc",
    ],
    hdrs = [
        "resource_update_copy_inserter.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:copy_into",
    ],
)

poplar_cc_library(
    name = "resource_update_elementwise_clustering",
    srcs = [
        "resource_update_elementwise_clustering.cc",
    ],
    hdrs = [
        "resource_update_elementwise_clustering.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:elementwise_cluster",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "resource_update_fixer",
    srcs = [
        "resource_update_fixer.cc",
    ],
    hdrs = [
        "resource_update_fixer.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "resource_update_merger",
    srcs = [
        "resource_update_merger.cc",
    ],
    hdrs = [
        "resource_update_merger.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "resource_update_schedule_optimizer",
    srcs = [
        "resource_update_schedule_optimizer.cc",
    ],
    hdrs = [
        "resource_update_schedule_optimizer.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "root_token_replacer",
    srcs = [
        "root_token_replacer.cc",
    ],
    hdrs = [
        "root_token_replacer.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "scatter_simplifier",
    srcs = [
        "scatter_simplifier.cc",
    ],
    hdrs = [
        "scatter_simplifier.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_matcher",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:multi_slice",
    ],
)

poplar_cc_library(
    name = "seed_hoisting",
    srcs = [
        "seed_hoisting.cc",
    ],
    hdrs = [
        "seed_hoisting.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:execution_counter",
    ],
)

poplar_cc_library(
    name = "serialize_gradient_accumulation",
    srcs = [
        "serialize_gradient_accumulation.cc",
    ],
    hdrs = [
        "serialize_gradient_accumulation.h",
    ],
    deps = [
        ":slice_optimizer",
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_matcher",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:hlo_remote_buffer_info",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:scaled_inplace",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:stateful_gradient_accumulate",
    ],
)

poplar_cc_library(
    name = "sharding_pass",
    srcs = [
        "sharding_pass.cc",
    ],
    hdrs = [
        "sharding_pass.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:find_all_users",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "slice_optimizer",
    srcs = [
        "slice_optimizer.cc",
    ],
    hdrs = [
        "slice_optimizer.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_matcher",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:slice_apply",
    ],
)

poplar_cc_library(
    name = "suggest_recompute",
    srcs = [
        "suggest_recompute.cc",
    ],
    hdrs = [
        "suggest_recompute.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_matcher",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:recompute",
    ],
)

poplar_cc_library(
    name = "variables_offload_and_partition",
    srcs = [
        "variables_offload_and_partition.cc",
    ],
    hdrs = [
        "variables_offload_and_partition.h",
    ],
    deps = [
        ":allocation_finder",
        ":pipeline_optimizer",
        "//tensorflow/compiler/plugin/poplar/driver/tools:poplar_replica_groups",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:all_gather",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:reduce_scatter",
    ],
)

poplar_cc_library(
    name = "while_loop_condition_simplify",
    srcs = [
        "while_loop_condition_simplify.cc",
    ],
    hdrs = [
        "while_loop_condition_simplify.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools:while_loop_util",
    ],
)

poplar_cc_library(
    name = "while_loop_optimiser",
    srcs = [
        "while_loop_optimiser.cc",
    ],
    hdrs = [
        "while_loop_optimiser.h",
    ],
    deps = [
        ":allocation_analysis",
        "//tensorflow/compiler/plugin/poplar/driver/tools:print_util",
        "//tensorflow/compiler/plugin/poplar/driver/tools:slice_util",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools:while_loop_util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops",
        "//tensorflow/compiler/xla/service:while_loop_analysis",
        "//tensorflow/compiler/xla/service:while_loop_invariant_code_motion",
    ],
)

poplar_cc_library(
    name = "while_loop_to_repeat_simplify",
    srcs = [
        "while_loop_to_repeat_simplify.cc",
    ],
    hdrs = [
        "while_loop_to_repeat_simplify.h",
    ],
    deps = [
        ":allocation_finder",
        "//tensorflow/compiler/plugin/poplar/driver:compiler_annotations",
        "//tensorflow/compiler/plugin/poplar/driver/tools:feed_info",
        "//tensorflow/compiler/plugin/poplar/driver/tools:input_output_aliasing_map",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools:while_loop_util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:host_embedding",
        "//tensorflow/compiler/xla/service:tuple_simplifier",
        "//tensorflow/compiler/xla/service:while_loop_analysis",
    ],
)

poplar_cc_library(
    name = "wide_const_finder",
    srcs = [
        "wide_const_finder.cc",
    ],
    hdrs = [
        "wide_const_finder.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
    ],
)

poplar_cc_library(
    name = "passes",
    deps = [
        ":add_block_recompute",
        ":add_stochastic_rounding_options",
        ":all_reduce_simplifier",
        ":all_to_all_finder",
        ":allocation_analysis",
        ":allocation_finder",
        ":apply_recompute_suggestion",
        ":call_optimizer",
        ":casts_elimination",
        ":combine_instructions",
        ":commutative_instruction_reorder_operands",
        ":computation_flattener",
        ":constant_nan",
        ":constant_slice_folding",
        ":conv_bwd_input_to_fwd_weights_transpose",
        ":convolution_classifier",
        ":copy_inserter",
        ":custom_op_replacer",
        ":dead_control_dependencies_elimination",
        ":dependency_replacer",
        ":distributed_batch_norm_decomposer",
        ":dynamic_slice_replacer",
        ":elementwise_broadcast_converter",
        ":elementwise_simplifier",
        ":embeddings_gradient_optimizer",
        ":expression_outliner",
        ":f16_constant_folding",
        ":fix_root_instruction",
        ":forward_allocation",
        ":function_combiner",
        ":function_optimizer",
        ":fuse_ops_early",
        ":fuse_ops_into_poplar_ops",
        ":fuse_ops_late",
        ":fuse_wide_const",
        ":fusion_inliner",
        ":gather_simplifier",
        ":gradient_accumulation_buffers_offload",
        ":gradient_accumulation_fuser",
        ":gradient_accumulation_verifier",
        ":hlo_computation_name_uniquify",
        ":host_compute_barrier_inserter",
        ":host_compute_schedule_optimizer",
        ":host_compute_util",
        ":host_embedding_notification",
        ":inplace_finder",
        ":inter_ipu_copy_inserter",
        ":inter_tileset_copy_inserter",
        ":io_tiles_placer",
        ":lift_recompute_suggestion",
        ":lower_frontend_attributes",
        ":mark_replica_identical_instructions",
        ":matmul_combiner",
        ":module_flatten",
        ":multi_conv_fixer",
        ":multi_slice_combiner",
        ":multi_slice_simplifier",
        ":multi_update_apply",
        ":multi_update_combiner",
        ":multi_update_scale_apply",
        ":multi_use_feeds_finder",
        ":not_supported_gather_expander",
        ":not_supported_scatter_expander",
        ":outline_remote_buffers",
        ":parse_poplar_backend_config",
        ":pipeline_batch_serialization_buffer_inserter",
        ":pipeline_batch_serialization_loop_inserter",
        ":pipeline_communication_optimizer",
        ":pipeline_control_dependency_inserter",
        ":pipeline_copy_inserter",
        ":pipeline_feed_hoisting",
        ":pipeline_fifo_inserter",
        ":pipeline_fixer",
        ":pipeline_gradient_accumulation_optimizer",
        ":pipeline_optimizer",
        ":pipeline_recomputation",
        ":pipeline_recomputation_stage_inserter",
        ":pipeline_stage_merger",
        ":pipeline_tuple_remover",
        ":pipeline_verifier",
        ":poplar_algebraic_simplifier",
        ":post_serialize_gradient_accumulation",
        ":recomputation_checkpoint_remover",
        ":recomputation_input_remover",
        ":recompute_casts",
        ":recompute_instructions",
        ":redundant_triangular_mask_remover",
        ":remote_buffer_canonicalizer",
        ":remote_buffer_merger",
        ":remote_parameter_parallel_combiner",
        ":remove_blocked_recompute_suggestions",
        ":remove_recompute_suggestions",
        ":repeat_loop_copy_inserter",
        ":replicated_resource_update_elementwise_clustering",
        ":replication_factor_to_constant",
        ":resource_update_copy_inserter",
        ":resource_update_elementwise_clustering",
        ":resource_update_fixer",
        ":resource_update_merger",
        ":resource_update_schedule_optimizer",
        ":root_token_replacer",
        ":scatter_simplifier",
        ":seed_hoisting",
        ":serialize_gradient_accumulation",
        ":sharding_pass",
        ":slice_optimizer",
        ":suggest_recompute",
        ":variables_offload_and_partition",
        ":while_loop_condition_simplify",
        ":while_loop_optimiser",
        ":while_loop_to_repeat_simplify",
        ":wide_const_finder",
    ],
)
