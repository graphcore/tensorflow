licenses(["restricted"])

package(default_visibility = ["//tensorflow/compiler/plugin/poplar:__subpackages__"])

load("//tensorflow/compiler/xla/tests:build_defs.bzl", "xla_test")
load("//tensorflow/compiler/tests:build_defs.bzl", "tf_xla_py_test")
load("@local_config_poplar//poplar:build_defs.bzl", "tf_poplar_build_tag")
load("@local_config_ipu_horovod//:build_defs_horovod.bzl", "horovod_py_test", "if_horovod", "poprun_py_test")
load("//tensorflow:tensorflow.bzl", "py_binary", "tf_cc_binary", "tf_cc_test")
load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library")

cc_library(
    name = "tf_graph_sources",
    hdrs = [
        "//tensorflow/compiler/plugin/poplar/vertices:convert_tf_asm_graph_sources_to_literal",
        "//tensorflow/compiler/plugin/poplar/vertices:convert_tf_cpp_graph_sources_to_literal",
    ],
)

# Rules for generating a binary with the Poplar package hash in it.
cc_binary(
    name = "poplar_version_generator",
    srcs = ["poplar_version_generator.cc"],
    deps = [
        "@local_config_poplar//poplar:poplar_libs",
    ],
)

genrule(
    name = "generate_poplar_version",
    outs = ["driver/poplar_version.cc"],
    cmd = "./$(location :poplar_version_generator) > $@",
    tools = [":poplar_version_generator"],
)

cc_library(
    name = "poplar_version",
    srcs = [":generate_poplar_version"],
)

# Rule for generating the custom codelet example
genrule(
    name = "custom_codelet_example",
    srcs = ["docs/custom_codelet.cpp"],
    outs = ["custom_codelet.gp"],
    cmd = "$(location @local_config_poplar//poplar:popc) -DNDEBUG -O3 -I . -o $@ $<",
    tools = ["@local_config_poplar//poplar:popc"],
)

tf_proto_library(
    name = "option_flag",
    srcs = [
        "driver/option_flag.proto",
    ],
    cc_api_version = 2,
    j2objc_api_version = 1,
    make_default_target_header_only = True,
)

tf_proto_library(
    name = "threestate",
    srcs = [
        "driver/threestate.proto",
    ],
    cc_api_version = 2,
    j2objc_api_version = 1,
    make_default_target_header_only = True,
)

tf_proto_library(
    name = "pipeline_config",
    srcs = [
        "driver/pipeline_config.proto",
    ],
    cc_api_version = 2,
    j2objc_api_version = 1,
    make_default_target_header_only = True,
    protodeps = [
        ":option_flag",
    ],
)

tf_proto_library(
    name = "config_protos",
    srcs = [
        "driver/config.proto",
        "driver/trace.proto",
    ],
    cc_api_version = 2,
    j2objc_api_version = 1,
    make_default_target_header_only = True,
    protodeps = [
        ":option_flag",
        ":threestate",
    ],
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "alias_info",
    srcs = [
        "driver/tools/alias_info.proto",
    ],
    cc_api_version = 2,
    j2objc_api_version = 1,
    make_default_target_header_only = True,
)

tf_proto_library(
    name = "backend_protos",
    srcs = [
        "driver/backend_config.proto",
        "driver/poplar_executable.proto",
        "driver/poplar_feed_config.proto",
    ],
    protodeps = [
        ":alias_info",
        ":option_flag",
        ":pipeline_config",
        ":threestate",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:protos_all",
    ],
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "ops",
    srcs = ["kernels/ops.proto"],
    cc_api_version = 2,
    j2objc_api_version = 1,
    make_default_target_header_only = True,
)

tf_proto_library(
    name = "popfloat_config_protos",
    srcs = ["kernels/popfloat/gfloat_config_utils.proto"],
    cc_api_version = 2,
    j2objc_api_version = 1,
    make_default_target_header_only = True,
)

cc_library(
    name = "custom_kernels_util",
    srcs = [
        "kernels/custom_kernels_util.cc",
    ],
    hdrs = [
        "driver/tools/flags.h",
        "driver/tools/util.h",
        "kernels/custom_kernels_util.h",
    ],
    deps = [
        ":backend_protos_cc",
        ":config_protos_cc_impl",
        ":ops_cc_impl",
        ":popfloat_config_protos_cc_impl",
        "//tensorflow/compiler/xla:xla_headers_lib",
        "//tensorflow/compiler/xla/service:hlo",
        "@com_google_absl//absl/types:any",
        "@com_google_absl//absl/types:bad_any_cast",
        "@jsoncpp_git//:jsoncpp",
    ],
    alwayslink = True,
)

cc_library(
    name = "optimizers",
    srcs = [
        "driver/passes/add_stochastic_rounding_options.cc",
        "driver/passes/all_reduce_simplifier.cc",
        "driver/passes/all_to_all_finder.cc",
        "driver/passes/allocation_finder.cc",
        "driver/passes/add_block_recompute.cc",
        "driver/passes/apply_recompute_suggestion.cc",
        "driver/passes/call_optimizer.cc",
        "driver/passes/casts_elimination.cc",
        "driver/passes/combine_instructions.cc",
        "driver/passes/commutative_instruction_reorder_operands.cc",
        "driver/passes/computation_flattener.cc",
        "driver/passes/constant_nan.cc",
        "driver/passes/constant_slice_folding.cc",
        "driver/passes/conv_bwd_input_to_fwd_weights_transpose.cc",
        "driver/passes/convolution_classifier.cc",
        "driver/passes/copy_inserter.cc",
        "driver/passes/custom_op_replacer.cc",
        "driver/passes/dead_control_dependencies_elimination.cc",
        "driver/passes/dependency_replacer.cc",
        "driver/passes/distributed_batch_norm_decomposer.cc",
        "driver/passes/elementwise_broadcast_converter.cc",
        "driver/passes/elementwise_simplifier.cc",
        "driver/passes/embeddings_gradient_optimizer.cc",
        "driver/passes/expression_outliner.cc",
        "driver/passes/fix_root_instruction.cc",
        "driver/passes/f16_constant_folding.cc",
        "driver/passes/forward_allocation.cc",
        "driver/passes/function_combiner.cc",
        "driver/passes/function_optimizer.cc",
        "driver/passes/fuse_ops_early.cc",
        "driver/passes/fuse_ops_into_poplar_ops.cc",
        "driver/passes/fuse_ops_late.cc",
        "driver/passes/fuse_wide_const.cc",
        "driver/passes/fusion_inliner.cc",
        "driver/passes/gather_simplifier.cc",
        "driver/passes/gradient_accumulation_buffers_offload.cc",
        "driver/passes/gradient_accumulation_fuser.cc",
        "driver/passes/gradient_accumulation_verifier.cc",
        "driver/passes/hlo_computation_name_uniquify.cc",
        "driver/passes/host_compute_barrier_inserter.cc",
        "driver/passes/host_compute_schedule_optimizer.cc",
        "driver/passes/host_compute_util.cc",
        "driver/passes/host_embedding_notification.cc",
        "driver/passes/inplace_finder.cc",
        "driver/passes/inter_ipu_copy_inserter.cc",
        "driver/passes/inter_tileset_copy_inserter.cc",
        "driver/passes/io_tiles_placer.cc",
        "driver/passes/lift_recompute_suggestion.cc",
        "driver/passes/lower_frontend_attributes.cc",
        "driver/passes/mark_replica_identical_instructions.cc",
        "driver/passes/matmul_combiner.cc",
        "driver/passes/module_flatten.cc",
        "driver/passes/multi_conv_fixer.cc",
        "driver/passes/multi_slice_combiner.cc",
        "driver/passes/multi_update_apply.cc",
        "driver/passes/multi_update_combiner.cc",
        "driver/passes/multi_update_scale_apply.cc",
        "driver/passes/multi_use_feeds_finder.cc",
        "driver/passes/not_supported_gather_expander.cc",
        "driver/passes/not_supported_scatter_expander.cc",
        "driver/passes/outline_remote_buffers.cc",
        "driver/passes/parse_poplar_backend_config.cc",
        "driver/passes/pipeline_batch_serialization_buffer_inserter.cc",
        "driver/passes/pipeline_batch_serialization_loop_inserter.cc",
        "driver/passes/pipeline_communication_optimizer.cc",
        "driver/passes/pipeline_control_dependency_inserter.cc",
        "driver/passes/pipeline_copy_inserter.cc",
        "driver/passes/pipeline_feed_hoisting.cc",
        "driver/passes/pipeline_fifo_inserter.cc",
        "driver/passes/pipeline_fixer.cc",
        "driver/passes/pipeline_gradient_accumulation_optimizer.cc",
        "driver/passes/pipeline_optimizer.cc",
        "driver/passes/pipeline_recomputation.cc",
        "driver/passes/pipeline_recomputation_stage_inserter.cc",
        "driver/passes/pipeline_stage_merger.cc",
        "driver/passes/pipeline_tuple_remover.cc",
        "driver/passes/pipeline_verifier.cc",
        "driver/passes/variables_offload_and_partition.cc",
        "driver/passes/poplar_algebraic_simplifier.cc",
        "driver/passes/poplar_algebraic_simplifier/poplar_algebraic_simplifier_convolution.cc",
        "driver/passes/poplar_algebraic_simplifier/poplar_algebraic_simplifier_dot.cc",
        "driver/passes/poplar_algebraic_simplifier/poplar_algebraic_simplifier_utils.cc",
        "driver/passes/post_serialize_gradient_accumulation.cc",
        "driver/passes/recompute_casts.cc",
        "driver/passes/recompute_instructions.cc",
        "driver/passes/recomputation_checkpoint_remover.cc",
        "driver/passes/recomputation_input_remover.cc",
        "driver/passes/remote_buffer_canonicalizer.cc",
        "driver/passes/remote_buffer_merger.cc",
        "driver/passes/remote_parameter_parallel_combiner.cc",
        "driver/passes/remove_blocked_recompute_suggestions.cc",
        "driver/passes/remove_recompute_suggestions.cc",
        "driver/passes/repeat_loop_copy_inserter.cc",
        "driver/passes/replicated_resource_update_elementwise_clustering.cc",
        "driver/passes/replication_factor_to_constant.cc",
        "driver/passes/resource_update_copy_inserter.cc",
        "driver/passes/resource_update_elementwise_clustering.cc",
        "driver/passes/resource_update_fixer.cc",
        "driver/passes/resource_update_merger.cc",
        "driver/passes/resource_update_schedule_optimizer.cc",
        "driver/passes/root_token_replacer.cc",
        "driver/passes/scatter_simplifier.cc",
        "driver/passes/seed_hoisting.cc",
        "driver/passes/serialize_gradient_accumulation.cc",
        "driver/passes/sharding_pass.cc",
        "driver/passes/slice_optimizer.cc",
        "driver/passes/suggest_recompute.cc",
        "driver/passes/while_loop_condition_simplify.cc",
        "driver/passes/while_loop_to_repeat_simplify.cc",
        "driver/passes/wide_const_finder.cc",

        # DO NOT build any files which create a HloPoplarInstructionFactory before this point
        # See https://phabricator.sourcevertex.net/T8409 for further information.
        "driver/tools/custom_ops/hlo_poplar_instruction.cc",

        # Place HloPoplarInstructionFactory files below here.
        "driver/tools/conv_util.cc",
        "driver/tools/hlo_instruction_extensions.cc",
        "driver/tools/custom_ops/arg_min_max.cc",
        "driver/tools/custom_ops/all_gather.cc",
        "driver/tools/custom_ops/assert.cc",
        "driver/tools/custom_ops/assume_equal_across_replicas.cc",
        "driver/tools/custom_ops/batch_norm_stats.cc",
        "driver/tools/custom_ops/codelet_expression_op.cc",
        "driver/tools/custom_ops/collective_reorder.cc",
        "driver/tools/custom_ops/conv_with_reverse.cc",
        "driver/tools/custom_ops/dropout_hlo.cc",
        "driver/tools/custom_ops/fifo.cc",
        "driver/tools/custom_ops/cast_to_gfloat_hlo.cc",
        "driver/tools/custom_ops/copy_into.cc",
        "driver/tools/custom_ops/ctc_loss.cc",
        "driver/tools/custom_ops/group_norm_infer.cc",
        "driver/tools/custom_ops/group_norm_train.cc",
        "driver/tools/custom_ops/group_norm_grad.cc",
        "driver/tools/custom_ops/group_norm_stats.cc",
        "driver/tools/custom_ops/gru.cc",
        "driver/tools/custom_ops/histogram.cc",
        "driver/tools/custom_ops/elementwise.cc",
        "driver/tools/custom_ops/execution_counter.cc",
        "driver/tools/custom_ops/hlo_extensions.cc",
        "driver/tools/custom_ops/host_embedding.cc",
        "driver/tools/custom_ops/ipu_inter_copy.cc",
        "driver/tools/custom_ops/inter_tileset_copy.cc",
        "driver/tools/custom_ops/candidate_sampler.cc",
        "driver/tools/custom_ops/lstm.cc",
        "driver/tools/custom_ops/multi_conv.cc",
        "driver/tools/custom_ops/multi_slice.cc",
        "driver/tools/custom_ops/non_linearity.cc",
        "driver/tools/custom_ops/normalise_image.cc",
        "driver/tools/custom_ops/onehot.cc",
        "driver/tools/custom_ops/norm.cc",
        "driver/tools/custom_ops/pooling.cc",
        "driver/tools/custom_ops/print_tensor.cc",
        "driver/tools/custom_ops/recv_from_host.cc",
        "driver/tools/custom_ops/reduce_many.cc",
        "driver/tools/custom_ops/reduce_scatter.cc",
        "driver/tools/custom_ops/recompute.cc",
        "driver/tools/custom_ops/remap.cc",
        "driver/tools/custom_ops/remap_deduce.cc",
        "driver/tools/custom_ops/remote_parameter.cc",
        "driver/tools/custom_ops/replication_factor.cc",
        "driver/tools/custom_ops/rnn.cc",
        "driver/tools/custom_ops/scaled_inplace.cc",
        "driver/tools/custom_ops/seed.cc",
        "driver/tools/custom_ops/send_recv_barrier.cc",
        "driver/tools/custom_ops/send_to_host.cc",
        "driver/tools/custom_ops/sequence_slice.cc",
        "driver/tools/custom_ops/slice_apply.cc",
        "driver/tools/custom_ops/sparse.cc",
        "driver/tools/custom_ops/stateful_gradient_accumulate.cc",
        "driver/tools/custom_ops/stateful_noop.cc",
        "driver/tools/custom_ops/stateless_random.cc",
        "driver/tools/custom_ops/topk.cc",
        "driver/tools/custom_ops/truncated_normal.cc",
        "driver/tools/custom_ops/user_op_hlo.cc",
        "driver/tools/custom_ops/weights_transpose_chans_flip_xy.cc",
        "driver/tools/elementwise_cluster.cc",
        "driver/tools/feed_info.cc",
        "driver/tools/find_all_users.cc",
        "driver/tools/hlo_hash.cc",
        "driver/tools/hlo_matcher.cc",
        "driver/tools/hlo_poplar_buffer_util.cc",
        "driver/tools/hlo_poplar_buffer.cc",
        "driver/tools/hlo_poplar_dataflow_analysis.cc",
        "driver/tools/inplace_util.cc",
        "driver/tools/input_output_aliasing_map.cc",
        "driver/tools/instruction_colocator_helper.cc",
        "driver/tools/matcher_predicates.cc",
        "driver/tools/matmul_util.cc",
        "driver/tools/ml_type_helper.cc",
        "driver/tools/offloading_util.cc",
        "driver/tools/pipeline_util.cc",
        "driver/tools/poplar_replica_groups.cc",
        "driver/tools/replica_identical_dataflow_analysis.cc",
        "driver/tools/single_hlo_matcher.cc",
        "driver/tools/util.cc",
        "driver/tools/while_loop_util.cc",
        "driver/schedulers/clustering_scheduler.cc",
        "driver/schedulers/ipu_scheduler.cc",
        "driver/schedulers/liveness_look_ahead_scheduler.cc",
        "driver/schedulers/shortest_path_scheduler.cc",
        "driver/schedulers/sync_list_scheduler.cc",
    ],
    hdrs = [
        "driver/compiler_annotations.h",
        "driver/compiler_information.h",
        "driver/passes/add_block_recompute.h",
        "driver/passes/add_stochastic_rounding_options.h",
        "driver/passes/all_reduce_simplifier.h",
        "driver/passes/all_to_all_finder.h",
        "driver/passes/allocation_finder.h",
        "driver/passes/apply_recompute_suggestion.h",
        "driver/passes/call_optimizer.h",
        "driver/passes/casts_elimination.h",
        "driver/passes/combine_instructions.h",
        "driver/passes/commutative_instruction_reorder_operands.h",
        "driver/passes/computation_flattener.h",
        "driver/passes/constant_nan.h",
        "driver/passes/constant_slice_folding.h",
        "driver/passes/conv_bwd_input_to_fwd_weights_transpose.h",
        "driver/passes/convolution_classifier.h",
        "driver/passes/copy_inserter.h",
        "driver/passes/custom_op_replacer.h",
        "driver/passes/dead_control_dependencies_elimination.h",
        "driver/passes/dependency_replacer.h",
        "driver/passes/distributed_batch_norm_decomposer.h",
        "driver/passes/elementwise_broadcast_converter.h",
        "driver/passes/elementwise_simplifier.h",
        "driver/passes/embeddings_gradient_optimizer.h",
        "driver/passes/expression_outliner.h",
        "driver/passes/f16_constant_folding.h",
        "driver/passes/fix_root_instruction.h",
        "driver/passes/forward_allocation.h",
        "driver/passes/function_combiner.h",
        "driver/passes/function_optimizer.h",
        "driver/passes/fuse_ops_early.h",
        "driver/passes/fuse_ops_into_poplar_ops.h",
        "driver/passes/fuse_ops_late.h",
        "driver/passes/fuse_wide_const.h",
        "driver/passes/fusion_inliner.h",
        "driver/passes/gather_simplifier.h",
        "driver/passes/gradient_accumulation_buffers_offload.h",
        "driver/passes/gradient_accumulation_fuser.h",
        "driver/passes/gradient_accumulation_verifier.h",
        "driver/passes/hlo_computation_name_uniquify.h",
        "driver/passes/host_compute_barrier_inserter.h",
        "driver/passes/host_compute_schedule_optimizer.h",
        "driver/passes/host_compute_util.h",
        "driver/passes/host_embedding_notification.h",
        "driver/passes/inplace_finder.h",
        "driver/passes/inter_ipu_copy_inserter.h",
        "driver/passes/inter_tileset_copy_inserter.h",
        "driver/passes/io_tiles_placer.h",
        "driver/passes/lift_recompute_suggestion.h",
        "driver/passes/lower_frontend_attributes.h",
        "driver/passes/mark_replica_identical_instructions.h",
        "driver/passes/matmul_combiner.h",
        "driver/passes/module_flatten.h",
        "driver/passes/multi_conv_fixer.h",
        "driver/passes/multi_slice_combiner.h",
        "driver/passes/multi_update_apply.h",
        "driver/passes/multi_update_combiner.h",
        "driver/passes/multi_update_scale_apply.h",
        "driver/passes/multi_use_feeds_finder.h",
        "driver/passes/not_supported_gather_expander.h",
        "driver/passes/not_supported_scatter_expander.h",
        "driver/passes/outline_remote_buffers.h",
        "driver/passes/parse_poplar_backend_config.h",
        "driver/passes/pipeline_batch_serialization_buffer_inserter.h",
        "driver/passes/pipeline_batch_serialization_loop_inserter.h",
        "driver/passes/pipeline_communication_optimizer.h",
        "driver/passes/pipeline_control_dependency_inserter.h",
        "driver/passes/pipeline_copy_inserter.h",
        "driver/passes/pipeline_feed_hoisting.h",
        "driver/passes/pipeline_fifo_inserter.h",
        "driver/passes/pipeline_fixer.h",
        "driver/passes/pipeline_gradient_accumulation_optimizer.h",
        "driver/passes/pipeline_optimizer.h",
        "driver/passes/pipeline_recomputation.h",
        "driver/passes/pipeline_recomputation_stage_inserter.h",
        "driver/passes/pipeline_stage_merger.h",
        "driver/passes/pipeline_tuple_remover.h",
        "driver/passes/pipeline_verifier.h",
        "driver/passes/poplar_algebraic_simplifier.h",
        "driver/passes/poplar_algebraic_simplifier/poplar_algebraic_simplifier_convolution.h",
        "driver/passes/poplar_algebraic_simplifier/poplar_algebraic_simplifier_dot.h",
        "driver/passes/poplar_algebraic_simplifier/poplar_algebraic_simplifier_utils.h",
        "driver/passes/post_serialize_gradient_accumulation.h",
        "driver/passes/recomputation_checkpoint_remover.h",
        "driver/passes/recomputation_input_remover.h",
        "driver/passes/recompute_casts.h",
        "driver/passes/recompute_instructions.h",
        "driver/passes/remote_buffer_canonicalizer.h",
        "driver/passes/remote_buffer_merger.h",
        "driver/passes/remote_parameter_parallel_combiner.h",
        "driver/passes/remove_blocked_recompute_suggestions.h",
        "driver/passes/remove_recompute_suggestions.h",
        "driver/passes/repeat_loop_copy_inserter.h",
        "driver/passes/replicated_resource_update_elementwise_clustering.h",
        "driver/passes/replication_factor_to_constant.h",
        "driver/passes/resource_update_copy_inserter.h",
        "driver/passes/resource_update_elementwise_clustering.h",
        "driver/passes/resource_update_fixer.h",
        "driver/passes/resource_update_merger.h",
        "driver/passes/resource_update_schedule_optimizer.h",
        "driver/passes/root_token_replacer.h",
        "driver/passes/scatter_simplifier.h",
        "driver/passes/seed_hoisting.h",
        "driver/passes/serialize_gradient_accumulation.h",
        "driver/passes/sharding_pass.h",
        "driver/passes/slice_optimizer.h",
        "driver/passes/suggest_recompute.h",
        "driver/passes/variables_offload_and_partition.h",
        "driver/passes/while_loop_condition_simplify.h",
        "driver/passes/while_loop_to_repeat_simplify.h",
        "driver/passes/wide_const_finder.h",
        "driver/schedulers/clustering_scheduler.h",
        "driver/schedulers/ipu_scheduler.h",
        "driver/schedulers/liveness_look_ahead_scheduler.h",
        "driver/schedulers/schedule_tree.h",
        "driver/schedulers/schedule_utils.h",
        "driver/schedulers/shortest_path_scheduler.h",
        "driver/schedulers/sync_list_scheduler.h",
        "driver/tools/conv_util.h",
        "driver/tools/custom_ops/all_gather.h",
        "driver/tools/custom_ops/arg_min_max.h",
        "driver/tools/custom_ops/assert.h",
        "driver/tools/custom_ops/assume_equal_across_replicas.h",
        "driver/tools/custom_ops/candidate_sampler.h",
        "driver/tools/custom_ops/cast_to_gfloat_hlo.h",
        "driver/tools/custom_ops/codelet_expression_op.h",
        "driver/tools/custom_ops/collective_reorder.h",
        "driver/tools/custom_ops/conv_with_reverse.h",
        "driver/tools/custom_ops/copy_into.h",
        "driver/tools/custom_ops/ctc_loss.h",
        "driver/tools/custom_ops/dropout_hlo.h",
        "driver/tools/custom_ops/elementwise.h",
        "driver/tools/custom_ops/execution_counter.h",
        "driver/tools/custom_ops/fifo.h",
        "driver/tools/custom_ops/gru.h",
        "driver/tools/custom_ops/histogram.h",
        "driver/tools/custom_ops/hlo_extensions.h",
        "driver/tools/custom_ops/hlo_poplar_instruction.h",
        "driver/tools/custom_ops/hlo_remote_buffer_info.h",
        "driver/tools/custom_ops/host_embedding.h",
        "driver/tools/custom_ops/inter_tileset_copy.h",
        "driver/tools/custom_ops/ipu_inter_copy.h",
        "driver/tools/custom_ops/lstm.h",
        "driver/tools/custom_ops/multi_conv.h",
        "driver/tools/custom_ops/multi_slice.h",
        "driver/tools/custom_ops/non_linearity.h",
        "driver/tools/custom_ops/norm.h",
        "driver/tools/custom_ops/normalise_image.h",
        "driver/tools/custom_ops/onehot.h",
        "driver/tools/custom_ops/pooling.h",
        "driver/tools/custom_ops/print_tensor.h",
        "driver/tools/custom_ops/recompute.h",
        "driver/tools/custom_ops/recv_from_host.h",
        "driver/tools/custom_ops/reduce_many.h",
        "driver/tools/custom_ops/reduce_scatter.h",
        "driver/tools/custom_ops/remap.h",
        "driver/tools/custom_ops/remap_deduce.h",
        "driver/tools/custom_ops/remote_parameter.h",
        "driver/tools/custom_ops/replication_factor.h",
        "driver/tools/custom_ops/rnn.h",
        "driver/tools/custom_ops/scaled_inplace.h",
        "driver/tools/custom_ops/seed.h",
        "driver/tools/custom_ops/send_recv_barrier.h",
        "driver/tools/custom_ops/send_to_host.h",
        "driver/tools/custom_ops/sequence_slice.h",
        "driver/tools/custom_ops/slice_apply.h",
        "driver/tools/custom_ops/sparse.h",
        "driver/tools/custom_ops/stateful_gradient_accumulate.h",
        "driver/tools/custom_ops/stateful_noop.h",
        "driver/tools/custom_ops/stateless_random.h",
        "driver/tools/custom_ops/topk.h",
        "driver/tools/custom_ops/truncated_normal.h",
        "driver/tools/custom_ops/user_op_hlo.h",
        "driver/tools/custom_ops/weights_transpose_chans_flip_xy.h",
        "driver/tools/elementwise_cluster.h",
        "driver/tools/extension_registry.h",
        "driver/tools/feed_info.h",
        "driver/tools/find_all_users.h",
        "driver/tools/flags.h",
        "driver/tools/hlo_hash.h",
        "driver/tools/hlo_instruction_extensions.h",
        "driver/tools/hlo_matcher.h",
        "driver/tools/hlo_poplar_buffer.h",
        "driver/tools/hlo_poplar_buffer_util.h",
        "driver/tools/hlo_poplar_dataflow_analysis.h",
        "driver/tools/inplace_util.h",
        "driver/tools/input_output_aliasing_map.h",
        "driver/tools/instruction_colocator_helper.h",
        "driver/tools/isomorphic_functions_map.h",
        "driver/tools/matcher_predicates.h",
        "driver/tools/matmul_util.h",
        "driver/tools/meta_graph.h",
        "driver/tools/ml_type_helper.h",
        "driver/tools/offloading_util.h",
        "driver/tools/pipeline_util.h",
        "driver/tools/poplar_replica_groups.h",
        "driver/tools/reduction_util.h",
        "driver/tools/replica_identical_dataflow_analysis.h",
        "driver/tools/single_hlo_matcher.h",
        "driver/tools/tensor_location.h",
        "driver/tools/tracepoint.h",
        "driver/tools/util.h",
        "driver/tools/while_loop_util.h",
    ],
    deps = [
        ":alias_info_cc_impl",
        ":backend_protos_cc",
        ":common",
        ":config_protos_cc_impl",
        ":custom_kernels_util",
        ":flags",
        ":hash",
        ":option_flag_cc_impl",
        ":pipeline_config_cc_impl",
        ":threestate_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver/invariant_passes:no_control_deps_checker",
        "//tensorflow/compiler/plugin/poplar/driver/invariant_passes:resource_update_checker",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/xla:xla_headers_lib",
        "//tensorflow/compiler/xla/service:call_inliner",
        "//tensorflow/compiler/xla/service:dynamic_padder",
        "//tensorflow/compiler/xla/service:flatten_call_graph",
        "//tensorflow/compiler/xla/service:gather_expander",
        "//tensorflow/compiler/xla/service:hlo_memory_scheduler",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:qr_expander",
        "//tensorflow/compiler/xla/service:scatter_expander",
        "//tensorflow/compiler/xla/service:tuple_simplifier",
        "//tensorflow/compiler/xla/service:while_loop_analysis",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf_headers",
    ],
    alwayslink = True,
)

# This is temporary and will be removed after refactor is done.
cc_library(
    name = "tools_headers",
    srcs = [],
    hdrs = [
        "driver/tools/flags.h",
        "driver/tools/util.h",
    ],
    visibility = ["//tensorflow/compiler/plugin/poplar:__subpackages__"],
    deps = [],
)

cc_library(
    name = "hash",
    srcs = [],
    hdrs = [
        "driver/tools/hash.h",
    ],
    deps = [
        "//tensorflow/core:framework_headers_lib",
    ],
)

cc_library(
    name = "flags",
    srcs = [
        "driver/tools/flags.cc",
    ],
    hdrs = [
        "driver/tools/flags.h",
    ],
    deps = [
        ":config_protos_cc_impl",
        ":hash",
        ":option_flag_cc_impl",
        ":threestate_cc_impl",
        "//tensorflow/compiler/xla:xla_headers_lib",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "common",
    srcs = [
        "driver/ipu_devices.cc",
    ],
    hdrs = [
        "driver/ipu_devices.h",
        "driver/xla_ipu_common.h",
    ],
    deps = [
        ":flags",
        "//tensorflow/compiler/jit:xla_device",
    ],
)

cc_library(
    name = "infeed_utils",
    srcs = [
        "driver/tools/infeed_allocator.cc",
        "driver/tools/infeed_iterator.cc",
    ],
    hdrs = [
        "driver/tools/infeed_allocator.h",
        "driver/tools/infeed_iterator.h",
        "driver/tools/spsc_queue.h",
    ],
    linkstatic = 1,
    deps = [
        ":backend_protos_cc",
        ":flags",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_headers_lib",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core/kernels/data:unbounded_thread_pool",
        "//third_party/eigen3",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "popef_util",
    srcs = [
        "driver/tools/popef_util.cc",
    ],
    hdrs = [
        "driver/tools/popef_util.h",
    ],
    linkstatic = 1,
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "@local_config_poplar//poplar:poplar_libs",
    ],
)

cc_library(
    name = "driver",
    srcs = [
        "driver/ops/cast_to_gfloat_ops.cc",
        "driver/ops/custom_ops/execution_counter.cc",
        "driver/ops/custom_ops/host_embedding.cc",
        "driver/ops/custom_ops/popfloat/cast_to_gfloat.cc",
        "driver/ops/custom_ops/popfloat/gfloat_ops_utils.cc",
        "driver/ops/custom_ops/poplin/cholesky.cc",
        "driver/ops/custom_ops/poplin/triangular_solve.cc",
        "driver/ops/custom_ops/poplin/weights_transpose_chans_flip_xy.cc",
        "driver/ops/custom_ops/popnn/arg_min_max.cc",
        "driver/ops/custom_ops/popnn/ctc_loss.cc",
        "driver/ops/custom_ops/popnn/gru.cc",
        "driver/ops/custom_ops/popnn/lstm.cc",
        "driver/ops/custom_ops/popnn/non_linearity.cc",
        "driver/ops/custom_ops/popnn/norm.cc",
        "driver/ops/custom_ops/popnn/onehot.cc",
        "driver/ops/custom_ops/popnn/pooling.cc",
        "driver/ops/custom_ops/popnn/topk.cc",
        "driver/ops/custom_ops/popops/all_gather.cc",
        "driver/ops/custom_ops/popops/collective_reorder.cc",
        "driver/ops/custom_ops/popops/conv_ops.cc",
        "driver/ops/custom_ops/popops/elementwise.cc",
        "driver/ops/custom_ops/popops/expression_helpers.cc",
        "driver/ops/custom_ops/popops/fusions.cc",
        "driver/ops/custom_ops/popops/gather_ops.cc",
        "driver/ops/custom_ops/popops/histogram.cc",
        "driver/ops/custom_ops/popops/multi_slice.cc",
        "driver/ops/custom_ops/popops/normalise_image.cc",
        "driver/ops/custom_ops/popops/random.cc",
        "driver/ops/custom_ops/popops/reduce_many.cc",
        "driver/ops/custom_ops/popops/reduce_scatter.cc",
        "driver/ops/custom_ops/popops/scaled_inplace.cc",
        "driver/ops/custom_ops/popops/scatter_ops.cc",
        "driver/ops/custom_ops/popops/select_scalars.cc",
        "driver/ops/custom_ops/popops/sequence_slice.cc",
        "driver/ops/custom_ops/popops/slice_apply.cc",
        "driver/ops/custom_ops/popops/sort_ops.cc",
        "driver/ops/custom_ops/poprand/candidate_sampler.cc",
        "driver/ops/custom_ops/poprand/dropout.cc",
        "driver/ops/custom_ops/poprand/random.cc",
        "driver/ops/custom_ops/poputil/assert.cc",
        "driver/ops/custom_ops/poputil/assume_equal_across_replicas.cc",
        "driver/ops/custom_ops/poputil/codelet_expression_op.cc",
        "driver/ops/custom_ops/poputil/copy_into.cc",
        "driver/ops/custom_ops/poputil/fifo.cc",
        "driver/ops/custom_ops/poputil/inter_tileset_copy.cc",
        "driver/ops/custom_ops/poputil/ipu_inter_copy.cc",
        "driver/ops/custom_ops/poputil/print_tensor.cc",
        "driver/ops/custom_ops/poputil/remap.cc",
        "driver/ops/custom_ops/poputil/remap_deduce.cc",
        "driver/ops/custom_ops/poputil/remote_parameter.cc",
        "driver/ops/custom_ops/poputil/replication_factor.cc",
        "driver/ops/custom_ops/poputil/replication_index.cc",
        "driver/ops/custom_ops/poputil/stateful_gradient_accumulate.cc",
        "driver/ops/custom_ops/poputil/stateful_noop.cc",
        "driver/ops/custom_ops/poputil/user_op.cc",
        "driver/ops/custom_ops/recv_from_host.cc",
        "driver/ops/custom_ops/send_recv_barrier.cc",
        "driver/ops/custom_ops/send_to_host.cc",
        "driver/ops/feed_ops.cc",
        "driver/ops/graph_caching_util.cc",
        "driver/ops/map_ops.cc",
        "driver/ops/maths_ops.cc",
        "driver/ops/ops_helper.cc",
        "driver/ops/reduction_ops.cc",
        "driver/ops/tensor_ops.cc",
        "driver/poplar_compiler.cc",
        "driver/poplar_executable.cc",
        "driver/poplar_executable_cache.cc",
        "driver/poplar_executor.cc",
        "driver/poplar_passes/convolution_preplanning.cc",
        "driver/poplar_passes/ctc_preplanning.cc",
        "driver/poplar_passes/embedding_plans_preplanning.cc",
        "driver/poplar_passes/map_hlo_instruction_to_debug_id.cc",
        "driver/poplar_passes/matmul_preplanning.cc",
        "driver/poplar_platform.cc",
        "driver/poplar_platform_id.cc",
        "driver/poplar_transfer_manager.cc",
        "driver/prng_seed_state.cc",
        "driver/tensor.cc",
        "driver/tools/conv_poplar_util.cc",
        "driver/tools/conversions.cc",
        "driver/tools/data_initializer.cc",
        "driver/tools/debug_info.cc",
        "driver/tools/execution_counter_util.cc",
        "driver/tools/generic_graph_caching.cc",
        "driver/tools/io_thread.cc",
        "driver/tools/mapping_helper.cc",
        "driver/tools/poplar_executable_binary_file.cc",
        "driver/tools/poplar_range_sampler.cc",
        "driver/tools/poplar_util.cc",
        "driver/tools/progress_bar.cc",
        "driver/tools/reduction_util.cc",
        "driver/tools/rnn_util.cc",
        "driver/tools/seed_generator.cc",
        "driver/tools/subcomputation_graph_caching.cc",
        "driver/tools/tensor_map.cc",
        "driver/tools/tracepoint.cc",
        "driver/visitors/deferred_visitor.cc",
        "driver/visitors/entry_visitor.cc",
        "driver/visitors/grouped_overlap_pipeline_visitor.cc",
        "driver/visitors/partitioned_elementwise_cluster_visitor.cc",
        "driver/visitors/pipeline_stage_visitor.cc",
        "driver/visitors/pipeline_visitor.cc",
        "driver/visitors/pipeline_visitor_creator.cc",
        "driver/visitors/repeat_loop_overlap_io_visitor.cc",
        "driver/visitors/repeat_loop_visitor.cc",
        "driver/visitors/visitor_arithmetic_expr.cc",
        "driver/visitors/visitor_base.cc",
        "driver/visitors/visitor_full.cc",
        "driver/visitors/visitor_map.cc",
    ],
    hdrs = [
        "driver/compiler_resources.h",
        "driver/ops/custom_ops/popfloat/gfloat_ops_utils.h",
        "driver/ops/custom_ops/poplar_ops.h",
        "driver/ops/custom_ops/popops/expression_helpers.h",
        "driver/ops/graph_caching_util.h",
        "driver/ops/ops.h",
        "driver/ops/ops_helper.h",
        "driver/poplar_compiler.h",
        "driver/poplar_executable.h",
        "driver/poplar_executable_cache.h",
        "driver/poplar_executor.h",
        "driver/poplar_passes/convolution_preplanning.h",
        "driver/poplar_passes/ctc_preplanning.h",
        "driver/poplar_passes/embedding_plans_preplanning.h",
        "driver/poplar_passes/map_hlo_instruction_to_debug_id.h",
        "driver/poplar_passes/matmul_preplanning.h",
        "driver/poplar_platform.h",
        "driver/poplar_platform_id.h",
        "driver/poplar_transfer_manager.h",
        "driver/poplar_version.h",
        "driver/prng_seed_state.h",
        "driver/tensor.h",
        "driver/tools/attributes_utils.h",
        "driver/tools/conv_poplar_util.h",
        "driver/tools/conversions.h",
        "driver/tools/data_initializer.h",
        "driver/tools/debug_info.h",
        "driver/tools/execution_counter_util.h",
        "driver/tools/generic_graph_caching.h",
        "driver/tools/io_thread.h",
        "driver/tools/make_visitor.h",
        "driver/tools/mapping_helper.h",
        "driver/tools/poplar_executable_binary_file.h",
        "driver/tools/poplar_range_sampler.h",
        "driver/tools/poplar_util.h",
        "driver/tools/progress_bar.h",
        "driver/tools/reduction_util.h",
        "driver/tools/rnn_util.h",
        "driver/tools/seed_generator.h",
        "driver/tools/spsc_outfeed_queue.h",
        "driver/tools/subcomputation_graph_caching.h",
        "driver/tools/tensor_map.h",
        "driver/tools/tracepoint.h",
        "driver/vertex_templates.h",
        "driver/visitors/deferred_visitor.h",
        "driver/visitors/entry_visitor.h",
        "driver/visitors/grouped_overlap_pipeline_visitor.h",
        "driver/visitors/partitioned_elementwise_cluster_visitor.h",
        "driver/visitors/pipeline_stage_visitor.h",
        "driver/visitors/pipeline_visitor.h",
        "driver/visitors/pipeline_visitor_creator.h",
        "driver/visitors/pipeline_visitor_utils.h",
        "driver/visitors/repeat_loop_overlap_io_visitor.h",
        "driver/visitors/repeat_loop_visitor.h",
        "driver/visitors/visitor_arithmetic_expr.h",
        "driver/visitors/visitor_base.h",
        "driver/visitors/visitor_full.h",
        "driver/visitors/visitor_map.h",
    ],
    copts = [
        "-DTF_POPLAR_BUILD_TAG=" + tf_poplar_build_tag(),
    ],
    deps = [
        ":common",
        ":config_protos_cc_impl",
        ":custom_kernels_util",
        ":infeed_utils",
        ":optimizers",
        ":option_flag_cc_impl",
        ":pipeline_config_cc_impl",
        ":popef_util",
        ":poplar_version",
        ":tf_graph_sources",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/xla:xla_headers_lib",
        "//tensorflow/compiler/xla/service:dynamic_index_splitter",
        "//tensorflow/compiler/xla/service:generic_transfer_manager",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_constant_folding",
        "//tensorflow/compiler/xla/service:hlo_cost_analysis",
        "//tensorflow/compiler/xla/service:hlo_cse",
        "//tensorflow/compiler/xla/service:hlo_dce",
        "//tensorflow/compiler/xla/service:hlo_graph_dumper",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:hlo_pass_pipeline",
        "//tensorflow/compiler/xla/service:map_inliner",
        "//tensorflow/compiler/xla/service:reshape_mover",
        "//tensorflow/compiler/xla/service:sort_simplifier",
        "//tensorflow/compiler/xla/service:transfer_manager",
        "//tensorflow/compiler/xla/service:tuple_simplifier",
        "//tensorflow/compiler/xla/service:while_loop_constant_sinking",
        "//tensorflow/compiler/xla/service:zero_sized_hlo_elimination",
        "//tensorflow/compiler/xla/service/cpu:cpu_runtime",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core/kernels/data:unbounded_thread_pool",
        "//third_party/eigen3",
        "//third_party/ipus/tools:poplar_executable_data",
        "@com_google_protobuf//:protobuf_headers",
        "@jsoncpp_git//:jsoncpp",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_poplar//poplar:poplar_headers",
    ],
    alwayslink = True,
)

cc_library(
    name = "graph_optimizer_passes",
    srcs = [
        "graph_optimizer_passes/extract_outside_compilation_pass.cc",
        "graph_optimizer_passes/gradient_accumulation_optimization_pass.cc",
        "graph_optimizer_passes/graph_optimizer_pass_registration.cc",
        "graph_optimizer_passes/reorder_gradient_accumulation_pass.cc",
        "graph_optimizer_passes/static_shape_inference_pass.cc",
        "graph_optimizer_passes/util.cc",
        "graph_optimizer_passes/verify_gradient_accumulation_pass.cc",
    ],
    hdrs = [
        "graph_optimizer_passes/extract_outside_compilation_pass.h",
        "graph_optimizer_passes/gradient_accumulation_optimization_pass.h",
        "graph_optimizer_passes/reorder_gradient_accumulation_pass.h",
        "graph_optimizer_passes/static_shape_inference_pass.h",
        "graph_optimizer_passes/util.h",
        "graph_optimizer_passes/verify_gradient_accumulation_pass.h",
    ],
    deps = [
        "//tensorflow/compiler/jit:compilation_passes",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
    ],
    alwayslink = True,
)

cc_library(
    name = "dataset_exporters_kernel",
    srcs = [
        "kernels/datastream/exporters.cc",
    ],
    deps = [
        ":backend_protos_cc",
        ":infeed_utils",
        ":option_flag_cc_impl",
        ":popef_util",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:core_cpu",
        "//third_party/eigen3",
        "//third_party/ipus/tools:poplar_executable_data",
        "@com_google_absl//absl/strings",
        "@jsoncpp_git//:jsoncpp",
        "@local_config_poplar//poplar:poplar_libs",
    ],
    alwayslink = True,
)

cc_library(
    name = "xla_util",
    srcs = [
        "driver/tools/xla_util.cc",
    ],
    hdrs = [
        "driver/tools/xla_util.h",
    ],
    deps = [
        ":ops_cc_impl",
        "//tensorflow/compiler/xla/client:xla_builder",
    ],
)

cc_library(
    name = "kernels",
    srcs = [
        "kernels/application_runtime/application_compile.cc",
        "kernels/application_runtime/application_runtime.cc",
        "kernels/datastream/dataset_benchmark.cc",
        "kernels/datastream/feeds.cc",
        "kernels/datastream/host_embedding.cc",
        "kernels/functional/functional.cc",
        "kernels/functional/functional_util.cc",
        "kernels/functional/pipelining.cc",
        "kernels/host_compute_kernels.cc",
        "kernels/ipu_kernels.cc",
        "kernels/popfloat/cast_to_gfloat.cc",
        "kernels/popnn/arg_min_max.cc",
        "kernels/popnn/ctc_loss.cc",
        "kernels/popnn/gru.cc",
        "kernels/popnn/lstm.cc",
        "kernels/popnn/non_linearity.cc",
        "kernels/popnn/norm.cc",
        "kernels/popnn/onehot.cc",
        "kernels/popnn/pooling.cc",
        "kernels/popnn/topk.cc",
        "kernels/popops/all_reduce.cc",
        "kernels/popops/all_to_all.cc",
        "kernels/popops/erf.cc",
        "kernels/popops/histogram.cc",
        "kernels/popops/multi_slice.cc",
        "kernels/popops/normalise_image.cc",
        "kernels/popops/reduce_scatter.cc",
        "kernels/popops/sequence_slice.cc",
        "kernels/popops/sparse_softmax_cross_entropy.cc",
        "kernels/poprand/candidate_sampler.cc",
        "kernels/poprand/categorical.cc",
        "kernels/poprand/dropout_xla.cc",
        "kernels/poprand/random.cc",
        "kernels/poprand/random_ops.cc",
        "kernels/poputil/assert.cc",
        "kernels/poputil/assume_equal_across_replicas.cc",
        "kernels/poputil/codelet_expression.cc",
        "kernels/poputil/device_sync.cc",
        "kernels/poputil/execution_counter.cc",
        "kernels/poputil/fifo.cc",
        "kernels/poputil/print_tensor.cc",
        "kernels/poputil/recompute.cc",
        "kernels/poputil/remap.cc",
        "kernels/poputil/replication_factor.cc",
        "kernels/poputil/replication_index.cc",
        "kernels/poputil/stateful_gradient_accumulate.cc",
        "kernels/poputil/user_operation.cc",
        "kernels/sendrecv_kernels.cc",
        "kernels/shape_op.cc",
        "kernels/training_ops.cc",
    ],
    hdrs = [
        "kernels/functional/functional_util.h",
        "kernels/ipu_kernels_common.h",
    ],
    deps = [
        ":common",
        ":custom_kernels_util",
        ":dataset_exporters_kernel",
        ":driver",
        ":xla_util",
        "//tensorflow/compiler/jit:xla_device",
        "//tensorflow/compiler/jit/kernels:xla_ops",
        "//tensorflow/compiler/plugin/poplar/kernels/dataset:kernels",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla/kernels:tensor_list_utils",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/compiler/xla/client/lib:pooling",
        "//tensorflow/compiler/xla/client/lib:prng",
        "//tensorflow/compiler/xla/client/lib:sorting",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core/kernels/data:dataset_utils",
        "//third_party/eigen3",
        "@com_google_protobuf//:protobuf_headers",
        "@jsoncpp_git//:jsoncpp",
    ],
    alwayslink = True,
)

cc_library(
    name = "poplar_test_base",
    testonly = True,
    srcs = [
        "driver/tools/hlo_poplar_test_base.cc",
        "driver/tools/hlo_poplar_test_util.cc",
    ],
    hdrs = [
        "driver/tools/hlo_poplar_test_base.h",
        "driver/tools/hlo_poplar_test_util.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":driver",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

cc_library(
    name = "poplar_lib",
    srcs = [
        "driver/xla_ipu_device.cc",
    ],
    hdrs = [],
    data = [
        ":config_protos_py",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":driver",
        ":graph_optimizer_passes",
        ":kernels",
        ":poplar_backend",
        "//tensorflow/compiler/jit:xla_device",
        "//tensorflow/compiler/jit:xla_jit_headers_lib",
        "//tensorflow/compiler/jit/kernels:xla_ops",
        "//tensorflow/compiler/plugin/poplar/ops:ipu_ops",
        "//tensorflow/compiler/plugin/poplar/ops:sendrecv_ops",
        "//tensorflow/compiler/plugin/poplar/ops/application_runtime:ops",
        "//tensorflow/compiler/plugin/poplar/ops/dataset:ops",
        "//tensorflow/compiler/plugin/poplar/ops/datastream:ops",
        "//tensorflow/compiler/plugin/poplar/ops/functional:ops",
        "//tensorflow/compiler/plugin/poplar/ops/popfloat:ops",
        "//tensorflow/compiler/plugin/poplar/ops/popnn:ops",
        "//tensorflow/compiler/plugin/poplar/ops/popops:ops",
        "//tensorflow/compiler/plugin/poplar/ops/poprand:ops",
        "//tensorflow/compiler/plugin/poplar/ops/poputil:ops",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core/kernels:constant_op",
        "//tensorflow/core/kernels:no_op",
        "@local_config_poplar//poplar:poplar_libs",
    ],
    alwayslink = True,
)

cc_binary(
    name = "libuser_op_test.so",
    srcs = ["tests/user_op_test_target.cc"],
    copts = ["-fexceptions"],
    linkshared = True,
    deps = [
        "@local_config_poplar//poplar:poplar_libs",
    ],
)

cc_library(
    name = "poplar_backend",
    srcs = [
        "driver/xla_ipu_backend.cc",
    ],
    hdrs = [],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@com_google_protobuf//:protobuf_headers",
    ],
    alwayslink = True,
)

cc_library(
    name = "ipu_config",
    srcs = [
        "config/ipu_config.cc",
    ],
    hdrs = [
        "config/ipu_config.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":config_protos_cc_impl",
        ":poplar_lib",
        "//tensorflow/core:direct_session",
        "@local_config_poplar//poplar:poplar_libs",
    ],
)

tf_cc_binary(
    name = "grpc_service_main_ipu",
    deps = [
        ":poplar_lib",
        "//tensorflow/compiler/xla/rpc:grpc_service_main_library",
    ],
)

cc_library(
    name = "test_utils",
    srcs = [],
    hdrs = [
        "tests/test_utils.h",
    ],
    deps = [
        "//tensorflow/compiler/xla:array3d",
    ],
)

xla_test(
    name = "add_stochastic_rounding_options_test",
    srcs = ["tests/add_stochastic_rounding_options_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        ":test_utils",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:pattern_matcher_gmock",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "mark_replica_identical_instructions_test",
    srcs = ["tests/mark_replica_identical_instructions_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        ":test_utils",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:pattern_matcher_gmock",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "replica_identical_dataflow_analysis_test",
    srcs = ["tests/replica_identical_dataflow_analysis_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        ":test_utils",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:pattern_matcher_gmock",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "poplar_algebraic_simplifier_test",
    srcs = ["tests/poplar_algebraic_simplifier_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:pattern_matcher_gmock",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "all_reduce_simplifier_test",
    srcs = ["tests/all_reduce_simplifier_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:pattern_matcher_gmock",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "elementwise_simplifier_test",
    srcs = ["tests/elementwise_simplifier_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:pattern_matcher_gmock",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "module_flatten_test",
    srcs = ["tests/module_flatten_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:pattern_matcher_gmock",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "map_hlo_instruction_to_debug_id_test",
    srcs = ["tests/map_hlo_instruction_to_debug_id_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:pattern_matcher_gmock",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "graph_compile_io_map_test",
    srcs = ["tests/graph_compile_io_map_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "layout_strip_test",
    srcs = ["tests/layout_strip_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "conversions_test",
    srcs = ["tests/conversions_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "allreduce_lookahead_scheduler_test",
    srcs = ["tests/allreduce_lookahead_scheduler_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "shortest_path_scheduler_test",
    srcs = ["tests/shortest_path_scheduler_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "dynamic_slice_layout_test",
    srcs = ["tests/dynamic_slice_layout_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "sort_op_cc_test",
    srcs = ["tests/sort_op_cc_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "convolution_classifier_test",
    size = "small",
    srcs = ["tests/convolution_classifier_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "inplace_test",
    size = "small",
    srcs = ["tests/inplace_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "instruction_extensions_test",
    size = "small",
    srcs = ["tests/instruction_extensions_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "hlo_hash_test",
    srcs = ["tests/hlo_hash_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "hlo_matcher_test",
    srcs = ["tests/hlo_matcher_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "expression_outliner_test",
    srcs = ["tests/expression_outliner_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        ":test_utils",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "allocation_finder_test",
    srcs = ["tests/allocation_finder_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "wide_const_finder_test",
    srcs = ["tests/wide_const_finder_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "add_block_recompute_test",
    srcs = ["tests/add_block_recompute_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "apply_recompute_suggestion_test",
    srcs = ["tests/apply_recompute_suggestion_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "lift_recompute_suggestion_test",
    srcs = ["tests/lift_recompute_suggestion_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "remove_blocked_recompute_test",
    srcs = ["tests/remove_blocked_recompute_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "recompute_casts_test",
    srcs = ["tests/recompute_casts_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "recompute_suggestion_test",
    srcs = ["tests/recompute_suggestion_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "repeat_loop_copy_inserter_test",
    srcs = ["tests/repeat_loop_copy_inserter_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "while_loop_condition_simplify_test",
    srcs = ["tests/while_loop_condition_simplify_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "while_loop_to_repeat_simplify_test",
    srcs = ["tests/while_loop_to_repeat_simplify_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "while_loop_alias_copy_test",
    srcs = ["tests/while_loop_alias_copy_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "infeed_queue_test",
    srcs = ["tests/infeed_queue_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "commutative_instruction_reorder_operands_test",
    srcs = ["tests/commutative_instruction_reorder_operands_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "fuse_wide_const_test",
    srcs = ["tests/fuse_wide_const_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "scheduler_tree_test",
    size = "small",
    srcs = ["tests/scheduler_tree_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "sync_list_scheduler_test",
    size = "small",
    srcs = ["tests/sync_list_scheduler_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "combine_instructions_test",
    size = "large",
    srcs = ["tests/combine_instructions_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    tags = ["hw_poplar_test"],
    deps = [
        ":optimizers",
        ":poplar_test_base",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "remote_buffer_merger_test",
    size = "small",
    srcs = ["tests/remote_buffer_merger_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "remote_parameter_parallel_combiner_test",
    size = "small",
    srcs = ["tests/remote_parameter_parallel_combiner_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "hlo_computation_name_uniquify_test",
    size = "small",
    srcs = ["tests/hlo_computation_name_uniquify_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "inter_tileset_copy_inserter_test",
    size = "small",
    srcs = ["tests/inter_tileset_copy_inserter_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "io_tiles_placer_test",
    size = "small",
    srcs = ["tests/io_tiles_placer_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "fused_batch_norm_test",
    size = "medium",
    srcs = ["tests/fused_batch_norm_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        ":test_utils",
        "//tensorflow/compiler/xla:array3d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:reference_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/compiler/xla/client/lib:math",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "sharding_pass_test",
    size = "small",
    srcs = ["tests/sharding_pass_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "inter_ipu_copy_inserter_test",
    size = "small",
    srcs = ["tests/inter_ipu_copy_inserter_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "dependency_replacer_test",
    size = "small",
    srcs = ["tests/dependency_replacer_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "host_compute_barrier_inserter_test",
    size = "small",
    srcs = ["tests/host_compute_barrier_inserter_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "host_compute_schedule_optimizer_test",
    size = "small",
    srcs = ["tests/host_compute_schedule_optimizer_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "host_embedding_notify_pass_test",
    size = "small",
    srcs = ["tests/host_embedding_notify_pass_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "f16_constant_folding_test",
    size = "small",
    srcs = ["tests/f16_constant_folding_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "constant_nan_test",
    size = "small",
    srcs = ["tests/constant_nan_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "constant_slice_folding_test",
    size = "small",
    srcs = ["tests/constant_slice_folding_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "find_all_users_test",
    srcs = ["tests/find_all_users_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "not_supported_gather_expander_test",
    srcs = ["tests/not_supported_gather_expander_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "not_supported_scatter_expander_test",
    srcs = ["tests/not_supported_scatter_expander_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "pipeline_visitor_test",
    srcs = ["tests/pipeline_visitor_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        ":poplar_test_base",
        ":test_utils",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "pipeline_verifier_test",
    srcs = ["tests/pipeline_verifier_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        ":poplar_test_base",
        ":test_utils",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "pipeline_visitor_utils_test",
    srcs = ["tests/pipeline_visitor_utils_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "pipeline_grouped_visitor_test",
    srcs = ["tests/pipeline_grouped_visitor_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        ":poplar_test_base",
        ":test_utils",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "pipeline_sequential_visitor_test",
    srcs = ["tests/pipeline_sequential_visitor_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        ":poplar_test_base",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "prng_seed_state_test",
    srcs = ["tests/prng_seed_state_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    tags = ["hw_poplar_test"],
    deps = [
        ":optimizers",
        ":poplar_test_base",
        ":test_utils",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "root_token_replacer_test",
    size = "small",
    srcs = ["tests/root_token_replacer_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "debug_info_test",
    size = "small",
    srcs = ["tests/debug_info_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "meta_graph_test",
    size = "small",
    srcs = ["tests/meta_graph_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "deferred_visitor_test",
    srcs = ["tests/deferred_visitor_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":poplar_test_base",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_runner",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

xla_test(
    name = "slice_plan_test",
    srcs = ["tests/slice_plan_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":poplar_test_base",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_runner",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

tf_xla_py_test(
    name = "random_ops",
    size = "small",
    srcs = ["tests/random_ops.py"],
    enabled_backends = ["poplar"],
    tags = ["hw_poplar_test"],
    deps = [
        ":test_utils_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

xla_test(
    name = "recomputation_test",
    srcs = ["tests/recomputation_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "data_initializer_test",
    srcs = ["tests/data_initializer_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "elementwise_broadcast_converter_test",
    srcs = ["tests/elementwise_broadcast_converter_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        ":poplar_test_base",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "gradient_accumulation_fuser_test",
    srcs = ["tests/gradient_accumulation_fuser_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "sparse_softmax_test",
    size = "medium",
    srcs = ["tests/sparse_softmax_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        ":test_utils",
        "//tensorflow/compiler/xla:array3d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:reference_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/compiler/xla/client/lib:math",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "user_op_internal_test",
    srcs = ["tests/user_op_internal_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    linkopts = ["-lstdc++fs"],
    deps = [
        ":libuser_op_test.so",
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "all_gather_replacer_test",
    size = "medium",
    srcs = ["tests/all_gather_replacer_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        ":test_utils",
        "//tensorflow/compiler/xla:array3d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:reference_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/compiler/xla/client/lib:math",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "matmul_combiner_test",
    size = "medium",
    srcs = ["tests/matmul_combiner_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        ":test_utils",
        "//tensorflow/compiler/xla:array3d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:reference_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/compiler/xla/client/lib:math",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "multi_slice_combiner_test",
    size = "medium",
    srcs = ["tests/multi_slice_combiner_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        ":test_utils",
        "//tensorflow/compiler/xla:array3d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:reference_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/compiler/xla/client/lib:math",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "multi_update_combiner_test",
    size = "medium",
    srcs = ["tests/multi_update_combiner_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        ":test_utils",
        "//tensorflow/compiler/xla:array3d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:reference_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/compiler/xla/client/lib:math",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "dynamic_slice_test",
    size = "medium",
    srcs = ["tests/dynamic_slice_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        ":test_utils",
        "//tensorflow/compiler/xla:array3d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:reference_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/compiler/xla/client/lib:math",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "create_buffer_test",
    size = "medium",
    srcs = ["tests/create_buffer_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        ":test_utils",
        "//tensorflow/compiler/xla:array3d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:reference_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/compiler/xla/client/lib:math",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "arithmetic_expr_test",
    size = "medium",
    srcs = ["tests/arithmetic_expr_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        ":test_utils",
        "//tensorflow/compiler/xla:array3d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:reference_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/compiler/xla/client/lib:math",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "pipeline_util_test",
    size = "small",
    srcs = ["tests/pipeline_util_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "pipeline_fixer_test",
    size = "small",
    srcs = ["tests/pipeline_fixer_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "pipeline_optimizer_test",
    size = "small",
    srcs = ["tests/pipeline_optimizer_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "pipeline_fifo_inserter_test",
    size = "small",
    srcs = ["tests/pipeline_fifo_inserter_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "pipeline_copy_inserter_test",
    size = "small",
    srcs = ["tests/pipeline_copy_inserter_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "stateful_noop_test",
    size = "small",
    srcs = ["tests/stateful_noop_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "pipeline_feed_hoisting_test",
    size = "small",
    srcs = ["tests/pipeline_feed_hoisting_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "pipeline_tuple_remover_test",
    size = "small",
    srcs = ["tests/pipeline_tuple_remover_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "pipeline_stage_merger_test",
    size = "small",
    srcs = ["tests/pipeline_stage_merger_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "pipeline_grouped_recomputation_test",
    size = "small",
    srcs = ["tests/pipeline_grouped_recomputation_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "pipeline_recomputation_test",
    size = "small",
    srcs = ["tests/pipeline_recomputation_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "pipeline_recomputation_stage_inserter_test",
    size = "small",
    srcs = ["tests/pipeline_recomputation_stage_inserter_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "scatter_simplifier_test",
    size = "small",
    srcs = ["tests/scatter_simplifier_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "gather_simplifier_test",
    size = "small",
    srcs = ["tests/gather_simplifier_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "conv_bwd_input_to_fwd_weights_transpose_test",
    size = "small",
    srcs = ["tests/conv_bwd_input_to_fwd_weights_transpose_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "resource_update_fixer_test",
    size = "small",
    srcs = ["tests/resource_update_fixer_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "resource_update_merger_test",
    size = "small",
    srcs = ["tests/resource_update_merger_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "pipeline_communication_optimizer_test",
    size = "small",
    srcs = ["tests/pipeline_communication_optimizer_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "pipeline_gradient_accumulation_optimizer_test",
    size = "small",
    srcs = ["tests/pipeline_gradient_accumulation_optimizer_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "multi_update_apply_test",
    size = "small",
    srcs = ["tests/multi_update_apply_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "multi_update_scale_apply_test",
    size = "small",
    srcs = ["tests/multi_update_scale_apply_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "all_to_all_finder_test",
    size = "small",
    srcs = ["tests/all_to_all_finder_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "variables_offload_and_partition_test",
    size = "small",
    srcs = ["tests/variables_offload_and_partition_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "resource_update_schedule_optimizer_test",
    size = "small",
    srcs = ["tests/resource_update_schedule_optimizer_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

tf_cc_test(
    name = "extract_outside_compilation_pass_test",
    size = "small",
    srcs = [
        "graph_optimizer_passes/extract_outside_compilation_pass_test.cc",
    ],
    deps = [
        ":graph_optimizer_passes",
        ":poplar_lib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:session_options",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "poplar_replica_groups_test",
    size = "small",
    srcs = ["tests/poplar_replica_groups_test.cc"],
    deps = [
        ":poplar_lib",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "reorder_gradient_accumulation_pass_test",
    size = "small",
    srcs = [
        "graph_optimizer_passes/reorder_gradient_accumulation_pass_test.cc",
    ],
    deps = [
        ":graph_optimizer_passes",
        ":poplar_lib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:session_options",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "gradient_accumulation_optimization_pass_test",
    size = "small",
    srcs = [
        "graph_optimizer_passes/gradient_accumulation_optimization_pass_test.cc",
    ],
    deps = [
        ":graph_optimizer_passes",
        ":poplar_lib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:resource_variable_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:session_options",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

xla_test(
    name = "slice_optimizer_test",
    size = "medium",
    srcs = ["tests/slice_optimizer_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "serialize_gradient_accumulation_test",
    size = "small",
    srcs = ["tests/serialize_gradient_accumulation_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "post_serialize_gradient_accumulation_test",
    size = "small",
    srcs = ["tests/post_serialize_gradient_accumulation_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "execution_counter_util_test",
    srcs = ["tests/execution_counter_util_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":poplar_test_base",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_runner",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

xla_test(
    name = "multi_conv_fixer_test",
    size = "small",
    srcs = ["tests/multi_conv_fixer_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "copy_inserter_test",
    size = "small",
    srcs = ["tests/copy_inserter_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "seed_hoisting_test",
    size = "small",
    srcs = ["tests/seed_hoisting_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "function_optimizer_test",
    size = "small",
    srcs = ["tests/function_optimizer_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "pipeline_batch_serialization_buffer_inserter_test",
    size = "small",
    srcs = ["tests/pipeline_batch_serialization_buffer_inserter_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "pipeline_batch_serialization_loop_inserter_test",
    size = "small",
    srcs = ["tests/pipeline_batch_serialization_loop_inserter_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "gradient_accumulation_buffers_offload_test",
    size = "small",
    srcs = ["tests/gradient_accumulation_buffers_offload_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "recomputation_checkpoint_remover_test",
    size = "small",
    srcs = ["tests/recomputation_checkpoint_remover_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "recomputation_input_remover_test",
    size = "small",
    srcs = ["tests/recomputation_input_remover_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "fusion_inliner_test",
    srcs = ["tests/fusion_inliner_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        ":test_utils",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "computation_flattener_test",
    size = "small",
    srcs = ["tests/computation_flattener_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "multi_use_feeds_finder_test",
    srcs = ["tests/multi_use_feeds_finder_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        ":test_utils",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "graph_caching_with_remote_buffer_test",
    srcs = ["tests/graph_caching_with_remote_buffer_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    tags = ["hw_poplar_test"],
    deps = [
        ":optimizers",
        ":poplar_test_base",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "remote_buffer_canonicalizer_test",
    srcs = ["tests/remote_buffer_canonicalizer_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "offloading_util_test",
    srcs = ["tests/offloading_util_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        ":poplar_test_base",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "opaque_handling_test",
    srcs = ["tests/opaque_handling_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        ":poplar_test_base",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "outline_remote_buffers_test",
    srcs = ["tests/outline_remote_buffers_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "function_combiner_test",
    size = "small",
    srcs = ["tests/function_combiner_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "distributed_batch_norm_decomposer_test",
    srcs = ["tests/distributed_batch_norm_decomposer_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "no_control_deps_checker_test",
    srcs = ["tests/no_control_deps_checker_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "resource_update_checker_test",
    srcs = ["tests/resource_update_checker_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "feed_info_test",
    srcs = ["tests/feed_info_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "call_optimizer_test",
    size = "small",
    srcs = ["tests/call_optimizer_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "util_test",
    size = "small",
    srcs = ["tests/util_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

py_library(
    name = "test_utils_py",
    testonly = 1,
    srcs = ["tests/test_utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/compiler/tf2xla/python:xla",
        "//tensorflow/python/compiler/xla:compiler_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "arg_min_max_test",
    size = "medium",
    srcs = ["tests/arg_min_max_test.py"],
    enabled_backends = ["poplar"],
    shard_count = 2,
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "ml_type_classify_test",
    size = "large",
    srcs = ["tests/ml_type_classify_test.py"],
    enabled_backends = ["poplar"],
    shard_count = 2,
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "ipu_model_device_test",
    size = "large",
    srcs = ["tests/ipu_model_device_test.py"],
    enabled_backends = ["poplar"],
    shard_count = 4,
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "topk_onehot_test",
    size = "large",
    srcs = ["tests/topk_onehot_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "topk_test",
    size = "large",
    srcs = ["tests/topk_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "simple_network_test",
    size = "large",
    srcs = ["tests/simple_network_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "matmul_test",
    size = "large",
    srcs = ["tests/matmul_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "matmul_option_test",
    size = "large",
    srcs = ["tests/matmul_option_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "conv_test",
    size = "large",
    srcs = ["tests/conv_test.py"],
    enabled_backends = ["poplar"],
    shard_count = 8,
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "poplar_executable_runner_test",
    size = "large",
    srcs = ["tests/poplar_executable_runner_test.py"],
    data = [
        "//tensorflow/compiler/plugin/poplar/tools:tensorflow_weights_extractor",
        "//third_party/ipus/tools:PoplarExecutableRunner",
    ],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "serialization_test",
    size = "large",
    srcs = ["tests/serialization_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "conv3d_test",
    size = "large",
    srcs = ["tests/conv3d_test.py"],
    enabled_backends = ["poplar"],
    shard_count = 2,
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "triangular_solve_test",
    size = "large",
    srcs = ["tests/triangular_solve_test.py"],
    enabled_backends = ["poplar"],
    shard_count = 4,  # Run all tests in parallel to improve performance
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "cholesky_test",
    size = "large",
    srcs = ["tests/cholesky_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "constant_test",
    size = "large",
    srcs = ["tests/constant_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "var_handle_name_test",
    size = "large",
    srcs = ["tests/var_handle_name_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "variable_test",
    size = "large",
    srcs = ["tests/variable_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "synthetic_data_test",
    size = "large",
    srcs = ["tests/synthetic_data_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "synthetic_data_with_outfeeds_test",
    size = "large",
    srcs = ["tests/synthetic_data_with_outfeeds_test.py"],
    enabled_backends = ["poplar"],
    shard_count = 3,
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "synthetic_data_categories_test",
    size = "large",
    srcs = ["tests/synthetic_data_categories_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_test(
    name = "multi_run_test",
    size = "large",
    srcs = ["tests/multi_run_test.py"],
    enabled_backends = ["poplar"],
    shard_count = 3,
    tags = ["hw_poplar_test"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "reduce_test",
    size = "large",
    srcs = ["tests/reduce_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "f16_test",
    size = "large",
    srcs = ["tests/f16_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "64_bit_test",
    size = "large",
    srcs = ["tests/64_bit_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "batch_norm_test",
    size = "large",
    srcs = ["tests/batch_norm_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "distributed_batch_norm_test",
    size = "large",
    srcs = ["tests/distributed_batch_norm_test.py"],
    enabled_backends = ["poplar"],
    shard_count = 2,
    tags = ["hw_poplar_test"],
    deps = [
        ":test_utils_py",
    ],
)

poprun_py_test(
    name = "distributed_tf2_test",
    size = "large",
    srcs = ["tests/distributed_tf2_test.py"],
    main = "tests/distributed_tf2_test.py",
    num_instances = 2,
    num_replicas = 8,
    deps = [
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

xla_test(
    name = "replicated_resource_update_elementwise_clustering_hw_test",
    srcs = ["tests/replicated_resource_update_elementwise_clustering_hw_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    tags = ["hw_poplar_test"],
    deps = [
        ":optimizers",
        ":poplar_test_base",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "replicated_resource_update_elementwise_clustering_test",
    srcs = ["tests/replicated_resource_update_elementwise_clustering_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        ":poplar_test_base",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "resource_update_elementwise_clustering_test",
    srcs = ["tests/resource_update_elementwise_clustering_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        ":test_utils",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "resource_update_copy_inserter_test",
    srcs = ["tests/resource_update_copy_inserter_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "embeddings_gradient_optimizer_test",
    srcs = ["tests/embeddings_gradient_optimizer_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "fuse_ops_early_test",
    srcs = ["tests/fuse_ops_early_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        ":test_utils",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "fuse_ops_into_poplar_ops_test",
    srcs = ["tests/fuse_ops_into_poplar_ops_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "hlo_poplar_buffer_test",
    srcs = ["tests/hlo_poplar_buffer_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "hlo_poplar_buffer_util_test",
    srcs = ["tests/hlo_poplar_buffer_util_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "nonlinearity_hlo_test",
    srcs = ["tests/nonlinearity_hlo_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "dead_control_dependencies_elimination_test",
    srcs = ["tests/dead_control_dependencies_elimination_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "hlo_poplar_dataflow_analysis_test",
    srcs = ["tests/hlo_poplar_dataflow_analysis_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        ":poplar_test_base",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

tf_xla_py_test(
    name = "fuse_ops_test",
    size = "medium",
    srcs = ["tests/fuse_ops_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "assert_test",
    size = "large",
    srcs = ["tests/assert_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "gather_simplifier_embedding_lookup_test",
    size = "large",
    srcs = ["tests/gather_simplifier_embedding_lookup_test.py"],
    enabled_backends = ["poplar"],
    shard_count = 5,
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "tensor_array_test",
    size = "large",
    srcs = ["tests/tensor_array_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "monitored_session_test",
    size = "large",
    srcs = ["tests/monitored_session_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "casts_elimination_test",
    size = "medium",
    srcs = ["tests/casts_elimination_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "sendrecv_ops_test",
    size = "large",
    srcs = ["tests/sendrecv_ops_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "outside_compilation_scope_test",
    size = "large",
    srcs = ["tests/outside_compilation_scope_test.py"],
    enabled_backends = ["poplar"],
    shard_count = 8,
    tags = ["isolated_poplar_tests"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "update_op_dependencies_test",
    size = "medium",
    srcs = ["tests/update_op_dependencies_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "norm_graph_caching_test",
    size = "medium",
    srcs = ["tests/norm_graph_caching_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "norm_graph_caching_sharded_test",
    size = "medium",
    srcs = ["tests/norm_graph_caching_sharded_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "conv_graph_caching_test",
    size = "medium",
    srcs = ["tests/conv_graph_caching_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "conv_graph_caching_sharded_test",
    size = "medium",
    srcs = ["tests/conv_graph_caching_sharded_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "lstm_test",
    size = "medium",
    srcs = ["tests/lstm_test.py"],
    enabled_backends = ["poplar"],
    shard_count = 8,
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "group_norm_test",
    size = "large",
    srcs = ["tests/group_norm_test.py"],
    enabled_backends = ["poplar"],
    shard_count = 6,
    deps = [
        ":test_utils_py",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_test(
    name = "wide_const_expansion_test",
    size = "medium",
    srcs = ["tests/wide_const_expansion_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "forward_allocation_test",
    size = "medium",
    srcs = ["tests/forward_allocation_test.py"],
    enabled_backends = ["poplar"],
    shard_count = 4,
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "stateful_gradient_accumulate_test",
    size = "medium",
    srcs = ["tests/stateful_gradient_accumulate_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "case_test",
    size = "large",
    srcs = ["tests/case_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "while_loop_perf_test",
    size = "large",
    srcs = ["tests/while_loop_perf_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "mapping_test",
    size = "large",
    srcs = ["tests/mapping_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "remapping_test",
    size = "large",
    srcs = ["tests/remapping_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "dump_poplar_info",
    size = "large",
    srcs = ["tests/dump_poplar_info.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "candidate_sampler_test",
    size = "large",
    srcs = ["tests/candidate_sampler_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "conditional_test",
    size = "large",
    srcs = ["tests/conditional_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "combined_weights_test",
    size = "large",
    srcs = ["tests/size_speed_tests/combined_weights_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "resnet_size_test",
    size = "large",
    srcs = ["tests/size_speed_tests/resnet_size_test.py"],
    enabled_backends = ["poplar"],
    shard_count = 8,
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "matmul_size_test",
    size = "large",
    srcs = ["tests/size_speed_tests/matmul_size_test.py"],
    enabled_backends = ["poplar"],
    shard_count = 4,
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "gru_size_test",
    size = "large",
    srcs = ["tests/size_speed_tests/gru_size_test.py"],
    enabled_backends = ["poplar"],
    shard_count = 6,
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "lstm_size_test",
    size = "large",
    srcs = ["tests/size_speed_tests/lstm_size_test.py"],
    enabled_backends = ["poplar"],
    shard_count = 4,
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "gru_training_test",
    size = "large",
    srcs = ["tests/training_tests/gru_training_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "lstm_training_test",
    size = "large",
    srcs = ["tests/training_tests/lstm_training_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "while_loop_test",
    size = "medium",
    srcs = ["tests/while_loop_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "while_loop_sharded_test",
    size = "medium",
    srcs = ["tests/while_loop_sharded_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "subcomputation_graph_caching_test",
    size = "medium",
    srcs = ["tests/subcomputation_graph_caching_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "multi_ipu_test",
    size = "medium",
    srcs = ["tests/multi_ipu_test.py"],
    enabled_backends = ["poplar"],
    tags = ["isolated_poplar_tests"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "scalar_elementwise_graph_test",
    size = "medium",
    srcs = ["tests/scalar_elementwise_graph_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "sort_op_py_test",
    size = "medium",
    srcs = ["tests/sort_op_py_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "sharded_inplace_test",
    size = "large",
    srcs = ["tests/sharded_inplace_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

cc_binary(
    name = "libadd_incrementing_custom_with_metadata.so",
    srcs = ["tests/add_incrementing_custom_with_metadata.cc"],
    copts = ["-fexceptions"],
    linkshared = True,
    deps = [
        "@local_config_poplar//poplar:poplar_libs",
    ],
)

tf_xla_py_test(
    name = "clear_xla_cache_test",
    size = "large",
    srcs = ["tests/clear_xla_cache_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "executable_cache_test",
    size = "medium",
    srcs = ["tests/executable_cache_test.py"],
    data = [
        ":libadd_incrementing_custom_with_metadata.so",
    ],
    enabled_backends = ["poplar"],
    shard_count = 8,
    tags = ["hw_poplar_test"],
    deps = [
        ":test_utils_py",
    ],
)

poprun_py_test(
    name = "poprun_basic_test",
    size = "large",
    srcs = ["tests/poprun_basic_test.py"],
    main = "tests/poprun_basic_test.py",
    num_instances = 2,
    num_replicas = 4,
    deps = [
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "device_connection_test",
    size = "medium",
    srcs = ["tests/device_connection_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "perf_training_example",
    size = "large",
    srcs = ["docs/perf_training_example.py"],
    enabled_backends = ["poplar"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

py_binary(
    name = "pre_compile_example",
    srcs = ["docs/pre_compile_example.py"],
    python_version = "PY3",
)

tf_xla_py_test(
    name = "pre_compile_example_runner",
    size = "large",
    srcs = ["docs/pre_compile_example_runner.py"],
    data = [":pre_compile_example"],
    enabled_backends = ["poplar"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "outfeed_example",
    size = "large",
    srcs = ["docs/outfeed_example.py"],
    enabled_backends = ["poplar"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

py_test(
    name = "distributed_training_example",
    size = "large",
    srcs = ["docs/distributed_training_example.py"],
    args = ["--num-steps=10"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

horovod_py_test(
    name = "distributed_training_horovod_example",
    size = "large",
    srcs = ["docs/distributed_training_horovod_example.py"],
    args = ["--num-steps=10"],
    main = "docs/distributed_training_horovod_example.py",
    num_processes = 2,
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

py_test(
    name = "ipu_estimator_example",
    size = "large",
    srcs = ["docs/ipu_estimator_example.py"],
    args = [
        "--iterations-per-loop=1",
        "--training-steps=1",
    ],
    tags = ["isolated_poplar_tests"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

py_test(
    name = "ipu_estimator_example_replicated",
    size = "large",
    srcs = ["docs/ipu_estimator_example.py"],
    args = [
        "--iterations-per-loop=1",
        "--training-steps=1",
        "--replicas=2",
    ],
    main = "ipu_estimator_example.py",
    tags = ["hw_poplar_test"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

py_test(
    name = "ipu_pipeline_estimator_example",
    size = "large",
    srcs = ["docs/ipu_pipeline_estimator_example.py"],
    args = [
        "--gradient-accumulation-count=4",
        "--iterations-per-loop=4",
        "--training-steps=4",
    ],
    tags = ["isolated_poplar_tests"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "example_tf2_model_fit",
    size = "large",
    srcs = ["docs/example_tf2_model_fit.py"],
    enabled_backends = ["poplar"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "example_tf2_fit",
    size = "large",
    srcs = ["docs/example_tf2_fit.py"],
    enabled_backends = ["poplar"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "example_tf2_custom_training",
    size = "large",
    srcs = ["docs/example_tf2_custom_training.py"],
    enabled_backends = ["poplar"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

py_test(
    name = "example_tf2_pipelined_model",
    size = "large",
    srcs = ["docs/example_tf2_pipelined_model.py"],
    args = [
        "--epochs=1",
        "--steps-per-epoch=16",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

py_test(
    name = "example_tf2_pipeline_sequential",
    size = "large",
    srcs = ["docs/example_tf2_pipeline_sequential.py"],
    args = [
        "--epochs=1",
        "--steps-per-epoch=2",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "bias_apply_graph_caching_test",
    size = "medium",
    srcs = ["tests/bias_apply_graph_caching_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "matmul_graph_caching_test",
    size = "medium",
    srcs = ["tests/matmul_graph_caching_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "auto_report_directory_test",
    size = "medium",
    srcs = ["tests/auto_report_directory_test.py"],
    enabled_backends = ["poplar"],
    tags = ["exclusive"],
    deps = [
        ":test_utils_py",
    ],
)

cc_binary(
    name = "libcustom_rotate_op.so",
    srcs = ["docs/custom_rotate_op.cc"],
    copts = ["-fexceptions"],
    linkshared = True,
    deps = ["@local_config_poplar//poplar:poplar_libs"],
)

tf_xla_py_test(
    name = "tutorial_custom_codelet",
    size = "large",
    srcs = ["docs/tutorial_custom_codelet.py"],
    data = [
        ":custom_codelet_example",
        ":libcustom_rotate_op.so",
    ],
    enabled_backends = ["poplar"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

cc_binary(
    name = "libcustom_add_inplace.so",
    srcs = ["docs/custom_add_inplace.cc"],
    copts = ["-fexceptions"],
    linkshared = True,
    deps = ["@local_config_poplar//poplar:poplar_libs"],
)

tf_xla_py_test(
    name = "custom_add_inplace",
    size = "large",
    srcs = ["docs/custom_add_inplace.py"],
    data = [
        ":libcustom_add_inplace.so",
    ],
    enabled_backends = ["poplar"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "gru_test",
    size = "medium",
    srcs = ["tests/gru_test.py"],
    enabled_backends = ["poplar"],
    shard_count = 6,
    deps = [
        ":test_utils_py",
    ],
)

xla_test(
    name = "weights_transpose_chans_flip_xy_test",
    srcs = ["tests/weights_transpose_chans_flip_xy_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

tf_xla_py_test(
    name = "allocation_finder_priority_test",
    size = "large",
    srcs = ["tests/allocation_finder_priority_test.py"],
    enabled_backends = ["poplar"],
    shard_count = 2,
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "function_example1",
    size = "large",
    srcs = ["docs/function_example1.py"],
    enabled_backends = ["poplar"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "function_example2",
    size = "large",
    srcs = ["docs/function_example2.py"],
    enabled_backends = ["poplar"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "function_block1",
    size = "large",
    srcs = ["docs/function_block1.py"],
    enabled_backends = ["poplar"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "function_block2",
    size = "large",
    srcs = ["docs/function_block2.py"],
    enabled_backends = ["poplar"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "function_block3",
    size = "large",
    srcs = ["docs/function_block3.py"],
    enabled_backends = ["poplar"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

cc_binary(
    name = "libtutorial_attributes_example.so",
    srcs = ["docs/tutorial_attributes_example.cc"],
    copts = ["-fexceptions"],
    linkshared = True,
    deps = [
        "@jsoncpp_git//:jsoncpp",
        "@local_config_poplar//poplar:poplar_libs",
    ],
)

tf_xla_py_test(
    name = "tutorial_attributes_example",
    size = "large",
    srcs = ["docs/tutorial_attributes_example.py"],
    data = [
        ":libtutorial_attributes_example.so",
    ],
    enabled_backends = ["poplar"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "pre_compile_test",
    size = "large",
    srcs = ["tests/pre_compile_test.py"],
    data = [
        ":libadd_incrementing_custom_with_metadata.so",
    ],
    enabled_backends = ["poplar"],
    shard_count = 4,
    deps = [
        ":test_utils_py",
    ],
)

tf_xla_py_test(
    name = "targeting_tf2_example1",
    size = "large",
    srcs = ["docs/targeting_tf2_example1.py"],
    enabled_backends = ["poplar"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "targeting_tf2_example2",
    size = "large",
    srcs = ["docs/targeting_tf2_example2.py"],
    enabled_backends = ["poplar"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "targeting_tf2_example3",
    size = "large",
    srcs = ["docs/targeting_tf2_example3.py"],
    enabled_backends = ["poplar"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "keras_tf2_example1",
    size = "large",
    srcs = ["docs/keras_tf2_example1.py"],
    enabled_backends = ["poplar"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "keras_tf2_example2",
    size = "large",
    srcs = ["docs/keras_tf2_example2.py"],
    enabled_backends = ["poplar"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "keras_tf2_example3",
    size = "large",
    srcs = ["docs/keras_tf2_example3.py"],
    enabled_backends = ["poplar"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "keras_tf2_example4",
    size = "large",
    srcs = ["docs/keras_tf2_example4.py"],
    enabled_backends = ["poplar"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "keras_tf2_example5",
    size = "large",
    srcs = ["docs/keras_tf2_example5.py"],
    enabled_backends = ["poplar"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "keras_tf2_example6",
    size = "large",
    srcs = ["docs/keras_tf2_example6.py"],
    enabled_backends = ["poplar"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "keras_tf2_example7",
    size = "large",
    srcs = ["docs/keras_tf2_example7.py"],
    enabled_backends = ["poplar"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "keras_tf2_example8",
    size = "large",
    srcs = ["docs/keras_tf2_example8.py"],
    enabled_backends = ["poplar"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "keras_tf2_example9",
    size = "large",
    srcs = ["docs/keras_tf2_example9.py"],
    enabled_backends = ["poplar"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "keras_tf2_example10",
    size = "large",
    srcs = ["docs/keras_tf2_example10.py"],
    enabled_backends = ["poplar"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "keras_porting_example",
    size = "large",
    srcs = ["docs/keras_porting_example.py"],
    enabled_backends = ["poplar"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "compile_cache_test",
    size = "large",
    srcs = ["tests/compile_cache_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
    ],
)

xla_test(
    name = "pipeline_control_dep_test",
    size = "small",
    srcs = ["tests/pipeline_control_dep_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

test_suite(
    name = "all_tests",
    tests = [
        "64_bit_test",
        "add_block_recompute_test",
        "add_stochastic_rounding_options_test",
        "all_to_all_finder_test",
        "allocation_finder_priority_test",
        "allocation_finder_test",
        "apply_recompute_suggestion_test",
        "arg_min_max_test",
        "arithmetic_expr_test",
        "assert_test",
        "auto_report_directory_test",
        "batch_norm_test",
        "bias_apply_graph_caching_test",
        "candidate_sampler_test",
        "case_test",
        "casts_elimination_test",
        "cholesky_test",
        "clear_xla_cache_test",
        "combine_instructions_test",
        "commutative_instruction_reorder_operands_test",
        "compile_cache_test",
        "computation_flattener_test",
        "conditional_test",
        "constant_nan_test",
        "constant_slice_folding_test",
        "conv3d_test",
        "conv_bwd_input_to_fwd_weights_transpose_test",
        "conv_graph_caching_sharded_test",
        "conv_graph_caching_test",
        "conv_test",
        "conversions_test",
        "convolution_classifier_test",
        "copy_inserter_test",
        "create_buffer_test",
        "data_initializer_test",
        "dead_control_dependencies_elimination_test",
        "debug_info_test",
        "deferred_visitor_test",
        "dependency_replacer_test",
        "device_connection_test",
        "distributed_batch_norm_decomposer_test",
        "distributed_batch_norm_test",
        "distributed_tf2_test",
        "dump_poplar_info",
        "dynamic_slice_layout_test",
        "dynamic_slice_test",
        "elementwise_broadcast_converter_test",
        "elementwise_simplifier_test",
        "embeddings_gradient_optimizer_test",
        "executable_cache_test",
        "execution_counter_util_test",
        "expression_outliner_test",
        "extract_outside_compilation_pass_test",
        "f16_constant_folding_test",
        "f16_test",
        "feed_info_test",
        "find_all_users_test",
        "forward_allocation_test",
        "function_combiner_test",
        "function_optimizer_test",
        "fuse_ops_early_test",
        "fuse_ops_into_poplar_ops_test",
        "fuse_ops_test",
        "fuse_wide_const_test",
        "fused_batch_norm_test",
        "fusion_inliner_test",
        "gather_simplifier_embedding_lookup_test",
        "gradient_accumulation_buffers_offload_test",
        "gradient_accumulation_fuser_test",
        "gradient_accumulation_optimization_pass_test",
        "graph_caching_with_remote_buffer_test",
        "graph_compile_io_map_test",
        "group_norm_test",
        "gru_test",
        "hlo_computation_name_uniquify_test",
        "hlo_hash_test",
        "hlo_matcher_test",
        "hlo_poplar_buffer_test",
        "hlo_poplar_buffer_util_test",
        "hlo_poplar_dataflow_analysis_test",
        "host_compute_barrier_inserter_test",
        "host_compute_schedule_optimizer_test",
        "inplace_test",
        "instruction_extensions_test",
        "inter_ipu_copy_inserter_test",
        "inter_tileset_copy_inserter_test",
        "io_tiles_placer_test",
        "ipu_model_device_test",
        "layout_strip_test",
        "lift_recompute_suggestion_test",
        "lstm_test",
        "map_hlo_instruction_to_debug_id_test",
        "mapping_test",
        "mark_replica_identical_instructions_test",
        "matmul_combiner_test",
        "matmul_graph_caching_test",
        "matmul_option_test",
        "matmul_test",
        "meta_graph_test",
        "ml_type_classify_test",
        "module_flatten_test",
        "monitored_session_test",
        "multi_conv_fixer_test",
        "multi_ipu_test",
        "multi_run_test",
        "multi_slice_combiner_test",
        "multi_update_apply_test",
        "multi_update_combiner_test",
        "multi_update_scale_apply_test",
        "multi_use_feeds_finder_test",
        "no_control_deps_checker_test",
        "nonlinearity_hlo_test",
        "norm_graph_caching_sharded_test",
        "norm_graph_caching_test",
        "not_supported_gather_expander_test",
        "not_supported_scatter_expander_test",
        "offloading_util_test",
        "opaque_handling_test",
        "outline_remote_buffers_test",
        "outside_compilation_scope_test",
        "pipeline_batch_serialization_buffer_inserter_test",
        "pipeline_batch_serialization_loop_inserter_test",
        "pipeline_communication_optimizer_test",
        "pipeline_control_dep_test",
        "pipeline_copy_inserter_test",
        "pipeline_feed_hoisting_test",
        "pipeline_fifo_inserter_test",
        "pipeline_fixer_test",
        "pipeline_gradient_accumulation_optimizer_test",
        "pipeline_grouped_recomputation_test",
        "pipeline_grouped_visitor_test",
        "pipeline_optimizer_test",
        "pipeline_recomputation_stage_inserter_test",
        "pipeline_recomputation_test",
        "pipeline_sequential_visitor_test",
        "pipeline_stage_merger_test",
        "pipeline_tuple_remover_test",
        "pipeline_util_test",
        "pipeline_visitor_test",
        "pipeline_visitor_utils_test",
        "poplar_algebraic_simplifier_test",
        "poplar_executable_runner_test",
        "poplar_replica_groups_test",
        "poprun_basic_test",
        "post_serialize_gradient_accumulation_test",
        "pre_compile_test",
        "prng_seed_state_test",
        "random_ops",
        "recomputation_checkpoint_remover_test",
        "recomputation_input_remover_test",
        "recomputation_test",
        "recompute_casts_test",
        "recompute_suggestion_test",
        "reduce_test",
        "remapping_test",
        "remote_buffer_canonicalizer_test",
        "remote_parameter_parallel_combiner_test",
        "remove_blocked_recompute_test",
        "reorder_gradient_accumulation_pass_test",
        "repeat_loop_copy_inserter_test",
        "replica_identical_dataflow_analysis_test",
        "replicated_resource_update_elementwise_clustering_hw_test",
        "replicated_resource_update_elementwise_clustering_test",
        "resource_update_checker_test",
        "resource_update_copy_inserter_test",
        "resource_update_elementwise_clustering_test",
        "resource_update_fixer_test",
        "resource_update_merger_test",
        "resource_update_schedule_optimizer_test",
        "root_token_replacer_test",
        "scalar_elementwise_graph_test",
        "scatter_simplifier_test",
        "scheduler_tree_test",
        "seed_hoisting_test",
        "sendrecv_ops_test",
        "serialization_test",
        "serialize_gradient_accumulation_test",
        "sharded_inplace_test",
        "sharding_pass_test",
        "shortest_path_scheduler_test",
        "simple_network_test",
        "slice_optimizer_test",
        "slice_plan_test",
        "sort_op_cc_test",
        "sort_op_py_test",
        "sparse_softmax_test",
        "stateful_gradient_accumulate_test",
        "stateful_noop_test",
        "subcomputation_graph_caching_test",
        "sync_list_scheduler_test",
        "synthetic_data_categories_test",
        "synthetic_data_test",
        "synthetic_data_with_outfeeds_test",
        "tensor_array_test",
        "topk_onehot_test",
        "topk_test",
        "triangular_solve_test",
        "update_op_dependencies_test",
        "user_op_internal_test",
        "util_test",
        "var_handle_name_test",
        "variable_test",
        "variables_offload_and_partition_test",
        "weights_transpose_chans_flip_xy_test",
        "while_loop_alias_copy_test",
        "while_loop_condition_simplify_test",
        "while_loop_perf_test",
        "while_loop_sharded_test",
        "while_loop_test",
        "while_loop_to_repeat_simplify_test",
        "wide_const_expansion_test",
        "wide_const_finder_test",
    ],
)

test_suite(
    name = "size_speed_tests",
    tests = [
        "combined_weights_test",
        "gru_size_test",
        "lstm_size_test",
        "matmul_size_test",
        "resnet_size_test",
    ],
)

test_suite(
    name = "training_tests",
    tests = [
        "gru_training_test",
        "lstm_training_test",
    ],
)

test_suite(
    name = "doc_examples",
    tests = [
        "custom_add_inplace",
        "distributed_training_example",
        "example_tf2_custom_training",
        "example_tf2_fit",
        "example_tf2_model_fit",
        "example_tf2_pipelined_model",
        "function_block1",
        "function_block2",
        "function_block3",
        "function_example1",
        "function_example2",
        "ipu_estimator_example",
        "ipu_pipeline_estimator_example",
        "keras_tf2_example1",
        "keras_tf2_example2",
        "keras_tf2_example3",
        "keras_tf2_example4",
        "keras_tf2_example5",
        "keras_tf2_example6",
        "keras_tf2_example7",
        "keras_tf2_example8",
        "keras_tf2_example9",
        "keras_tf2_example10",
        "keras_porting_example",
        "outfeed_example",
        "perf_training_example",
        "pre_compile_example_runner",
        "targeting_tf2_example1",
        "targeting_tf2_example2",
        "targeting_tf2_example3",
        "tutorial_attributes_example",
        "tutorial_custom_codelet",
    ] + if_horovod(["distributed_training_horovod_example"]),
)

test_suite(
    name = "poplar_test_suite",
    tests = [
        ":all_tests",
        "//tensorflow/compiler/plugin/poplar/config/test:all_ipu_config_tests",
        "//tensorflow/compiler/plugin/poplar/kernels/dataset:dataset_tests",
        "//tensorflow/compiler/tests:poplar_tests",
        "//tensorflow/compiler/xla/client/lib:poplar_tests",
        "//tensorflow/compiler/xla/tests:poplar_tests",
    ],
)

test_suite(
    name = "poplar_ci_test_suite",
    tests = [
        ":doc_examples",
        ":poplar_test_suite",
        ":size_speed_tests",
        ":training_tests",
    ],
)

exports_files(
    [
        "disabled_manifest.txt",
        "disabled_xla_tests_manifest.txt",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)
