licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

load("//tensorflow/compiler/xla/tests:build_defs.bzl", "xla_test")
load("//tensorflow/compiler/tests:build_defs.bzl", "tf_xla_py_test")
load("@local_config_poplar//poplar:build_defs.bzl", "tf_poplar_build_tag")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_cc_test", "tf_custom_op_py_library", "tf_gen_op_wrapper_py")
load(
    "//tensorflow/core/platform:default/build_config.bzl",
    "tf_proto_library",
    "tf_proto_library_py",
)
load("//tensorflow/compiler/xla:xla.bzl", "xla_proto_library")

# Rule for creating the vertex program pre-compiled object
genrule(
    name = "tf_graph_program",
    srcs = ["vertices/tf.cpp"],
    outs = ["tf.gp"],
    cmd = "$(location @local_config_poplar//poplar:popc) -DNDEBUG -O3 -I . -o $@ $<",
    tools = ["@local_config_poplar//poplar:popc"],
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
    ],
)

xla_proto_library(
    name = "xla_backend_protos",
    srcs = [
        "driver/backend_config.proto",
        "driver/poplar_executable.proto",
        "driver/poplar_feed_config.proto",
    ],
    deps = [
        ":option_flag_cc",
        ":pipeline_config_cc",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_proto_library_py(
    name = "backend_protos",  # bzl adds a _py suffix
    srcs = [
        "driver/backend_config.proto",
        "driver/poplar_executable.proto",
        "driver/poplar_feed_config.proto",
    ],
    deps = [
        ":option_flag_py",
        ":pipeline_config_py",
        "//tensorflow/compiler/xla:xla_data_proto_py",
        "//tensorflow/core:protos_all_py",
    ],
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
        "driver/tools/util.h",
        "kernels/custom_kernels_util.h",
    ],
    deps = [
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
        "driver/passes/all_to_all_finder.cc",
        "driver/passes/allocation_finder.cc",
        "driver/passes/add_block_recompute.cc",
        "driver/passes/apply_recompute_suggestion.cc",
        "driver/passes/casts_elimination.cc",
        "driver/passes/combine_instructions.cc",
        "driver/passes/commutative_instruction_reorder_operands.cc",
        "driver/passes/computation_flattener.cc",
        "driver/passes/constant_nan.cc",
        "driver/passes/constant_slice_folding.cc",
        "driver/passes/convolution_classifier.cc",
        "driver/passes/custom_op_replacer.cc",
        "driver/passes/dependency_replacer.cc",
        "driver/passes/elementwise_broadcast_converter.cc",
        "driver/passes/expression_outliner.cc",
        "driver/passes/f16_constant_folding.cc",
        "driver/passes/forward_allocation.cc",
        "driver/passes/fuse_ops_early.cc",
        "driver/passes/fuse_ops_late.cc",
        "driver/passes/fuse_wide_const.cc",
        "driver/passes/gather_simplifier.cc",
        "driver/passes/gradient_accumulation_fuser.cc",
        "driver/passes/gradient_accumulation_verifier.cc",
        "driver/passes/hlo_computation_name_uniquify.cc",
        "driver/passes/host_compute_barrier_inserter.cc",
        "driver/passes/inplace_finder.cc",
        "driver/passes/inplace_util.cc",
        "driver/passes/inter_ipu_copy_inserter.cc",
        "driver/passes/lift_recompute_suggestion.cc",
        "driver/passes/lower_frontend_attributes.cc",
        "driver/passes/matmul_combiner.cc",
        "driver/passes/module_flatten.cc",
        "driver/passes/multi_slice_combiner.cc",
        "driver/passes/multi_update_apply.cc",
        "driver/passes/multi_update_canonicalize.cc",
        "driver/passes/multi_update_combiner.cc",
        "driver/passes/multi_update_scale_apply.cc",
        "driver/passes/not_supported_gather_expander.cc",
        "driver/passes/not_supported_scatter_expander.cc",
        "driver/passes/parse_poplar_backend_config.cc",
        "driver/passes/pipeline_communication_optimizer.cc",
        "driver/passes/pipeline_copy_inserter.cc",
        "driver/passes/pipeline_feed_hoisting.cc",
        "driver/passes/pipeline_fifo_inserter.cc",
        "driver/passes/pipeline_fixer.cc",
        "driver/passes/pipeline_optimizer.cc",
        "driver/passes/pipeline_recomputation.cc",
        "driver/passes/pipeline_resource_update_fixer.cc",
        "driver/passes/pipeline_resource_variables_offload.cc",
        "driver/passes/pipeline_verifier.cc",
        "driver/passes/poplar_algebraic_simplifier.cc",
        "driver/passes/recompute_instructions.cc",
        "driver/passes/remove_blocked_recompute_suggestions.cc",
        "driver/passes/remove_recompute_suggestions.cc",
        "driver/passes/replication_factor_to_constant.cc",
        "driver/passes/root_token_replacer.cc",
        "driver/passes/scatter_simplifier.cc",
        "driver/passes/sharding_pass.cc",
        "driver/passes/suggest_recompute.cc",
        "driver/passes/while_loop_condition_simplify.cc",
        "driver/passes/while_loop_to_repeat_simplify.cc",
        "driver/passes/wide_const_finder.cc",

        # DO NOT build any files which create a HloPoplarInstructionFactory before this point
        # See https://phabricator.sourcevertex.net/T8409 for further information.
        "driver/tools/custom_ops/hlo_poplar_instruction.cc",

        # Place HloPoplarInstructionFactory files below here.
        "driver/tools/conv_util.cc",
        "driver/tools/custom_ops/all_gather.cc",
        "driver/tools/custom_ops/codelet_expression_op.cc",
        "driver/tools/custom_ops/dropout_hlo.cc",
        "driver/tools/custom_ops/fifo.cc",
        "driver/tools/custom_ops/arg_min_max.cc",
        "driver/tools/custom_ops/cast_to_gfloat_hlo.cc",
        "driver/tools/custom_ops/gelu.cc",
        "driver/tools/custom_ops/group_norm_infer.cc",
        "driver/tools/custom_ops/group_norm_train.cc",
        "driver/tools/custom_ops/group_norm_grad.cc",
        "driver/tools/custom_ops/group_norm_stats.cc",
        "driver/tools/custom_ops/gru.cc",
        "driver/tools/custom_ops/host_embedding.cc",
        "driver/tools/custom_ops/ipu_inter_copy.cc",
        "driver/tools/custom_ops/lstm.cc",
        "driver/tools/custom_ops/onehot.cc",
        "driver/tools/custom_ops/multi_slice.cc",
        "driver/tools/custom_ops/norm.cc",
        "driver/tools/custom_ops/pooling.cc",
        "driver/tools/custom_ops/print_tensor.cc",
        "driver/tools/custom_ops/recv_from_host.cc",
        "driver/tools/custom_ops/reduce_scatter.cc",
        "driver/tools/custom_ops/relu.cc",
        "driver/tools/custom_ops/recompute.cc",
        "driver/tools/custom_ops/remap.cc",
        "driver/tools/custom_ops/remap_deduce.cc",
        "driver/tools/custom_ops/remote_parameter.cc",
        "driver/tools/custom_ops/replication_factor.cc",
        "driver/tools/custom_ops/replication_index.cc",
        "driver/tools/custom_ops/rnn.cc",
        "driver/tools/custom_ops/send_recv_barrier.cc",
        "driver/tools/custom_ops/send_to_host.cc",
        "driver/tools/custom_ops/sigmoid.cc",
        "driver/tools/custom_ops/sparse.cc",
        "driver/tools/custom_ops/stateful_gradient_accumulate.cc",
        "driver/tools/custom_ops/stateful_noop.cc",
        "driver/tools/custom_ops/stateless_random.cc",
        "driver/tools/custom_ops/tanh.cc",
        "driver/tools/custom_ops/topk.cc",
        "driver/tools/custom_ops/truncated_normal.cc",
        "driver/tools/custom_ops/user_op_hlo.cc",
        "driver/tools/find_all_users.cc",
        "driver/tools/hlo_hash.cc",
        "driver/tools/hlo_matcher.cc",
        "driver/tools/input_output_aliasing_map.cc",
        "driver/tools/instruction_colocator_helper.cc",
        "driver/tools/matcher_predicates.cc",
        "driver/tools/matmul_util.cc",
        "driver/tools/ml_type_helper.cc",
        "driver/tools/pipeline_util.cc",
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
        "driver/passes/all_to_all_finder.h",
        "driver/passes/allocation_finder.h",
        "driver/passes/apply_recompute_suggestion.h",
        "driver/passes/casts_elimination.h",
        "driver/passes/combine_instructions.h",
        "driver/passes/commutative_instruction_reorder_operands.h",
        "driver/passes/computation_flattener.h",
        "driver/passes/constant_nan.h",
        "driver/passes/constant_slice_folding.h",
        "driver/passes/convolution_classifier.h",
        "driver/passes/custom_op_replacer.h",
        "driver/passes/dependency_replacer.h",
        "driver/passes/elementwise_broadcast_converter.h",
        "driver/passes/expression_outliner.h",
        "driver/passes/f16_constant_folding.h",
        "driver/passes/forward_allocation.h",
        "driver/passes/fuse_ops_early.h",
        "driver/passes/fuse_ops_late.h",
        "driver/passes/fuse_wide_const.h",
        "driver/passes/gather_simplifier.h",
        "driver/passes/gradient_accumulation_fuser.h",
        "driver/passes/gradient_accumulation_verifier.h",
        "driver/passes/hlo_computation_name_uniquify.h",
        "driver/passes/host_compute_barrier_inserter.h",
        "driver/passes/inplace_finder.h",
        "driver/passes/inplace_util.h",
        "driver/passes/inter_ipu_copy_inserter.h",
        "driver/passes/lift_recompute_suggestion.h",
        "driver/passes/lower_frontend_attributes.h",
        "driver/passes/matmul_combiner.h",
        "driver/passes/module_flatten.h",
        "driver/passes/multi_slice_combiner.h",
        "driver/passes/multi_update_apply.h",
        "driver/passes/multi_update_canonicalize.h",
        "driver/passes/multi_update_combiner.h",
        "driver/passes/multi_update_scale_apply.h",
        "driver/passes/not_supported_gather_expander.h",
        "driver/passes/not_supported_scatter_expander.h",
        "driver/passes/parse_poplar_backend_config.h",
        "driver/passes/pipeline_communication_optimizer.h",
        "driver/passes/pipeline_copy_inserter.h",
        "driver/passes/pipeline_feed_hoisting.h",
        "driver/passes/pipeline_fifo_inserter.h",
        "driver/passes/pipeline_fixer.h",
        "driver/passes/pipeline_optimizer.h",
        "driver/passes/pipeline_recomputation.h",
        "driver/passes/pipeline_resource_update_fixer.h",
        "driver/passes/pipeline_resource_variables_offload.h",
        "driver/passes/pipeline_verifier.h",
        "driver/passes/poplar_algebraic_simplifier.h",
        "driver/passes/recompute_instructions.h",
        "driver/passes/remove_blocked_recompute_suggestions.h",
        "driver/passes/remove_recompute_suggestions.h",
        "driver/passes/replication_factor_to_constant.h",
        "driver/passes/root_token_replacer.h",
        "driver/passes/scatter_simplifier.h",
        "driver/passes/sharding_pass.h",
        "driver/passes/suggest_recompute.h",
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
        "driver/tools/custom_ops/cast_to_gfloat_hlo.h",
        "driver/tools/custom_ops/codelet_expression_op.h",
        "driver/tools/custom_ops/dropout_hlo.h",
        "driver/tools/custom_ops/fifo.h",
        "driver/tools/custom_ops/gelu.h",
        "driver/tools/custom_ops/gru.h",
        "driver/tools/custom_ops/hlo_poplar_instruction.h",
        "driver/tools/custom_ops/host_embedding.h",
        "driver/tools/custom_ops/ipu_inter_copy.h",
        "driver/tools/custom_ops/lstm.h",
        "driver/tools/custom_ops/multi_slice.h",
        "driver/tools/custom_ops/norm.h",
        "driver/tools/custom_ops/onehot.h",
        "driver/tools/custom_ops/pooling.h",
        "driver/tools/custom_ops/print_tensor.h",
        "driver/tools/custom_ops/recompute.h",
        "driver/tools/custom_ops/recv_from_host.h",
        "driver/tools/custom_ops/reduce_scatter.h",
        "driver/tools/custom_ops/relu.h",
        "driver/tools/custom_ops/remap.h",
        "driver/tools/custom_ops/remap_deduce.h",
        "driver/tools/custom_ops/remote_parameter.h",
        "driver/tools/custom_ops/replication_factor.h",
        "driver/tools/custom_ops/replication_index.h",
        "driver/tools/custom_ops/rnn.h",
        "driver/tools/custom_ops/send_recv_barrier.h",
        "driver/tools/custom_ops/send_to_host.h",
        "driver/tools/custom_ops/sigmoid.h",
        "driver/tools/custom_ops/sparse.h",
        "driver/tools/custom_ops/stateful_gradient_accumulate.h",
        "driver/tools/custom_ops/stateful_noop.h",
        "driver/tools/custom_ops/stateless_random.h",
        "driver/tools/custom_ops/tanh.h",
        "driver/tools/custom_ops/topk.h",
        "driver/tools/custom_ops/truncated_normal.h",
        "driver/tools/custom_ops/user_op_hlo.h",
        "driver/tools/find_all_users.h",
        "driver/tools/hlo_hash.h",
        "driver/tools/hlo_matcher.h",
        "driver/tools/input_output_aliasing_map.h",
        "driver/tools/instruction_colocator_helper.h",
        "driver/tools/matcher_predicates.h",
        "driver/tools/matmul_util.h",
        "driver/tools/meta_graph.h",
        "driver/tools/ml_type_helper.h",
        "driver/tools/pipeline_util.h",
        "driver/tools/single_hlo_matcher.h",
        "driver/tools/tensor_location.h",
        "driver/tools/util.h",
        "driver/tools/while_loop_util.h",
    ],
    deps = [
        ":common",
        ":config_protos_cc_impl",
        ":custom_kernels_util",
        ":flags",
        ":hash",
        ":option_flag_cc_impl",
        ":pipeline_config_cc_impl",
        ":xla_backend_protos",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/xla:xla_headers_lib",
        "//tensorflow/compiler/xla/service:call_inliner",
        "//tensorflow/compiler/xla/service:flatten_call_graph",
        "//tensorflow/compiler/xla/service:gather_expander",
        "//tensorflow/compiler/xla/service:hlo_memory_scheduler",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:scatter_expander",
        "//tensorflow/compiler/xla/service:tuple_simplifier",
        "//tensorflow/compiler/xla/service:while_loop_analysis",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf_headers",
    ],
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
        ":hash",
        "//tensorflow/compiler/xla:xla_headers_lib",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "common",
    srcs = [],
    hdrs = [
        "driver/xla_ipu_common.h",
    ],
    deps = [
        ":flags",
    ],
)

cc_library(
    name = "driver",
    srcs = [
        "driver/ops/cast_to_gfloat_ops.cc",
        "driver/ops/conv_graph_caching.cc",
        "driver/ops/custom_ops/custom_ops.cc",
        "driver/ops/custom_ops/host_embedding.cc",
        "driver/ops/custom_ops/popfloat/cast_to_gfloat.cc",
        "driver/ops/custom_ops/popfloat/gfloat_ops_utils.cc",
        "driver/ops/custom_ops/poplar_ops.cc",
        "driver/ops/custom_ops/popnn/arg_min_max.cc",
        "driver/ops/custom_ops/popnn/gelu.cc",
        "driver/ops/custom_ops/popnn/gru.cc",
        "driver/ops/custom_ops/popnn/lstm.cc",
        "driver/ops/custom_ops/popnn/norm.cc",
        "driver/ops/custom_ops/popnn/onehot.cc",
        "driver/ops/custom_ops/popnn/pooling.cc",
        "driver/ops/custom_ops/popnn/relu.cc",
        "driver/ops/custom_ops/popnn/sigmoid.cc",
        "driver/ops/custom_ops/popnn/tanh.cc",
        "driver/ops/custom_ops/popnn/topk.cc",
        "driver/ops/custom_ops/popops/all_gather.cc",
        "driver/ops/custom_ops/popops/conv_ops.cc",
        "driver/ops/custom_ops/popops/elementwise.cc",
        "driver/ops/custom_ops/popops/expression_helpers.cc",
        "driver/ops/custom_ops/popops/fusions.cc",
        "driver/ops/custom_ops/popops/multi_slice.cc",
        "driver/ops/custom_ops/popops/random.cc",
        "driver/ops/custom_ops/popops/reduce_scatter.cc",
        "driver/ops/custom_ops/popops/select_scalars.cc",
        "driver/ops/custom_ops/poprand/dropout.cc",
        "driver/ops/custom_ops/poprand/random.cc",
        "driver/ops/custom_ops/poputil/codelet_expression_op.cc",
        "driver/ops/custom_ops/poputil/fifo.cc",
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
        "driver/ops/gather_ops.cc",
        "driver/ops/graph_caching_util.cc",
        "driver/ops/map_ops.cc",
        "driver/ops/maths_ops.cc",
        "driver/ops/norm_ops.cc",
        "driver/ops/reduction_ops.cc",
        "driver/ops/scatter_ops.cc",
        "driver/ops/sort_ops.cc",
        "driver/ops/tensor_ops.cc",
        "driver/poplar_compiler.cc",
        "driver/poplar_executable.cc",
        "driver/poplar_executor.cc",
        "driver/poplar_platform.cc",
        "driver/poplar_platform_id.cc",
        "driver/poplar_transfer_manager.cc",
        "driver/tensor.cc",
        "driver/tools/conv_poplar_util.cc",
        "driver/tools/conversions.cc",
        "driver/tools/convolution_preplanning.cc",
        "driver/tools/data_initializer.cc",
        "driver/tools/embedding_plans_preplanning.cc",
        "driver/tools/generic_graph_caching.cc",
        "driver/tools/infeed_allocator.cc",
        "driver/tools/infeed_iterator.cc",
        "driver/tools/io_thread.cc",
        "driver/tools/mapping_helper.cc",
        "driver/tools/matmul_preplanning.cc",
        "driver/tools/poplar_util.cc",
        "driver/tools/rnn_util.cc",
        "driver/tools/seed_generator.cc",
        "driver/tools/subcomputation_graph_caching.cc",
        "driver/tools/tensor_map.cc",
        "driver/visitors/deferred_visitor.cc",
        "driver/visitors/entry_visitor.cc",
        "driver/visitors/pipeline_stage_visitor.cc",
        "driver/visitors/pipeline_visitor.cc",
        "driver/visitors/visitor_arithmetic_expr.cc",
        "driver/visitors/visitor_base.cc",
        "driver/visitors/visitor_full.cc",
        "driver/visitors/visitor_inline_call.cc",
        "driver/visitors/visitor_map.cc",
    ],
    hdrs = [
        "driver/compiler_resources.h",
        "driver/ops/conv_graph_caching.h",
        "driver/ops/custom_ops/custom_ops.h",
        "driver/ops/custom_ops/popfloat/gfloat_ops_utils.h",
        "driver/ops/custom_ops/poplar_ops.h",
        "driver/ops/custom_ops/popops/expression_helpers.h",
        "driver/ops/graph_caching_util.h",
        "driver/ops/ops.h",
        "driver/poplar_compiler.h",
        "driver/poplar_executable.h",
        "driver/poplar_executor.h",
        "driver/poplar_platform.h",
        "driver/poplar_platform_id.h",
        "driver/poplar_transfer_manager.h",
        "driver/tensor.h",
        "driver/tools/conv_poplar_util.h",
        "driver/tools/conversions.h",
        "driver/tools/convolution_preplanning.h",
        "driver/tools/data_initializer.h",
        "driver/tools/embedding_plans_preplanning.h",
        "driver/tools/generic_graph_caching.h",
        "driver/tools/infeed_allocator.h",
        "driver/tools/infeed_iterator.h",
        "driver/tools/io_thread.h",
        "driver/tools/mapping_helper.h",
        "driver/tools/matmul_preplanning.h",
        "driver/tools/poplar_util.h",
        "driver/tools/rnn_util.h",
        "driver/tools/seed_generator.h",
        "driver/tools/spsc_outfeed_queue.h",
        "driver/tools/spsc_queue.h",
        "driver/tools/subcomputation_graph_caching.h",
        "driver/tools/tensor_map.h",
        "driver/vertex_templates.h",
        "driver/visitors/deferred_visitor.h",
        "driver/visitors/entry_visitor.h",
        "driver/visitors/pipeline_stage_visitor.h",
        "driver/visitors/pipeline_visitor.h",
        "driver/visitors/visitor_arithmetic_expr.h",
        "driver/visitors/visitor_base.h",
        "driver/visitors/visitor_full.h",
        "driver/visitors/visitor_inline_call.h",
        "driver/visitors/visitor_map.h",
    ],
    copts = [
        "-DTF_POPLAR_BUILD_TAG=" + tf_poplar_build_tag(),
    ],
    deps = [
        ":common",
        ":config_protos_cc_impl",
        ":custom_kernels_util",
        ":optimizers",
        ":option_flag_cc_impl",
        ":pipeline_config_cc_impl",
        "//tensorflow/compiler/plugin/poplar/tools:poplar_executable_runner",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/xla:xla_headers_lib",
        "//tensorflow/compiler/xla/service:cholesky_expander",
        "//tensorflow/compiler/xla/service:dynamic_index_splitter",
        "//tensorflow/compiler/xla/service:generic_transfer_manager",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_constant_folding",
        "//tensorflow/compiler/xla/service:hlo_cost_analysis",
        "//tensorflow/compiler/xla/service:hlo_cse",
        "//tensorflow/compiler/xla/service:hlo_dce",
        "//tensorflow/compiler/xla/service:hlo_get_dimension_size_rewriter",
        "//tensorflow/compiler/xla/service:hlo_graph_dumper",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:hlo_pass_pipeline",
        "//tensorflow/compiler/xla/service:map_inliner",
        "//tensorflow/compiler/xla/service:reshape_mover",
        "//tensorflow/compiler/xla/service:sort_simplifier",
        "//tensorflow/compiler/xla/service:transfer_manager",
        "//tensorflow/compiler/xla/service:triangular_solve_expander",
        "//tensorflow/compiler/xla/service:tuple_simplifier",
        "//tensorflow/compiler/xla/service:while_loop_constant_sinking",
        "//tensorflow/compiler/xla/service:zero_sized_hlo_elimination",
        "//tensorflow/compiler/xla/service/cpu:cpu_runtime",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core/kernels/data:unbounded_thread_pool",
        "//third_party/eigen3",
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
    name = "rearrange_pipeline_stage_arguments",
    srcs = [
        "kernels/pipelining/rearrange_pipeline_stage_arguments.cc",
    ],
    hdrs = [
        "kernels/pipelining/rearrange_pipeline_stage_arguments.h",
    ],
    linkstatic = 1,
    deps = [
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/compiler/tf2xla/ops:xla_ops",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:optional",
    ],
    alwayslink = True,
)

cc_library(
    name = "kernels",
    srcs = [
        "kernels/datastream/dataset_benchmark.cc",
        "kernels/datastream/exporters.cc",
        "kernels/datastream/feeds.cc",
        "kernels/datastream/host_embedding.cc",
        "kernels/host_compute_kernels.cc",
        "kernels/ipu_kernels.cc",
        "kernels/pipelining/pipelining.cc",
        "kernels/popfloat/cast_to_gfloat.cc",
        "kernels/popnn/arg_min_max.cc",
        "kernels/popnn/gelu.cc",
        "kernels/popnn/gru.cc",
        "kernels/popnn/lstm.cc",
        "kernels/popnn/norm.cc",
        "kernels/popnn/onehot.cc",
        "kernels/popnn/pooling.cc",
        "kernels/popnn/relu.cc",
        "kernels/popnn/sigmoid.cc",
        "kernels/popnn/tanh.cc",
        "kernels/popnn/topk.cc",
        "kernels/popops/all_reduce.cc",
        "kernels/popops/all_to_all.cc",
        "kernels/popops/multi_slice.cc",
        "kernels/popops/reduce_scatter.cc",
        "kernels/popops/sparse_softmax_cross_entropy.cc",
        "kernels/poprand/categorical.cc",
        "kernels/poprand/dropout_xla.cc",
        "kernels/poprand/random.cc",
        "kernels/poputil/codelet_expression.cc",
        "kernels/poputil/fifo.cc",
        "kernels/poputil/print_tensor.cc",
        "kernels/poputil/recompute.cc",
        "kernels/poputil/remap.cc",
        "kernels/poputil/replication_factor.cc",
        "kernels/poputil/replication_index.cc",
        "kernels/poputil/stateful_gradient_accumulate.cc",
        "kernels/poputil/user_operation.cc",
        "kernels/sendrecv_kernels.cc",
        "kernels/training_ops.cc",
    ],
    hdrs = [
        "kernels/ipu_kernels_common.h",
    ],
    deps = [
        ":common",
        ":custom_kernels_util",
        ":driver",
        ":rearrange_pipeline_stage_arguments",
        "//tensorflow/compiler/plugin/poplar/kernels/dataset:kernels",
        "//tensorflow/compiler/plugin/poplar/tools:poplar_executable_runner",
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

# When creating a new set of ops the following needs to be done:
#
# 1. New cc_library:
# cc_library(
#     name = "new_ops",
#     srcs = ["new/new_ops.cc"],
#     deps = ["//tensorflow/core:framework"],
#     alwayslink = True,
# )
#
# 2. New python wrapper for those ops:
# tf_gen_op_wrapper_py(
#     name = "new_ops_wrapper_py",
#     out = "gen_new_ops.py",
#     deps = [":new_ops"],
#     visibility = ["//tensorflow:internal"],
# )
#
# 3. New custom op libary:
# tf_custom_op_py_library(
#     name = "new_ops_py",
#     kernels = [":new_ops"],
#     deps = [":new_ops_wrapper_py"],
#     srcs_version = "PY2AND3",
# )
#
# The cc library `new_ops` needs to be added as a dependency to:
# //tensorflow/compiler/plugin/poplar:poplar_lib
# The custom op py library `new_ops_py` needs to be added as a dependency to one
# of the relevant libs in //tensorflow/python/ipu.

cc_library(
    name = "dataset_ops",
    srcs = ["ops/dataset/dataset_ops.cc"],
    deps = ["//tensorflow/core:framework"],
    alwayslink = True,
)

tf_gen_op_wrapper_py(
    name = "dataset_ops_wrapper_py",
    out = "ops/gen_dataset_ops.py",
    visibility = ["//tensorflow:internal"],
    deps = [":dataset_ops"],
)

tf_custom_op_py_library(
    name = "dataset_ops_py",
    kernels = [":dataset_ops"],
    srcs_version = "PY2AND3",
    deps = [":dataset_ops_wrapper_py"],
)

cc_library(
    name = "ipu_ops",
    srcs = ["ops/ipu_ops.cc"],
    deps = ["//tensorflow/core:framework"],
    alwayslink = True,
)

tf_gen_op_wrapper_py(
    name = "ipu_ops_wrapper_py",
    out = "ops/gen_ipu_ops.py",
    deps = [":ipu_ops"],
)

tf_custom_op_py_library(
    name = "ipu_ops_py",
    kernels = [":ipu_ops"],
    deps = [":ipu_ops_wrapper_py"],
)

cc_library(
    name = "popnn_ops",
    srcs = [
        "ops/popnn/gelu.cc",
        "ops/popnn/gru.cc",
        "ops/popnn/lstm.cc",
        "ops/popnn/norm.cc",
    ],
    deps = ["//tensorflow/core:framework"],
    alwayslink = True,
)

tf_gen_op_wrapper_py(
    name = "popnn_ops_wrapper_py",
    out = "ops/gen_popnn_ops.py",
    visibility = ["//tensorflow:internal"],
    deps = [":popnn_ops"],
)

tf_custom_op_py_library(
    name = "popnn_ops_py",
    kernels = [":popnn_ops"],
    srcs_version = "PY2AND3",
    deps = [":popnn_ops_wrapper_py"],
)

cc_library(
    name = "popops_ops",
    srcs = [
        "ops/popops/all_to_all.cc",
        "ops/popops/cross_replica_sum.cc",
        "ops/popops/multi_slice.cc",
        "ops/popops/reduce_scatter.cc",
    ],
    deps = ["//tensorflow/core:framework"],
    alwayslink = True,
)

tf_gen_op_wrapper_py(
    name = "popops_ops_wrapper_py",
    out = "ops/gen_popops_ops.py",
    visibility = ["//tensorflow:internal"],
    deps = [":popops_ops"],
)

tf_custom_op_py_library(
    name = "popops_ops_py",
    kernels = [":popops_ops"],
    srcs_version = "PY2AND3",
    deps = [":popops_ops_wrapper_py"],
)

cc_library(
    name = "popfloat_ops",
    srcs = [
        "ops/popfloat/cast_to_gfloat.cc",
    ],
    deps = ["//tensorflow/core:framework"],
    alwayslink = True,
)

tf_gen_op_wrapper_py(
    name = "popfloat_ops_wrapper_py",
    out = "ops/gen_popfloat_ops.py",
    visibility = ["//tensorflow:internal"],
    deps = [":popfloat_ops"],
)

tf_custom_op_py_library(
    name = "popfloat_ops_py",
    kernels = [":popfloat_ops"],
    srcs_version = "PY2AND3",
    deps = [":popfloat_ops_wrapper_py"],
)

cc_library(
    name = "poprand_ops",
    srcs = [
        "ops/poprand/dropout_tf.cc",
    ],
    deps = ["//tensorflow/core:framework"],
    alwayslink = True,
)

tf_gen_op_wrapper_py(
    name = "poprand_ops_wrapper_py",
    out = "ops/gen_poprand_ops.py",
    visibility = ["//tensorflow:internal"],
    deps = [":poprand_ops"],
)

tf_custom_op_py_library(
    name = "poprand_ops_py",
    kernels = [":poprand_ops"],
    deps = [":poprand_ops_wrapper_py"],
)

cc_library(
    name = "poputil_ops",
    srcs = [
        "ops/poputil/codelet_expression.cc",
        "ops/poputil/fifo.cc",
        "ops/poputil/print_tensor.cc",
        "ops/poputil/recompute.cc",
        "ops/poputil/remap.cc",
        "ops/poputil/remap_deduce.cc",
        "ops/poputil/replication_factor.cc",
        "ops/poputil/replication_index.cc",
        "ops/poputil/stateful_gradient_accumulate.cc",
        "ops/poputil/user_operation.cc",
    ],
    deps = ["//tensorflow/core:framework"],
    alwayslink = True,
)

tf_gen_op_wrapper_py(
    name = "poputil_ops_wrapper_py",
    out = "ops/gen_poputil_ops.py",
    visibility = ["//tensorflow:internal"],
    deps = [":poputil_ops"],
)

tf_custom_op_py_library(
    name = "poputil_ops_py",
    kernels = [":poputil_ops"],
    deps = [":poputil_ops_wrapper_py"],
)

cc_library(
    name = "pop_datastream_ops",
    srcs = [
        "ops/datastream/feeds.cc",
        "ops/datastream/host_embedding.cc",
    ],
    deps = ["//tensorflow/core:framework"],
    alwayslink = True,
)

tf_gen_op_wrapper_py(
    name = "pop_datastream_ops_wrapper_py",
    out = "ops/gen_pop_datastream_ops.py",
    visibility = ["//tensorflow:internal"],
    deps = [":pop_datastream_ops"],
)

tf_custom_op_py_library(
    name = "pop_datastream_ops_py",
    kernels = [":pop_datastream_ops"],
    srcs_version = "PY2AND3",
    deps = [":pop_datastream_ops_wrapper_py"],
)

cc_library(
    name = "sendrecv_ops",
    srcs = ["ops/sendrecv_ops.cc"],
    deps = ["//tensorflow/core:framework"],
    alwayslink = True,
)

cc_library(
    name = "host_compute_ops",
    srcs = ["ops/host_compute_ops.cc"],
    deps = ["//tensorflow/core:framework"],
    alwayslink = True,
)

tf_gen_op_wrapper_py(
    name = "sendrecv_ops_wrapper_py",
    out = "ops/gen_sendrecv_ops.py",
    visibility = ["//tensorflow:internal"],
    deps = [":sendrecv_ops"],
)

tf_custom_op_py_library(
    name = "sendrecv_ops_py",
    kernels = [":sendrecv_ops"],
    srcs_version = "PY2AND3",
    deps = [":sendrecv_ops_wrapper_py"],
)

cc_library(
    name = "pipelining_ops",
    srcs = ["ops/pipelining/pipelining.cc"],
    deps = ["//tensorflow/core:framework"],
    alwayslink = True,
)

tf_gen_op_wrapper_py(
    name = "pipelining_ops_wrapper_py",
    out = "ops/gen_pipelining_ops.py",
    visibility = ["//tensorflow:internal"],
    deps = [":pipelining_ops"],
)

tf_custom_op_py_library(
    name = "pipelining_ops_py",
    kernels = [":pipelining_ops"],
    srcs_version = "PY2AND3",
    deps = [":pipelining_ops_wrapper_py"],
)

cc_library(
    name = "dataset_benchmark",
    srcs = ["ops/datastream/dataset_benchmark.cc"],
    deps = ["//tensorflow/core:framework"],
    alwayslink = True,
)

tf_gen_op_wrapper_py(
    name = "dataset_benchmark_wrapper_py",
    out = "ops/gen_dataset_benchmark.py",
    visibility = ["//tensorflow:internal"],
    deps = [":dataset_benchmark"],
)

tf_custom_op_py_library(
    name = "dataset_benchmark_py",
    kernels = [":dataset_benchmark"],
    srcs_version = "PY2AND3",
    deps = [":dataset_benchmark_wrapper_py"],
)

cc_library(
    name = "dataset_exporters",
    srcs = ["ops/datastream/exporters.cc"],
    deps = ["//tensorflow/core:framework"],
    alwayslink = True,
)

tf_gen_op_wrapper_py(
    name = "dataset_exporters_wrapper_py",
    out = "ops/gen_dataset_exporters.py",
    visibility = ["//tensorflow:internal"],
    deps = [":dataset_exporters"],
)

tf_custom_op_py_library(
    name = "dataset_exporters_py",
    kernels = [":dataset_exporters"],
    srcs_version = "PY2AND3",
    deps = [":dataset_exporters_wrapper_py"],
)

cc_library(
    name = "poplar_lib",
    srcs = [
        "driver/xla_ipu_device.cc",
    ],
    hdrs = [],
    data = [
        ":config_protos_py",
        ":tf_graph_program",
    ],
    linkstatic = 1,
    deps = [
        ":dataset_benchmark",
        ":dataset_exporters",
        ":dataset_ops",
        ":driver",
        ":graph_optimizer_passes",
        ":host_compute_ops",
        ":ipu_ops",
        ":kernels",
        ":pipelining_ops",
        ":pop_datastream_ops",
        ":popfloat_ops",
        ":poplar_backend",
        ":popnn_ops",
        ":popops_ops",
        ":poprand_ops",
        ":poputil_ops",
        ":sendrecv_ops",
        "//tensorflow/compiler/jit:xla_device",
        "//tensorflow/compiler/jit:xla_jit_headers_lib",
        "//tensorflow/compiler/jit/kernels:xla_ops",
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
    deps = [
        ":common",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@com_google_protobuf//:protobuf_headers",
    ],
    alwayslink = True,
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
    ],
)

xla_test(
    name = "combine_instructions_test",
    size = "small",
    srcs = ["tests/combine_instructions_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
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
        "//tensorflow/compiler/xla:xla_data_proto",
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
    ],
)

xla_test(
    name = "pipeline_visitor_test",
    srcs = ["tests/pipeline_visitor_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "pipeline_grouped_visitor_test",
    srcs = ["tests/pipeline_grouped_visitor_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "pipeline_sequential_visitor_test",
    srcs = ["tests/pipeline_sequential_visitor_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
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
    ],
)

xla_test(
    name = "deferred_visitor_test",
    srcs = ["tests/deferred_visitor_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
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
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
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
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu:ipu_lib",
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
    ],
)

xla_test(
    name = "elementwise_broadcast_converter_test",
    srcs = ["tests/elementwise_broadcast_converter_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
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
        "//tensorflow/compiler/xla:xla_data_proto",
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
        "//tensorflow/compiler/xla:xla_data_proto",
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
        "//tensorflow/compiler/xla:xla_data_proto",
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
        "//tensorflow/compiler/xla:xla_data_proto",
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
        "//tensorflow/compiler/xla:xla_data_proto",
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
        "//tensorflow/compiler/xla:xla_data_proto",
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
        "//tensorflow/compiler/xla:xla_data_proto",
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
    ],
)

xla_test(
    name = "multi_update_canonicalize_test",
    size = "small",
    srcs = ["tests/multi_update_canonicalize_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "pipeline_resource_update_fixer_test",
    size = "small",
    srcs = ["tests/pipeline_resource_update_fixer_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
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
    ],
)

xla_test(
    name = "pipeline_resource_variables_offload_test",
    size = "small",
    srcs = ["tests/pipeline_resource_variables_offload_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":optimizers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
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

py_library(
    name = "test_utils_py",
    srcs = ["tests/test_utils.py"],
    deps = [
        ":ipu_ops_py",
        "//tensorflow/compiler/tf2xla/python:xla",
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
        ":ipu_ops_py",
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
)

tf_xla_py_test(
    name = "ml_type_classify_test",
    size = "small",
    srcs = ["tests/ml_type_classify_test.py"],
    enabled_backends = ["poplar"],
    shard_count = 2,
    deps = [
        ":ipu_ops_py",
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras:layers_base",
    ],
)

tf_xla_py_test(
    name = "ipu_model_device_test",
    size = "large",
    srcs = ["tests/ipu_model_device_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":ipu_ops_py",
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "topk_onehot_test",
    size = "small",
    srcs = ["tests/topk_onehot_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":ipu_ops_py",
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
)

tf_xla_py_test(
    name = "topk_test",
    size = "small",
    srcs = ["tests/topk_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":ipu_ops_py",
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
)

tf_xla_py_test(
    name = "infeed_prefetch_test",
    size = "small",
    srcs = ["tests/infeed_prefetch_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":ipu_ops_py",
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "simple_network_test",
    size = "small",
    srcs = ["tests/simple_network_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:platform",
        "//tensorflow/python:variables",
    ],
)

tf_xla_py_test(
    name = "invalid_scheduler_selection",
    size = "small",
    srcs = ["tests/invalid_scheduler_selection.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:platform",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "matmul_test",
    size = "small",
    srcs = ["tests/matmul_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:platform",
        "//tensorflow/python:variables",
    ],
)

tf_xla_py_test(
    name = "matmul_option_test",
    size = "small",
    srcs = ["tests/matmul_option_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:platform",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "conv_test",
    size = "large",
    srcs = ["tests/conv_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":ipu_ops_py",
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "poplar_executable_runner_test",
    size = "large",
    srcs = ["tests/poplar_executable_runner_test.py"],
    data = [
        "//tensorflow/compiler/plugin/poplar/tools:PoplarExecutableRunner",
        "//tensorflow/compiler/plugin/poplar/tools:tensorflow_weights_extractor",
    ],
    enabled_backends = ["poplar"],
    deps = [
        ":ipu_ops_py",
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
    ],
)

tf_xla_py_test(
    name = "serialization_test",
    size = "large",
    srcs = ["tests/serialization_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":ipu_ops_py",
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
    ],
)

tf_xla_py_test(
    name = "conv3d_test",
    size = "large",
    srcs = ["tests/conv3d_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":ipu_ops_py",
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:nn_ops",
    ],
)

tf_xla_py_test(
    name = "constant_test",
    size = "large",
    srcs = ["tests/constant_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":ipu_ops_py",
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:constant_op",
    ],
)

tf_xla_py_test(
    name = "variable_test",
    size = "small",
    srcs = ["tests/variable_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":ipu_ops_py",
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/keras:layers_base",
    ],
)

tf_xla_py_test(
    name = "synthetic_data_test",
    size = "small",
    srcs = ["tests/synthetic_data_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":ipu_ops_py",
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/keras:layers_base",
    ],
)

tf_xla_py_test(
    name = "synthetic_data_with_outfeeds_test",
    size = "small",
    srcs = ["tests/synthetic_data_with_outfeeds_test.py"],
    enabled_backends = ["poplar"],
    shard_count = 2,
    deps = [
        ":ipu_ops_py",
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/keras:layers_base",
    ],
)

tf_xla_py_test(
    name = "multi_run_test",
    size = "small",
    srcs = ["tests/multi_run_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
)

tf_xla_py_test(
    name = "reduce_test",
    size = "small",
    srcs = ["tests/reduce_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":ipu_ops_py",
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
    ],
)

tf_xla_py_test(
    name = "f16_test",
    size = "small",
    srcs = ["tests/f16_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
)

tf_xla_py_test(
    name = "64_bit_test",
    size = "small",
    srcs = ["tests/64_bit_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
)

tf_xla_py_test(
    name = "batch_norm_test",
    size = "small",
    srcs = ["tests/batch_norm_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":ipu_ops_py",
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:platform",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras:layers_base",
    ],
)

tf_xla_py_test(
    name = "fuse_ops_test",
    size = "medium",
    srcs = ["tests/fuse_ops_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":ipu_ops_py",
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/keras:layers_base",
    ],
)

tf_xla_py_test(
    name = "gather_simplifier_embedding_lookup_test",
    size = "medium",
    srcs = ["tests/gather_simplifier_embedding_lookup_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":ipu_ops_py",
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras:layers_base",
    ],
)

tf_xla_py_test(
    name = "tensor_array_test",
    size = "small",
    srcs = ["tests/tensor_array_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:tensor_array_grad",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:variables",
    ],
)

tf_xla_py_test(
    name = "monitored_session_test",
    size = "small",
    srcs = ["tests/monitored_session_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:variables",
        "//tensorflow/python/keras:layers_base",
    ],
)

tf_xla_py_test(
    name = "casts_elimination_test",
    size = "medium",
    srcs = ["tests/casts_elimination_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":ipu_ops_py",
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
)

tf_xla_py_test(
    name = "sendrecv_ops_test",
    size = "large",
    srcs = ["tests/sendrecv_ops_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":sendrecv_ops_py",
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "outside_compilation_scope_test",
    size = "large",
    srcs = ["tests/outside_compilation_scope_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "update_op_dependencies_test",
    size = "medium",
    srcs = ["tests/update_op_dependencies_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":ipu_ops_py",
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
)

tf_xla_py_test(
    name = "norm_graph_caching_test",
    size = "medium",
    srcs = ["tests/norm_graph_caching_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":ipu_ops_py",
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras:layers_base",
    ],
)

tf_xla_py_test(
    name = "norm_graph_caching_sharded_test",
    size = "medium",
    srcs = ["tests/norm_graph_caching_sharded_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":ipu_ops_py",
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras:layers_base",
    ],
)

tf_xla_py_test(
    name = "conv_graph_caching_test",
    size = "medium",
    srcs = ["tests/conv_graph_caching_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":ipu_ops_py",
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/keras:layers_base",
    ],
)

tf_xla_py_test(
    name = "conv_graph_caching_sharded_test",
    size = "medium",
    srcs = ["tests/conv_graph_caching_sharded_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras:layers_base",
    ],
)

tf_xla_py_test(
    name = "lstm_test",
    size = "medium",
    srcs = ["tests/lstm_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":ipu_ops_py",
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras:layers_base",
    ],
)

tf_xla_py_test(
    name = "group_norm_test",
    size = "large",
    srcs = ["tests/group_norm_test.py"],
    enabled_backends = ["poplar"],
    shard_count = 5,
    deps = [
        ":ipu_ops_py",
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras:layers_base",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_test(
    name = "wide_const_expansion_test",
    size = "medium",
    srcs = ["tests/wide_const_expansion_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":ipu_ops_py",
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "forward_allocation_test",
    size = "medium",
    srcs = ["tests/forward_allocation_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":ipu_ops_py",
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/keras:layers_base",
    ],
)

tf_xla_py_test(
    name = "stateful_gradient_accumulate_test",
    size = "medium",
    srcs = ["tests/stateful_gradient_accumulate_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":ipu_ops_py",
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
)

tf_xla_py_test(
    name = "replicated_stateful_gradient_accumulate_test",
    size = "large",
    srcs = ["tests/replicated_stateful_gradient_accumulate_test.py"],
    enabled_backends = ["poplar"],
    shard_count = 2,
    deps = [
        ":ipu_ops_py",
        ":popops_ops_py",
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
)

tf_xla_py_test(
    name = "case_test",
    size = "small",
    srcs = ["tests/case_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras:layers_base",
    ],
)

tf_xla_py_test(
    name = "while_loop_perf_test",
    size = "small",
    srcs = ["tests/while_loop_perf_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/plugin/poplar:ipu_ops_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "mapping_test",
    size = "small",
    srcs = ["tests/mapping_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/plugin/poplar:ipu_ops_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/compiler/xla:compiler_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "remapping_test",
    size = "small",
    srcs = ["tests/remapping_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/plugin/poplar:ipu_ops_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "dump_poplar_info",
    size = "small",
    srcs = ["tests/dump_poplar_info.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/plugin/poplar:ipu_ops_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "conditional_test",
    size = "small",
    srcs = ["tests/conditional_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "combined_weights_test",
    size = "large",
    srcs = ["tests/size_speed_tests/combined_weights_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras:layers_base",
    ],
)

tf_xla_py_test(
    name = "resnet_size_test",
    size = "large",
    srcs = ["tests/size_speed_tests/resnet_size_test.py"],
    enabled_backends = ["poplar"],
    shard_count = 4,
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras:layers_base",
    ],
)

tf_xla_py_test(
    name = "matmul_size_test",
    size = "large",
    srcs = ["tests/size_speed_tests/matmul_size_test.py"],
    enabled_backends = ["poplar"],
    shard_count = 3,
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras:layers_base",
    ],
)

tf_xla_py_test(
    name = "gru_size_test",
    size = "large",
    srcs = ["tests/size_speed_tests/gru_size_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras:layers_base",
    ],
)

tf_xla_py_test(
    name = "lstm_size_test",
    size = "large",
    srcs = ["tests/size_speed_tests/lstm_size_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras:layers_base",
    ],
)

tf_xla_py_test(
    name = "gru_training_test",
    size = "large",
    srcs = ["tests/training_tests/gru_training_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:layers",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "lstm_training_test",
    size = "large",
    srcs = ["tests/training_tests/lstm_training_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:layers",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "while_loop_test",
    size = "medium",
    srcs = ["tests/while_loop_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "while_loop_sharded_test",
    size = "medium",
    srcs = ["tests/while_loop_sharded_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "subcomputation_graph_caching_test",
    size = "medium",
    srcs = ["tests/subcomputation_graph_caching_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "multi_ipu_test",
    size = "medium",
    srcs = ["tests/multi_ipu_test.py"],
    enabled_backends = ["poplar"],
    tags = ["exclusive"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "scalar_elementwise_graph_test",
    size = "medium",
    srcs = ["tests/scalar_elementwise_graph_test.py"],
    enabled_backends = ["poplar"],
    tags = ["exclusive"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "sort_op_py_test",
    size = "medium",
    srcs = ["tests/sort_op_py_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "replicated_graph_test",
    size = "large",
    srcs = ["tests/replicated_graph_test.py"],
    enabled_backends = ["poplar"],
    shard_count = 2,
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "sharded_inplace_test",
    size = "small",
    srcs = ["tests/sharded_inplace_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "sharded_and_replicated_test",
    size = "large",
    srcs = ["tests/sharded_and_replicated_test.py"],
    enabled_backends = ["poplar"],
    shard_count = 2,
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "dropout_hw_test",
    size = "small",
    srcs = ["tests/dropout_hw_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "executable_cache_test",
    size = "medium",
    srcs = ["tests/executable_cache_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
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
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "logging_example_1",
    size = "small",
    srcs = ["docs/logging_example_1.py"],
    enabled_backends = ["poplar"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "perf_training_example",
    size = "small",
    srcs = ["docs/perf_training_example.py"],
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
    args = ["--num-steps=100"],
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
        "--iterations-per-loop=1",
        "--training-steps=1",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "tutorial_basic",
    size = "small",
    srcs = ["docs/tutorial_basic.py"],
    enabled_backends = ["poplar"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "tutorial_sharding",
    size = "small",
    srcs = ["docs/tutorial_sharding.py"],
    enabled_backends = ["poplar"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "tutorial_xla_compile",
    size = "small",
    srcs = ["docs/tutorial_xla_compile.py"],
    enabled_backends = ["poplar"],
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
        ":ipu_ops_py",
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/keras:layers_base",
    ],
)

tf_xla_py_test(
    name = "matmul_graph_caching_test",
    size = "medium",
    srcs = ["tests/matmul_graph_caching_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":ipu_ops_py",
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/keras:layers_base",
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
    size = "small",
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

tf_xla_py_test(
    name = "gru_test",
    size = "medium",
    srcs = ["tests/gru_test.py"],
    enabled_backends = ["poplar"],
    deps = [
        ":ipu_ops_py",
        ":test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras:layers_base",
    ],
)

test_suite(
    name = "all_tests",
    tests = [
        "64_bit_test",
        "add_block_recompute_test",
        "all_to_all_finder_test",
        "allocation_finder_test",
        "apply_recompute_suggestion_test",
        "arg_min_max_test",
        "arithmetic_expr_test",
        "batch_norm_test",
        "bias_apply_graph_caching_test",
        "case_test",
        "casts_elimination_test",
        "combine_instructions_test",
        "commutative_instruction_reorder_operands_test",
        "conditional_test",
        "constant_nan_test",
        "constant_slice_folding_test",
        "conv3d_test",
        "conv_graph_caching_sharded_test",
        "conv_graph_caching_test",
        "conv_test",
        "conversions_test",
        "convolution_classifier_test",
        "data_initializer_test",
        "deferred_visitor_test",
        "dependency_replacer_test",
        "device_connection_test",
        "dump_poplar_info",
        "dynamic_slice_layout_test",
        "dynamic_slice_test",
        "elementwise_broadcast_converter_test",
        "executable_cache_test",
        "expression_outliner_test",
        "extract_outside_compilation_pass_test",
        "f16_constant_folding_test",
        "f16_test",
        "find_all_users_test",
        "forward_allocation_test",
        "fuse_ops_test",
        "fuse_wide_const_test",
        "fused_batch_norm_test",
        "gather_simplifier_embedding_lookup_test",
        "gradient_accumulation_fuser_test",
        "gradient_accumulation_optimization_pass_test",
        "graph_compile_io_map_test",
        "group_norm_test",
        "gru_test",
        "hlo_computation_name_uniquify_test",
        "hlo_hash_test",
        "hlo_matcher_test",
        "host_compute_barrier_inserter_test",
        "infeed_prefetch_test",
        "inplace_test",
        "inter_ipu_copy_inserter_test",
        "invalid_scheduler_selection",
        "ipu_model_device_test",
        "layout_strip_test",
        "lift_recompute_suggestion_test",
        "lstm_test",
        "mapping_test",
        "matmul_combiner_test",
        "matmul_graph_caching_test",
        "matmul_option_test",
        "matmul_test",
        "ml_type_classify_test",
        "module_flatten_test",
        "monitored_session_test",
        "multi_ipu_test",
        "multi_run_test",
        "multi_slice_combiner_test",
        "multi_update_apply_test",
        "multi_update_canonicalize_test",
        "multi_update_combiner_test",
        "multi_update_scale_apply_test",
        "norm_graph_caching_sharded_test",
        "norm_graph_caching_test",
        "not_supported_gather_expander_test",
        "not_supported_scatter_expander_test",
        "outside_compilation_scope_test",
        "pipeline_communication_optimizer_test",
        "pipeline_copy_inserter_test",
        "pipeline_feed_hoisting_test",
        "pipeline_fifo_inserter_test",
        "pipeline_fixer_test",
        "pipeline_grouped_recomputation_test",
        "pipeline_grouped_visitor_test",
        "pipeline_optimizer_test",
        "pipeline_recomputation_test",
        "pipeline_resource_update_fixer_test",
        "pipeline_sequential_visitor_test",
        "pipeline_util_test",
        "pipeline_visitor_test",
        "poplar_algebraic_simplifier_test",
        "poplar_executable_runner_test",
        "recomputation_test",
        "recompute_suggestion_test",
        "reduce_test",
        "remapping_test",
        "remove_blocked_recompute_test",
        "reorder_gradient_accumulation_pass_test",
        "root_token_replacer_test",
        "scalar_elementwise_graph_test",
        "scatter_simplifier_test",
        "scheduler_tree_test",
        "sendrecv_ops_test",
        "serialization_test",
        "sharded_inplace_test",
        "sharding_pass_test",
        "shortest_path_scheduler_test",
        "simple_network_test",
        "slice_plan_test",
        "sort_op_cc_test",
        "sort_op_py_test",
        "sparse_softmax_test",
        "stateful_gradient_accumulate_test",
        "stateful_noop_test",
        "subcomputation_graph_caching_test",
        "sync_list_scheduler_test",
        "synthetic_data_test",
        "synthetic_data_with_outfeeds_test",
        "tensor_array_test",
        "topk_onehot_test",
        "update_op_dependencies_test",
        "variable_test",
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
        "distributed_training_example",
        "ipu_estimator_example",
        "ipu_pipeline_estimator_example",
        "logging_example_1",
        "perf_training_example",
        "tutorial_basic",
        "tutorial_custom_codelet",
        "tutorial_sharding",
        "tutorial_xla_compile",
    ],
)

test_suite(
    name = "poplar_test_suite",
    tests = [
        ":all_tests",
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

test_suite(
    name = "replicated_tests",
    tests = [
        "replicated_graph_test",
        "replicated_stateful_gradient_accumulate_test",
        "sharded_and_replicated_test",
    ],
)

test_suite(
    name = "hardware_tests",
    tests = [
        ":dropout_hw_test",
        ":executable_cache_test",
        ":random_ops",
        ":replicated_tests",
    ],
)

exports_files([
    "disabled_manifest.txt",
    "disables_xla_tests_manifest.txt",
])

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
