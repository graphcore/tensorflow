load("//tensorflow/compiler/xla/tests:build_defs.bzl", "xla_test")
load("//tensorflow/compiler/tests:build_defs.bzl", "tf_xla_py_test")
load("@local_config_poplar//poplar:build_defs.bzl", "tf_poplar_build_tag")
load("@local_config_ipu_horovod//:build_defs_horovod.bzl", "horovod_py_test", "if_horovod", "poprun_py_test")
load("//tensorflow:tensorflow.bzl", "py_binary", "tf_cc_binary", "tf_cc_test")
load("//tensorflow/compiler/plugin/poplar:poplar.bzl", "poplar_cc_library")

licenses(["restricted"])

package(default_visibility = [
    "//tensorflow/compiler/plugin/poplar:__subpackages__",
    "//tensorflow/python/ipu:__subpackages__",
])

poplar_cc_library(
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

poplar_cc_library(
    name = "poplar_version",
    srcs = ["//tensorflow/compiler/plugin/poplar/driver:generate_poplar_version"],
)

# Rule for generating the custom codelet example
genrule(
    name = "custom_codelet_example",
    srcs = ["docs/custom_codelet.cpp"],
    outs = ["custom_codelet.gp"],
    cmd = "$(location @local_config_poplar//poplar:popc) -DNDEBUG -O3 -I . -o $@ $<",
    tools = ["@local_config_poplar//poplar:popc"],
)

poplar_cc_library(
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

poplar_cc_library(
    name = "poplar_test_base",
    testonly = True,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_poplar_test_base",
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_poplar_test_util",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

poplar_cc_library(
    name = "poplar_lib",
    srcs = [
    ],
    hdrs = [],
    copts = [
        "-DTF_POPLAR_BUILD_TAG=" + tf_poplar_build_tag(),
    ],
    data = [
        "//tensorflow/compiler/plugin/poplar/driver:config_py",
        "//tensorflow/compiler/plugin/poplar/driver:trace_py",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":graph_optimizer_passes",
        ":poplar_version",
        ":poplar_version_generator",
        "//tensorflow/compiler/plugin/poplar/driver",
        "//tensorflow/compiler/plugin/poplar/kernels",
        "//tensorflow/compiler/plugin/poplar/ops",
        "//tensorflow/compiler/xla/service:fusion_queue",
        "//tensorflow/compiler/xla/service:instruction_fusion",
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

poplar_cc_library(
    name = "ipu_config",
    srcs = [
        "config/ipu_config.cc",
    ],
    hdrs = [
        "config/ipu_config.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":poplar_lib",
        "//tensorflow/compiler/plugin/poplar/driver:config_cc_impl",
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

poplar_cc_library(
    name = "test_utils",
    testonly = True,
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
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:pattern_matcher_gmock",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "all_reduce_simplifier_test",
    srcs = ["tests/all_reduce_simplifier_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
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
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:pattern_matcher_gmock",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "map_hlo_instruction_to_debug_id_test",
    srcs = ["tests/map_hlo_instruction_to_debug_id_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
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
        "//tensorflow/compiler/plugin/poplar/driver:backend_config_cc_impl",
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
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "instruction_extensions_test",
    size = "small",
    srcs = ["tests/instruction_extensions_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
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
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "recompute_casts_test",
    srcs = ["tests/recompute_casts_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
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
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "repeat_loop_copy_inserter_test",
    srcs = ["tests/repeat_loop_copy_inserter_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
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
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
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
        ":poplar_test_base",
        ":test_utils",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "remote_buffer_merger_test",
    size = "small",
    srcs = ["tests/remote_buffer_merger_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "remote_parameter_parallel_combiner_test",
    size = "small",
    srcs = ["tests/remote_parameter_parallel_combiner_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
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
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "inter_tileset_copy_inserter_test",
    size = "small",
    srcs = ["tests/inter_tileset_copy_inserter_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "io_tiles_placer_test",
    size = "small",
    srcs = ["tests/io_tiles_placer_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
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
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "host_compute_schedule_optimizer_test",
    size = "small",
    srcs = ["tests/host_compute_schedule_optimizer_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "host_embedding_notify_pass_test",
    size = "small",
    srcs = ["tests/host_embedding_notify_pass_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
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
        ":poplar_test_base",
        ":test_utils",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "pipeline_verifier_test",
    srcs = ["tests/pipeline_verifier_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
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
        ":poplar_test_base",
        ":test_utils",
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
        ":poplar_test_base",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "prng_seed_state_test",
    srcs = ["tests/prng_seed_state_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    tags = ["hw_poplar_test"],
    deps = [
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
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "debug_info_test",
    size = "small",
    srcs = ["tests/debug_info_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
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
        ":poplar_test_base",
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
        ":poplar_test_base",
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
    name = "multi_slice_update_constant_indices_test",
    size = "large",
    srcs = ["tests/multi_slice_update_constant_indices_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":test_utils",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "dynamic_slice_test",
    size = "medium",
    srcs = ["tests/dynamic_slice_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
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
    name = "create_buffer_test",
    size = "medium",
    srcs = ["tests/create_buffer_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
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
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "pipeline_tuple_remover_test",
    size = "small",
    srcs = ["tests/pipeline_tuple_remover_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "pipeline_stage_merger_test",
    size = "small",
    srcs = ["tests/pipeline_stage_merger_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
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
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "pipeline_recomputation_stage_inserter_test",
    size = "small",
    srcs = ["tests/pipeline_recomputation_stage_inserter_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
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
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "conv_bwd_input_to_fwd_weights_transpose_test",
    size = "small",
    srcs = ["tests/conv_bwd_input_to_fwd_weights_transpose_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "resource_update_fixer_test",
    size = "small",
    srcs = ["tests/resource_update_fixer_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "resource_update_merger_test",
    size = "small",
    srcs = ["tests/resource_update_merger_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
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
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "pipeline_gradient_accumulation_optimizer_test",
    size = "small",
    srcs = ["tests/pipeline_gradient_accumulation_optimizer_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
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
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "variables_offload_and_partition_test",
    size = "small",
    srcs = ["tests/variables_offload_and_partition_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "resource_update_schedule_optimizer_test",
    size = "small",
    srcs = ["tests/resource_update_schedule_optimizer_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
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
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "serialize_gradient_accumulation_test",
    size = "small",
    srcs = ["tests/serialize_gradient_accumulation_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "post_serialize_gradient_accumulation_test",
    size = "small",
    srcs = ["tests/post_serialize_gradient_accumulation_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
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
        "//tensorflow/compiler/xla:xla_data_proto",
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
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "copy_inserter_test",
    size = "small",
    srcs = ["tests/copy_inserter_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "seed_hoisting_test",
    size = "small",
    srcs = ["tests/seed_hoisting_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "function_optimizer_test",
    size = "small",
    srcs = ["tests/function_optimizer_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "pipeline_batch_serialization_buffer_inserter_test",
    size = "small",
    srcs = ["tests/pipeline_batch_serialization_buffer_inserter_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "pipeline_batch_serialization_loop_inserter_test",
    size = "small",
    srcs = ["tests/pipeline_batch_serialization_loop_inserter_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "gradient_accumulation_buffers_offload_test",
    size = "small",
    srcs = ["tests/gradient_accumulation_buffers_offload_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "recomputation_checkpoint_remover_test",
    size = "small",
    srcs = ["tests/recomputation_checkpoint_remover_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "recomputation_input_remover_test",
    size = "small",
    srcs = ["tests/recomputation_input_remover_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "fusion_inliner_test",
    srcs = ["tests/fusion_inliner_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":test_utils",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "computation_flattener_test",
    size = "small",
    srcs = ["tests/computation_flattener_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "multi_use_feeds_finder_test",
    srcs = ["tests/multi_use_feeds_finder_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":test_utils",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "graph_caching_with_remote_buffer_test",
    srcs = ["tests/graph_caching_with_remote_buffer_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    tags = ["hw_poplar_test"],
    deps = [
        ":poplar_test_base",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "remote_buffer_canonicalizer_test",
    srcs = ["tests/remote_buffer_canonicalizer_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "offloading_util_test",
    srcs = ["tests/offloading_util_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
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
        ":poplar_test_base",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "outline_remote_buffers_test",
    srcs = ["tests/outline_remote_buffers_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "function_combiner_test",
    size = "small",
    srcs = ["tests/function_combiner_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "distributed_batch_norm_decomposer_test",
    srcs = ["tests/distributed_batch_norm_decomposer_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
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
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "util_test",
    size = "small",
    srcs = ["tests/util_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

py_library(
    name = "test_utils_py",
    testonly = True,
    srcs = ["tests/test_utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/tests:xla_test",
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

xla_test(
    name = "replicated_resource_update_elementwise_clustering_hw_test",
    srcs = ["tests/replicated_resource_update_elementwise_clustering_hw_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    tags = ["hw_poplar_test"],
    deps = [
        ":poplar_test_base",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "replicated_resource_update_elementwise_clustering_test",
    srcs = ["tests/replicated_resource_update_elementwise_clustering_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":poplar_test_base",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "resource_update_elementwise_clustering_test",
    srcs = ["tests/resource_update_elementwise_clustering_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        ":test_utils",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "resource_update_copy_inserter_test",
    srcs = ["tests/resource_update_copy_inserter_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "embeddings_gradient_optimizer_test",
    srcs = ["tests/embeddings_gradient_optimizer_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "fuse_ops_early_test",
    srcs = ["tests/fuse_ops_early_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
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
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "hlo_poplar_buffer_test",
    srcs = ["tests/hlo_poplar_buffer_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "hlo_poplar_buffer_util_test",
    srcs = ["tests/hlo_poplar_buffer_util_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "nonlinearity_hlo_test",
    srcs = ["tests/nonlinearity_hlo_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
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
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

xla_test(
    name = "hlo_poplar_dataflow_analysis_test",
    srcs = ["tests/hlo_poplar_dataflow_analysis_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
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
    name = "tutorial_basic",
    size = "large",
    srcs = ["docs/tutorial_basic.py"],
    enabled_backends = ["poplar"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "tutorial_sharding",
    size = "large",
    srcs = ["docs/tutorial_sharding.py"],
    enabled_backends = ["poplar"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_xla_py_test(
    name = "tutorial_xla_compile",
    size = "large",
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
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
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
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
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
        "multi_slice_update_constant_indices_test",
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
        "matmul_size_test",
        "resnet_size_test",
    ],
)

test_suite(
    name = "doc_examples",
    tests = [
        "custom_add_inplace",
        "distributed_training_example",
        "function_block1",
        "function_block2",
        "function_block3",
        "function_example1",
        "function_example2",
        "ipu_estimator_example",
        "ipu_pipeline_estimator_example",
        "outfeed_example",
        "perf_training_example",
        "pre_compile_example_runner",
        "tutorial_attributes_example",
        "tutorial_basic",
        "tutorial_custom_codelet",
        "tutorial_sharding",
        "tutorial_xla_compile",
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
