load("//tensorflow/compiler/plugin/poplar:poplar.bzl", "poplar_cc_library")

licenses(["restricted"])

package(default_visibility = ["//tensorflow/compiler/plugin/poplar:__subpackages__"])

poplar_cc_library(
    name = "custom_ops_dependencies",
    visibility = ["//visibility:private"],
    deps = [
        ":hlo_poplar_instruction",
        "//tensorflow/compiler/plugin/poplar/driver/tools:conv_util",
    ],
)

poplar_cc_library(
    name = "all_gather",
    srcs = [
        "all_gather.cc",
    ],
    hdrs = [
        "all_gather.h",
    ],
    deps = [
        ":custom_ops_dependencies",
        "//tensorflow/compiler/plugin/poplar/driver/tools:poplar_replica_groups",
    ],
)

poplar_cc_library(
    name = "within_replicas",
    srcs = [
        "within_replicas.cc",
    ],
    hdrs = [
        "within_replicas.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "arg_min_max",
    srcs = [
        "arg_min_max.cc",
    ],
    hdrs = [
        "arg_min_max.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "assert",
    srcs = [
        "assert.cc",
    ],
    hdrs = [
        "assert.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "assume_equal_across_replicas",
    srcs = [
        "assume_equal_across_replicas.cc",
    ],
    hdrs = [
        "assume_equal_across_replicas.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "batch_norm_stats",
    srcs = [
        "batch_norm_stats.cc",
    ],
    deps = [
        ":norm",
    ],
)

poplar_cc_library(
    name = "barrier",
    srcs = [
        "barrier.cc",
    ],
    hdrs = [
        "barrier.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "candidate_sampler",
    srcs = [
        "candidate_sampler.cc",
    ],
    hdrs = [
        "candidate_sampler.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "cast_to_gfloat_hlo",
    srcs = [
        "cast_to_gfloat_hlo.cc",
    ],
    hdrs = [
        "cast_to_gfloat_hlo.h",
    ],
    deps = [
        ":custom_ops_dependencies",
        "//tensorflow/compiler/plugin/poplar/kernels:popfloat_config_protos_cc_impl",
        "//tensorflow/compiler/tf2xla:common",
    ],
)

poplar_cc_library(
    name = "codelet_expression_op",
    srcs = [
        "codelet_expression_op.cc",
    ],
    hdrs = [
        "codelet_expression_op.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "collective_reorder",
    srcs = [
        "collective_reorder.cc",
    ],
    hdrs = [
        "collective_reorder.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "conv_with_reverse",
    srcs = [
        "conv_with_reverse.cc",
    ],
    hdrs = [
        "conv_with_reverse.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "copy_into",
    srcs = [
        "copy_into.cc",
    ],
    hdrs = [
        "copy_into.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "ctc_loss",
    srcs = [
        "ctc_loss.cc",
    ],
    hdrs = [
        "ctc_loss.h",
    ],
    deps = [
        ":custom_ops_dependencies",
        "//tensorflow/compiler/tf2xla:common",
    ],
)

poplar_cc_library(
    name = "dropout_hlo",
    srcs = [
        "dropout_hlo.cc",
    ],
    hdrs = [
        "dropout_hlo.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "elementwise",
    srcs = [
        "elementwise.cc",
    ],
    hdrs = [
        "elementwise.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "execution_counter",
    srcs = [
        "execution_counter.cc",
    ],
    hdrs = [
        "execution_counter.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "fifo",
    srcs = [
        "fifo.cc",
    ],
    hdrs = [
        "fifo.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "gru",
    srcs = [
        "gru.cc",
    ],
    hdrs = [
        "gru.h",
    ],
    deps = [
        ":rnn",
    ],
)

poplar_cc_library(
    name = "histogram",
    srcs = [
        "histogram.cc",
    ],
    hdrs = [
        "histogram.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "hlo_extensions",
    srcs = [
        "hlo_extensions.cc",
    ],
    hdrs = [
        "hlo_extensions.h",
        "hlo_poplar_instruction.h",
        "//tensorflow/compiler/plugin/poplar/driver/tools:extension_registry.h",
        "//tensorflow/compiler/plugin/poplar/driver/tools:hash.h",
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_instruction_extensions.h",
        "//tensorflow/compiler/plugin/poplar/driver/tools:inplace_util.h",
        "//tensorflow/compiler/plugin/poplar/driver/tools:matcher_predicates.h",
        "//tensorflow/compiler/plugin/poplar/driver/tools:pipeline_util.h",
        "//tensorflow/compiler/plugin/poplar/driver/tools:tensor_location.h",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:backend_config_proto_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:config_proto_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:option_flag_proto_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:threestate_proto_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver/tools:alias_info_proto_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver/tools:flags",
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_poplar_buffer",
        "//tensorflow/compiler/plugin/poplar/kernels:ops_proto_cc",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc_impl",
        "//tensorflow/compiler/xla/service:buffer_value",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_group",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:hlo_reachability",
        "//tensorflow/compiler/xla/service:hlo_value",
        "//tensorflow/core/protobuf:error_codes_proto_impl_cc_impl",
        "//tensorflow/stream_executor:dnn_proto_cc_impl",
        "//third_party/eigen3",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_protobuf//:protobuf_headers",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "hlo_poplar_instruction",
    srcs = [
        "hlo_poplar_instruction.cc",
    ],
    hdrs = [
        "hlo_poplar_instruction.h",
        "//tensorflow/compiler/plugin/poplar/driver/tools:inplace_util.h",
    ],
    deps = [
        ":hlo_extensions",
        "//tensorflow/compiler/plugin/poplar/driver/tools:hash",
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_poplar_buffer",
        "//tensorflow/compiler/plugin/poplar/driver/tools:tensor_location",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_reachability",
        "//tensorflow/stream_executor:platform",
        "//tensorflow/stream_executor/cuda:cuda_platform_id",
        "//tensorflow/stream_executor/host:host_platform_id",
        "//tensorflow/stream_executor/platform:dso_loader",
        "//tensorflow/stream_executor/rocm:rocm_platform_id",
        "//third_party/eigen3",
        "@com_google_absl//absl/strings",
    ],
)

poplar_cc_library(
    name = "hlo_remote_buffer_info",
    hdrs = [
        "hlo_remote_buffer_info.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "host_embedding",
    srcs = [
        "host_embedding.cc",
    ],
    hdrs = [
        "host_embedding.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "inter_tileset_copy",
    srcs = [
        "inter_tileset_copy.cc",
    ],
    hdrs = [
        "inter_tileset_copy.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "inter_ipu_copy",
    srcs = [
        "inter_ipu_copy.cc",
    ],
    hdrs = [
        "inter_ipu_copy.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "lstm",
    srcs = [
        "lstm.cc",
    ],
    hdrs = [
        "lstm.h",
    ],
    deps = [
        ":rnn",
    ],
)

poplar_cc_library(
    name = "multi_conv",
    srcs = [
        "multi_conv.cc",
    ],
    hdrs = [
        "multi_conv.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "multi_slice",
    srcs = [
        "multi_slice.cc",
    ],
    hdrs = [
        "multi_slice.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "non_linearity",
    srcs = [
        "non_linearity.cc",
    ],
    hdrs = [
        "non_linearity.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "normalise_image",
    srcs = [
        "normalise_image.cc",
    ],
    hdrs = [
        "normalise_image.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "norm",
    srcs = [
        "group_norm_grad.cc",
        "group_norm_infer.cc",
        "group_norm_stats.cc",
        "group_norm_train.cc",
        "norm.cc",
    ],
    hdrs = [
        "norm.h",
    ],
    deps = [
        ":custom_ops_dependencies",
        "//tensorflow/compiler/tf2xla:common",
    ],
)

poplar_cc_library(
    name = "onehot",
    srcs = [
        "onehot.cc",
    ],
    hdrs = [
        "onehot.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "pooling",
    srcs = [
        "pooling.cc",
    ],
    hdrs = [
        "pooling.h",
    ],
    deps = [
        ":custom_ops_dependencies",
        "//tensorflow/compiler/tf2xla:common",
    ],
)

poplar_cc_library(
    name = "print_tensor",
    srcs = [
        "print_tensor.cc",
    ],
    hdrs = [
        "print_tensor.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "recompute",
    srcs = [
        "recompute.cc",
    ],
    hdrs = [
        "recompute.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "recv_from_host",
    srcs = [
        "recv_from_host.cc",
    ],
    hdrs = [
        "recv_from_host.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "reduce_many",
    srcs = [
        "reduce_many.cc",
    ],
    hdrs = [
        "reduce_many.h",
    ],
    deps = [
        ":custom_ops_dependencies",
        "//tensorflow/compiler/plugin/poplar/driver/tools:poplar_replica_groups",
        "//tensorflow/compiler/plugin/poplar/driver/tools:reduction_util",
    ],
)

poplar_cc_library(
    name = "reduce_scatter",
    srcs = [
        "reduce_scatter.cc",
    ],
    hdrs = [
        "reduce_scatter.h",
    ],
    deps = [
        ":custom_ops_dependencies",
        "//tensorflow/compiler/plugin/poplar/driver/tools:poplar_replica_groups",
    ],
)

poplar_cc_library(
    name = "remap_deduce",
    srcs = [
        "remap_deduce.cc",
    ],
    hdrs = [
        "remap_deduce.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "remap",
    srcs = [
        "remap.cc",
    ],
    hdrs = [
        "remap.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "remote_parameter",
    srcs = [
        "remote_parameter.cc",
    ],
    hdrs = [
        "remote_parameter.h",
    ],
    deps = [
        ":custom_ops_dependencies",
        ":hlo_remote_buffer_info",
        "//tensorflow/compiler/plugin/poplar/driver/tools:offloading_util",
    ],
)

poplar_cc_library(
    name = "replication_factor",
    srcs = [
        "replication_factor.cc",
    ],
    hdrs = [
        "replication_factor.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "rnn",
    srcs = [
        "rnn.cc",
    ],
    hdrs = [
        "rnn.h",
    ],
    deps = [
        ":custom_ops_dependencies",
        "//tensorflow/compiler/tf2xla:common",
    ],
)

poplar_cc_library(
    name = "scaled_inplace",
    srcs = [
        "scaled_inplace.cc",
    ],
    hdrs = [
        "scaled_inplace.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "seed",
    srcs = [
        "seed.cc",
    ],
    hdrs = [
        "seed.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "send_recv_barrier",
    srcs = [
        "send_recv_barrier.cc",
    ],
    hdrs = [
        "send_recv_barrier.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "send_to_host",
    srcs = [
        "send_to_host.cc",
    ],
    hdrs = [
        "send_to_host.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "sequence_slice",
    srcs = [
        "sequence_slice.cc",
    ],
    hdrs = [
        "sequence_slice.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "slice_apply",
    srcs = [
        "slice_apply.cc",
    ],
    hdrs = [
        "slice_apply.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "sparse",
    srcs = [
        "sparse.cc",
    ],
    hdrs = [
        "sparse.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "stateful_gradient_accumulate",
    srcs = [
        "stateful_gradient_accumulate.cc",
    ],
    hdrs = [
        "stateful_gradient_accumulate.h",
    ],
    deps = [
        ":custom_ops_dependencies",
        ":hlo_remote_buffer_info",
    ],
)

poplar_cc_library(
    name = "stateful_noop",
    srcs = [
        "stateful_noop.cc",
    ],
    hdrs = [
        "stateful_noop.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "stateless_random",
    srcs = [
        "stateless_random.cc",
    ],
    hdrs = [
        "stateless_random.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "topk",
    srcs = [
        "topk.cc",
    ],
    hdrs = [
        "topk.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "truncated_normal",
    srcs = [
        "truncated_normal.cc",
    ],
    hdrs = [
        "truncated_normal.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "user_op_hlo",
    srcs = [
        "user_op_hlo.cc",
    ],
    hdrs = [
        "user_op_hlo.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "weights_transpose_chans_flip_xy",
    srcs = [
        "weights_transpose_chans_flip_xy.cc",
    ],
    hdrs = [
        "weights_transpose_chans_flip_xy.h",
    ],
    deps = [
        ":custom_ops_dependencies",
    ],
)

poplar_cc_library(
    name = "custom_ops",
    deps = [
        ":all_gather",
        ":arg_min_max",
        ":assert",
        ":assume_equal_across_replicas",
        ":barrier",
        ":batch_norm_stats",
        ":candidate_sampler",
        ":cast_to_gfloat_hlo",
        ":codelet_expression_op",
        ":collective_reorder",
        ":conv_with_reverse",
        ":copy_into",
        ":ctc_loss",
        ":dropout_hlo",
        ":elementwise",
        ":execution_counter",
        ":fifo",
        ":gru",
        ":histogram",
        ":hlo_extensions",
        ":hlo_poplar_instruction",
        ":hlo_remote_buffer_info",
        ":host_embedding",
        ":inter_ipu_copy",
        ":inter_tileset_copy",
        ":lstm",
        ":multi_conv",
        ":multi_slice",
        ":non_linearity",
        ":norm",
        ":normalise_image",
        ":onehot",
        ":pooling",
        ":print_tensor",
        ":recompute",
        ":recv_from_host",
        ":reduce_many",
        ":reduce_scatter",
        ":remap",
        ":remap_deduce",
        ":remote_parameter",
        ":replication_factor",
        ":rnn",
        ":scaled_inplace",
        ":seed",
        ":send_recv_barrier",
        ":send_to_host",
        ":sequence_slice",
        ":slice_apply",
        ":sparse",
        ":stateful_gradient_accumulate",
        ":stateful_noop",
        ":stateless_random",
        ":topk",
        ":truncated_normal",
        ":user_op_hlo",
        ":weights_transpose_chans_flip_xy",
        ":within_replicas",
    ],
    alwayslink = True,
)
