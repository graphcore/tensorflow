load("//tensorflow/compiler/plugin/poplar:poplar.bzl", "poplar_cc_library")

licenses(["restricted"])

package(default_visibility = ["//tensorflow/compiler/plugin/poplar:__subpackages__"])

poplar_cc_library(
    name = "poplar_ops",
    srcs = [
        "execution_counter.cc",
        "host_embedding.cc",
        "recv_from_host.cc",
        "send_recv_barrier.cc",
        "send_to_host.cc",
    ],
    hdrs = [
        "poplar_ops.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:execution_counter",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:host_embedding",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:recv_from_host",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:send_recv_barrier",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:send_to_host",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "ops",
    deps = [
        ":poplar_ops",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popfloat:cast_to_gfloat",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popfloat:gfloat_ops_utils",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplin:cholesky",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplin:fp8_ops",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplin:triangular_solve",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplin:weights_transpose_chans_flip_xy",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popnn:arg_min_max",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popnn:ctc_loss",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popnn:gru",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popnn:lstm",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popnn:non_linearity",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popnn:norm",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popnn:onehot",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popnn:pooling",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popnn:topk",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popops:all_gather",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popops:collective_reorder",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popops:conv_ops",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popops:elementwise",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popops:expression_helpers",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popops:f8_convert",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popops:fusions",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popops:gather_ops",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popops:histogram",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popops:mask",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popops:matmul_ops",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popops:multi_slice",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popops:normalise_image",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popops:random",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popops:reduce_many",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popops:reduce_scatter",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popops:scaled_inplace",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popops:scatter_ops",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popops:select_scalars",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popops:sequence_slice",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popops:slice_apply",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popops:sort_ops",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popops:within_replicas",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poprand:candidate_sampler",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poprand:dropout",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poprand:random",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poputil:assert",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poputil:assume_equal_across_replicas",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poputil:barrier",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poputil:codelet_expression_op",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poputil:copy_into",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poputil:fifo",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poputil:inter_ipu_copy",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poputil:inter_tileset_copy",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poputil:print_tensor",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poputil:remap",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poputil:remap_deduce",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poputil:remote_parameter",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poputil:replication_factor",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poputil:replication_index",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poputil:stateful_gradient_accumulate",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poputil:stateful_noop",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poputil:uninitialised",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poputil:user_op",
    ],
)
