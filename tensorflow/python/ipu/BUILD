# Description:
# Python support for IPU.
#

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("@local_config_ipu_horovod//:build_defs_horovod.bzl", "if_horovod", "poprun_py_test")

py_library(
    name = "ipu_ops_lib",
    srcs = [
        "ops/application_compile_op.py",
        "ops/control_flow_ops.py",
        "ops/cross_replica_ops.py",
        "ops/embedded_runtime.py",
        "ops/embedding_ops.py",
        "ops/functional_ops.py",
        "ops/image_ops.py",
        "ops/internal_ops.py",
        "ops/math_ops.py",
        "ops/nn_ops.py",
        "ops/normalization_ops.py",
        "ops/op_util.py",
        "ops/pipelining_ops.py",
        "ops/rand_ops.py",
        "ops/slicing_ops.py",
        "ops/statistics_ops.py",
        "ops/within_replica_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:backend_config_py",
        "//tensorflow/compiler/plugin/poplar/driver:pipeline_config_py",
        "//tensorflow/compiler/plugin/poplar/driver:poplar_executable_protos_py",
        "//tensorflow/compiler/plugin/poplar/driver:poplar_feed_config_py",
        "//tensorflow/compiler/plugin/poplar/ops:application_runtime_py",
        "//tensorflow/compiler/plugin/poplar/ops:dataset_benchmark_py",
        "//tensorflow/compiler/plugin/poplar/ops:functional_ops_py",
        "//tensorflow/compiler/plugin/poplar/ops:ipu_ops_py",
        "//tensorflow/compiler/plugin/poplar/ops:pop_datastream_ops_py",
        "//tensorflow/compiler/plugin/poplar/ops:popfloat_ops_py",
        "//tensorflow/compiler/plugin/poplar/ops:popnn_ops_py",
        "//tensorflow/compiler/plugin/poplar/ops:popops_ops_py",
        "//tensorflow/compiler/plugin/poplar/ops:poprand_ops_py",
        "//tensorflow/compiler/plugin/poplar/ops:poputil_ops_py",
        "//tensorflow/compiler/xla:xla_data_proto_py",
    ],
)

py_library(
    name = "ipu_grad_ops_lib",
    srcs = [
        "ops/all_to_all_op_grad.py",
        "ops/control_flow_ops_grad.py",
        "ops/cross_replica_ops_grad.py",
        "ops/embedding_ops_grad.py",
        "ops/experimental/popfloat_ops_grad.py",
        "ops/functional_ops_grad.py",
        "ops/internal_ops_grad.py",
        "ops/nn_ops_grad.py",
        "ops/normalization_ops_grad.py",
        "ops/pipelining_ops_grad.py",
        "ops/rand_ops_grad.py",
        "ops/rnn_ops_grad.py",
        "ops/slicing_ops_grad.py",
        "ops/within_replica_ops_grad.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/compiler/plugin/poplar/ops:functional_ops_py",
        "//tensorflow/compiler/plugin/poplar/ops:ipu_ops_py",
        "//tensorflow/compiler/plugin/poplar/ops:popnn_ops_py",
        "//tensorflow/compiler/plugin/poplar/ops:popops_ops_py",
        "//tensorflow/compiler/plugin/poplar/ops:poprand_ops_py",
        "//tensorflow/compiler/plugin/poplar/ops:poputil_ops_py",
    ],
)

py_library(
    name = "data",
    srcs = [
        "data/__init__.py",
        "data/ops/__init__.py",
        "data/ops/dataset_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/compiler/plugin/poplar/ops:dataset_ops_py",
        "//tensorflow/compiler/xla:xla_data_proto_py",
        "//tensorflow/python/distribute:values",
    ],
)

py_library(
    name = "dataset_extractor",
    srcs = [
        "dataset_extractor.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/compiler/plugin/poplar/ops:dataset_exporters_py",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

py_library(
    name = "config_lib",
    srcs = ["config.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/compiler/plugin/poplar/ops:ipu_ops_py",
        "//tensorflow/compiler/plugin/poplar/ops:poputil_ops_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:session",
        "//tensorflow/python/eager:context",
    ],
)

py_library(
    name = "ipu_lib",
    srcs = [
        "__init__.py",
        "config.py",
        "dataset_benchmark.py",
        "gradient_accumulation.py",
        "ipu_compiler.py",
        "ipu_estimator.py",
        "ipu_infeed_queue.py",
        "ipu_multi_worker_strategy.py",
        "ipu_outfeed_queue.py",
        "ipu_pipeline_estimator.py",
        "ipu_run_config.py",
        "ipu_session_run_hooks.py",
        "keras/__init__.py",
        "keras/layers/__init__.py",
        "keras/layers/assume_equal_across_replicas.py",
        "keras/layers/dropout.py",
        "keras/layers/embedding_lookup.py",
        "keras/layers/normalization.py",
        "keras/layers/recomputation.py",
        "loops.py",
        "ops/all_to_all_op.py",
        "ops/control_flow_ops.py",
        "ops/cross_replica_ops.py",
        "ops/custom_ops.py",
        "ops/embedding_ops.py",
        "ops/experimental/popfloat_cast_to_gfloat.py",
        "ops/functional_ops.py",
        "ops/image_ops.py",
        "ops/internal_ops.py",
        "ops/math_ops.py",
        "ops/nn_ops.py",
        "ops/normalization_ops.py",
        "ops/op_util.py",
        "ops/pipelining_ops.py",
        "ops/rand_ops.py",
        "ops/reduce_scatter_op.py",
        "ops/replication_ops.py",
        "ops/slicing_ops.py",
        "ops/statistics_ops.py",
        "ops/within_replica_ops.py",
        "optimizers/__init__.py",
        "optimizers/cross_replica_optimizer.py",
        "optimizers/gradient_accumulation_optimizer.py",
        "optimizers/ipu_optimizer.py",
        "optimizers/map_gradient_optimizer.py",
        "optimizers/sharded_optimizer.py",
        "scopes.py",
        "serving.py",
        "sharding.py",
        "sharding_utils.py",
        "utils.py",
        "vertex_edsl.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":config_lib",
        ":data",
        ":dataset_extractor",
        ":ipu_ops_lib",
        ":ipu_grad_ops_lib",
        "//tensorflow/python/ipu/test_utils:test_utils",
        "//tensorflow/compiler/plugin/poplar/ops:pop_datastream_ops_py",
        "//tensorflow/compiler/plugin/poplar/ops:sendrecv_ops_py",
        "//tensorflow/compiler/plugin/poplar/ops:ipu_ops_py",
        "//tensorflow/compiler/plugin/poplar/tools:tensorflow_weights_extractor_lib",
        "//tensorflow/compiler/xla:xla_data_proto_py",
        "//tensorflow/python/compiler/xla:compiler_py",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/estimator:estimator_py",
    ] + if_horovod(["//tensorflow/python/ipu/horovod:horovod_lib"]),
)

tf_py_test(
    name = "all_gather_test",
    size = "small",
    srcs = ["tests/all_gather_test.py"],
    tags = ["hw_poplar_test_4_ipus"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/ipu/test_utils",
    ],
)

tf_py_test(
    name = "config_test",
    size = "small",
    srcs = ["tests/config_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow/python/keras:layers",
    ],
)

tf_py_test(
    name = "reduce_scatter_test",
    size = "small",
    srcs = ["tests/reduce_scatter_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/ipu/test_utils",
    ],
    tags = ["hw_poplar_test_4_ipus"],
)

tf_py_test(
    name = "reconfigure_ipu_test",
    size = "small",
    srcs = ["tests/reconfigure_ipu_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/compiler/xla:compiler_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras:layers",
        "//tensorflow/python/ipu/test_utils",
    ],
    shard_count = 7,
)

tf_py_test(
    name = "gcl_test",
    size = "medium",
    srcs = ["tests/gcl_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
    tags = ["hw_poplar_test_4_ipus"],
)

tf_py_test(
    name = "within_replicas_test",
    size = "large",
    srcs = ["tests/within_replicas_test.py"],
    shard_count = 2,
    tags = ["hw_poplar_test_4_ipus"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/ipu/test_utils",
    ],
)

tf_py_test(
    name = "io_tiles_test",
    size = "medium",
    srcs = ["tests/io_tiles_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
    shard_count = 4,
)

tf_py_test(
    name = "io_tiles_hw_test",
    size = "medium",
    srcs = ["tests/io_tiles_hw_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
    shard_count = 10,
    tags = ["hw_poplar_test_2_ipus"],
)

tf_py_test(
    name = "map_gradient_optimizer_test",
    size = "medium",
    srcs = ["tests/map_gradient_optimizer_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
)

tf_py_test(
    name = "expression_op_test",
    size = "medium",
    srcs = ["tests/expression_op_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
)

tf_py_test(
    name = "dropout_test",
    size = "medium",
    srcs = ["tests/dropout_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
    shard_count = 4,
    tags = ["hw_poplar_test_1_ipus"],
)

tf_py_test(
    name = "application_compile_test",
    size = "large",
    srcs = ["tests/application_compile_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/ipu/test_utils",
    ],
)

tf_py_test(
    name = "application_compile_and_run_test",
    size = "large",
    srcs = ["tests/application_compile_and_run_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/ipu/test_utils",
    ],
    tags = ["hw_poplar_test_2_ipus"],
)

tf_py_test(
    name = "exception_test",
    size = "large",
    srcs = ["tests/exception_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/ipu/test_utils",
    ],
    tags = ["hw_poplar_test_1_ipus"],
)

tf_py_test(
    name = "application_runtime_test",
    size = "large",
    srcs = ["tests/application_runtime_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/ipu:ipu_ops_lib",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
    shard_count = 2,
    tags = ["hw_poplar_test_2_ipus"],
)

tf_py_test(
    name = "nn_ops_test",
    size = "medium",
    srcs = ["tests/nn_ops_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
    shard_count = 8,
    tags = ["disabled_poplar_test"],
)

tf_py_test(
    name = "gelu_erf_test",
    size = "medium",
    srcs = ["tests/gelu_erf_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
    shard_count = 8,
)

tf_py_test(
    name = "embedding_lookup_test",
    size = "large",
    srcs = ["tests/embedding_lookup_test.py"],
    additional_deps = [
        ":ipu_grad_ops_lib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras:layers",
    ],
    shard_count = 12,
    tags = ["hw_poplar_test_1_ipus"],
)

tf_py_test(
    name = "math_ops_test",
    size = "large",
    srcs = ["tests/math_ops_test.py"],
    additional_deps = [
        ":ipu_grad_ops_lib",
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras:layers",
        "//tensorflow/python:framework",
    ],
    shard_count = 40,
)

py_test(
    name = "estimator_test",
    size = "medium",
    srcs = ["tests/estimator_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:platform",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/estimator:model_fn",
        "//tensorflow/python/estimator:run_config",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow/python/keras:layers",
    ],
)

tf_py_test(
    name = "fifo_test",
    size = "large",
    srcs = ["tests/fifo_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
    tags = ["hw_poplar_test_1_ipus"],
)

tf_py_test(
    name = "op_util_test",
    size = "small",
    srcs = ["tests/op_util_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "iteration_counter_test",
    size = "large",
    srcs = ["tests/iteration_counter_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
)

tf_py_test(
    name = "host_embedding_lookup_test",
    size = "large",
    srcs = ["tests/host_embedding_lookup_test.py"],
    additional_deps = [
        ":ipu_grad_ops_lib",
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras:layers",
    ],
    tags = ["hw_poplar_test_1_ipus"],
)

tf_py_test(
    name = "host_embedding_lookup_ga_test",
    size = "large",
    srcs = ["tests/host_embedding_lookup_ga_test.py"],
    additional_deps = [
        ":ipu_grad_ops_lib",
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras:layers",
    ],
)

tf_py_test(
    name = "random_constant_matmul_test",
    size = "large",
    srcs = ["tests/random_constant_matmul_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
)

tf_py_test(
    name = "infeed_outfeed_test",
    size = "large",
    srcs = ["tests/infeed_outfeed_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
    shard_count = 10,
)

tf_py_test(
    name = "infeed_outfeed_overlap_test",
    size = "large",
    srcs = ["tests/infeed_outfeed_overlap_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
    shard_count = 10,
)

tf_py_test(
    name = "outfeed_ordinal_test",
    size = "medium",
    srcs = ["tests/outfeed_ordinal_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "dataset_benchmark_test",
    size = "medium",
    srcs = ["tests/dataset_benchmark_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "ipu_compiler_test",
    size = "large",
    srcs = ["tests/ipu_compiler_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "ipu_estimator_test",
    size = "large",
    srcs = ["tests/ipu_estimator_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/contrib/compiler:xla",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
    shard_count = 8,
    tags = ["isolated_poplar_tests"],
)

tf_py_test(
    name = "ipu_estimator_replicated_test",
    size = "large",
    srcs = ["tests/ipu_estimator_replicated_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/contrib/compiler:xla",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
    shard_count = 8,
    tags = ["hw_poplar_test_4_ipus"],
)

tf_py_test(
    name = "ipu_pipeline_estimator_test",
    size = "large",
    srcs = ["tests/ipu_pipeline_estimator_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/compiler/xla:compiler_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
    shard_count = 8,
    tags = ["isolated_poplar_tests"],
)

tf_py_test(
    name = "ipu_multi_worker_strategy_test",
    size = "large",
    srcs = ["tests/ipu_multi_worker_strategy_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
    shard_count = 4,
    tags = [
        "hw_poplar_test_4_ipus",
        "isolated_poplar_tests",
    ],
)

tf_py_test(
    name = "ipu_session_run_hooks_test",
    size = "large",
    srcs = ["tests/ipu_session_run_hooks_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/contrib/compiler:xla",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

py_test(
    name = "pop_datastream_test",
    size = "large",
    srcs = ["tests/pop_datastream_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/compiler/plugin/poplar/ops:pop_datastream_ops_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/ipu/test_utils",
    ],
)

tf_py_test(
    name = "frontend_attributes_test",
    size = "medium",
    srcs = ["tests/frontend_attributes_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/compiler/xla:xla_data_proto_py",
    ],
    tags = ["hw_poplar_test_1_ipus"],
)

tf_py_test(
    name = "popnn_normalization_test",
    size = "medium",
    srcs = ["tests/popnn_normalization_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/keras:layers",
    ],
    shard_count = 6,
)

tf_py_test(
    name = "popops_cross_replica_sum_test",
    size = "medium",
    srcs = ["tests/popops_cross_replica_sum_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

py_test(
    name = "popops_cross_replica_mean_test",
    size = "large",
    srcs = ["tests/popops_cross_replica_mean_test.py"],
    tags = ["hw_poplar_test_4_ipus"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/ipu/test_utils",
    ],
)

py_test(
    name = "replication_normalise_test",
    size = "medium",
    srcs = ["tests/replication_normalise_test.py"],
    srcs_version = "PY2AND3",
    tags = ["hw_poplar_test_2_ipus"],
    deps = [
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow/python/keras:layers",
    ],
)

tf_py_test(
    name = "softmax_test",
    size = "medium",
    srcs = ["tests/softmax_test.py"],
    shard_count = 8,
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/ipu/test_utils",
    ],
)

cc_binary(
    name = "libadd_incrementing_custom.so",
    srcs = ["tests/add_incrementing_custom.cc"],
    copts = ["-fexceptions"],
    linkshared = True,
    deps = [
        "@local_config_poplar//poplar:poplar_libs",
    ],
)

cc_binary(
    name = "libadd_tensors_custom.so",
    srcs = ["tests/add_tensors_custom.cc"],
    copts = ["-fexceptions"],
    linkshared = True,
    deps = [
        "@local_config_poplar//poplar:poplar_libs",
    ],
)

cc_binary(
    name = "libadd_partial_gradients_custom.so",
    srcs = ["tests/add_partial_gradients_custom.cc"],
    copts = ["-fexceptions"],
    linkshared = True,
    deps = [
        "@local_config_poplar//poplar:poplar_libs",
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

cc_binary(
    name = "libwrong_api_level_custom.so",
    srcs = ["tests/wrong_api_level_custom.cc"],
    copts = ["-fexceptions"],
    linkshared = True,
    deps = [
        "@local_config_poplar//poplar:poplar_libs",
    ],
)

tf_py_test(
    name = "user_ops_test",
    size = "large",
    srcs = ["tests/user_ops_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
    data = [
        "tests/add_scaled_vector_add_codelet.cc",
        ":libadd_incrementing_custom.so",
        ":libadd_incrementing_custom_with_metadata.so",
        ":libadd_partial_gradients_custom.so",
        ":libadd_tensors_custom.so",
        ":libwrong_api_level_custom.so",
    ],
)

tf_py_test(
    name = "utils_test",
    size = "medium",
    srcs = ["tests/utils_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/compiler/xla:compiler_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras:layers",
    ],
)

tf_py_test(
    name = "utils_use_synthetic_data_for_test",
    size = "medium",
    srcs = ["tests/utils_use_synthetic_data_for_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/compiler/xla:compiler_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras:layers",
    ],
    # Shard count set equal to the number of tests as one process per test is
    # required
    shard_count = 7,
)

tf_py_test(
    name = "utils_get_config_test",
    size = "medium",
    srcs = ["tests/utils_get_config_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/ipu/test_utils",
    ],
    shard_count = 4,
)

py_library(
    name = "pipelining_test_util",
    testonly = 1,
    srcs = ["tests/pipelining_test_util.py"],
    deps = [
        ":ipu_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu/test_utils",
    ],
)

tf_py_test(
    name = "pipelining_test",
    size = "large",
    srcs = ["tests/pipelining_test.py"],
    additional_deps = [
        ":ipu_lib",
        ":pipelining_test_util",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu/test_utils",
    ],
    # Shard count needs to match the number of tests.
    shard_count = 37,
)

tf_py_test(
    name = "pipelining_grouped_recomputation_test",
    size = "large",
    srcs = ["tests/pipelining_grouped_recomputation_test.py"],
    additional_deps = [
        ":ipu_lib",
        ":pipelining_test_util",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu/test_utils",
    ],
    data = [
        ":libpipelining_stateful_op.so",
    ],
    shard_count = 25,
)

tf_py_test(
    name = "pipelining_grouped_test",
    size = "large",
    srcs = ["tests/pipelining_grouped_test.py"],
    additional_deps = [
        ":ipu_lib",
        ":pipelining_test_util",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu/test_utils",
    ],
    shard_count = 16,
)

tf_py_test(
    name = "pipelining_grouped_overlap_test",
    size = "large",
    srcs = ["tests/pipelining_grouped_overlap_test.py"],
    additional_deps = [
        ":ipu_lib",
        ":pipelining_test_util",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu/test_utils",
    ],
    shard_count = 16,
)

cc_binary(
    name = "libpipelining_stateful_op.so",
    srcs = ["tests/pipelining_stateful_op.cc"],
    copts = ["-fexceptions"],
    linkshared = True,
    deps = [
        "@local_config_poplar//poplar:poplar_libs",
    ],
)

tf_py_test(
    name = "pipelining_recomputation_test",
    size = "large",
    srcs = ["tests/pipelining_recomputation_test.py"],
    additional_deps = [
        ":ipu_lib",
        ":pipelining_test_util",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow/contrib/compiler:xla",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu/test_utils",
    ],
    data = [
        ":libpipelining_stateful_op.so",
    ],
    shard_count = 16,
)

tf_py_test(
    name = "pipelining_sequential_recomputation_test",
    size = "large",
    srcs = ["tests/pipelining_sequential_recomputation_test.py"],
    additional_deps = [
        ":ipu_lib",
        ":pipelining_test_util",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu/test_utils",
    ],
    shard_count = 16,
    tags = ["hw_poplar_test_2_ipus"],
)

tf_py_test(
    name = "pipelining_sequential_test",
    size = "large",
    srcs = ["tests/pipelining_sequential_test.py"],
    additional_deps = [
        ":ipu_lib",
        ":pipelining_test_util",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu/test_utils",
    ],
    shard_count = 16,
)

tf_py_test(
    name = "pipelining_batch_serial_sequential_recomputation_test",
    size = "large",
    srcs = ["tests/pipelining_batch_serial_sequential_recomputation_test.py"],
    additional_deps = [
        ":ipu_lib",
        ":pipelining_test_util",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu/test_utils",
    ],
    shard_count = 16,
)

tf_py_test(
    name = "pipelining_batch_serial_sequential_test",
    size = "large",
    srcs = ["tests/pipelining_batch_serial_sequential_test.py"],
    additional_deps = [
        ":ipu_lib",
        ":pipelining_test_util",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu/test_utils",
    ],
    shard_count = 16,
)

tf_py_test(
    name = "pipelining_conv_classify_test",
    size = "large",
    srcs = ["tests/pipelining_conv_classify_test.py"],
    additional_deps = [
        ":ipu_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow/contrib/compiler:xla",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu/test_utils",
    ],
    shard_count = 4,
)

tf_py_test(
    name = "pipelining_recomputation_conv_classify_test",
    size = "large",
    srcs = ["tests/pipelining_recomputation_conv_classify_test.py"],
    additional_deps = [
        ":ipu_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow/contrib/compiler:xla",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu/test_utils",
    ],
    shard_count = 2,
)

tf_py_test(
    name = "dataset_ops_test",
    size = "medium",
    srcs = ["tests/dataset_ops_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu/test_utils",
    ],
)

tf_py_test(
    name = "keras_embedding_lookup_test",
    size = "medium",
    srcs = ["tests/keras/keras_embedding_lookup_test.py"],
    additional_deps = [
        "//tensorflow/python/keras:layers",
        "//third_party/py/numpy",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:framework_test_lib",
    ],
)

tf_py_test(
    name = "keras_normalization_test",
    size = "medium",
    srcs = ["tests/keras/keras_normalization_test.py"],
    additional_deps = [
        "//tensorflow/python/keras:layers",
        "//third_party/py/numpy",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:framework_test_lib",
    ],
)

tf_py_test(
    name = "keras_dropout_test",
    size = "medium",
    srcs = ["tests/keras/keras_dropout_test.py"],
    additional_deps = [
        "//tensorflow/python/keras:layers",
        "//third_party/py/numpy",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:framework_test_lib",
    ],
)

tf_py_test(
    name = "functional_ops_test",
    size = "large",
    srcs = ["tests/functional_ops_test.py"],
    additional_deps = [
        ":ipu_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu/test_utils",
    ],
    shard_count = 4,
)

tf_py_test(
    name = "gradient_accumulation_test",
    size = "large",
    srcs = ["tests/gradient_accumulation_test.py"],
    additional_deps = [
        ":ipu_lib",
        ":pipelining_test_util",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu/test_utils",
    ],
    # Shard count needs to match the number of tests.
    shard_count = 11,
    tags = ["hw_poplar_test_2_ipus"],
)

tf_py_test(
    name = "image_ops_test",
    size = "medium",
    srcs = ["tests/image_ops_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
)

tf_py_test(
    name = "multi_conv_test",
    size = "large",
    srcs = ["tests/multi_conv_test.py"],
    additional_deps = [
        ":ipu_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu/test_utils",
    ],
    shard_count = 5,
)

tf_py_test(
    name = "group_conv_test",
    size = "large",
    srcs = ["tests/group_conv_test.py"],
    additional_deps = [
        ":ipu_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu/test_utils",
    ],
)

tf_py_test(
    name = "candidate_sampler_test",
    size = "large",
    srcs = ["tests/candidate_sampler_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
    shard_count = 7,
    tags = ["hw_poplar_test_1_ipus"],
)

tf_py_test(
    name = "seed_control_test",
    size = "large",
    srcs = ["tests/seed_control_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
    tags = ["hw_poplar_test_2_ipus"],
)

tf_py_test(
    name = "reductions_test",
    size = "large",
    srcs = ["tests/reductions_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
    tags = ["hw_poplar_test_1_ipus"],
)

tf_py_test(
    name = "random_number_generation_test",
    size = "large",
    srcs = ["tests/random_number_generation_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
    # Shard count needs to match the number of tests.
    shard_count = 4,
    tags = ["hw_poplar_test_1_ipus"],
)

tf_py_test(
    name = "assume_equal_test",
    size = "large",
    srcs = ["tests/assume_equal_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
    tags = ["hw_poplar_test_4_ipus"],
)

tf_py_test(
    name = "replicated_graph_test",
    size = "large",
    srcs = ["tests/replicated_graph_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
    # Shard count needs to match the number of tests.
    shard_count = 12,
    tags = ["hw_poplar_test_2_ipus"],
)

tf_py_test(
    name = "replicated_index_test",
    size = "large",
    srcs = ["tests/replicated_index_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
    tags = ["hw_poplar_test_2_ipus"],
)

tf_py_test(
    name = "replicated_reduce_scatter_test",
    size = "large",
    srcs = ["tests/replicated_reduce_scatter_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
    tags = ["hw_poplar_test_4_ipus"],
)

tf_py_test(
    name = "replicated_seed_control_test",
    size = "large",
    srcs = ["tests/replicated_seed_control_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
    # Shard count needs to match the number of tests.
    shard_count = 2,
    tags = ["hw_poplar_test_4_ipus"],
)

tf_py_test(
    name = "replicated_stateful_gradient_accumulate_test",
    size = "large",
    srcs = ["tests/replicated_stateful_gradient_accumulate_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
    # Shard count needs to match the number of tests.
    shard_count = 4,
    tags = ["hw_poplar_test_2_ipus"],
)

tf_py_test(
    name = "replication_all_to_all_test",
    size = "large",
    srcs = ["tests/replication_all_to_all_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
    shard_count = 2,
    tags = ["hw_poplar_test_8_ipus"],
)

tf_py_test(
    name = "replication_grouped_collectives_test",
    size = "large",
    srcs = ["tests/replication_grouped_collectives_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
    shard_count = 2,
    tags = ["hw_poplar_test_4_ipus"],
)

tf_py_test(
    name = "sharded_and_replicated_test",
    size = "large",
    srcs = ["tests/sharded_and_replicated_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
    # Shard count needs to match the number of tests.
    shard_count = 5,
    tags = ["hw_poplar_test_4_ipus"],
)

tf_py_test(
    name = "floating_point_control_bits_test",
    size = "large",
    srcs = ["tests/floating_point_control_bits_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
    # Shard count needs to match the number of tests.
    shard_count = 11,
    tags = ["hw_poplar_test_2_ipus"],
)

tf_py_test(
    name = "dropout_hw_test",
    size = "large",
    srcs = ["tests/dropout_hw_test.py"],
    additional_deps = [
        ":pipelining_test_util",
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
    # Shard count needs to match the number of tests.
    shard_count = 7,
    tags = ["hw_poplar_test_8_ipus"],
)

tf_py_test(
    name = "pipelining_batch_serial_sequential_hw_test",
    size = "large",
    srcs = ["tests/pipelining_batch_serial_sequential_hw_test.py"],
    additional_deps = [
        ":pipelining_test_util",
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
    # Shard count needs to match the number of tests.
    shard_count = 4,
    tags = ["hw_poplar_test_2_ipus"],
)

tf_py_test(
    name = "pipelining_hw_test",
    size = "large",
    srcs = ["tests/pipelining_hw_test.py"],
    additional_deps = [
        ":pipelining_test_util",
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
    # Shard count needs to match the number of tests.
    shard_count = 10,
    tags = ["hw_poplar_test_8_ipus"],
)

poprun_py_test(
    name = "poprun_replica_partitioning_test",
    size = "large",
    srcs = ["tests/poprun_replica_partitioning_test.py"],
    ipus_per_replica = 2,
    main = "tests/poprun_replica_partitioning_test.py",
    num_instances = 2,
    num_replicas = 8,
    deps = [
        ":pipelining_test_util",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/ipu/test_utils",
    ],
)

tf_py_test(
    name = "prng_stability_models_test",
    size = "large",
    srcs = [
        "tests/prng_stability_models_test.py",
    ],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
    ],
    shard_count = 4,
    tags = ["hw_poplar_test_4_ipus"],
)

tf_py_test(
    name = "sequence_slice_test",
    size = "large",
    srcs = ["tests/sequence_slice_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
    shard_count = 12,
)

tf_py_test(
    name = "histogram_test",
    size = "medium",
    srcs = ["tests/histogram_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/ipu/test_utils",
    ],
    shard_count = 8,
)

tf_py_test(
    name = "rnn_test",
    size = "large",
    srcs = ["tests/rnn_test.py"],
    additional_deps = [
        ":ipu_lib",
        ":pipelining_test_util",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu/test_utils",
    ],
    shard_count = 7,
    tags = ["disabled_poplar_test"],
)

tf_py_test(
    name = "control_flow_ops_test",
    size = "large",
    srcs = [
        "tests/control_flow_ops_test.py",
    ],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "serving_export_test",
    size = "large",
    srcs = ["tests/serving_export_test.py"],
    additional_deps = [
        "//tensorflow/python/ipu/test_utils",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//third_party/py/numpy",
    ],
    tags = ["hw_poplar_test_1_ipus"],
)

test_suite(
    name = "all_tests",
)

test_suite(
    name = "poplar_ci_test_suite",
    tests = [
        ":all_tests",
        "//tensorflow/compiler/plugin/poplar:poplar_ci_test_suite",
    ] + if_horovod(["//tensorflow/python/ipu/horovod:horovod_test_suite"]),
)
