licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
)

py_library(
    name = "ipu",
    srcs = [
      "__init__.py",
      "python/autoshard.py",
      "python/autoshard_cnn.py",
      "python/gradient_accumulation_optimizer.py",
      "python/internal.py",
      "python/ipu_compiler.py",
      "python/ipu_infeed_queue.py",
      "python/ipu_optimizer.py",
      "python/ipu_outfeed_queue.py",
      "python/loops.py",
      "python/ops.py",
      "python/popnn_embedding.py",
      "python/popnn_normalization.py",
      "python/popnn_rnn.py",
      "python/poprand.py",
      "python/sharded_optimizer.py",
      "python/sharding_utils.py",
      "python/popops_cross_replica_sum.py",
      "python/sharding.py",
      "python/utils.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/compiler/plugin/poplar:ipu_ops_py",
        "//tensorflow/compiler/plugin/poplar:popnn_ops_py",
        "//tensorflow/compiler/plugin/poplar:popops_ops_py",
        "//tensorflow/compiler/plugin/poplar:poprand_ops_py",
        "//tensorflow/compiler/plugin/poplar:poputil_ops_py",
        "//tensorflow/compiler/xla:xla_data_proto_py",
        "//tensorflow/contrib/compiler:compiler_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/compiler/plugin/poplar:pop_datastream_ops_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

test_suite(
    name = "poplar_ci_test_suite",
    tests = [
      "//tensorflow/compiler/plugin/poplar:poplar_ci_test_suite",
      "//tensorflow/python/ipu:all_tests",
    ],
)
