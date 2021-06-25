load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

tf_cc_test(
    name = "ipu_config_test",
    size = "small",
    srcs = ["ipu_config_test.cc"],
    extra_copts = ["-fexceptions"],
    shard_count = 4,
    tags = ["hw_poplar_test"],
    deps = [
        "//tensorflow/compiler/plugin/poplar:ipu_config",
        "//tensorflow/compiler/plugin/poplar:poplar_test_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

test_suite(
    name = "all_ipu_config_tests",
    tests = [
        "ipu_config_test",
    ],
)
