"""Build configurations for testing documentation examples."""

SUPPORTED_NUM_IPUS = [1, 2, 4, 8, 16]

def tf_docs_py_test(
        name,
        srcs,
        main,
        num_ipus = 0,
        args = [],
        tags = [],
        **kwargs):

    if num_ipus > 0:
        if not num_ipus in SUPPORTED_NUM_IPUS:
            fail("The number of IPUs (" + str(num_ipus) + ") must be one of " +
                 str(SUPPORTED_NUM_IPUS) + ".")

        hw_poplar_tag = "hw_poplar_test_" + str(num_ipus) + "_ipus"
        tags = tags + [hw_poplar_tag]

    native.py_test(
        name = name,
        srcs = srcs + [
            "//tensorflow/compiler/plugin/poplar/docs/tf_docs_py_test:docs_test_wrapper.py",
        ],
        main = "docs_test_wrapper.py",
        tags = tags,
        args = [
            "--source",
            "$(location {})".format(main),
            "--num-ipus",
            str(num_ipus),
        ] + args,
        **kwargs
    )
