"""Build configurations for Horovod."""

def if_horovod(if_true, if_false = []):
    """Tests whether Horovod is enabled in this build."""
    return {IF_HOROVOD}

def horovod_py_test(
        name,
        srcs,
        main,
        num_processes = 1,
        args = [],
        **kwargs):
    native.py_test(
        name = name,
        srcs = srcs + [
            "//third_party/ipus/horovod:horovod_test_wrapper.py",
        ],
        main = "horovod_test_wrapper.py",
        args = [
            "$(location @local_config_poplar//poplar:mpirun)",
            str(num_processes),
            "$(location {})".format(main),
        ] + args,
        data = ["@local_config_poplar//poplar:mpirun"],
        **kwargs
    )
