package(default_visibility = ["//visibility:public"])

filegroup(
    name = "horovod_tensorflow_sources",
    srcs = glob([
        "horovod/tensorflow/**/*.cc",
    ]),
)

filegroup(
    name = "horovod_tensorflow_headers",
    srcs = glob([
        "horovod/tensorflow/**/*.h",
    ]),
)

filegroup(
    name = "horovod_common_sources",
    srcs = glob([
        "horovod/common/*.cc",
        "horovod/common/mpi/mpi_*.cc",
        "horovod/common/optim/*.cc",
        "horovod/common/utils/*.cc",
    ]) + [
        "horovod/common/ops/adasum/adasum_mpi.cc",
        "horovod/common/ops/adasum_mpi_operations.cc",
        "horovod/common/ops/collective_operations.cc",
        "horovod/common/ops/mpi_operations.cc",
        "horovod/common/ops/operation_manager.cc",
    ],
)

filegroup(
    name = "horovod_common_headers",
    srcs = glob([
        "horovod/common/**/*.h",
    ]),
)
