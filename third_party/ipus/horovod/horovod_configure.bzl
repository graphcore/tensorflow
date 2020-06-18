"""
Configuration for Horovod.

If Horovod is enabled, find the installed MPI headers and libraries and make
them available as filegroups.
"""

_TF_NEED_IPU_HOROVOD = "TF_NEED_IPU_HOROVOD"
_MPI_COMPILER = "mpic++"

def _exec(repository_ctx, cmd):
    result = repository_ctx.execute(cmd)
    if result.return_code != 0:
        fail("Command failed: {}: {}".format(cmd, result.stderr))
    output = result.stdout.splitlines()
    if not output:
        fail("Command had no output: {}".format(cmd))
    return output[0]

def _enable_horovod(repository_ctx):
    return int(repository_ctx.os.environ.get(_TF_NEED_IPU_HOROVOD, 0)) == 1

def _create_dummy_repository(repository_ctx):
    repository_ctx.file("BUILD", "")

    repository_ctx.template(
        "build_defs_horovod.bzl",
        Label("//third_party/ipus/horovod:build_defs_horovod.tpl"),
        {"{IF_HOROVOD}": "if_false"},
    )

def _impl(repository_ctx):
    if not _enable_horovod(repository_ctx):
        _create_dummy_repository(repository_ctx)
        return

    if not repository_ctx.which(_MPI_COMPILER):
        fail(("MPI installation not found ({} was not found). MPI is required for Horovod " +
              "support. Either install MPI or build without Horovod by setting {}=0").format(
            _MPI_COMPILER,
            _TF_NEED_IPU_HOROVOD,
        ))

    incdirs = _exec(repository_ctx, [_MPI_COMPILER, "--showme:incdirs"]).split(" ")
    incdir = _exec(repository_ctx, ["find"] + incdirs + ["-name", "mpi.h"])

    # Symlink the parent of the directory containing mpi.h.
    repository_ctx.symlink(incdir + "/..", "mpi/include")

    # Find and symlink the libraries.
    libdirs = _exec(repository_ctx, [_MPI_COMPILER, "--showme:libdirs"]).split(" ")
    libs = _exec(repository_ctx, [_MPI_COMPILER, "--showme:libs"]).split(" ")
    for lib in libs:
        filename = "lib{}.so".format(lib)
        path = _exec(repository_ctx, ["find"] + libdirs + ["-name", filename])
        repository_ctx.symlink(path, "mpi/lib/" + filename)

    repository_ctx.file("BUILD", """
filegroup(
    name = "mpi_headers",
    srcs = glob(["mpi/include/**/*.h"]),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "mpi_libs",
    srcs = glob(["mpi/lib/*.so"]),
    visibility = ["//visibility:public"],
)
""")

    repository_ctx.template(
        "build_defs_horovod.bzl",
        Label("//third_party/ipus/horovod:build_defs_horovod.tpl"),
        {"{IF_HOROVOD}": "if_true"},
    )

ipu_horovod_configure = repository_rule(
    implementation = _impl,
    local = True,
    environ = [_TF_NEED_IPU_HOROVOD],
)
