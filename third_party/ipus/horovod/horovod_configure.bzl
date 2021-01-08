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
    # TODO(T29900): Horovod disabled during transition to embedded OpenMPI.
    _create_dummy_repository(repository_ctx)

ipu_horovod_configure = repository_rule(
    implementation = _impl,
    local = True,
    environ = [_TF_NEED_IPU_HOROVOD],
)
