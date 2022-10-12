"""
Configuration for PopDist.

If PopDist is enabled, find the installed MPI headers and libraries and make
them available as filegroups.
"""

def _exec(repository_ctx, cmd):
    result = repository_ctx.execute(cmd)
    if result.return_code != 0:
        fail("Command failed: {}: {}".format(cmd, result.stderr))
    output = result.stdout.splitlines()
    if not output:
        fail("Command had no output: {}".format(cmd))
    return output[0]

def _impl(repository_ctx):
    repository_ctx.file("BUILD", "")

    repository_ctx.template(
        "build_defs_popdist.bzl",
        Label("//third_party/ipus/popdist_lib:build_defs_popdist.tpl"),
        {
            "{PYTHON_INTERPRETER}": str(repository_ctx.which("python")),
        },
    )

popdist_configure = repository_rule(
    implementation = _impl,
    local = True,
)
