"""
Configuration for Popit.

If Popit is enabled build it's libraries.
"""

_TF_NEED_POPIT = "TF_NEED_POPIT"

def _popit_is_enabled(repository_ctx):
    return int(repository_ctx.os.environ.get(_TF_NEED_POPIT, 0)) == 1

def _impl(repository_ctx):
    # need to create build file for this bzl file
    repository_ctx.file("BUILD", "")
    enabled = _popit_is_enabled(repository_ctx)
    repository_ctx.template(
        "build_defs_popit.bzl",
        Label("//third_party/ipus/popit_lib:build_defs_popit.tpl"),
        {
            "{IF_POPIT}": str(enabled),
        },
    )

popit_configure = repository_rule(
    implementation = _impl,
    local = True,
    environ = [_TF_NEED_POPIT],
)
