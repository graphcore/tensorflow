# Copyright 2017 Graphcore Ltd
"""Helpers to generate Poplar BUILD files from templates
"""

def _poplar_autoconf_impl(repository_ctx):
    # Tensorflow build tag
    tf_poplar_build_tag = "UNKNOWN"
    tf_custom_poplibs_available = "False"
    if not "TF_POPLAR_SANDBOX" in repository_ctx.os.environ and \
       not "TF_POPLAR_BASE" in repository_ctx.os.environ:
        tf_poplar_build_tag = ""
        tf_poplar_available = "False"
    else:
        tf_poplar_available = "True"
        if "TF_POPLAR_BUILD_TAG" in repository_ctx.os.environ:
            tf_poplar_build_tag = repository_ctx.os.environ["TF_POPLAR_BUILD_TAG"].strip()

        # Poplar release
        if "TF_POPLAR_BASE" in repository_ctx.os.environ:
            poplar_base = repository_ctx.os.environ["TF_POPLAR_BASE"].strip()
            poplibs_base = ""
            if "TF_POPLIBS_BASE" in repository_ctx.os.environ:
                poplibs_base = repository_ctx.os.environ["TF_POPLIBS_BASE"].strip()

                if poplibs_base == "":
                    fail("TF_POPLIBS_BASE not specified")
                tf_custom_poplibs_available = "True"

            if poplar_base == "":
                fail("TF_POPLAR_BASE not specified")

            if not repository_ctx.path(poplar_base + "/include").exists:
                fail("Cannot find poplar include path.")

            if not repository_ctx.path(poplar_base + "/lib").exists:
                fail("Cannot find poplar libary path.")

            if not repository_ctx.path(poplar_base + "/bin").exists:
                fail("Cannot find poplar bin path.")

            repository_ctx.symlink(poplar_base + "/lib", "poplar/lib/poplar")
            repository_ctx.symlink(poplar_base + "/lib", "poplar/lib/popit")
            repository_ctx.symlink(poplar_base + "/lib", "poplar/lib/popir")
            repository_ctx.symlink(poplar_base + "/lib", "poplar/lib/poprithms")
            repository_ctx.symlink(poplar_base + "/lib/python", "poplar/lib/python")
            repository_ctx.symlink(poplar_base + "/bin", "poplar/poplar/bin")
            repository_ctx.symlink(poplar_base + "/python", "poplar/poplar/python")

            if repository_ctx.path(poplar_base + "/lib64").exists:
                repository_ctx.symlink(poplar_base + "/lib64", "poplar/lib64/poplar")
                repository_ctx.symlink(poplar_base + "/lib64", "poplar/lib64/popit")
                repository_ctx.symlink(poplar_base + "/lib64", "poplar/lib64/popir")
                repository_ctx.symlink(poplar_base + "/lib64", "poplar/lib64/poprithms")
                repository_ctx.symlink(poplar_base + "/lib64/python", "poplar/lib64/python")

            if tf_custom_poplibs_available == "False":
                repository_ctx.symlink(poplar_base + "/include", "poplar/poplar/include")
            else:
                # Don't include any PopLibs headers from the Poplar path.
                repository_ctx.symlink(poplar_base + "/include/poplar", "poplar/poplar/include/poplar")
                repository_ctx.symlink(poplar_base + "/include/gcl", "poplar/poplar/include/gcl")
                repository_ctx.symlink(poplar_base + "/include/gccs", "poplar/poplar/include/gccs")
                repository_ctx.symlink(poplibs_base + "/include", "poplar/poplibs/include")
                repository_ctx.symlink(poplibs_base + "/lib", "poplar/lib/poplibs")

                if repository_ctx.path(poplibs_base + "/lib64").exists:
                    repository_ctx.symlink(poplibs_base + "/lib64", "poplar/lib64/poplibs")

            if "TF_POPSEC_BASE" in repository_ctx.os.environ:
                popsec_base = repository_ctx.os.environ["TF_POPSEC_BASE"].strip()
                if popsec_base != "":
                    repository_ctx.symlink(popsec_base + "/include", "poplar/popsec/include")
                    repository_ctx.symlink(popsec_base + "/lib", "poplar/popsec/lib")
                    if repository_ctx.path(popsec_base + "/lib64").exists:
                        repository_ctx.symlink(popsec_base + "/lib64", "poplar/popsec/lib64")

        else:
            # Poplar sandbox
            poplar_base = repository_ctx.os.environ["TF_POPLAR_SANDBOX"].strip()

            if poplar_base == "":
                fail("TF_POPLAR_SANDBOX not specified")

            if not repository_ctx.path(poplar_base + "/poplar/include").exists:
                fail("Cannot find poplar/include path.")

            if not repository_ctx.path(poplar_base + "/poplibs/include").exists:
                fail("Cannot find poplibs/include path.")

            repository_ctx.symlink(poplar_base + "/graphcore_target_access/include", "poplar/graphcore_target_access/include")
            repository_ctx.symlink(poplar_base + "/gccs/include", "poplar/gccs/include")
            repository_ctx.symlink(poplar_base + "/gccs/lib", "poplar/lib/gccs")
            repository_ctx.symlink(poplar_base + "/popef/include", "poplar/popef/include")
            repository_ctx.symlink(poplar_base + "/popef/lib", "poplar/lib/popef")
            repository_ctx.symlink(poplar_base + "/popef/lib/python", "poplar/lib/python")
            repository_ctx.symlink(poplar_base + "/poplar/include", "poplar/poplar/include")
            repository_ctx.symlink(poplar_base + "/poplibs/include", "poplar/poplibs/include")
            repository_ctx.symlink(poplar_base + "/popit/include", "poplar/popit/include")
            repository_ctx.symlink(poplar_base + "/poprithms/include", "poplar/poprithms/include")
            repository_ctx.symlink(poplar_base + "/poplar/bin", "poplar/poplar/bin")
            repository_ctx.symlink(poplar_base + "/poplibs/lib", "poplar/lib/poplibs")
            repository_ctx.symlink(poplar_base + "/popit/lib", "poplar/lib/popit")
            repository_ctx.symlink(poplar_base + "/popir/lib", "poplar/lib/popir")
            repository_ctx.symlink(poplar_base + "/poprithms/lib", "poplar/lib/poprithms")
            repository_ctx.symlink(poplar_base + "/poplar/lib", "poplar/lib/poplar")
            repository_ctx.symlink(poplar_base + "/tbb/lib", "poplar/lib/tbb")
            repository_ctx.symlink(poplar_base + "/popsec/include", "poplar/popsec/include")
            repository_ctx.symlink(poplar_base + "/popsec/lib", "poplar/popsec/lib")
            repository_ctx.symlink(poplar_base + "/libpvti/lib", "poplar/libpvti/lib")
            repository_ctx.symlink(poplar_base + "/libpvti/include", "poplar/libpvti/include")
            repository_ctx.symlink(poplar_base + "/openmpi/bin", "poplar/openmpi/bin")
            repository_ctx.symlink(poplar_base + "/openmpi/lib", "poplar/openmpi/lib")
            repository_ctx.symlink(poplar_base + "/openmpi/include", "poplar/openmpi/include")
            repository_ctx.symlink(poplar_base + "/libpva/lib", "poplar/libpva/lib")
            repository_ctx.symlink(poplar_base + "/libpva/include", "poplar/libpva/include")
            repository_ctx.symlink(poplar_base + "/libpva/python", "poplar/libpva/python")
            repository_ctx.symlink(poplar_base + "/popdist/lib", "poplar/lib/popdist")
            repository_ctx.symlink(poplar_base + "/popdist/python", "poplar/popdist/python")
            repository_ctx.symlink(poplar_base + "/popdist/include", "poplar/popdist/include")
            repository_ctx.symlink(poplar_base + "/poprun/bin", "poplar/poprun/bin")
            repository_ctx.symlink(poplar_base + "/ipu_arch_info/lib", "poplar/lib/ipu_arch_info")
            repository_ctx.symlink(poplar_base + "/ipuof_lib/lib", "poplar/ipuof_lib/lib")
            if repository_ctx.path(poplar_base + "/popsec/lib64").exists:
                repository_ctx.symlink(poplar_base + "/popsec/lib64", "poplar/popsec/lib64")
            if repository_ctx.path(poplar_base + "/libpvti/lib64").exists:
                repository_ctx.symlink(poplar_base + "/libpvti/lib64", "poplar/libpvti/lib64")
            if repository_ctx.path(poplar_base + "/openmpi/lib64").exists:
                repository_ctx.symlink(poplar_base + "/openmpi/lib64", "poplar/openmpi/lib64")
            if repository_ctx.path(poplar_base + "/popdist/lib64").exists:
                repository_ctx.symlink(poplar_base + "/popdist/lib64", "poplar/lib64/popdist")
            if repository_ctx.path(poplar_base + "/libpva/lib64").exists:
                repository_ctx.symlink(poplar_base + "/libpva/lib64", "poplar/libpva/lib64")
            if repository_ctx.path(poplar_base + "/ipu_arch_info/lib64").exists:
                repository_ctx.symlink(poplar_base + "/ipu_arch_info/lib64", "poplar/lib64/ipu_arch_info")
            if repository_ctx.path(poplar_base + "/ipuof_lib/lib64").exists:
                repository_ctx.symlink(poplar_base + "/ipuof_lib/lib64", "poplar/ipuof_lib/lib64")
            if repository_ctx.path(poplar_base + "/gccs/lib64").exists:
                repository_ctx.symlink(poplar_base + "/gccs/lib64", "poplar/gccs/lib64")

            repository_ctx.symlink(poplar_base + "/gcl/include", "poplar/gcl/include")
            repository_ctx.symlink(poplar_base + "/gcl/lib", "poplar/lib/gcl")
            if repository_ctx.path(poplar_base + "/gcl/lib64").exists:
                repository_ctx.symlink(poplar_base + "/gcl/lib64", "poplar/lib64/gcl")

            if repository_ctx.path(poplar_base + "/poplar/lib64").exists:
                repository_ctx.symlink(poplar_base + "/poplar/lib64", "poplar/poplar/lib64/poplar")
                repository_ctx.symlink(poplar_base + "/poplibs/lib64", "poplar/poplar/lib64/poplibs")
                repository_ctx.symlink(poplar_base + "/tbb/lib64", "poplar/lib64/tbb")
                repository_ctx.symlink(poplar_base + "/popit/lib64", "poplar/poplar/lib64/popit")
                repository_ctx.symlink(poplar_base + "/popir/lib64", "poplar/poplar/lib64/popir")
                repository_ctx.symlink(poplar_base + "/poprithms/lib64", "poplar/poplar/lib64/poprithms")

            if repository_ctx.path(poplar_base + "/popef/lib64").exists:
                repository_ctx.symlink(poplar_base + "/popef/lib64", "poplar/lib64/popef")

    repository_ctx.template(
        "poplar/BUILD",
        Label("//third_party/ipus/poplar_lib:BUILD_poplar.tpl"),
        {},
    )
    repository_ctx.template(
        "poplar/build_defs.bzl",
        Label("//third_party/ipus/poplar_lib:build_defs_poplar.tpl"),
        {
            "TF_POPLAR_BUILD_TAG": tf_poplar_build_tag,
            "TF_POPLAR_AVAILABLE": tf_poplar_available,
            "TF_CUSTOM_POPLIBS_AVAILABLE": tf_custom_poplibs_available,
        },
    )

poplar_configure = repository_rule(
    implementation = _poplar_autoconf_impl,
    local = True,
    environ = ["TF_POPLAR_BASE", "TF_POPLIBS_BASE", "TF_POPLAR_SANDBOX", "TF_POPLAR_BUILD_TAG", "TF_POPSEC_BASE"],
)
