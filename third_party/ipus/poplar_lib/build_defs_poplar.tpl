"""Macros for supporting Poplar library
"""

def poplar_available():
    """Returns true because Poplar library was configured
    """
    return True

def tf_poplar_build_tag():
    """Returns a build tag/hash for displaying along with the Poplar version
    """
    return "TF_POPLAR_BUILD_TAG"
