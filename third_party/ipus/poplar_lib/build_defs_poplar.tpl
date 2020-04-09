"""Macros for supporting Poplar library
"""

def poplar_available():
    """Returns true because Poplar library was configured
    """
    return TF_POPLAR_AVAILABLE

def tf_poplar_build_tag():
    """Returns a build tag/hash for displaying along with the Poplar version
    """
    return "TF_POPLAR_BUILD_TAG"

def if_poplar(if_true, if_false = []):
    if poplar_available():
        return if_true
    return if_false

def if_no_poplar(if_true, if_false = []):
    return if_poplar(if_false, if_true)
