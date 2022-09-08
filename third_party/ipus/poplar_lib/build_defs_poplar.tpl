"""Macros for supporting Poplar library
"""

def poplar_available():
    """Returns true because Poplar library was configured
    """
    return TF_POPLAR_AVAILABLE

def custom_poplibs_available():
    """Returns true when a custom PopLibs library was configured
    """
    return TF_CUSTOM_POPLIBS_AVAILABLE

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

def if_custom_poplibs(if_true, if_false = []):
    if custom_poplibs_available():
        return if_true
    return if_false

def ld_library_path():
    """Returns the value of the LD_LIBRARY_PATH env variable if set, else ""
    """
    return "LD_LIBRARY_PATH"
