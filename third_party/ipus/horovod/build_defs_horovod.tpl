"""Build configurations for Horovod."""

def if_horovod(if_true, if_false = []):
    """Tests whether Horovod is enabled in this build."""
    return {IF_HOROVOD}
