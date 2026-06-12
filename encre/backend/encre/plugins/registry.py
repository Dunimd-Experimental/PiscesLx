"""Stub: plugins module was removed."""


class PluginRegistry:
    """Stub — plugins are not available in EnTA mode."""

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return lambda *a, **kw: None
