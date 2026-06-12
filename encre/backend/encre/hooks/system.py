from __future__ import annotations
"""Stub: hooks module was removed, this stub prevents import errors."""


class EncreHookSystem:
    """Stub — hooks are not available in EnTA mode."""

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return lambda *a, **kw: None
