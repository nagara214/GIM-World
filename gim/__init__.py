"""GIM-World: geometry-aware implicit memory for video world models."""

__all__ = ['GIMWorldPipeline', 'Clip', 'RolloutResult']


def __getattr__(name):
    # Lazy import so light-weight tools (e.g. make_trajectory.py) do not pull
    # in torch / the backbone.
    if name in __all__:
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(f"module 'gim' has no attribute {name!r}")
