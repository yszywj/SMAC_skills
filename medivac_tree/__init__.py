"""Single shared action tree for Medivacs; no teacher or learned scorer."""

__all__ = ["MedivacTree"]


def __getattr__(name):
    if name == "MedivacTree":
        from .runtime import MedivacTree
        return MedivacTree
    raise AttributeError(name)
