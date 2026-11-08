"""Scoped one-decision skills layered on an unchanged baseline proposal."""

from .library import (FEATURE_NAMES, FAMILIES, generate_candidates, apply_candidate,
                      deduplicate_candidates, effect_key)

__all__ = ["FEATURE_NAMES", "FAMILIES", "generate_candidates", "apply_candidate",
           "deduplicate_candidates", "effect_key"]
