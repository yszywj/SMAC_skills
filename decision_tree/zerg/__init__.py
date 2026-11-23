"""Hierarchical decision-tree framework for the Zerg tactical_v1 skill.

Layered modules (source resolution -> features -> candidates -> scoring ->
selection -> training -> runtime); see README.md in this package. Nothing
here re-implements tactical_v1: candidates always originate from its
Coordinator/Skills, and its validator and memory-commit conditions are never
bypassed.
"""
from .features import DEFAULT_SCHEMA, ZergFeatureSchema, SCHEMA_NAME, \
    SCHEMA_VERSION
from .source import ZergSkillSourceResolver, source_fingerprint

__all__ = [
    "DEFAULT_SCHEMA", "ZergFeatureSchema", "SCHEMA_NAME", "SCHEMA_VERSION",
    "ZergSkillSourceResolver", "source_fingerprint",
]
