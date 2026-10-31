"""Medivac treatment-preference channel: features, scorers, contracts.

Dependency direction: tree_skills may import `skills`, never the reverse.
The injection point in skills/current/skills/heal_assign.py is duck-typed
(callable scorer) so the skills package stays importable without any
tree_skills or training dependency.
"""
