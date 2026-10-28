"""SMAC-Hard entry points with skills maintained in the workspace library.

Keep existing ``jev_micro.*`` imports and patch targets working while loading
the sole editable skill implementations from ``skills/current``. Environment
and experiment entry points live in sc2_skill_adapter/smacv1/jev_micro.
"""

from pathlib import Path as _Path

_workspace = _Path(__file__).resolve().parents[3]
GAME_ROOT = next((parent / "game_skill" for parent in (_workspace, _workspace.parent)
                  if (parent / "game_skill/micro/pysc2/__init__.py").is_file()),
                 _workspace / "game_skill")
_skill_source = _workspace / "skills" / "current"
if not _skill_source.is_dir():
    raise ImportError(f"Shared micro skills directory is missing: {_skill_source}")
__path__.insert(0, str(_skill_source))
