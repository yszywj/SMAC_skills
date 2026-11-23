"""Unified Zerg SkillManager factory (Phase 3 / Task 3.1).

Every entry point (shadow collection, pair experiments, offline tests) must
construct the SAME skill package from the SAME resolved source: base
``skills/tactical_v1`` + overlay merged by ``source.py``. The manager is
imported from the resolved tree through an explicit importlib package spec
(no sys.path package shadowing, no second copy of ``skills.tactical_v1``);
the module path, whole-source fingerprint and effective tactical config are
recorded in a :class:`ZergManagerBundle` for provenance.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .source import DEFAULT_OUTPUT_DIR, ZergSkillSourceResolver, \
    load_manifest

RESOLVED_MODULE_NAME = "zerg_resolved_tactical_v1"


@dataclass
class ZergManagerBundle:
    manager: Any
    source_fingerprint: str
    module_path: str
    effective_config: Dict[str, Any]
    manifest_path: str
    module_namespace: str = ""
    extra_module_paths: Dict[str, str] = field(default_factory=dict)

    @property
    def skill_manager(self) -> Any:
        return self.manager


def _ensure_workspace_on_path(workspace: Path) -> None:
    text = str(workspace)
    if text not in sys.path:
        sys.path.insert(0, text)


def load_resolved_package(resolved_dir: str | Path,
                          module_name: Optional[str] = None
                          ) -> Any:
    """Import the resolved tactical_v1 package under a dedicated namespace.

    Task 5.2: the module name carries the SOURCE FINGERPRINT suffix, so a
    re-resolved (different) source tree can never reuse a stale cached
    module, and same-fingerprint re-entry reuses the cache only after
    verifying ``__file__`` really matches. Modules are never evicted while
    alive managers may still reference them; a genuinely new source version
    loads under its own isolated namespace (or after a process restart).
    """
    resolved_dir = Path(resolved_dir).resolve()
    manifest_path = resolved_dir / "source_manifest.json"
    fingerprint = ""
    if manifest_path.is_file():
        try:
            fingerprint = str(json.loads(manifest_path.read_text(
                encoding="utf-8")).get("fingerprint", ""))[:12]
        except Exception:
            fingerprint = ""
    namespace = module_name or (
        f"{RESOLVED_MODULE_NAME}_{fingerprint}" if fingerprint
        else f"{RESOLVED_MODULE_NAME}_{abs(hash(str(resolved_dir))) % 99999}")
    init = resolved_dir / "__init__.py"
    if not init.is_file():
        raise FileNotFoundError(f"resolved package lacks __init__.py: "
                                f"{resolved_dir}")
    existing = sys.modules.get(namespace)
    if existing is not None:
        cached_file = str(getattr(existing, "__file__", "") or "")
        if Path(cached_file).resolve() == init.resolve():
            return existing          # same source, same tree: safe reuse
        # same namespace, different file: never silently reuse
        namespace = f"{namespace}_{abs(hash(str(init))) % 99999}"
    spec = importlib.util.spec_from_file_location(
        namespace, init, submodule_search_locations=[str(resolved_dir)])
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build import spec for {resolved_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_zerg_skill_manager(
        workspace: Optional[str] = None,
        resolver: Optional[ZergSkillSourceResolver] = None,
        output_dir: Optional[str] = None,
        verify_source: bool = True,
        manager_kwargs: Optional[Dict[str, Any]] = None,
        module_name: Optional[str] = None) -> ZergManagerBundle:
    """Resolve the source, import the resolved package and build ONE
    SkillManager bound to that exact tree. Repeated calls with the same
    inputs reuse the already-imported module (import caching) while the
    fingerprint is re-verified against the manifest on disk."""
    from .source import DEFAULT_BASE_DIR, DEFAULT_OVERLAY_DIR

    ws = Path(workspace) if workspace else Path(__file__).resolve().parents[2]
    if resolver is None:
        resolver = ZergSkillSourceResolver(
            base_dir=str(ws / DEFAULT_BASE_DIR),
            overlay_dir=str(ws / DEFAULT_OVERLAY_DIR),
            output_dir=output_dir or DEFAULT_OUTPUT_DIR,
            workspace=str(ws))
    result = resolver.resolve()          # idempotent; verifies drift
    manifest = load_manifest(str(result.output_dir))
    if verify_source and manifest["fingerprint"] != result.fingerprint:
        raise RuntimeError("resolved manifest fingerprint drift — re-run "
                           "resolve(clean_output=True)")

    _ensure_workspace_on_path(ws)        # shared helpers under skills/current
    package = load_resolved_package(result.output_dir, module_name)
    resolved_namespace = package.__name__
    manager_module = importlib.import_module(f"{package.__name__}.skill_manager")
    manager = (manager_module.SkillManager(**(manager_kwargs or {})))

    paths = {
        name: getattr(getattr(package, name, None), "__file__", "")
        for name in ("skill_manager", "registry", "coordinator", "validator")
    }
    return ZergManagerBundle(
        manager=manager,
        source_fingerprint=result.fingerprint,
        module_path=str(manager_module.__file__),
        module_namespace=resolved_namespace,
        effective_config=dict(getattr(manager, "config", {})),
        manifest_path=str(Path(result.output_dir) / "source_manifest.json"),
        extra_module_paths={k: v for k, v in paths.items() if v})
