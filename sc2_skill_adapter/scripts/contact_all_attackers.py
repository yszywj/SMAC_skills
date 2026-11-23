"""Install opt-in named approach grants in a private contact_response manager.

Apply the local-pack variant (if requested) FIRST, this variant LAST. The
installed runtime has no imports from this checkout or the launcher scripts.
"""
from functools import lru_cache
from hashlib import sha256
from inspect import signature
from pathlib import Path
from shutil import copytree
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory

if __package__:
    from .contact_muster_variant import FROZEN, PACKAGE, ROOT, RUN
    from .contact_response_micro_variant import FROZEN as PRE_MICRO, PACKAGE_DIGEST
else:
    from contact_muster_variant import FROZEN, PACKAGE, ROOT, RUN
    from contact_response_micro_variant import FROZEN as PRE_MICRO, PACKAGE_DIGEST


# The two released local-packs packages (without / with reserve dispatch).
_PACKS = {'d7a034fc07777ef663402e0d8d81cb86057bf3aa533657dde304d3f78564e621',
          'd6dc56292476dbd53d05570fa735d7ee9c82849b4964686ef6e71b959502dc24'}


def _digest(root):
    digest = sha256()
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root)
        if path.is_symlink():
            raise ValueError('expected private files without symlinks')
        if path.suffix in ('.py', '.json') and 'tests' not in rel.parts:
            digest.update(rel.as_posix().encode() + b'\0' + sha256(path.read_bytes()).digest())
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _marginal_packages():
    """Recognize exact installer outputs, never trust a marker in the target."""
    # The launcher and its older sibling scripts use top-level sibling imports.
    # Also support callers importing us via sc2_skill_adapter.scripts.
    known = set()
    scripts = str(Path(__file__).resolve().parent)
    sys.path.insert(0, scripts)
    try:
        from contact_local_packs_variant import apply_variant as install_packs
        if 'marginal' in signature(install_packs).parameters:
            for dispatch, horizon in ((False, 4), (True, 4), (True, 2)):
                with TemporaryDirectory() as folder:
                    root = Path(folder) / 'tactical_v1'
                    copytree(FROZEN, root)
                    install_packs(root, dispatch, marginal=True, reserve_horizon=horizon)
                    known.add(_digest(root))
    finally:
        sys.path.remove(scripts)
    return known


def apply_variant(skill_dir: Path) -> None:
    """Atomically patch only an independent, recognized policy copy."""
    source = Path(skill_dir).absolute()
    if any(p.is_symlink() for p in (source, *source.parents)):
        raise ValueError('expected an independent directory, not a link')
    root = source.resolve(strict=True)
    if any(p in (root, *root.parents) for p in (ROOT / 'skills', PRE_MICRO.parents[2], RUN)):
        raise ValueError('refusing to modify frozen or workspace skills')
    # Published experiments can themselves be the reference for another run.
    runs = ROOT / 'runs'
    if root.is_relative_to(runs) and any((p / 'design.json').exists()
            for p in root.parents if p.is_relative_to(runs)):
        raise ValueError('refusing to modify a frozen experiment')

    package = _digest(root)
    manager = root / 'skill_manager.py'
    if package == PACKAGE_DIGEST:
        # The pre-micro package differs only in its manager. Start from the
        # verified, released hydra hook without modifying either source.
        if _digest(FROZEN) != PACKAGE:
            raise ValueError('unverified hydra reference')
        raw = (FROZEN / manager.name).read_bytes()
    elif package in _PACKS or package == PACKAGE or (
            (root / 'local_engagement.py').is_file() and package in _marginal_packages()):
        raw = manager.read_bytes()
    else:
        raise ValueError('unrecognized contact_response package or installation order')

    runtime = (ROOT / 'skills/tactical_v1/micro_approach.py').read_text()
    text = raw.decode().replace('\r\n', '\n')
    pending = '        self._pending_commit = (snapshot, plans, list(actions), risk_now, context)'
    edits = (
        ('class SkillManager:', runtime + '\n\nclass SkillManager:\n    micro_all_attackers = False\n'),
        (pending, pending + '\n        if self.micro_all_attackers is True:\n'
         '            context.micro_approaches = extra_approaches(context, summary, plans, actions)'),
        ('        self.memory.commit(', '        if self.micro_all_attackers is True and final_actions == actions:\n'
         "            intents = merge_approaches(intents, getattr(context, 'micro_approaches', ()))\n"
         '        self.memory.commit('),
        ("        from .actions import attack_action_id\n        if any(r['actor'] == actor and "
         "attack_action_id(int(r['target'][1:])) == action\n               for r in self.last_micro_intents):\n"
         '            self.memory.last_committed_actions[actor] = action',
         '        record_approach(self.memory, self.last_micro_intents, actor, action)'),
    )
    for old, new in edits:
        if text.count(old) != 1:
            raise ValueError('unexpected frozen manager hook')
        text = text.replace(old, new)
    compile(text, str(manager), 'exec')
    with NamedTemporaryFile(dir=root, delete=False) as handle:
        tmp = Path(handle.name)
        try:
            handle.write(text.replace('\n', '\r\n' if b'\r\n' in raw else '\n').encode())
            handle.close()
            tmp.chmod(manager.stat().st_mode)
            tmp.replace(manager)  # Do not follow hard links back to a frozen source.
        finally:
            tmp.unlink(missing_ok=True)
