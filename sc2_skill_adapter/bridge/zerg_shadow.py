"""Strict collection on the ordinary tactical runner; imported only on opt-in."""
from pathlib import Path
import sys
from uuid import uuid4

from .config import WORKSPACE

if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from decision_tree.zerg.factory import build_zerg_skill_manager
from decision_tree.zerg.runtime import ZergTreeRuntime, default_config
from decision_tree.zerg.schema import IDENTIFIER


class ShadowRecorder(ZergTreeRuntime):
    def log_event(self, kind, **fields):
        # The runtime normally treats event I/O as best-effort. Collection must fail closed.
        self._events_for().log(kind, run_id=self.run_id, **fields)

    def record_decision(self, **fields):
        # Join the manifest seed at the sink, not in the tactical observation.
        fields['seed'] = self.episode_seed
        return super().record_decision(**fields)

    def check_step(self, episode, step):
        buffered = self.stats['decisions_seen'] - self.stats['decisions_recorded']
        if self.last_error or self.stats['model_errors'] or buffered != step:
            raise RuntimeError(f'Zerg Shadow episode {episode} step {step}: '
                               f'{buffered} decisions pending, {self.stats["model_errors"]} errors; '
                               f'last_error={self.last_error!r}')

    def finish(self, record, initial_score, final_score):
        def damage(field):
            # Zerg-only maps: no shields. Missing counters are not measured zeros.
            start = initial_score['score_details'][field]['life']
            end = final_score['score_details'][field]['life']
            return end - start if start is not None and end is not None else None

        episode, steps = record['episode'], record['steps']
        written = self.finalize_episode(
            episode_id=str(episode), outcome=record['outcome'], episode_return=record['reward'],
            ally_losses=record['info']['dead_allies'], enemy_losses=record['info']['dead_enemies'],
            damage_dealt=damage('total_damage_dealt'), damage_received=damage('total_damage_taken'))
        if written != steps:
            raise RuntimeError(f'Zerg Shadow episode {episode}: wrote {written} decisions for {steps} steps')
        self.check_step(episode, 0)
        return written


def start_shadow(directory, roster, manifest, method, args, manager_kwargs):
    if not directory or not directory.strip() or not roster or not roster.strip():
        raise ValueError('Set both ZERG_SHADOW_DIR and ZERG_SHADOW_ROSTER to non-empty values')
    if not IDENTIFIER.fullmatch(roster):
        raise ValueError('ZERG_SHADOW_ROSTER must be a valid scene_config_id')
    if (method != 'adaptive-skill' or args.skill_version != 'tactical_v1'
            or manifest['env_args']['map_name'] != '10gen_zerg' or args.local_policy is not None
            or getattr(args, 'skill_tree', None) is not None
            or getattr(args, 'native_teacher_control', 'off') != 'off'
            or getattr(args, 'pair_mode', 'off') != 'off'):
        raise ValueError('Zerg Shadow requires plain adaptive-skill tactical_v1 on 10gen_zerg')
    teams = [row['episode_config']['team_gen'] for row in manifest['episodes']]
    if len({(tuple(sorted(t['ally_team'])), tuple(sorted(t['enemy_team']))) for t in teams}) != 1:
        raise ValueError('ZERG_SHADOW_ROSTER requires a fixed composition within the worker manifest')
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)

    # The resolver rewrites its shared source manifest, even on a cache hit.
    # Serialize just factory startup across workers, never episode execution.
    resolved = WORKSPACE / 'decision_tree/zerg/resolved'
    resolved.mkdir(parents=True, exist_ok=True)
    with (resolved / '.shadow-factory.lock').open('a+b') as lock:
        if sys.platform == 'win32':
            import msvcrt
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
        bundle = build_zerg_skill_manager(workspace=str(WORKSPACE), manager_kwargs=manager_kwargs)

    config = default_config(str(WORKSPACE))
    config.update(mode='shadow', run_id=uuid4().hex,
                  scene_config={'scene_config_id': roster, 'source': 'ZERG_SHADOW_ROSTER'},
                  records_path=str(directory / 'shadow.jsonl'), events_path=str(directory / 'events.jsonl'))
    runtime = ShadowRecorder(config=config, tactical_config=bundle.effective_config)
    return runtime, runtime.wrap(bundle)
