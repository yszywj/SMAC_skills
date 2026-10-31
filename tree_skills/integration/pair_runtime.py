"""PairRuntime: glue between the bridge runner and the pair-scorer channel.

Built only when the runner's --pair-mode is not "off". Responsibilities:

- build the scorer for the requested mode (a-plug / shadow / teacher-score /
  tree-score);
- per environment step: (a) advance the teacher service with ALL agents'
  raw observations + native availability masks in ONE batched request
  (weights stay loaded; histories advance exactly once per game_loop);
  (b) maintain health-ratio histories for the student's history features —
  training and deployment share this same generator;
- episode resets clear teacher histories and health histories.

No global state, no monkeypatching; off means this module is never built.
"""

import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


class HealthHistories(dict):
    """target_id -> [health_ratio, previous, ...] newest first.

    Same generator for training collection and deployment scoring; targets
    without enough history stay simply short (features flag them missing).
    """

    def __init__(self, length=8):
        super().__init__()
        self.length = length

    def observe(self, allies):
        """allies: {target_id: {"health": h, "max_health": m}} snapshot."""
        for target_id, unit in allies.items():
            maximum = unit.get("max_health") or 1
            ratio = unit.get("health", maximum) / maximum
            series = self.setdefault(target_id, [])
            if not series or abs(series[0] - ratio) > 1e-12:
                series.insert(0, ratio)
                del series[self.length:]

    def reset(self):
        self.clear()


def env_dims(env):
    """Teacher obs layout dims from the live environment."""
    return dict(n_agents=env.n_agents, n_enemies=env.n_enemies,
                move_feats=env.get_obs_move_feats_size(),
                own_feats=env.get_obs_own_feats_size()[1]
                if isinstance(env.get_obs_own_feats_size(), tuple)
                else env.get_obs_own_feats_size(),
                obs_nf_en=env.get_obs_enemy_feats_size()[1],
                obs_nf_al=env.get_obs_ally_feats_size()[1])


class PairRuntime:
    def __init__(self, scorer, failure="strict"):
        self.scorer = scorer
        self.failure = failure
        self.histories = HealthHistories()
        self._teacher = getattr(scorer, "name", "") == "teacher"
        self._teacher_ready = not self._teacher
        self._episode = None
        self.steps = 0
        self.teacher_calls = 0

    @classmethod
    def from_args(cls, mode, model_dir=None, checkpoint=None, last_token="off",
                  failure="strict", shadow_of=None):
        from ..medivac.pair_scorer import (OriginalPairScorer, TreePairScorer,
                                           make_pair_scorer)
        if mode == "a-plug":
            return cls(OriginalPairScorer(), failure)
        if mode == "shadow":
            underlying = shadow_of or ("tree-score" if model_dir else None)
            if underlying == "teacher-score":
                scorer = _make_teacher_scorer(checkpoint, last_token)
            elif underlying == "tree-score":
                scorer = TreePairScorer(model_dir)
            else:
                raise ValueError("shadow needs --pair-shadow-of or --pair-model-dir")
            return cls(scorer, failure)
        if mode == "teacher-score":
            return cls(_make_teacher_scorer(checkpoint, last_token), failure)
        if mode == "tree-score":
            if not model_dir:
                raise ValueError("tree-score requires --pair-model-dir")
            return cls(TreePairScorer(model_dir), failure)
        raise ValueError(f"unknown pair mode {mode!r}")

    def _init_teacher(self, env):
        client = self.scorer.client
        client.init(env_dims(env))
        self._teacher_ready = True

    def begin_step(self, episode, game_loop, env, reset=False):
        """One real environment step: teacher histories + health histories."""
        from ..medivac.pair_scorer import OriginalPairScorer  # noqa: F401
        if reset or episode != self._episode:
            self._episode = episode
            self.histories.reset()
            if self._teacher:
                if not self._teacher_ready:
                    self._init_teacher(env)  # dims first, then histories
                self.scorer.client.reset_episode(episode)
        allies = {f"u{a}": {"health": unit.health, "max_health": unit.health_max}
                  for a, unit in env.agents.items()}
        self.histories.observe(allies)
        if self._teacher:
            if not self._teacher_ready:
                self._init_teacher(env)
            agents = []
            for a in range(env.n_agents):
                agents.append({
                    "agent": a,
                    "obs": [float(v) for v in env.get_obs_agent(a)],
                    "avail": [int(v) for v in env.get_avail_agent_actions(a)],
                })
            self.scorer.begin_step(episode, game_loop, agents)
            self.teacher_calls += 1
        self.steps += 1

    def close(self):
        if self._teacher:
            scorer = self.scorer
            if hasattr(scorer, "client") and hasattr(scorer.client, "close"):
                scorer.client.close()


def _make_teacher_scorer(checkpoint, last_token):
    from ..teacher.inference import TeacherPairScorer, TeacherServiceClient
    if checkpoint is None:
        from ..teacher import preflight
        status = preflight.teacher_status()
        audited = sorted(name for name, rec in (status.get("checkpoints") or {}).items()
                         if rec.get("status") == "audited")
        if not audited:
            raise RuntimeError("BLOCKED_TEACHER_WEIGHTS: no audited checkpoint")
        checkpoint = os.path.join(os.path.dirname(preflight.__file__),
                                  "checkpoints", audited[0])
    client = TeacherServiceClient(checkpoint, last_token=last_token)
    return TeacherPairScorer(client)
