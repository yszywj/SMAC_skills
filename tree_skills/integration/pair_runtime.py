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
    """target_id -> [health_ratio per DECISION STEP, ...] newest first.

    Semantics (one entry per real (episode, game_loop) decision step):
    - every NEW game_loop appends one ratio for every tracked target,
      including unchanged health — a constant prefix means "no net change
      over N decision steps", not "no history";
    - observing the same (episode, game_loop) twice is idempotent;
    - a dead unit records 0.0; a unit with invalid max_health (<=0/None)
      keeps its previous ratio (1.0 when none) — no division by zero, no
      fabricated large loss;
    - reset() clears everything; cross-episode staleness is prevented by
      PairRuntime.begin_step resetting on episode change.

    Same generator for training collection and deployment scoring.
    """

    def __init__(self, length=8):
        super().__init__()
        self.length = length
        self._last_key = None

    def observe(self, episode, game_loop, allies):
        """allies: {target_id: {"health": h, "max_health": m, "alive": bool}}."""
        key = (episode, game_loop)
        if key == self._last_key:
            return  # idempotent per decision step
        self._last_key = key
        for target_id, unit in allies.items():
            series = self.setdefault(target_id, [])
            if unit.get("alive") is False:
                ratio = 0.0
            else:
                maximum = unit.get("max_health")
                health = unit.get("health")
                if isinstance(maximum, (int, float)) and maximum > 0 \
                        and isinstance(health, (int, float)):
                    ratio = health / maximum
                else:
                    ratio = series[0] if series else 1.0
            series.insert(0, ratio)
            del series[self.length:]

    def reset(self):
        self.clear()
        self._last_key = None


def env_dims(env):
    """Teacher obs layout dims from the live environment."""
    return dict(n_agents=env.n_agents, n_enemies=env.n_enemies,
                move_feats=env.get_obs_move_feats_size(),
                own_feats=env.get_obs_own_feats_size()[1]
                if isinstance(env.get_obs_own_feats_size(), tuple)
                else env.get_obs_own_feats_size(),
                obs_nf_en=env.get_obs_enemy_feats_size()[1],
                obs_nf_al=env.get_obs_ally_feats_size()[1])


class CollectingScorer:
    """Wraps a scorer; records every (healer, candidate) score it serves.

    The wrapper enters assign_healing in place of the real scorer, so the
    recorded candidate set is exactly the legal set the injection layer
    scored, with the exact features and scores used at decision time.
    """

    def __init__(self, inner, collector):
        self.inner = inner
        self.collector = collector
        self._step_scores = {}  # healer -> {target: score}

    def __getattr__(self, item):
        return getattr(self.inner, item)

    @property
    def name(self):
        return getattr(self.inner, "name", "scorer")

    def begin_collect(self, episode, game_loop):
        self._step_scores = {}
        self._episode_loop = (episode, game_loop)

    def __call__(self, healer, target_id, context):
        value = self.inner(healer, target_id, context)
        self._step_scores.setdefault(healer, {})[target_id] = value
        episode, loop = getattr(self, "_episode_loop", (None, None))
        if episode is not None and self.collector is not None:
            self.collector.record_candidates(episode, loop, healer,
                                             context, {target_id: value})
        return value


class PairRuntime:
    def __init__(self, scorer, failure="strict", collect_dir=None, run_label="run"):
        self.failure = failure
        self.histories = HealthHistories()
        self.collector = None
        if collect_dir:
            from ..data.collect_real import PairCollector
            self.collector = PairCollector(collect_dir, run_label)
            scorer = CollectingScorer(scorer, self.collector)
        self.scorer = scorer
        self._teacher = getattr(scorer, "name", "") == "teacher"
        self._teacher_ready = not self._teacher
        self._episode = None
        self.steps = 0
        self.teacher_calls = 0

    @classmethod
    def from_args(cls, mode, model_dir=None, checkpoint=None, last_token="off",
                  failure="strict", shadow_of=None, collect_dir=None,
                  run_label="run"):
        from ..medivac.pair_scorer import OriginalPairScorer, TreePairScorer
        if mode == "a-plug":
            return cls(OriginalPairScorer(), failure, collect_dir=collect_dir,
                       run_label=run_label)
        if mode == "shadow":
            underlying = shadow_of or ("tree-score" if model_dir else None)
            if underlying == "teacher-score":
                scorer = _make_teacher_scorer(checkpoint, last_token)
            elif underlying == "tree-score":
                scorer = TreePairScorer(model_dir)
            else:
                raise ValueError("shadow needs --pair-shadow-of or --pair-model-dir")
            return cls(scorer, failure, collect_dir=collect_dir, run_label=run_label)
        if mode == "teacher-score":
            return cls(_make_teacher_scorer(checkpoint, last_token), failure,
                       collect_dir=collect_dir, run_label=run_label)
        if mode == "tree-score":
            if not model_dir:
                raise ValueError("tree-score requires --pair-model-dir")
            return cls(TreePairScorer(model_dir), failure,
                       collect_dir=collect_dir, run_label=run_label)
        raise ValueError(f"unknown pair mode {mode!r}")

    def _init_teacher(self, env):
        client = self.scorer.client
        client.init(env_dims(env))
        self._teacher_ready = True

    def begin_step(self, episode, game_loop, env, reset=False):
        """One real environment step: teacher histories + health histories.

        game_loop drives BOTH histories (decision-step semantics); health
        changes are never used as a clock.
        """
        if reset or episode != self._episode:
            self._episode = episode
            self.histories.reset()
            if self._teacher:
                if not self._teacher_ready:
                    self._init_teacher(env)  # dims first, then histories
                self.scorer.client.reset_episode(episode)
        allies = {f"u{a}": {"health": unit.health, "max_health": unit.health_max,
                            "alive": unit.health > 0}
                  for a, unit in env.agents.items()}
        self.histories.observe(episode, game_loop, allies)
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
            if self.collector is not None:
                cache = getattr(self.scorer.inner, "_logits", {}) or {}
                this_step = {agent: vec for (ep, lp, agent), vec in cache.items()
                             if ep == episode and lp == game_loop}
                self.collector.record_teacher_step(episode, game_loop, agents,
                                                   this_step)
        if self.collector is not None and hasattr(self.scorer, "begin_collect"):
            self.scorer.begin_collect(episode, game_loop)
        self.steps += 1

    def collect_step(self, episode, step, game_loop, telemetry, final_actions):
        """Runner hook after skill_manager.step: seal the collected row.

        episode arrives as the runner's INT; collection keys use the same
        episode string convention as begin_step ("ep%04d").
        """
        if self.collector is None:
            return
        healing = (telemetry or {}).get("race_telemetry", {}).get("healing", {})
        keep_decisions = (telemetry or {}).get("race_telemetry", {}).get(
            "terran_coordination", {}).get("keep_order")
        self.collector.finish_step(f"ep{episode:04d}", game_loop,
                                   healing.get("pair_scoring"), healing,
                                   final_actions, keep_decisions=keep_decisions)

    def describe(self, checkpoint=None, model_dir=None, last_token="off"):
        """Audit metadata for run.json / pair_configuration events."""
        import hashlib
        info = {"mode_scorer": getattr(self.scorer, "name", type(self.scorer).__name__),
                "failure_policy": self.failure,
                "checkpoint": checkpoint, "checkpoint_sha256": None,
                "model_dir": model_dir, "last_token": last_token,
                "feature_schema_version": None, "feature_schema_size": None,
                "tree_skills_source_sha256": {}}
        if checkpoint and os.path.isfile(checkpoint):
            digest = hashlib.sha256()
            with open(checkpoint, "rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    digest.update(block)
            info["checkpoint_sha256"] = digest.hexdigest()
        teacher = getattr(self.scorer, "client", None)
        if teacher is not None:
            info["teacher_model_sha256"] = teacher.model_sha256
            info["teacher_model_args"] = {
                k: teacher.model_args.get(k) for k in
                ("block_size", "history_len", "action_size", "n_attr", "vocab_size")}
            info["effective_history_len"] = teacher.history_len
            info["env_indicator"] = teacher.env_indicator
            info["teacher_last_token"] = teacher.last_token
        tree = getattr(self.scorer, "schema", None)
        if tree is not None:
            info["feature_schema_version"] = tree.version
            info["feature_schema_size"] = tree.size
        base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        for rel in ("integration/pair_runtime.py", "medivac/features.py",
                    "medivac/pair_scorer.py", "teacher/inference.py",
                    "teacher/service.py", "teacher/obs_builder.py"):
            path = os.path.join(base, rel.replace("/", os.sep))
            if os.path.isfile(path):
                info["tree_skills_source_sha256"][rel] = hashlib.sha256(
                    open(path, "rb").read()).hexdigest()
        return info

    def close(self):
        if self._teacher:
            scorer = self.scorer
            if hasattr(scorer, "client") and hasattr(scorer.client, "close"):
                scorer.client.close()


def _make_teacher_scorer(checkpoint, last_token):
    """Build the teacher scorer. For FORMAL evaluation both `checkpoint` and
    `last_token` must be given explicitly — the sorted-first fallback is a
    convenience for diagnostics only."""
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
