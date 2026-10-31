"""Real-trajectory collection for medivac pair distillation (P0 §1.5).

PairCollector wraps the LIVE scorer inside the real runner and records, per
(episode, game_loop, healer) sample group:

- the 33-dim feature row of EVERY legal candidate that entered scoring
  (only candidates the original logic allowed — no reopening);
- the scorer's per-candidate score (teacher heal-slot logit for
  teacher-score; original edge for a-plug);
- the raw 20-dim teacher actor logits + native availability mask + raw
  per-agent observation (teacher input semantics);
- original assignment, proposed assignment, changed pairs and influence
  rate from the injection diagnostics;
- the final submitted actions and key healing telemetry for that step.

Rows are JSONL; one line per decision step. Teacher labels are computed
OFFLINE from the raw logits (data/labeling.py) so collection and labeling
stay separately auditable. Losing/won episodes are kept alike; groups are
never split across datasets (DATA_PROTOCOL.md).
"""

import json
import os


class PairCollector:
    def __init__(self, out_dir, run_label):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, f"collect-{run_label}.jsonl")
        if os.path.exists(self.path):
            raise FileExistsError(f"refusing to overwrite {self.path}")
        self._pending = {}   # (episode, loop) -> row being assembled
        self.rows = 0

    # -- called by the wrapping scorer during the injection call ----------
    def record_candidates(self, episode, loop, healer, context, scores):
        """scores: {target_id: scorer_value}; called once per candidate as
        the injection layer scores it — entries accumulate, never replace."""
        from ..medivac.features import build_pair_features
        row = self._pending.setdefault((episode, loop), {
            "episode": episode, "game_loop": loop, "healers": {}})
        entry = row["healers"].setdefault(healer, {})
        candidates = entry.setdefault("candidates", {})
        for t, s in scores.items():
            candidates[t] = {"features": build_pair_features(healer, t, context),
                             "score": s}

    # -- teacher-side raw evidence ----------------------------------------
    def record_teacher_step(self, episode, loop, agents, logits_by_agent):
        row = self._pending.setdefault((episode, loop), {
            "episode": episode, "game_loop": loop, "healers": {}})
        row["teacher_raw"] = {
            "agents": agents, "logits": {str(a): v for a, v in logits_by_agent.items()}}

    # -- runner-side completion -------------------------------------------
    def finish_step(self, episode, loop, pair_diag, healing_telemetry, final_actions):
        row = self._pending.pop((episode, loop), None)
        if row is None:
            return
        row["pair_scoring"] = pair_diag
        row["final_actions"] = list(final_actions)
        row["healing"] = {
            k: healing_telemetry.get(k) for k in
            ("assignments", "retained_orders", "released", "rescue_requests",
             "self_preserving", "wounded_targets", "keep_decisions")}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        self.rows += 1

    def close(self):
        if self._pending:
            # unfinished steps (run interrupted) are dropped, never guessed
            with open(self.path + ".incomplete", "w", encoding="utf-8") as fh:
                json.dump([str(k) for k in self._pending], fh)
            self._pending.clear()
        return self.path
