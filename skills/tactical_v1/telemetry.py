"""Step telemetry and per-skill usage statistics (A5: concise)."""
from typing import Any, Dict, List, Optional

SKILL_NAMES = ("focus_fire", "coordinated_engage", "combat_reposition",
               "disengage_reenter", "regroup", "local_formation",
               "cohesive_search", "splash_fire", "heal_support", "burst_engage")

USAGE_COUNTERS = ("applicable", "proposed", "selected", "continued", "exited")


class Telemetry:
    def __init__(self, top_k_candidates: int = 3, debug: bool = False):
        self.top_k = top_k_candidates
        self.debug = debug
        self.usage: Dict[str, Dict[str, int]] = {
            skill: {counter: 0 for counter in USAGE_COUNTERS}
            for skill in (*SKILL_NAMES, "baseline")}
        self.specialization_stats: Dict[str, Dict[str, int]] = {
            "splash_fire": {"providers_active": 0, "plans_selected": 0},
            "heal_support": {"providers_active": 0, "plans_selected": 0},
            "burst_engage": {"providers_active": 0, "plans_selected": 0}}
        self.events: List[Dict[str, Any]] = []
        self.search_budget_events = 0

    def count(self, skill: str, counter: str, amount: int = 1) -> None:
        if skill in self.usage and counter in self.usage[skill]:
            self.usage[skill][counter] += amount

    def log(self, event: str, **fields: Any) -> None:
        row = {"event": event}
        row.update(fields)
        self.events.append(row)
        if len(self.events) > 200:
            del self.events[:len(self.events) - 200]

    def step_summary(self, snapshot_id: int, game_loop: int, actions: Dict[str, int],
                     per_unit: Dict[str, Dict[str, Any]], chosen: List[Dict[str, Any]],
                     alternates: List[Dict[str, Any]], repairs: List[str],
                     duration_ms: float, config_fingerprint: str) -> Dict[str, Any]:
        return {
            "snapshot_id": snapshot_id, "snapshot_game_loop": game_loop,
            "config_fingerprint": config_fingerprint,
            "actions": dict(actions),
            "units": per_unit,
            "chosen_plans": [
                {"id": p.get("id"), "skill": p.get("skill"),
                 "provider": p.get("provider"), "actors": p.get("actors"),
                 "goal": p.get("goal"),
                 "score": p.get("score"),   # joint plan score
                 "rationale": p.get("rationale")}
                for p in chosen],
            "top_alternates": [
                {"id": p.get("id"), "skill": p.get("skill"),
                 "rationale": p.get("rationale")}
                for p in alternates[:self.top_k]],
            "repairs": repairs,
            "events": self.events[-20:] if self.debug else [],
            "search_budget_events": self.search_budget_events,
            "duration_ms": round(duration_ms, 3),
        }
