"""Step telemetry and per-skill usage statistics (P5.2)."""
from typing import Any, Dict, List, Optional

SKILL_NAMES = ("focus_fire", "coordinated_engage", "combat_reposition",
               "disengage_reenter", "regroup", "local_formation",
               "cohesive_search", "splash_fire", "heal_support", "burst_engage")


class Telemetry:
    def __init__(self, top_k_candidates: int = 3):
        self.top_k = top_k_candidates
        self.usage: Dict[str, Dict[str, int]] = {
            skill: {"applicable": 0, "proposed": 0, "selected": 0,
                    "continued": 0, "exited": 0}
            for skill in (*SKILL_NAMES, "fallback")}
        self.specialization_stats: Dict[str, Dict[str, int]] = {
            "splash_fire": {"providers_active": 0, "plans_selected": 0},
            "heal_support": {"providers_active": 0, "plans_selected": 0},
            "burst_engage": {"providers_active": 0, "plans_selected": 0}}
        self.events: List[Dict[str, Any]] = []
        self.search_budget_events = 0

    def count(self, skill: str, counter: str, amount: int = 1) -> None:
        if skill in self.usage:
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
                 "goal": p.get("goal"), "score": round(p.get("score", 0.0), 4),
                 "rationale": p.get("rationale")}
                for p in chosen[:self.top_k * 3]],
            "top_alternates": alternates[:self.top_k],
            "repairs": repairs,
            "events": self.events[-20:],
            "search_budget_events": self.search_budget_events,
            "duration_ms": round(duration_ms, 3),
        }
