"""Teacher signals and training objectives for the Zerg tree framework.

Four strictly separated signal kinds (a row never mixes them):

1. rule imitation — what the original tactical_v1 legally chose;
2. candidate ranking — relative preference among candidates of ONE decision
   point; rule scores may seed it, but realised paired returns are the only
   source of preference evidence;
3. counterfactual gain — ``gain = candidate_return - base_return`` under the
   same scene, same observation prefix and frozen execution context; missing
   results stay missing, they are never fabricated to zero;
4. episode outcome — terminal win/loss/draw/timeout plus environment return
   and casualty statistics; terminal rewards never blend into local gains.

This module builds and audits labels only. Collecting the trajectories and
running the counterfactual re-executions stays with the project lead.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

from .dataset import candidate_rows, scene_group

GAIN_TOLERANCE = 1e-9


# -- 8.1 rule imitation -----------------------------------------------------

def imitation_rows(records: Sequence[Dict[str, Any]]
                   ) -> List[Dict[str, Any]]:
    """One row per legal candidate; ``y`` = 1.0 for the teacher (original
    rule) choice, 0.0 otherwise. Rows without a teacher choice are skipped
    (no fabricated negatives for choices the rules never made)."""
    rows: List[Dict[str, Any]] = []
    for record in records:
        teacher = record.get("teacher_choice")
        if teacher is None:
            continue
        point_rows = candidate_rows([record])
        for row in point_rows:
            row["y"] = row["teacher"]
            row["objective"] = "rule_imitation"
        rows.extend(point_rows)
    return rows


# -- 8.2 candidate ranking --------------------------------------------------

def ranking_pairs(rows: Sequence[Dict[str, Any]],
                  rule_score_prior: bool = True
                  ) -> List[Dict[str, Any]]:
    """Pairwise preference rows within one decision point.

    Evidence hierarchy per pair:
    - both candidates have realised gains -> preference from the gain
      difference (provenance ``paired_gain``);
    - otherwise, if ``rule_score_prior``, fall back to the rule score
      ordering (provenance ``rule_score_prior`` — weak labels, usable for
      initialisation only, never as outcome evidence).
    Unexecuted candidates never receive gain labels.
    """
    pairs: List[Dict[str, Any]] = []
    by_decision: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_decision.setdefault(row["decision_key"], []).append(row)
    for decision_key, group in sorted(by_decision.items()):
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                gain_a, gain_b = a.get("gain"), b.get("gain")
                if gain_a is not None and gain_b is not None:
                    margin = float(gain_a) - float(gain_b)
                    provenance = "paired_gain"
                elif rule_score_prior:
                    margin = _rule_margin(a, b)
                    provenance = "rule_score_prior"
                else:
                    continue
                if math.isnan(margin):
                    continue
                pairs.append({
                    "group_id": a["group_id"],
                    "decision_key": decision_key,
                    "higher_id": a["candidate_id"] if margin >= 0
                    else b["candidate_id"],
                    "lower_id": b["candidate_id"] if margin >= 0
                    else a["candidate_id"],
                    "margin": abs(margin),
                    "weight": min(a["weight"], b["weight"]),
                    "provenance": provenance,
                    "objective": "candidate_ranking",
                })
    return pairs


def _rule_margin(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    rank_a, rank_b = a.get("rule_rank", -1.0), b.get("rule_rank", -1.0)
    if rank_a < 0 or rank_b < 0:
        return float("nan")
    return rank_b - rank_a        # lower rule rank = preferred


# -- 8.3 counterfactual gain ------------------------------------------------

def gain_rows(rows: Sequence[Dict[str, Any]],
              strict: bool = True) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Counterfactual gain rows: ``gain = candidate_return - base_return``.

    Rows missing either return are retained as ``gain=None`` (absent never
    zero) and excluded from fitting. ``strict`` audits the arithmetic on
    every present gain and rejects fabricated pairs.
    """
    fit_rows: List[Dict[str, Any]] = []
    problems: List[str] = []
    for row in rows:
        candidate_return, base_return = row.get("candidate_return"), \
            row.get("base_return")
        gain = row.get("gain")
        if candidate_return is None or base_return is None:
            if gain is not None:
                problem = (f"candidate {row['candidate_id']} carries a gain "
                           "without both returns")
                if strict:
                    problems.append(problem)
                    continue
            continue
        expected = float(candidate_return) - float(base_return)
        if gain is None or abs(float(gain) - expected) > GAIN_TOLERANCE:
            problem = (f"candidate {row['candidate_id']} gain "
                       f"{gain} != candidate_return - base_return")
            if strict:
                problems.append(problem)
                continue
            gain = expected
        if not math.isfinite(float(gain)):
            problems.append(f"candidate {row['candidate_id']} non-finite gain")
            continue
        fitted = dict(row)
        fitted["gain"] = float(gain)
        fitted["objective"] = "counterfactual_gain"
        fit_rows.append(fitted)
    return fit_rows, problems


def group_weights_for(rows: Sequence[Dict[str, Any]]) -> List[float]:
    """Scene mass one, split across the decision points of the scene, then
    across that point's candidate rows (candidate_rows already normalised
    per point; this scales to scene level)."""
    scene_points: Dict[str, Dict[str, int]] = {}
    for row in rows:
        per_scene = scene_points.setdefault(row["group_id"], {})
        per_scene[row["decision_key"]] = per_scene.get(row["decision_key"], 0) + 1
    weights = []
    for row in rows:
        per_scene = scene_points[row["group_id"]]
        points = len(per_scene)
        rows_in_point = per_scene[row["decision_key"]]
        weights.append(1.0 / (points * rows_in_point))
    return weights


# -- 8.4 episode outcome ----------------------------------------------------

def outcome_rows(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Terminal outcome rows, one per episode: outcome, environment return,
    casualty and damage statistics from ``reward_provenance``."""
    by_episode: Dict[str, Dict[str, Any]] = {}
    for record in records:
        episode_id = record["episode_id"]
        current = by_episode.get(episode_id)
        if current is not None and current["outcome"] is not None:
            continue          # first terminal record per episode wins
        by_episode[episode_id] = {
            "group_id": scene_group(record),
            "episode_id": episode_id,
            "outcome": record.get("outcome"),
            "return": record.get("return"),
            "reward_provenance": dict(record.get("reward_provenance") or {}),
            "objective": "episode_outcome",
        }
    return [by_episode[key] for key in sorted(by_episode)]


def audit_outcome_row(row: Dict[str, Any]) -> List[str]:
    problems = []
    provenance = row.get("reward_provenance") or {}
    if row.get("outcome") is None and row.get("return") is None:
        problems.append(f"episode {row['episode_id']}: no terminal evidence")
    if row.get("outcome") is not None and row.get("return") is None:
        problems.append(f"episode {row['episode_id']}: outcome without return")
    for field in ("ally_losses", "enemy_losses"):
        if field in provenance and type(provenance[field]) is not int:
            problems.append(f"episode {row['episode_id']}: {field} not integer")
    return problems
