"""Teacher signals and training objectives for the Zerg tree framework (V1.1).

Four strictly separated signal kinds (a row never mixes them):

1. rule imitation — what the original tactical_v1 legally chose;
2. candidate ranking — REAL pairwise comparisons within one decision point;
   realised paired gains are the evidence, rule scores may seed a weak
   prior that is always flagged (``rule_score_prior``) and never presented
   as outcome evidence;
3. counterfactual gain — ``gain = candidate_return - baseline_return``
   under the same scene, same observation prefix and frozen execution
   context; missing results stay missing, never fabricated to zero;
4. episode outcome — terminal win/loss/draw/timeout plus environment return
   and casualty statistics; terminal rewards never blend into local gains.

Correction labels (Phase 6.4) replay the FROZEN old model over each
decision point: the veto target is minus the real gain of the OLD MODEL'S
own winner — never the empirically best candidate.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .dataset import candidate_rows, episode_group

GAIN_TOLERANCE = 1e-9


# -- 6.1 rule imitation -----------------------------------------------------

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


# -- 6.3 candidate ranking (real pairwise) ----------------------------------

def ranking_pairs(rows: Sequence[Dict[str, Any]],
                  rule_score_prior: bool = True
                  ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Pairwise comparisons within one decision point.

    Evidence hierarchy per pair:
    - both candidates have realised gains -> preference from the gain
      difference (provenance ``paired_gain`` — REAL evidence);
    - otherwise, if ``rule_score_prior``, fall back to the rule score
      ordering (provenance ``rule_score_prior`` — weak supervision, always
      flagged, never presented as outcome evidence).
    Unexecuted candidates never receive gain labels.

    Returns (pairs, provenance summary with the real-vs-prior ratio).
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
                if isinstance(margin, float) and math.isnan(margin):
                    continue
                higher, lower = (a, b) if margin >= 0 else (b, a)
                pairs.append({
                    "group_id": a["group_id"],
                    "decision_key": decision_key,
                    "higher_id": higher["candidate_id"],
                    "lower_id": lower["candidate_id"],
                    "higher_features": higher["features"],
                    "lower_features": lower["features"],
                    "margin": abs(margin),
                    "weight": min(a["weight"], b["weight"]),
                    "provenance": provenance,
                    "objective": "candidate_ranking",
                })
    summary = _pair_provenance_summary(pairs)
    return pairs, summary


def _pair_provenance_summary(pairs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(pairs)
    real = sum(1 for pair in pairs if pair["provenance"] == "paired_gain")
    return {
        "pairs_total": total,
        "pairs_real_gain": real,
        "pairs_rule_prior": total - real,
        "real_gain_ratio": (real / total) if total else None,
        "rule_prior_ratio": ((total - real) / total) if total else None,
    }


def _rule_margin(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    rank_a, rank_b = a.get("rule_rank", -1.0), b.get("rule_rank", -1.0)
    if rank_a < 0 or rank_b < 0:
        return float("nan")
    return rank_b - rank_a        # lower rule rank = preferred


# -- 6.2 counterfactual gain ------------------------------------------------

def gain_rows(rows: Sequence[Dict[str, Any]],
              strict: bool = True) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Counterfactual gain rows: ``gain = candidate_return - baseline_return``.

    Rows missing either return are retained as ``gain=None`` (absent never
    zero) and excluded from fitting. ``strict`` audits the arithmetic on
    every present gain and rejects fabricated pairs.
    """
    fit_rows: List[Dict[str, Any]] = []
    problems: List[str] = []
    for row in rows:
        candidate_return = row.get("candidate_return")
        baseline_return = row.get("baseline_return",
                                  row.get("base_return"))
        gain = row.get("gain")
        if candidate_return is None or baseline_return is None:
            if gain is not None:
                problem = (f"candidate {row['candidate_id']} carries a gain "
                           "without both returns")
                if strict:
                    problems.append(problem)
                    continue
            continue
        expected = float(candidate_return) - float(baseline_return)
        if gain is None or abs(float(gain) - expected) > GAIN_TOLERANCE:
            problem = (f"candidate {row['candidate_id']} gain "
                       f"{gain} != candidate_return - baseline_return")
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
    """Episode mass one, split across the decision points of the episode,
    then across that point's candidate rows (candidate_rows already
    normalised per point; this scales to episode level)."""
    episode_points: Dict[str, Dict[str, int]] = {}
    for row in rows:
        per_episode = episode_points.setdefault(row["group_id"], {})
        per_episode[row["decision_key"]] = per_episode.get(
            row["decision_key"], 0) + 1
    weights = []
    for row in rows:
        per_episode = episode_points[row["group_id"]]
        points = len(per_episode)
        rows_in_point = per_episode[row["decision_key"]]
        weights.append(1.0 / (points * rows_in_point))
    return weights


# -- 6.4 correction (frozen old model replay) -------------------------------

def correction_rows(rows: Sequence[Dict[str, Any]],
                    old_scorer) -> List[Dict[str, Any]]:
    """Veto labels from the FROZEN old model's own choice.

    For each decision point: score every candidate with the old model
    (real scoring path, ties by rule rank), find the old model's WINNER,
    then look up that winner's REAL gain. veto_target = -winner_gain.
    Points where the old model has no supported winner or the winner has no
    real gain are SKIPPED — the empirically best candidate is never
    substituted for the old model's winner.

    ``old_scorer`` is a ``scorer.ScorerModel``; rows must carry flat
    ``features`` per candidate (as produced by ``dataset.candidate_rows``).
    """
    from .schema import missing_report

    by_decision: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_decision.setdefault(row["decision_key"], []).append(row)
    out: List[Dict[str, Any]] = []
    for decision_key, group in sorted(by_decision.items()):
        scored = []
        for row in group:
            try:
                leaves = old_scorer.estimate(
                    row["features"], candidate_skill=row["candidate_skill"])
            except Exception:
                leaves = []
            if not leaves or not leaves[0].get("supported"):
                continue
            scored.append((float(leaves[0]["value"]),
                           row.get("rule_rank", -1.0), row))
        if not scored:
            continue        # old model has no supported winner here
        scored.sort(key=lambda triple: (-triple[0], triple[1]))
        winner_row = scored[0][2]
        winner_gain = winner_row.get("gain")
        if winner_gain is None or not math.isfinite(float(winner_gain)):
            continue        # no real gain for the old model's winner: skip
        veto = dict(winner_row)
        veto["veto_target"] = -float(winner_gain)
        veto["objective"] = "correction"
        out.append(veto)
    return out


# -- 6.4-old / episode outcome ----------------------------------------------

def outcome_rows(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Terminal outcome rows, one per episode: outcome, environment return,
    casualty and damage statistics from ``reward_provenance``."""
    by_episode: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for record in records:
        key = (record["run_id"], record["episode_id"])
        current = by_episode.get(key)
        if current is not None and current["outcome"] is not None:
            continue          # first terminal record per episode wins
        by_episode[key] = {
            "group_id": episode_group(record),
            "run_id": record["run_id"],
            "episode_id": record["episode_id"],
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
