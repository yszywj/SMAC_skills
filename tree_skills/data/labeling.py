"""Teacher label construction for medivac pair distillation.

Label definition (DATA_PROTOCOL.md §labels): for one sample group
(episode_id, game_loop, healer_id) with the original rule's legal heal
candidates {V}, and the teacher's RAW actor logits over base actions z_T:

    y(H, U) = z_T(H, heal(U)) - mean_V z_T(H, heal(V))

- Only finite, legal logits enter; masked/pad slots and the project's
  keep-heal-order extension (not a teacher action) never touch the mean.
- The full action distribution is stored separately: the teacher's argmax
  overall is NOT assumed to be "heal", this is conditional ranking only.
- Sample groups are never split across datasets; each group's training
  weight is normalized to 1/N_groups regardless of candidate count.
- Episode outcome, teacher labels and the original policy's own choice are
  three separate stored fields, never merged into one label.
"""

import json
import math
import os

KEEP_ACTION_ID_NOT_TEACHER = True  # the keep slot is never a teacher slot


def pair_labels(teacher_logits, heal_slot_of):
    """Compute relative labels for one healer's candidate group.

    teacher_logits: {unit_slot: {action_id: logit}} raw finite logits of the
        teacher for that healer, base actions only.
    heal_slot_of: {candidate_target_id: action_id} the explicit environment
        unit-id -> teacher heal-action-slot mapping (heal targets are allies;
        never reuse the attack-enemy mapping).

    Returns ({target_id: y}, diagnostics) and raises on any missing slot or
    non-finite logit — invalid groups are rejected, not silently repaired.
    """
    healer = next(iter(teacher_logits)) if len(teacher_logits) == 1 else None
    if healer is None:
        raise ValueError("pair_labels expects exactly one healer's logits")
    logits = teacher_logits[healer]
    used = {}
    for target_id, action_id in heal_slot_of.items():
        if action_id not in logits:
            raise KeyError(f"missing teacher slot {action_id} for {target_id}")
        value = float(logits[action_id])
        if not math.isfinite(value):
            raise ValueError(f"non-finite teacher logit for {target_id}")
        used[target_id] = value
    if not used:
        raise ValueError("empty candidate group")
    mean = sum(used.values()) / len(used)
    labels = {t: v - mean for t, v in used.items()}
    return labels, {"group_mean": mean, "n_candidates": len(used),
                    "teacher_logits_raw": dict(logits)}


def group_weight(total_groups):
    """Per-group training weight so groups, not candidate rows, sum to 1."""
    if total_groups <= 0:
        raise ValueError("total_groups must be positive")
    return 1.0 / total_groups


def write_sample_group(writer_path, episode_id, game_loop, healer_id,
                       labels, diagnostics, features_by_target):
    """Append one validated sample group to a JSONL shard."""
    row = {"episode_id": episode_id, "game_loop": game_loop,
           "healer_id": healer_id, "labels": labels,
           "features": features_by_target, "diagnostics": diagnostics}
    with open(writer_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return row
