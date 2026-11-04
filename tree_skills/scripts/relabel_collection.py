"""Re-label a recorded trajectory with the corrected, frozen teacher.

This never changes the recorded actions/outcomes and is NOT a new rollout.
Only agents that have scored candidates in an episode need inference; their
complete recorded observation histories are replayed, including steps without
candidates. Output is a new collection, training shard and provenance report.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tree_skills.data.build_shard import (analyze, build_shard, iter_rows,
                                         native_teacher_diagnostics)
from tree_skills.teacher.action_mapping import SmacActionMap


DEFAULT_DIMS = dict(n_agents=10, n_enemies=10, move_feats=4, own_feats=6,
                    obs_nf_en=8, obs_nf_al=8)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def replay_rows(rows, client, mapping, provenance, progress=None):
    """Yield copied rows. Never mix corrected predictions with old actions.

    The original logits and candidate scores are retained explicitly as
    trajectory evidence. pair_scoring/final_actions still describe the old
    rollout; new logits alone provide the re-labelled supervision.
    """
    requested = {}
    for row in rows:
        requested.setdefault(row["episode"], set()).update(
            int(h[1:]) for h in (row.get("healers") or {}))
    current_episode, last_loop = None, None
    finished_episodes = set()
    for i, original in enumerate(rows):
        row = copy.deepcopy(original)
        episode, loop = row["episode"], row["game_loop"]
        if episode != current_episode:
            if episode in finished_episodes:
                raise ValueError("input episodes must be contiguous")
            if current_episode is not None:
                finished_episodes.add(current_episode)
            current_episode, last_loop = episode, None
            if requested[episode]:
                client.reset_episode(episode)
        if last_loop is not None and loop <= last_loop:
            raise ValueError("input game loops must increase within an episode")
        last_loop = loop
        teacher = row.get("teacher_raw") or {}
        source_logits = teacher.get("logits") or {}
        agents = [a for a in teacher.get("agents", []) if int(a["agent"]) in requested[episode]]
        if {int(a["agent"]) for a in agents} != requested[episode]:
            raise ValueError(f"missing replay observations at {episode}/{loop}")
        logits = client.logits(episode, loop, agents) if agents else {}
        row["trajectory_teacher_logits"] = source_logits
        row["teacher_raw"] = {**teacher, "logits": {str(a): v for a, v in logits.items()}}
        row["label_provenance"] = provenance
        for healer, entry in (row.get("healers") or {}).items():
            for target, candidate in entry["candidates"].items():
                candidate["trajectory_score"] = candidate["score"]
                candidate["score"] = logits[int(healer[1:])][mapping.heal_slot(int(target[1:]))]
        if progress and (i + 1) % 100 == 0:
            progress(i + 1, len(rows))
        yield row


def compare_labels(old, new, mapping):
    result = {"groups": 0, "multi_candidate_groups": 0,
              "multi_candidate_top1_changed": 0,
              "old_masked_argmax": {}, "corrected_masked_argmax": {}}
    for before, after in zip(old, new):
        for healer, entry in (before.get("healers") or {}).items():
            result["groups"] += 1
            for key, row in (("old_masked_argmax", before), ("corrected_masked_argmax", after)):
                a = str(native_teacher_diagnostics(row, healer, mapping)["teacher_native_argmax"])
                result[key][a] = result[key].get(a, 0) + 1
            candidates = entry["candidates"]
            if len(candidates) > 1:
                result["multi_candidate_groups"] += 1
                agent = healer[1:]
                def top(row):
                    logits = row["teacher_raw"]["logits"][agent]
                    return min(candidates, key=lambda t: (-logits[mapping.heal_slot(int(t[1:]))], t))
                result["multi_candidate_top1_changed"] += int(top(before) != top(after))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", required=True, help="one existing collection JSONL")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--teacher-python")
    parser.add_argument("--output-dir", required=True, help="must not already exist")
    parser.add_argument("--dims-json", help="optional JSON object overriding frozen 10v10 dims")
    args = parser.parse_args(argv)
    source, output = Path(args.collect).resolve(), Path(args.output_dir).resolve()
    if not source.is_file():
        raise SystemExit("--collect must name one existing JSONL file")
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing directory {output}")
    rows = list(iter_rows(str(source)))
    dims = json.loads(args.dims_json) if args.dims_json else DEFAULT_DIMS
    mapping = SmacActionMap(n_agents=dims["n_agents"], n_enemies=dims["n_enemies"])
    from tree_skills.teacher.inference import TeacherServiceClient
    kwargs = {"python": args.teacher_python} if args.teacher_python else {}
    with TeacherServiceClient(str(Path(args.checkpoint).resolve()), last_token="on", **kwargs) as client:
        init = client.init(dims)
        if client.env_indicator != 0 or init.get("env_indicator") != 0:
            raise RuntimeError("relabel requires corrected official-default env_indicator=0")
        provenance = {
            "kind": "offline_relabel_of_recorded_old_policy_trajectory",
            "source_collection": str(source), "source_sha256": sha256(source),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": client.model_sha256,
            "last_token": True, "env_indicator": client.env_indicator,
            "env_specific": client.env_specific, "history_len": client.history_len,
            "actions_and_pair_diagnostics": "unchanged recorded old-policy actions; not new-policy outcomes",
            "replayed_agents": "all agents with scored candidates, complete per-episode recorded history",
        }
        output.mkdir(parents=True)
        manifest = {"status": "running", **provenance, "dims": dims,
                    "model_args": client.model_args,
                    "source_code_sha256": {str(p.relative_to(REPO)): sha256(p) for p in
                        [Path(__file__).resolve(), REPO / "tree_skills/teacher/obs_builder.py",
                         REPO / "tree_skills/teacher/service.py"]}}
        manifest_path = output / "relabel_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        relabelled = []
        collection_path = output / "collect-corrected-labels-on-old-trajectory.jsonl"
        with collection_path.open("x", encoding="utf-8") as out:
            for row in replay_rows(rows, client, mapping, provenance,
                                   lambda done, total: print(f"relabelled {done}/{total} steps", flush=True)):
                out.write(json.dumps(row) + "\n")
                relabelled.append(row)
        shard = output / "corrected_labels_shard.jsonl"
        written, skipped = build_shard(relabelled, mapping, str(shard))
        report = {"comparison": compare_labels(rows, relabelled, mapping),
                  "stats": analyze(relabelled, mapping), "shard_rows": written,
                  "skipped": skipped, "shard": str(shard)}
        (output / "relabel_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        manifest.update(status="complete", collection_sha256=sha256(collection_path),
                        shard_sha256=sha256(shard))
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
