"""Synthetic repeated-control logs check audit behavior, never game performance."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from decision_tree.conditional.runtime import CARRIER, ConditionalController
from decision_tree.conditional.policy import normalize_policy
from decision_tree.local.runtime import digest, execution_context, load_policy
from decision_tree.phase2.policy import REGROUP_POLICY
from decision_tree.tests.test_local_dataset import receipt
from skills.local_v2.tests.test_retarget import combat
from skills.local_v3.run import check_completion, source_hashes


def synthetic_run(output, forced=None, scene_index=None):
    env = {"map_name": "10gen_protoss", "step_mul": 8, "unit_balance": "native",
           "use_unit_ranges": True, "min_attack_range": 2,
           "capability_config": {"n_units": 10, "n_enemies": 10,
                                 "team_gen": {"unit_types": ["stalker"]}, "start_positions": {}}}
    scenes = [{"episode_config": {"scene": i}, "sha256": digest({"scene": i}), "sc2_seed": i, "index": i}
              for i in (range(2) if scene_index is None else [scene_index])]
    manifest = {"version": 1, "seed": 0, "env_args": env, "episodes": scenes}
    manifest["sha256"] = digest(manifest)
    policy = normalize_policy()
    metadata = {"kind": "conditional_local_v3", "policy": policy, "policy_sha256": digest(policy),
                "source_hashes": source_hashes(), "forced": forced}
    native = {"conditional_skills": metadata, "manifest_sha256": manifest["sha256"], "env_args": env,
              "source_hashes": {"synthetic.py": "fixture"}, "versions": {"fixture": "1"},
              "method": "adaptive-skill", "skill_version": "current", "native_teacher_control": "off",
              "medivac_policy": "legacy", "local_policy": {"policy": load_policy(CARRIER)}}
    context = execution_context(native, load_policy(CARRIER))
    native["local_policy"]["context_hash"] = context
    summary = {"status": "complete", "error": None, "completed_episodes": len(scenes), "expected_episodes": len(scenes),
               "conditional_skills": metadata, "manifest_sha256": manifest["sha256"], "pair_mode": "off",
               "outcomes": {"win": len(scenes), "loss": 0, "draw": 0, "timeout": 0}, "win_rate": 1.0}
    payload, baseline = combat()
    payload["state"]["visible_enemies"]["e0"]["health"] = 8.0
    for slot in range(2, 10):
        actor = f"u{slot}"
        payload["state"]["allies"][actor] = {**deepcopy(payload["state"]["allies"]["u0"]), "alive": False}
        payload["questions"][actor] = {"criteria": {"0": "noop"}}
        baseline["actions"].append(0)
        baseline["roles"][actor] = "dead"
    events, records = [], []
    for number, scene in enumerate(scenes, 1):
        owner = ConditionalController(SimpleNamespace(race_profile="off", _spatial_features_enabled=lambda: False),
                                      policy, context_hash=REGROUP_POLICY["model"]["context_hash"], forced=forced)
        events.append({"event": "episode_started", "episode": number, "scenario_sha256": scene["sha256"],
                       "initial_state_sha256": f"initial-{scene['index']+1}", "game_version": REGROUP_POLICY["model"]["game_version"],
                       "game_loop": 0})
        for step in range(3):
            payload["state"].update(game_loop=step * 8, episode=number)
            owner.step_number = step
            owner.events = []
            result = owner._decide(payload, baseline)
            actions = result["actions"]
            events.extend([
                {"event": "skill_observation", "episode": number, "step": step, "payload": deepcopy(payload)},
                {**deepcopy(owner.events[-1]), "episode": number},
                {"event": "step", "episode": number, "step": step + 1, "game_loop": step * 8,
                 "actions": deepcopy(actions), "reward": 2.0 if step == 2 else 0.0, "terminated": step == 2,
                 "feedback": receipt(payload, actions)},
            ])
        record = {"episode": number, "scenario_sha256": scene["sha256"], "initial_state_sha256": f"initial-{scene['index']+1}",
                  "steps": 3, "reward": 2.0, "outcome": "win", "info": {"battle_won": True}}
        records.append(record)
        events.append({"event": "episode_completed", **record})
    files = {"run.json": native, "summary.json": summary, "manifest.json": manifest, "episodes.json": records}
    for name, value in files.items():
        (output/name).write_text(json.dumps(value), encoding="utf-8")
    return metadata, context, events


class NativeAuditTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)
        self.metadata, self.context, self.events = synthetic_run(self.path)

    def verify(self, events=None):
        (self.path/"events.jsonl").write_text("\n".join(json.dumps(e) for e in (self.events if events is None else events)),
                                             encoding="utf-8")
        with patch.dict(REGROUP_POLICY["model"], {"context_hash": self.context}):
            check_completion(self.path, self.metadata, 2)

    def test_two_episodes_multiple_interventions_with_receipts(self):
        changes = [e for e in self.events if e["event"] == "conditional_local_decision"]
        self.assertEqual([e["interventions"] for e in changes], [1, 2, 3, 1, 2, 3])
        self.assertEqual([e["selected_id"] for e in changes[:3]],
                         ["retarget:u0:7", "retarget:u1:7", "retarget:u0:7"])
        self.verify()

    def test_missing_and_reordered_events_are_rejected(self):
        for event_name in ("episode_started", "skill_observation", "conditional_local_decision", "step", "episode_completed"):
            with self.subTest(event=event_name):
                altered = deepcopy(self.events)
                altered.pop(next(i for i, event in enumerate(altered) if event["event"] == event_name))
                with self.assertRaises(ValueError):
                    self.verify(altered)
        altered = deepcopy(self.events)
        altered[1:4], altered[4:7] = altered[4:7], altered[1:4]
        with self.assertRaisesRegex(ValueError, "step number"):
            self.verify(altered)

    def test_actual_version_and_native_acceptance_are_required(self):
        altered = deepcopy(self.events)
        altered[0]["game_version"] = "different-build"
        with self.assertRaisesRegex(ValueError, "SC2 version"):
            self.verify(altered)
        for change in ("rejected", "missing_receipt", "wrong_action"):
            altered = deepcopy(self.events)
            submitted = next(e for e in altered if e["event"] == "step")
            if change == "missing_receipt":
                submitted["feedback"]["actors"].pop("u0")
            elif change == "wrong_action":
                submitted["actions"][0] = 6
            else:
                submitted["feedback"]["actors"]["u0"]["submission"] = "rejected"
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.verify(altered)

    def test_policy_choice_cooldown_counters_and_candidate_menu_are_replayed(self):
        for change in ("selected", "counter", "menu", "cooldown"):
            altered = deepcopy(self.events)
            decision = next(e for e in altered if e["event"] == "conditional_local_decision" and e["step"] == 1)
            if change == "selected":
                decision["selected_id"] = "retarget:u9:7"
            elif change == "counter":
                decision["interventions"] = 1
            elif change == "menu":
                decision["catalog"]["bindings"].clear()
            else:
                decision["selected_id"] = decision["proposed_id"] = "retarget:u0:7"
                decision["final_actions"][0], decision["final_actions"][1] = 7, 6
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "differ from replay"):
                self.verify(altered)

    def test_return_outcome_scene_and_terminal_length_are_bound(self):
        for key, value in (("reward", 3.0), ("outcome", "loss"), ("scenario_sha256", "other"), ("steps", 2)):
            altered = deepcopy(self.events)
            end = next(e for e in altered if e["event"] == "episode_completed")
            end[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.verify(altered)
        altered = deepcopy(self.events)
        next(e for e in altered if e["event"] == "step")["reward"] = 1.0
        with self.assertRaisesRegex(ValueError, "return_sum"):
            self.verify(altered)


if __name__ == "__main__":
    unittest.main()
