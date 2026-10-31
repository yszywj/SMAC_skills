"""Part B of the synthetic smoke: run the exported rule model through the
real injection channel (assign_healing, mode "on") in the SMACv2 venv.

Adapter-venv python; stdlib + existing skills deps only (no sklearn/torch).
"""

import argparse
import copy
import math
import random
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from skills.current.skills import heal_assign
from tree_skills.medivac.pair_scorer import TreePairScorer


def unit(uid, health, max_health, type_="marine", pos=(0.0, 0.0), **extra):
    base = {"health": health, "max_health": max_health, "type": type_,
            "alive": True, "is_flying": type_ == "medivac",
            "position": list(pos)}
    base.update(extra)
    return base


def make_state(seed=0):
    rng = random.Random(seed)
    allies = {
        "u0": unit("u0", 20, 45, pos=(rng.random(), rng.random()),
                   health_lost_since_previous_decision=10),
        "u1": unit("u1", 38, 45, pos=(rng.random(), rng.random()),
                   health_lost_since_previous_decision=5),
        "u2": unit("u2", 30, 45, pos=(rng.random(), rng.random()),
                   health_lost_since_previous_decision=8),
        "u4": unit("u4", 150, 150, "medivac", pos=(0.5, 0.5), energy=100),
        "u5": unit("u5", 150, 150, "medivac", pos=(1.5, 0.5), energy=100),
    }
    state = {"allies": allies, "visible_enemies": {}, "relations": {a: {} for a in allies},
             "heal_candidates": {}, "movement_candidates": {}}
    for healer in ("u4", "u5"):
        state["heal_candidates"][healer] = {
            str(6 + j): {"target": t, "distance": 1.0 + rng.random() * 3,
                         "missing_health": allies[t]["max_health"] - allies[t]["health"]}
            for j, t in enumerate(("u0", "u1", "u2"))}
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()
    scorer = TreePairScorer(args.model_dir)
    for seed in range(10):
        state = make_state(seed)
        base = heal_assign.assign_healing(copy.deepcopy(state), 6)
        scored = heal_assign.assign_healing(copy.deepcopy(state), 6,
                                            pair_scorer=scorer, pair_scoring="on")
        diag = scored["pair_scoring"]
        assert set(scored["assignments"]) == set(base["assignments"])
        assert len(set(scored["assignments"].values())) == len(scored["assignments"])
        print(f"seed {seed}: base={base['assignments']} scored={scored['assignments']} "
              f"influence={diag['model_influence_rate']} failures={diag['failures']}")
    print("CHANNEL CHECK OK")


if __name__ == "__main__":
    main()
