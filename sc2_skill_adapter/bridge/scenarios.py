"""Deterministic manifests using upstream distributions, including nested RNGs."""
import copy
import random

import numpy as np

from .config import digest, source_path, validate_config


def seed_distribution(obj, seed_sequence):
    if hasattr(obj, "rng"):
        obj.rng = np.random.default_rng(seed_sequence.spawn(1)[0])
    for name, child in sorted(vars(obj).items()):
        if name != "rng" and hasattr(child, "generate") and hasattr(child, "__dict__"):
            seed_distribution(child, seed_sequence)


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def generate_manifest(args, count, seed):
    source_path()
    from smacv2.env.starcraft2.distributions import get_distribution
    validate_config(args)
    if count <= 0:
        raise ValueError("Episode count must be positive")
    cap = args["capability_config"]
    distributions = []
    sequence = np.random.SeedSequence(seed)
    for key, config in cap.items():
        if key in ("n_units", "n_enemies"):
            continue
        config = dict(copy.deepcopy(config), env_key=key,
                      n_units=cap["n_units"], n_enemies=cap["n_enemies"])
        distribution = get_distribution(config["dist_type"])(config)
        seed_distribution(distribution, sequence.spawn(1)[0])
        distributions.append(distribution)
    previous = random.getstate()
    episodes = []
    try:
        random.seed(seed)
        for index in range(count):
            episode = {}
            for distribution in distributions:
                episode.update(plain(distribution.generate()))
            episodes.append({"index": index, "sc2_seed": seed + index,
                             "episode_config": episode, "sha256": digest(episode)})
    finally:
        random.setstate(previous)
    body = {"version": 1, "seed": seed, "env_args": copy.deepcopy(args), "episodes": episodes}
    return dict(body, sha256=digest(body))


def validate_manifest(manifest):
    body = {k: v for k, v in manifest.items() if k != "sha256"}
    if manifest.get("version") != 1 or digest(body) != manifest.get("sha256"):
        raise ValueError("Manifest version/hash mismatch")
    validate_config(manifest["env_args"])
    if not manifest["episodes"]:
        raise ValueError("Empty scenario manifest")
    for record in manifest["episodes"]:
        if digest(record["episode_config"]) != record["sha256"]:
            raise ValueError("Episode hash mismatch")


def reset_config(record):
    config = copy.deepcopy(record["episode_config"])
    for name in ("ally_start_positions", "enemy_start_positions"):
        config[name]["item"] = np.asarray(config[name]["item"], dtype=float)
    return config
