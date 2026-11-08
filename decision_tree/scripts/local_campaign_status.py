"""Read-only progress for the six-batch local experiment; never starts games."""
import argparse
import json
from pathlib import Path


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return {}


def candidate_counts(plan):
    paths = {Path(a['path']) for a in plan.get('arms', [])}
    complete, failed = set(), set()
    for parent in {p.parent for p in paths}:
        for directory in parent.iterdir():
            stem, marker, attempt = directory.name.partition('.recovery-')
            original = parent / stem
            if original not in paths or (marker and not attempt.isdigit()) or not directory.is_dir():
                continue
            summary = read_json(directory / 'summary.json')
            if summary.get('status') == 'complete' and summary.get('completed_episodes') == 1:
                complete.add(original)
            elif summary.get('status') == 'failed':
                failed.add(original)
    return len(complete), len(failed - complete)


def status(root):
    rounds = []
    for folder in sorted(Path(root).glob('round_*')):
        study = folder / 'study'
        events = study / 'base/events.jsonl'
        base_completed = 0
        if events.is_file():
            with events.open(encoding='utf-8') as stream:
                base_completed = sum('"event": "episode_completed"' in line for line in stream)
        plan = read_json(study / 'plan.json')
        total = plan.get('candidate_games')
        complete, failed = candidate_counts(plan)
        results = [(p, read_json(p)) for p in folder.glob('*/result.json')]
        # Partial probes and the original collector may coexist. The largest
        # audited set is a checkpoint, distinct from raw successful rollouts.
        paired = [(p, r) for p, r in results if 'accepted_rows' in r and (p.parent / 'paired.json').is_file()]
        best = max(paired, key=lambda entry: entry[1]['accepted_rows'], default=(None, {}))
        accepted = best[1].get('accepted_rows', 0)
        rounds.append({'round': folder.name, 'base_completed': base_completed,
                       'candidate_planned': total, 'candidate_complete': complete,
                       'candidate_failed_unrecovered': failed, 'audited_pairs': accepted,
                       'paired_ready': best[0] is not None and total == accepted,
                       'model_directories': [p.parent.name for p in folder.glob('*/policy.json')]})
    return rounds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    args = parser.parse_args()
    print(json.dumps(status(args.campaign), ensure_ascii=False, indent=2))
