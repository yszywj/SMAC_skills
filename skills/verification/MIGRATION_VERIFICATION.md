# Skill migration verification

All checks passed. No new games or model calls were made.

| Package | Saved episodes | Frames | Actions, roles and assignments |
| --- | ---: | ---: | --- |
| current | 40 | 2667 | Exact match |
| baseline_v2_70pct | 30 | 2124 | Exact match |

Current: all 14 original source files match the frozen v6 recorded SHA-256 hashes; no algorithm copies remain at the old locations.
Baseline: all 10 original source files match the historical v2 recorded SHA-256 hashes.

Fresh processes verified module paths, version isolation, same-namespace protocol exception identity and monkeypatch behavior. Adapter loading does not change sys.path or import PySC2/SMAC-Hard. The legacy jev_micro namespace resolves the moved files through its package path.

Replay also matches predicted damage, All-in decisions and retreat reasons, and verifies that observation payloads stay unchanged.

This verifies migration and historical decision reproduction only. It does not establish a future 70% win rate or compare policies on new game states.

Machine-readable evidence: migration.json. Run with the adapter virtual environment: skills/tools/verify_migration.py.
