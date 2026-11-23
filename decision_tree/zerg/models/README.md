# models/

Trained scorer model files land here, one JSON per scorer slot:

- `zergling_scorer.json`
- `hydralisk_scorer.json`
- `baneling_scorer.json`
- `joint_scorer.json`

They are produced by `decision_tree/zerg/training.py` (ZergTrainingBackend)
or exported via `decision_tree/scripts/linux/export_zerg_model.py`.

A model file embeds its execution-context hash and feature-schema digest;
a runtime refuses to load a model whose context does not match the live one.
Absent files are valid: the scorer slot then runs as a pure rule fallback.
No trained models ship with this framework deliverable.
