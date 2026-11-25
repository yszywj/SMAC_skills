import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from decision_tree.scripts.linux.fit_zerg_gains import fit
from decision_tree.scripts.linux.zerg_gain_exports import cumulative, lines, load, verify_export
from decision_tree.zerg.dataset import build_pair_record, dumps_key, episode_group, record_key
from decision_tree.zerg.experiment_manifest import make_anchor_manifest
from decision_tree.zerg.pair_dataset import build_gain_dataset, make_execution_receipt
from decision_tree.zerg.schema import digest, dumps
from decision_tree.zerg.tests.test_v11 import _records


def put(path, obj):
    path.write_text(json.dumps(obj))


def put_lines(path, rows):
    path.write_text(''.join(dumps(r) for r in rows))


def export(root, exp, n=8, **overrides):
    """Synthetic, independent raw runner facts and their v1 export, never games."""
    data = root / exp / 'export'
    data.mkdir(parents=True)
    anchors, pairs, receipts = {}, [], []
    source, ctx = overrides.get('source', '0'*64), overrides.get('context', '1'*64)
    records = _records(n, run_id=overrides.get('run_id', exp))
    for i, r in enumerate(records):
        r.update(source_hash=source, execution_context_hash=ctx,
                 seed=overrides.get('seed', int(digest(exp)[:7], 16)) + i)
        a, b, _ = r['candidates']
        for c in r['candidates']:
            del c['candidate_return']
        b['proposed_actions'] = {'u0': 9}
        r.update(teacher_choice=a['candidate_id'], selected_candidate=a['candidate_id'], baseline_return=None)
        key = dumps_key(record_key(r))
        initial = digest([overrides.get('initial', exp), i, 'initial'])
        scenario = digest([overrides.get('scenario', exp), i, 'scenario'])
        start = dict(initial_state_hash=initial, prefix_hash=digest([exp, i, 'prefix']))
        anchors[key] = dict(start, baseline_id=a['candidate_id'])
        common = dict(experiment_id=exp, anchor_id=key, baseline_id=a['candidate_id'],
                      intervention_type='single_step', intervention_horizon=1, execution_context_hash=ctx)
        ret_b = -1.5 + i
        name = f'pair-{i+1:03d}'  # Intentionally reused across exports.
        pair = build_pair_record(**common, **start, run_id=r['run_id'], episode_id=r['episode_id'],
                                 scene_config_id=r['scene_config_id'], pair_id=name, candidate_id=b['candidate_id'],
                                 baseline_return=1., candidate_return=ret_b, completion_status='complete', outcome='win')
        pairs.append(dict(pair, decision_key=key, scope='joint', source_fingerprint=source))
        for c, arm, role, reward in [(a, 'A', 'BASE', 1.), (b, 'B', 'candidate', ret_b)]:
            runner = data.parent / name / arm / 'runner'
            runner.mkdir(parents=True)
            raw = dict(kind='zerg_one_shot_fork_receipt_v1', arm=arm, prefix_verified=True, anchor_commits=1,
                       status='complete', error=None, engine_receipt='accepted', initial_state_sha256=initial,
                       scenario_sha256=scenario, episode_group=episode_group(r), roster=r['scene_config_id'], seed=r['seed'],
                       snapshot_id=r['snapshot_id'], game_loop=r['game_loop'], candidate_id=c['candidate_id'],
                       source_plan_id=c['source_plan_id'], final_actions=c['proposed_actions'], source_hash=source,
                       collection_context=ctx)
            put(runner / 'fork_receipt.json', raw)
            put(runner / 'episodes.json', [dict(initial_state_sha256=initial, scenario_sha256=scenario,
                                                outcome='win', reward=reward)])
            receipts.append(make_execution_receipt(**common, **start, candidate_id=c['candidate_id'], branch_id=role,
                source_fingerprint=source, execution_started=True, execution_completed=True, episode_finished=True,
                outcome='win', return_value=reward, receipt_source='sc2_runner', receipt_id=digest(raw)))
    manifest = make_anchor_manifest(experiment_id=exp, execution_context_hash=ctx, source_fingerprint=source, anchors=anchors)
    bundle = build_gain_dataset(records, pairs, anchor_manifest=manifest, execution_receipts=receipts)
    for key, obj in [('envelope', bundle.export_envelope()), ('audit', bundle.audit_report), ('anchor_manifest', manifest)]:
        put(data / (key + '.json'), obj)
    for key, rows in [('pairs', pairs), ('receipts', receipts), ('joint_records', bundle.joint_records)]:
        put_lines(data / (key + '.jsonl'), rows)
    return data


class CumulativeGainTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.a = export(self.root, 'pilot')

    def test_append_preserves_native_holdout_and_original_pair_identities(self):
        batches = [self.a, export(self.root, 'second', 7), export(self.root, 'third', 2)]
        prior = {}
        for n in range(1, 4):
            out = self.root / f'fit-{n}'
            fit(batches[:n], out)
            table = load(out / 'pair_facts.json')
            report = load(out / 'report.json')
            parts = {f['episode_group']: f['partition'] for f in table['pairs']}
            self.assertEqual({g: parts[g] for g in prior}, prior)
            self.assertEqual(set(parts), set(table['split']['train_groups'] + table['split']['validation_groups']))
            self.assertEqual({f['episode_group'] for f in table['pairs'] if f['partition'] == 'validation'},
                             {d['group'] for d in report['holdout_policy']['decisions']})
            self.assertEqual(digest(table), report['pair_facts_sha256'])
            train_table = load(out / 'train_pair_facts.json')
            self.assertEqual(digest(train_table), report['train_pair_facts_sha256'])
            self.assertEqual(train_table['pairs'], [p for p in table['pairs'] if p['partition'] == 'train'])
            ids = {(f['experiment_id'], f['pair_id']) for f in table['pairs']}
            self.assertEqual(len(ids), len(parts))
            self.assertEqual(sum(p == 'pair-001' for _, p in ids), n)
            for f in table['pairs']:
                self.assertEqual(f['gain'], f['alternate']['return'] - f['original']['return'])
                self.assertEqual(f['pre_action_visible_enemy_count'], 1.)
                self.assertEqual(f['original']['actions'], {'u0': 6})
            trained = load(out / 'model.json')
            self.assertEqual(trained['gain_audit']['pairs_trainable'], len(parts))
            if n == 1:
                self.assertEqual(trained['dataset_hash'], load(self.a / 'envelope.json')['dataset_hash'])
            prior = parts

    def test_mismatched_context_or_reused_underlying_identity(self):
        a = verify_export(self.a)
        for keyword, reason, original in [('seed', 'seed', a.records[0]['seed']), ('scenario', 'scenario_sha256', 'pilot'),
                ('initial', 'initial_state_hash', 'pilot'), ('run_id', 'decision_key', 'pilot'),
                ('source', 'source_fingerprint', '9'*64), ('context', 'execution_context_hash', '8'*64)]:
            with self.subTest(keyword=keyword):
                b = verify_export(export(self.root, 'overlap_'+keyword, 1, **{keyword: original}))
                with self.assertRaisesRegex(ValueError, reason):
                    cumulative([a, b])
        with self.assertRaisesRegex(ValueError, 'experiment_id'):
            cumulative([a, a])

    def test_ignores_no_extra_targets_even_when_envelope_resealed(self):
        dataset = self.a / 'joint_records.jsonl'
        records = lines(dataset)
        records[0]['candidates'][2]['candidate_return'] = 999.
        forged = build_gain_dataset(records, lines(self.a / 'pairs.jsonl'),
                    anchor_manifest=load(self.a / 'anchor_manifest.json'), execution_receipts=lines(self.a / 'receipts.jsonl'))
        put_lines(dataset, forged.joint_records)
        put(self.a / 'envelope.json', forged.export_envelope())
        with self.assertRaisesRegex(ValueError, 'joined records/targets'):
            verify_export(self.a)

    def test_tampering_or_duplicate_rejected_before_writing(self):
        for filename, mutate in [
            ('joint_records.jsonl', lambda rs: rs[0].update(baseline_return=-99.)),
            ('joint_records.jsonl', lambda rs: rs[0]['candidates'][1].update(candidate_return=-99.)),
            ('receipts.jsonl', lambda rs: rs.append(rs[0])), ('receipts.jsonl', lambda rs: rs.pop()),
            ('receipts.jsonl', lambda rs: rs[0].update(receipt_id='0'*64)),
            ('receipts.jsonl', lambda rs: rs[0].update(execution_completed=False)),
            ('pairs.jsonl', lambda rs: rs[1].update(pair_id=rs[0]['pair_id'])),
            ('anchor_manifest.json', lambda m: m.update(manifest_sha256='0'*64)),
            ('audit.json', lambda m: m.update(pairs_total=999)), ('envelope.json', lambda m: m.update(audited=False)),
            ('../pair-001/B/runner/episodes.json', lambda rs: rs[0].update(reward=999.))]:
            with self.subTest(filename=filename):
                path = self.a / filename
                original = path.read_bytes()
                many = path.suffix == '.jsonl'
                contents = lines(path) if many else load(path)
                mutate(contents)
                (put_lines if many else put)(path, contents)
                with self.assertRaises(ValueError):
                    fit([self.a], self.root / 'should-not-exist')
                self.assertFalse((self.root / 'should-not-exist').exists())
                path.write_bytes(original)

    def test_review_requires_exact_audited_fit_and_does_not_train(self):
        out, review = self.root / 'fit', self.root / 'review'
        fit([self.a], out)
        with patch('decision_tree.zerg.training.ZergTrainingBackend.train', side_effect=AssertionError('unexpected fit')):
            fit([self.a], review, out)
            self.assertEqual(load(out / 'pair_facts.json'), load(review / 'pair_facts.json'))
            with self.assertRaisesRegex(ValueError, 'do not match'):
                fit([self.a, export(self.root, 'next', 2)], self.root / 'wrong', out)
            model = load(out / 'model.json')
            model['trees'][0]['weight'] = 0
            put(out / 'model.json', model)
            with self.assertRaisesRegex(ValueError, 'do not match'):
                fit([self.a], self.root / 'forged', out)


if __name__ == '__main__':
    unittest.main()
