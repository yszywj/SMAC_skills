import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from decision_tree.scripts.linux import fit_zerg_route_gate as fit
from decision_tree.zerg.policy_route_gate import CUTS, anchor_features, choose_route


def row(i, positive=True):
    return dict(scenario_sha256=f'{i:064x}', prefix_verified=True, micro_verified=True,
                anchor=dict(features={k: int(k == 'ready_in_range' and positive) for k in CUTS},
                            parent_plan=dict(actions=[2], plan_ids=['p']),
                            alt_plan=dict(actions=[3], plan_ids=['q'])),
                v2=dict(initial='c'*64, outcome='loss' if positive else 'win',
                        enemy_elimination_fraction=.5 if positive else 1),
                alternative=dict(initial='c'*64, outcome='win' if positive else 'loss',
                                 enemy_elimination_fraction=1 if positive else .5))


class TestRouteGate(unittest.TestCase):
    def test_anchor_features_are_predecision_only(self):
        s = dict(allies={'u0': dict(alive=True, type='Hydralisk', health=1, max_health=2,
                                    weapon_ready=True)}, relations={'u0': {'e0': dict(
                 in_weapon_range=True, smac_attack_available=True, in_enemy_weapon_range=True)}})
        p = dict(actions=[3], plan_ids=['p'])
        a = dict(actions=[6], plan_ids=['a'])
        f = anchor_features({'state': s}, p, a)
        self.assertEqual(f, {k: 1 for k in CUTS})
        self.assertEqual(f, anchor_features(dict(state=dict(s, outcome='win'), seed=7),
                                            dict(p, candidate_id='v2'), a))
        with self.assertRaises(ValueError):
            anchor_features(s, p, dict(actions=[]))

    def test_grouped_cv_and_heldout_veto(self):
        rows = [row(i, i % 2 == 0) for i in range(160)]
        model, report = fit.fit(rows, {})
        self.assertEqual((model, report), fit.fit(reversed(rows), {}))
        self.assertEqual(report['status'], 'supported_stump')
        self.assertEqual(len(report['cross_validation']), 4)
        for i in (0, 1):
            self.assertEqual(choose_route(model, rows[i]['anchor']['features']), ('alternative', 'v2')[i])
        sample = next(r for r in rows if r['v2']['outcome'] == 'loss' and fit.fold(r['scenario_sha256']) == 0)
        sample['v2'], sample['alternative'] = sample['alternative'], sample['v2']
        veto, later = fit.fit(rows, {})
        self.assertEqual(later['train_candidate'], report['train_candidate'])
        self.assertEqual(later['holdout_selected']['dropped'], 1)
        self.assertEqual(later['holdout_selected']['new_severe'], 1)
        self.assertIsNone(veto['tree'])

    def test_missing_evidence_abstains_or_rejects(self):
        r = row(1)
        r.update(anchor=None, alternative=dict(r['v2']))
        model, report = fit.fit([r], {})
        self.assertIsNone(model['tree'])
        self.assertEqual(report['cohort']['no_intervention'], 1)
        self.assertEqual(report['interventions']['n'], 0)
        for broken in (dict(r, alternative=row(1)['alternative']), dict(r, micro_verified=1),
                       dict(r, v2=dict(r['v2'], initial='b'*64))):
            with self.assertRaises(ValueError):
                fit.fit([broken], {})
        for flag in ('micro_verified', 'prefix_verified'):
            _, report = fit.fit([dict(row(1), **{flag: False})], {})
            self.assertEqual(report['cohort']['n'], 0)
            self.assertIn(r['scenario_sha256'], report['excluded'])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            fit.fit([r, r], {})

    def test_plan_and_feature_validation(self):
        r = row(1)
        r['anchor']['alt_plan'].update(actions=[2])  # micro-only difference is a full-plan divergence
        self.assertIsNotNone(fit.normalize(r)['features'])
        r['anchor']['alt_plan'].update(plan_ids=['p'], candidate_id='alias')
        with self.assertRaisesRegex(ValueError, 'no executable divergence'):
            fit.normalize(r)
        m = dict(schema='zerg_route_gate_v1', default='v2', tree=dict(feature='ready_in_range', op='gt', threshold=0))
        f = r['anchor']['features']
        for invalid in (dict(f, outcome=1), dict(f, ready_in_range=True), {}, None):
            self.assertEqual(choose_route(m, invalid), 'v2')
        for bad in (None, {}, dict(m, tree=dict(m['tree'], threshold=-1)), dict(m, schema='v3')):
            self.assertEqual(choose_route(bad, f), 'v2')

    def test_cli_empty_json_and_rejects_old_data(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'design.json').write_text(json.dumps({'arms': ['v2','v3']}))
            args = ['--panel', tmp, '--out', str(root/'model.json'), '--report', str(root/'stats.json')]
            with self.assertRaisesRegex(ValueError, 'modified-alternative'):
                fit.main(args)
            self.assertFalse((root/'model.json').exists())
            # Relabeling old scores or policies cannot turn them into evidence.
            (root/'design.json').write_text(json.dumps(dict(schema='zerg_route_pair_v1',
                arms=['v2', 'alternative'], policy_sha256={'alternative':'a'*64})))
            (root/'report.json').write_text(json.dumps(dict(paired_outcomes=[row(0)])))
            with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
                fit.main(args)
            # Exercise serialization separately from the runner's raw trajectory auditor.
            with patch.object(fit, 'load_panel', return_value=({'alternative':'a'*64}, [row(0)])):
                fit.main(args)
            self.assertIsNone(json.loads((root/'model.json').read_text())['tree'])
            self.assertEqual(json.loads((root/'stats.json').read_text())['status'], 'empty_default_v2')
