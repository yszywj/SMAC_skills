"""Offline regression tests, run with python -B scripts/test_hydra_switch_report.py."""
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

if __package__:
    from . import hydra_switch_report as report
else:
    import hydra_switch_report as report


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')


class ReportTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.batch = Path(tmp.name)
        self.results = {}
        self.design = dict(manifests='manifests', labels=list(report.LABELS), shards={})
        episodes = []
        for i, n in enumerate((3, 6), 1):
            team = ['zergling'] * n + ['hydralisk'] * (10-n)
            cfg = dict(team_gen=dict(ally_team=team, enemy_team=team),
                       ally_start_positions=dict(item=[[16, 16]]*10 if i == 1 else [[x, x] for x in range(10)]),
                       enemy_start_positions=dict(item=[[24, 24]]*10 if i == 1 else [[32-x, x] for x in range(10)]))
            episodes.append(dict(sc2_seed=123+i, sha256=report._digest(cfg), episode_config=cfg))
        manifest = dict(version=1, episodes=episodes)
        manifest['sha256'] = report._digest(manifest)
        dump(self.batch / 'manifests/shard.json', manifest)
        self.design['shards']['shard'] = manifest['sha256']
        for idx, label in enumerate(report.LABELS):
            self.results[label, 'shard'] = rows = []
            for i, episode in enumerate(episodes, 1):
                row = dict(episode=i, outcome=(('draw', 'win') if not idx else ('win', 'timeout'))[i-1],
                           initial_state_sha256=report._digest(i), scenario_sha256=episode['sha256'])
                rows.append(row)
                players = {}
                for side, value in (('1', 10+i+idx), ('2', 40-i-idx)):
                    amounts = dict.fromkeys(report.TYPES, value)
                    players[side] = dict(damage_dealt_by_source_type=amounts,
                                         ticks_by_source_type={k: v*4096 for k, v in amounts.items()})
                dump(self.artifact(label, 'damage_by_type', i), dict(episode=i, scenario_sha256=episode['sha256'],
                     manifest_sha256=manifest['sha256'], ticks_per_damage=4096, players=players))
                if idx:
                    self.trace(label, i, [])

    def artifact(self, label, kind, i):
        return self.batch / label / 'shard' / kind / f'episode-{i:04}.json'

    def trace(self, label, i, rows, **extra):
        states = Counter(f"{r['switch']}:{r['result']}" for r in rows if not r.get('followup'))
        self.results[label, 'shard'][i-1]['hydra_switches'] = dict(decisions=len(rows), states=dict(states))
        dump(self.artifact(label, 'hydra_switches', i), dict(episode=i, decisions=rows, **extra))

    def build(self, missing=None):
        data = report.build_report(self.batch, self.design, self.results, missing or {})
        self.assertEqual(data, json.loads((self.batch/'comparison.json').read_text()))
        return data

    def test_per_episode_buckets_transitions_damage(self):
        dump(self.artifact('both', 'damage_by_type', 99), {'stale': True})
        data = self.build()
        self.assertEqual(data['telemetry_issues'], [])
        self.assertEqual(data['pairing_initial_state_errors'], [])
        self.assertEqual(data['paired_scenarios'], 2)
        group = data['groups']['both']
        self.assertEqual(group['vs_baseline']['transitions'], {'draw->win': 1, 'win->timeout': 1})
        self.assertEqual(group['vs_baseline']['flipped_up_wins']['center'], 1)
        self.assertEqual(group['vs_baseline']['lost_wins']['scattered'], 1)
        self.assertEqual(group['vs_baseline']['center_win_rate_delta'], 1)
        self.assertEqual(group['by_composition_layout']['6ling_4hydra']['scattered']['games'], 1)
        self.assertEqual(group['vs_baseline']['ally_zergling_damage_delta'], 6)
        self.assertEqual(group['vs_baseline']['enemy_hydralisk_damage_delta'], -6)

    def test_mismatch_missing_and_bad_digests(self):
        self.results['both', 'shard'][0]['initial_state_sha256'] = report._digest('different')
        del self.results['escape_only', 'shard'][1]['initial_state_sha256']
        data = self.build()
        self.assertEqual(len(data['pairing_initial_state_errors']), 2)
        self.assertIsNone(data['groups']['baseline']['win_rate'])
        del self.results['shot_guard_only', 'shard']
        data = self.build({'shot_guard_only/shard': 'unfinished'})
        self.assertEqual(data['missing_shards']['shot_guard_only/shard'], 'unfinished')
        self.assertEqual(data['excluded_scenarios'][0]['missing_labels'], ['shot_guard_only'])
        self.assertIsNone(data['groups']['both']['vs_baseline']['center_win_rate_delta'])

    def test_wrong_or_missing_damage_not_zero_or_loss(self):
        path = self.artifact('baseline', 'damage_by_type', 1)
        data = json.loads(path.read_text())
        data['scenario_sha256'] = 'wrong'
        dump(path, data)
        group = self.build()['groups']['both']
        self.assertEqual(group['damage_paired_games'], 1)
        self.assertEqual(group['games'], 2)
        self.assertEqual(group['vs_baseline']['enemy_hydralisk_damage_delta'], -3)
        self.artifact('both', 'damage_by_type', 2).unlink()
        self.assertIsNone(self.build()['groups']['both']['vs_baseline']['enemy_hydralisk_damage_delta'])

    def test_design_and_duplicate_rejected(self):
        self.results['baseline', 'shard'].append(self.results['baseline', 'shard'][0])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.build()
        self.design['shards']['shard'] = 'old-config-only-hash'
        with self.assertRaisesRegex(ValueError, 'manifest checksum'):
            self.build()

    def test_applied_requires_receipt_followups_are_per_decision(self):
        label = 'escape_only'
        decision = dict(actor='u0', actor_tag='77', switch='A', game_loop=10, result='applied',
                        changed=True, original_action=6, final_action=3, candidate_action=3)
        samples = [dict(actor='u0', actor_tag='77', switch='A', followup=True, alive=True,
                        decision_loop=10, game_loop=10+off, cooldown_restarted=True,
                        threat_state={'e0': 'in_ally_range', 'e1': 'not_visible'}) for off in (8, 16, 24)]
        self.trace(label, 1, [decision, *samples, dict(decision, game_loop=34)], terminal_loop=42)
        self.assertEqual(self.build()['groups'][label]['switch_decisions']['A:receipt:missing_receipt'], 2)
        actor = dict(requested_action_id='3', submission='accepted', command=dict(ability_id=16), response=dict(code=1))
        events = [dict(event='step', episode=1, game_loop=t, actions=[3], feedback=dict(actors={'u0': actor}))
                  for t in (10, 34)]
        (self.batch/label/'shard/events.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in events))
        group = self.build()['groups'][label]
        self.assertEqual(group['switch_decisions']['A:changed_commands'], 2)
        track = group['switch_focus']['A']
        for name in ('survival_16:alive', 'survival_24:alive', 'survival_16:terminal_censored',
                     'observed_threat_in_ally_range', 'observed_threat_not_visible', 'macro_cd_restart_proxy'):
            self.assertEqual(track[name], 1, name)
        self.assertEqual(track['observed_dead_within_24'], 0)

    def test_b_shot_proxy_death_visibility_and_missing_are_distinct(self):
        row = dict(game_loop=10, switch='B', actor='u0', actor_tag='7', target_tag='9')
        follow = dict(game_loop=18, actor='u0', alive=True, cooldown_restarted=False)
        track = Counter()
        frame = dict(live={'u0': '7'}, cooldown={'u0': 7}, visible={'9'}, attack_target={'u0': '9'})
        # 7 -> 6 in eight loops is a restart, even though 6 is not above 7.
        report._followup(row, [follow], 80, track, {10: frame, 18: dict(frame, cooldown={'u0': 6})}, {'7': 31})
        self.assertEqual(track['next_8_macro_proxy:actor_cd_restart_matching_order'], 1)
        self.assertEqual(track['restart_then_died_within_24'], 1)
        self.assertEqual(track['survival_16:alive'], 1)
        self.assertEqual(track['survival_24:dead'], 1)
        for samples, end, frames, deaths, expected in [
                ([follow], 80, {18: dict(frame, visible=set())}, {}, 'target_not_visible_no_restart_observed'),
                ([follow], 80, {}, {'9': 15}, 'target_dead_no_restart_observed'),
                ([], 80, {}, {'7': 16}, 'dead_by_next_sample_shot_unknown'),
                ([], 18, {}, {}, 'terminal_censored'),
                ([], 80, {}, None, 'missing_observation')]:
            c = Counter()
            report._followup(row, samples, end, c, frames, deaths)
            self.assertEqual(c['next_8_macro_proxy:' + expected], 1)
            self.assertEqual(c['macro_cd_restart_proxy'], 0)

    def test_receipt_rejected_retargeted_or_overridden(self):
        row = dict(actor='u0', game_loop=2, switch='B', target_tag='9', final_action=6)
        rec = dict(requested_action_id='6', submission='accepted', response={'code': 1},
                   command=dict(ability_id=23, target_tag='9'))
        for action, receipt, expected in [(6, rec, 'accepted'), (3, rec, 'not_submitted_as_selected'),
                (6, dict(rec, submission='rejected'), 'rejected'),
                (6, dict(rec, command=dict(ability_id=23, target_tag='99')), 'receipt_mismatch')]:
            self.assertEqual(report._receipt({(1, 2): ([action], {'u0': receipt})}, 1, row), expected)

    def test_protection_snapshot_counts_a_target_once_not_per_pair(self):
        actor = dict(type='Hydralisk', alive=True, weapon_cooldown_loops=5, orders=[])
        rel = dict(smac_attack_available=True, in_weapon_range=True, in_enemy_weapon_range=True)
        snapshot = dict(allies={a: dict(actor, tag=a) for a in ('u0','u1','u2')}, visible_enemies={'e0': {'tag': '8'}},
                        relations={a: {'e0': rel} for a in ('u0','u1','u2')})
        f = report._snapshot(snapshot)
        self.assertEqual((f['threats'], f['jointly_covered'], f['isolated']), (1, 1, set()))
        snapshot['allies']['u0']['alive'] = False
        g = report._snapshot(snapshot)
        p = report._protection({2:f, 10:g}, 2)
        self.assertEqual(p['first_observed_ally_hydra_death_prior_macro_state'], 'supported')
        self.assertIsNone(report._protection({2:f, 18:g}, 2))


if __name__ == '__main__':
    unittest.main()
