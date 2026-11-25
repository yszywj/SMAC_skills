import unittest

from decision_tree.zerg.combat_relay_r8 import CombatRelayR8


def frame(step, *, alive=False, casualty='Hydralisk', distance=10, visible=True,
          contact=True, restored=False, health=35, protected=(), anchor_action=6):
    # u0 owns a firing lane, u1 is a disconnected helper, u2 is the first loss.
    def unit(x, typ, living=True, hp=35):
        return dict(position=[x, 0], alive=living, type=typ, health=hp, max_health=35)
    base = [anchor_action if visible and contact else 1, 1, int(alive)]
    masks = [[0,1,1,1,1,1,int(visible and contact)], [0,1,1,1,1,1,int(restored)],
             [int(not alive),int(alive),0,0,0,0,0]]
    s = dict(game_loop=2+8*step, step_mul=8,
             allies={'u0':unit(1,'Hydralisk'), 'u1':unit(distance,'Zergling',hp=health),
                     'u2':unit(20,casualty,alive)},
             visible_enemies={'e0':unit(0,'Hydralisk')} if visible else {},
             relations={f'u{i}':{'e0':dict(native_weapon_range=1, in_weapon_range=i==0)} for i in range(3)},
             movement_candidates={'u1':{'4':{'destination':[distance+2,0]},'5':{'destination':[distance-2,0]}}})
    return dict(state=s, relay_protected_actors=protected), masks, base, {'BASE':base}


class RelayTest(unittest.TestCase):
    def armed(self, mode='full', **kw):
        c = CombatRelayR8(mode)
        self.assertEqual(c.propose(*frame(0, alive=True, **kw))['phase'], 'r6')
        self.assertEqual(c.propose(*frame(1, **kw))['phase'], 'r6')
        return c

    def test_one_native_choice_or_new_complete_vector(self):
        for mode in ('menu', 'full'):
            c = self.armed(mode)
            p,m,b,menu = frame(2)
            r = c.propose(p,m,b,menu)
            self.assertEqual(r['actions'], b if mode=='menu' else [6,5,0])
            if mode == 'full':
                self.assertTrue(r['audit']['outside_menu'])
            else:
                menu['native'] = [6,5,0]
                r = c.propose(*frame(3)[:3],menu)
                self.assertEqual((r['actions'],r['audit']['native_label']), (menu['native'],'native'))

    def test_casualty_contact_health_and_witness(self):
        for kw in ({'casualty':'Baneling'}, {'contact':False}, {'health':8}, {'distance':13}):
            c = self.armed(**kw)
            self.assertEqual(c.propose(*frame(2, **kw))['phase'], 'r6')
        c = self.armed()
        self.assertEqual(c.propose(*frame(3))['phase'],'r6')  # missing observation
        self.assertEqual(c.propose(*frame(4))['phase'],'relay')
        c = CombatRelayR8('full')
        for n in range(3):  # pre-contact casualty never retroactively arms relay
            self.assertEqual(c.propose(*frame(n, alive=n==0, contact=n==2))['phase'],'r6')
        self.assertIsNone(c.loss)

    def test_preserves_any_attack_option_and_protected_duties(self):
        for mode in ('menu','full'):
            c = self.armed(mode)
            p,m,b,menu = frame(2, protected=['u1'], anchor_action=1)
            menu['tempting'] = [6,5,0]
            self.assertEqual(c.propose(p,m,b,menu)['actions'],b)
            p,m,b,menu = frame(3,anchor_action=1)
            menu['retask_anchor'] = [6,5,0]
            r = c.propose(p,m,b,menu)
            self.assertEqual(r['actions'][0],1)  # don't overwrite intentional attack-capable R6 actor

    def test_runtime_veto_and_invalid_vectors(self):
        for mode in ('menu','full'):
            c = self.armed(mode)
            p,m,b,_ = frame(2)
            self.assertEqual(c.propose(p,m,b,{})['actions'], b)
            self.assertEqual(c.propose(*frame(3)[:3], {'partial':[5], 'illegal':[6,6,0]})['actions'], b)
            with self.assertRaises(ValueError):
                c.propose(p,m,[6,6,0],{})

    def test_early_exit_permanently_hands_back(self):
        for kw,reason in (({'visible':False},'target_absent_or_front_lost'),
                          ({'restored':True},'support_restored'), ({},'no_observed_progress')):
            c = self.armed()
            self.assertEqual(c.propose(*frame(2))['phase'],'relay')
            self.assertEqual(c.propose(*frame(3, **kw))['reason'], reason)
            self.assertEqual(c.propose(*frame(4,distance=9))['reason'],reason)

    def test_four_step_budget(self):
        c = self.armed()
        for n in range(2,6):
            self.assertEqual(c.propose(*frame(n,distance=12-n))['phase'],'relay')
        self.assertEqual(c.propose(*frame(6,distance=6))['reason'],'step_limit')

    def test_does_not_replace_R6_that_already_closes(self):
        for mode in ('menu','full'):
            c = self.armed(mode)
            p,m,b,menu = frame(2)
            b[1] = 5
            self.assertEqual(c.propose(p,m,b,menu)['reason'],'no_improving_command')
