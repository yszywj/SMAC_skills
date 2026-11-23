"""Native proposal, lifecycle and historical decision replay tests (no SC2)."""
from collections import Counter
import copy
import importlib
from math import dist
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sc2_skill_adapter.tests import test_contact_local_packs_variant as installed
from contact_local_packs_report import _audit


class LocalEngagementTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Install the actual template via the production copy/hash/patch path.
        installed.LocalPacksTest.setUpClass()
        cls.addClassCleanup(installed.LocalPacksTest.doClassCleanups)
        cls.manager = installed.LocalPacksTest.reserve.SkillManager
        cls.L = importlib.import_module('skills.reserve.local_engagement')
        cls.context = importlib.import_module('skills.reserve.context')
        cls.memory = importlib.import_module('skills.reserve.memory').TacticalMemory
        cls.payload = next(installed.frames(3,0,1))[0]

    def window(self, allies, enemies, loop=2, *, health=None, tags=None, dogs=3, enemy_dogs=()):
        """Known catalog + native Context, with explicit synthetic geometry."""
        p = copy.deepcopy(self.payload)
        s = p['state']
        s['game_loop'] = loop
        enemy_template = next(iter(s['visible_enemies'].values()))
        gun = s['allies']['u9'].copy()
        dog = s['allies']['u0'].copy()
        s['visible_enemies'] = {e:dict(enemy_template,position=pos,tag=(tags or {}).get(e,str(201+int(e[1:]))),
                                  health=(health or {}).get(e,90.)) for e,pos in enemies.items()}
        for e in enemy_dogs:
            t = s['visible_enemies'][e]
            t.update({key: dog[key] for key in ('type', 'unit_type_id', 'radius', 'max_health')})
            t['health'] = t['effective_health'] = (health or {}).get(e, dog['max_health'])
        s['legal_actions'] = {}
        for a, unit in s['allies'].items():
            i = int(a[1:])
            unit.update(dog if i<dogs else gun)
            unit.update(position=allies.get(a,(100+i*4,100.)),alive=a in allies,
                        tag=(tags or {}).get(a,str(101+i)),orders=[])
            unit['health'] = (health or {}).get(a, unit['max_health']) if unit['alive'] else 0.
            unit['effective_health'] = unit['health']
            x,y = unit['position']
            s['movement_candidates'][a] = {str(k):dict(destination=(x+dx,y+dy))
                    for k,(dx,dy) in enumerate(((0,2),(0,-2),(2,0),(-2,0)),2)}
            s['relations'][a] = {}
            menu = [1,2,3,4,5]
            for e,t in s['visible_enemies'].items():
                d = dist(unit['position'],t['position'])
                reach = .1 if i<dogs else 5.
                limit = 2. if i<dogs else 5.
                incoming = d <= (.1 if e in enemy_dogs else 5.)+unit['radius']+t['radius']
                outgoing = d <= reach+unit['radius']+t['radius']
                s['relations'][a][e] = dict(distance=d, in_weapon_range=outgoing,
                        weapon_range_status='in_range' if outgoing else 'out_of_range',native_weapon_range=reach,
                        in_enemy_weapon_range=incoming,enemy_weapon_range_status='in_range' if incoming else 'out_of_range',
                        smac_attack_available=d<=limit,smac_attack_distance_limit=limit,
                        smac_center_distance_allowed=d<=limit)
                if d<=limit:
                    menu.append(6+int(e[1:]))
            s['legal_actions'][a] = menu if unit['alive'] else [0]
        ctx = self.context.Context(self.context.build_snapshot(p,(loop-2)//8+1))
        return ctx,{a:(1 if u.alive else 0) for a,u in ctx.snapshot.allies.items()}

    def commit(self, ctx, memory, ref, audit, plans=(), mismatch=False):
        joint = dict(ref)
        if not mismatch:
            for p in plans:
                joint.update(p.current_actions)
        memory.begin_step(ctx.snapshot)
        memory.commit(ctx.snapshot,joint,list(plans))
        self.L.commit_local(ctx,memory,audit)

    def test_multiple_fronts_not_joined_by_shooter_or_enemy_chain(self):
        ctx, ref = self.window({'u3':(-4,0),'u4':(0,0),'u5':(4,0),'u6':(0,5.5)},
                               {'e0':(-2.5,0),'e1':(0,1),'e2':(2.5,0)})
        m = self.memory()
        offers, trace = self.L.propose_local(ctx,m,ref,False)
        self.assertGreaterEqual(len(trace['packs']),2)
        self.assertFalse(any({'e0','e2'} <= set(p['enemies']) for p in trace['packs']))
        self.assertTrue(all(p.goal_data['witnesses'][a] for p in offers for a in p.controlled_actors))
        self.assertTrue(any(p.atomic for p in offers))
        self.assertEqual(trace['requests'],[])
        self.assertLessEqual(len(offers),self.L.MAX_OFFERS)
        self.assertTrue(all(set(p.current_actions)==set(p.claims.controls) for p in offers))

    def test_wounded_beneficiary_is_not_a_control_claim_and_kill_is_kept(self):
        ctx,ref = self.window({'u3':(0,0),'u4':(-2,1),'u5':(0,3)},
                              {'e0':(1,0),'e1':(2,1)}, health={'u3':8,'e1':3})
        ref['u4']=7   # the sole actual finishing shot on e1
        offers,audit = self.L.propose_local(ctx,self.memory(),ref,True)
        self.assertTrue(any('u3' in p.beneficiaries for p in offers))
        self.assertFalse(any({'u3','u4'} & set(p.controlled_actors) for p in offers))
        self.assertTrue(any('u3' in p['protected'] for p in audit['packs']))

    def test_direct_cooldown_rays_and_projected_moves_do_not_claim_damage(self):
        ctx,ref = self.window({'u3':(0,0),'u4':(-2,1),'u5':(6.5,2)}, {'e0':(1,0)})
        plans,_ = self.L.propose_local(ctx,self.memory(),ref,False)
        moves = [p for p in plans if p.current_actions.keys()=={'u5'}]
        self.assertTrue(moves)
        self.assertTrue(all(not p.claims.fire and p.goal_data['roles']['u5']=='approach_attack' for p in moves))
        self.assertTrue(all(p.goal_data['witnesses']['u5'] and not p.goal_data['explicit_attack']['u5'] for p in moves))

    def test_commit_bound_claims_handoff_stall_expiry_and_stable_tags(self):
        m = self.memory()
        def state(loop, x, enemy='e0',tag='201'):
            return self.window({'u3':(2,0),'u4':(x,.8),'u5':(7.4,-1)}, {enemy:(0,0)},
                               loop,tags={enemy:tag})
        def dispatch(ctx,ref):
            ps,d = self.L.propose_local(ctx,m,ref,True)
            self.assertEqual((ps,d),self.L.propose_local(ctx,m,ref,True))
            return [p for p in ps if p.primary_skill==self.L.RESERVE],d
        ctx,ref = state(2,7.4)
        plans,tr = dispatch(ctx,ref)
        self.assertEqual({p.controlled_actors for p in plans},{('u4',),('u5',)})
        self.assertFalse(m.instances)
        self.assertFalse(hasattr(m,'_local_observed'))
        self.assertTrue(self.L.local_conflicts(ctx,plans,{**ref,**plans[0].current_actions,**plans[1].current_actions}))
        self.commit(ctx,m,ref,tr,[plans[0]],mismatch=True)
        self.assertFalse(m.instances)
        ctx,ref = state(10,7.4)
        plans,tr = dispatch(ctx,ref)
        a = next(p for p in plans if p.controlled_actors==('u4',))
        expires = tr['requests'][0]['expires_loop']
        self.assertEqual(expires,34)
        self.commit(ctx,m,ref,tr,[a]); iid = a.instance_id
        self.assertIn(iid,m.instances)
        # The native enemy slot changes; the tagged request, seat, and
        # deadline do not. A closer competing bidder cannot steal a live seat.
        ctx,ref = state(18,6.6,'e7')
        plans,tr = dispatch(ctx,ref)
        self.assertEqual(tr['requests'][0]['claimed_by'],['u4'])
        self.assertEqual([(p.instance_id,p.goal) for p in plans],[(iid,'e7')])
        self.assertEqual(plans[0].goal_data['expires_loop'],expires)
        self.commit(ctx,m,ref,tr,plans)
        # Same observation slot, different native tag cannot inherit the task.
        other,oref = state(26,6.6,'e7','999')
        opts,other_trace = dispatch(other,oref)
        self.assertNotEqual(other_trace['packs'][0]['pack_id'],tr['packs'][0]['pack_id'])
        self.assertFalse(any(p.instance_id==iid for p in opts))
        # Reached a real contact: local ownership immediately wins against
        # the previously claimed reservation; it cannot keep walking blindly.
        ctx,ref = state(26,4.5,'e7')
        opts,tr = dispatch(ctx,ref)
        self.assertFalse(any('u4' in p.current_actions for p in opts))
        self.assertIn('local_contact_handoff',[r['reason'] for r in tr['releases']])
        self.commit(ctx,m,ref,tr)
        self.assertNotIn(iid,m.instances)

    def test_unclaimed_request_does_not_extend_forever(self):
        m = self.memory()
        initial = None
        for loop in range(2,59,8):
            ctx,ref = self.window({'u3':(2,0)}, {'e0':(0,0)},loop)
            plans,d = self.L.propose_local(ctx,m,ref,True)
            self.assertFalse(plans)
            if loop < 34:
                rid = d['requests'][0]['request_id']
                initial = initial or rid
                self.assertEqual(initial,rid)
                self.assertEqual(d['requests'][0]['expires_loop'],34)
            elif loop < 50:
                self.assertFalse(d['requests'])
            else:
                self.assertNotEqual(d['requests'][0]['request_id'],initial)
            self.commit(ctx,m,ref,d)

    def test_two_fronts_bid_for_one_idle_hydra_then_honor_or_release_lease(self):
        m = self.memory()
        def bid(loop,x):
            ctx,ref = self.window({'u3':(-8.,0.),'u4':(8.,0.),'u5':(x,0.)},
                                  {'e0':(-7.,0.),'e1':(7.,0.)},loop)
            plans,d = self.L.propose_local(ctx,m,ref,True)
            return ctx,ref,[p for p in plans if p.primary_skill==self.L.RESERVE],d
        ctx,ref,ps,d = bid(2,0.)
        self.assertEqual({p.goal for p in ps},{'e0','e1'})
        self.assertTrue(all(p.controlled_actors==('u5',) for p in ps))
        self.assertTrue(self.L.local_conflicts(ctx,ps,{**ref,**ps[0].current_actions}))
        left = next(p for p in ps if p.goal=='e0')
        self.commit(ctx,m,ref,d,[left])
        _,_,ps,d = bid(10,-.5)
        self.assertEqual([(p.goal,p.instance_id) for p in ps],[('e0',left.instance_id)])
        _,_,ps,d = bid(10,0.)  # same clock, different fixture: progress stopped
        self.assertNotIn('e0',[p.goal for p in ps])
        self.assertIn('blocked_or_preempted',[r['reason'] for r in d['releases']])

    def test_branch_spends_once_on_enemy_and_checks_crossing_paths(self):
        ctx,ref = self.window({'u3':(-1,0),'u4':(1,0),'u5':(-2,2)},
                              {'e0':(0,1)},health={'e0':8})
        ps,_=self.L.propose_local(ctx,self.memory(),ref,False)
        p,q = [next(p for p in ps if p.controlled_actors==(a,)) for a in ('u3','u4')]
        q.goal_data={**q.goal_data,'pack_id':'other-front'}
        for plan in (p,q):
            self.assertFalse(self.L.local_conflicts(ctx,[plan],{**ref,**plan.current_actions}))
        self.assertIn('redundant_fire:e0',self.L.local_conflicts(ctx,[p,q],{**ref,**p.current_actions,**q.current_actions}))
        # Even different seats/targets cannot issue mutually crossing moves.
        p.current_actions={'u3':4}; q.current_actions={'u4':5}
        self.assertTrue(self.L.local_conflicts(ctx,[p,q],{**ref,'u3':4,'u4':5})[0].startswith('path:'))

    def test_strict_composition_gate(self):
        for dogs in (3,4,5,6,7):
            ctx,ref = self.window({'u3':(2,0)}, {'e0':(0,0)},dogs=dogs)
            _,d = self.L.propose_local(ctx,self.memory(),ref,True)
            self.assertEqual(d['in_scope'],dogs in (3,6))

    def test_installed_policy_real_observations_through_report_audit(self):
        selected = Counter()
        for dogs, shard, count in ((3,0,16),(6,10,10)):
            m = self.manager()
            for p,_,_ in installed.frames(dogs,shard,count):
                actions,summary=m.step(p,10)
                state=p['state']; trace=summary['local_coord']; loop=state['game_loop']
                _,_,accepted = _audit({**summary,'game_loop':loop},'packs_reserve',p['questions'],
                    {a:u for a,u in state['allies'].items() if u['alive']},
                    {e:u['tag'] for e,u in state['visible_enemies'].items() if u['alive']},f'{dogs}:{loop}')
                m.commit_final_actions(actions)
                for a,row in accepted.items():
                    selected[row['skill']] += row['changed']
                    if row['skill']==self.L.RESERVE:
                        self.assertTrue(any(i.skill==self.L.RESERVE and i.actors==(a,) and
                           i.extra['request_id']==row['request_id'] for i in m.memory.instances.values()))
                self.assertEqual(trace['candidate_count'],sum(trace['candidate_skills'].values()))
        self.assertGreater(selected[self.L.LOCAL],0)
        self.assertGreater(selected[self.L.RESERVE],0)


if __name__=='__main__':
    unittest.main()
