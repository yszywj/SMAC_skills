import json
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from decision_tree.scripts import local_v2_collect as collector
from skills.local_v2 import explore


class CollectionTests(unittest.TestCase):
    @staticmethod
    def menu(families):
        effects = {'BASE': {'aliases':['retarget:u0:6']}}
        for i, family in enumerate(families):
            effects[str(i)] = {'aliases':[f'{family}:u{i}:7']}
        return {'effects':effects}

    def test_scene_budget_allows_boundary_and_ignores_base_only_steps(self):
        full = ['retarget','retreat_step','regroup_step','retreat_veto'] * 2
        entries = {(scene,step): self.menu(full) for scene in ('a','b') for step in range(4)}
        entries['a',4] = self.menu([])
        collector.check_scene_budget(entries)

    def test_scene_budget_rejects_extra_steps_effects_and_family_aliases(self):
        with self.assertRaisesRegex(ValueError, '4 intervention steps'):
            collector.check_scene_budget({('a',step):self.menu(['retarget']) for step in range(5)})
        with self.assertRaisesRegex(ValueError, 'limit of 8'):
            collector.check_scene_budget({('a',0):self.menu(['retarget']*9)})
        entry = self.menu(['retarget','retarget','retreat_step'])
        entry['effects']['2']['aliases'].append('retarget:u2:7')
        with self.assertRaisesRegex(ValueError, 'retarget has 3 distinct effects'):
            collector.check_scene_budget({('a',0):entry})
        # Repeated aliases of one effect do not create additional branch budget.
        entry['effects'].pop('1')
        entry['effects']['0']['aliases'].append('retarget:u0:7')
        collector.check_scene_budget({('a',0):entry})

    def test_max_games_does_not_bypass_budget_or_create_output(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); out=root/'collection'
            over={('a',step):self.menu(['retarget']) for step in range(5)}
            with patch.object(collector,'checked_directory',return_value=out), \
                    patch.object(collector,'load_plan',return_value=({},over,[])), \
                    patch.object(collector,'open_db') as db:
                with self.assertRaisesRegex(ValueError, '4 intervention steps'):
                    collector.collect(root/'plan',out,1,120,1,12,False)
                db.assert_not_called()
            self.assertFalse(out.exists())

    def test_stopped_marker_refuses_run_without_loading_or_creating(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); out=root/'collection'
            (root/'STOPPED_BY_USER.json').touch()
            with patch.object(collector,'checked_directory',return_value=out), \
                    patch.object(collector,'load_plan') as load:
                with self.assertRaisesRegex(ValueError, 'stopped by user'):
                    collector.collect(root/'plan',out,1,120,None,12,False)
                load.assert_not_called()
            self.assertFalse(out.exists())

    def test_archive_and_restore_exact_event_bytes(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in explore.FILES[:-1]:
                (root/name).write_text('{}',encoding='utf-8')
            raw=b'{"byte":"\xc3\xa9"}\r\n'
            (root/'events.jsonl').write_bytes(raw)
            original=explore.checksum(root/'events.jsonl')
            collector.archive_events(root,{'run_hashes':{'events.jsonl':original}})
            self.assertFalse((root/'events.jsonl').exists())
            with collector.raw_view(root) as view:
                self.assertEqual((view/'events.jsonl').read_bytes(),raw)
            self.assertEqual(collector.compressed_digest(root/'events.jsonl.gz'),original)

    def test_load_binds_all_jobs_and_interleaves_scenes(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'base';base.mkdir()
            for f in explore.FILES:
                (base/f).write_text('{}')
            rows=[]
            for ep in (1,2):
                for step in (0,1):
                    name='retarget:u1:7'
                    rows.append(dict(episode=ep,seed=100+ep,scene_hash=str(ep),step=step,anchor_sha256='x',
                                     effects={'BASE':{'id':'BASE','aliases':[]},'ef':{'id':name,'aliases':[name]}}))
            menu=root/'menus.jsonl';menu.write_text('\n'.join(json.dumps(x) for x in rows),encoding='utf-8')
            summary={'kind':'local_v2_exploration_plan','scope':explore.SCOPE,'base':str(base),
                     'base_hashes':{k:explore.checksum(base/k) for k in explore.FILES},
                     'source_hashes':explore.source_hashes(),'menus_sha256':explore.checksum(menu),
                     'counts':{'decision_steps':4,'nonbase_effects':4}}
            (root/'summary.json').write_text(json.dumps(summary))
            _,_,jobs=collector.load_plan(root)
            self.assertEqual([(j[1],j[4]) for j in jobs],[(1,0),(2,0),(1,1),(2,1)])
            menu.write_text('')
            with self.assertRaisesRegex(ValueError,'menu'):
                collector.load_plan(root)

    def test_ledger_identity_and_partial_counts(self):
        with TemporaryDirectory() as tmp:
            out=Path(tmp)
            with closing(collector.open_db(out)) as db:
                db.execute('CREATE TABLE tasks(id INTEGER,episode INTEGER,scene_hash TEXT,seed INTEGER,step INTEGER,effect TEXT,binding TEXT,state TEXT,attempts INTEGER)')
                jobs=[(0,1,'scene',5,10,'ef','retarget:u1:7')]
                db.execute("INSERT INTO tasks VALUES(1,1,'scene',5,10,'ef','retarget:u1:7','pending',0)")
                collector.validate_job_keys(db,jobs)
                self.assertEqual(collector.status(db)['accepted'],0)
                db.execute("UPDATE tasks SET step=11")
                with self.assertRaisesRegex(ValueError,'differ'):
                    collector.validate_job_keys(db,jobs)
