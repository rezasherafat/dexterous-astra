"""Behavioral regression checks on constructed trajectories, no simulator needed."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
from rewards import evaluate_trial
from synthetic import fixture, JOINTS

class RewardTests(unittest.TestCase):
    def evaluate(self, kind):
        return evaluate_trial(fixture(kind), JOINTS)

    def test_success_has_bounded_budgets(self):
        c,d=self.evaluate('success')
        self.assertTrue(d['proxy_success'])
        for k,v in [('rotation',20),('timely_turn',9),('participation',6),('completion_proxy',40)]:
            self.assertAlmostEqual(c[k].sum(),v)
        self.assertAlmostEqual(c['hold_shaping'].sum()+c['hold_bonus'].sum(),10)
        self.assertAlmostEqual(c['thumb_shaping'].sum()+c['thumb_bonus'].sum(),8)

    def test_rocking_cannot_repeat_rotation_credit(self):
        c,_=self.evaluate('rocking')
        self.assertLessEqual(c['rotation'].sum(),20*.4/(6*np.pi)+1e-8)

    def test_static_jitter_gets_no_participation(self):
        c,_=self.evaluate('static_jitter')
        self.assertEqual(c['participation'].sum(),0)
        self.assertLess(c['action_change'].sum(),0)

    def test_no_rewards_after_drop(self):
        c,d=self.evaluate('drop_then_spin')
        self.assertEqual(d['terminal'],'drop')
        self.assertEqual(c['drop'].sum(),-40)
        self.assertTrue(all(np.all(v[d['stop_step']:]==0) for v in c.values()))
        self.assertLess(c['rotation'].sum(),2)

    def test_unsupported_release_earns_no_thumb_bonus(self):
        c,d=self.evaluate('unsupported_thumb')
        self.assertEqual(c['thumb_bonus'].sum(),0)
        self.assertFalse(d['proxy_success'])

    def test_overspin_capped_and_no_hold_bonus(self):
        c,_=self.evaluate('overspin')
        self.assertAlmostEqual(c['rotation'].sum(),20)
        self.assertEqual(c['hold_bonus'].sum(),0)
        self.assertLess(c['braking'].sum(),0)

    def test_penetration_blocks_proxy_success(self):
        c,d=self.evaluate('bad_penetration')
        self.assertFalse(d['proxy_success'])
        self.assertLess(c['penetration_proxy'].sum(),0)

    def test_discounted_plain_potential_cycle_is_known_risk(self):
        c,_=self.evaluate('thumb_cycles')
        self.assertAlmostEqual(c['thumb_shaping'].sum(),0)
        self.assertGreater(c['thumb_shaping'] @ (.99**np.arange(720)),0)

    def test_initial_heading_wrap_is_not_an_extra_turn(self):
        r=fixture();theta=np.full(720,-np.pi+.01)
        r['heading0']=np.pi-.01
        r['pen_axis']=np.c_[np.cos(theta),np.sin(theta),np.zeros(720)]
        c,_=evaluate_trial(r,JOINTS)
        self.assertAlmostEqual(c['rotation'].sum(),20*.02/(6*np.pi))

if __name__=='__main__':unittest.main()
