"""Reward behavior, scalar-reference agreement, and row reset isolation."""
import sys
from pathlib import Path
import unittest
import numpy as np
import torch
PEN=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PEN/'training'),str(PEN/'analysis')]
from replay import replay
from reward import PenReward
from synthetic import fixture,JOINTS
from rewards import evaluate_trial

class TrainingRewards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kinds=('success','drop_then_spin','near_zero_forward_jitter','thumb_cycles','unsupported_thumb','overspin')
        cls.records=[fixture(k) for k in cls.kinds]
        cls.c,cls.stop=replay(cls.records,JOINTS)

    def test_event_budgets(self):
        for k,v in [('rotation',20),('timely_turn',9),('hold',10),('thumb',8),('participation',6)]:
            self.assertAlmostEqual(self.c[k][0].sum(),v,places=4)

    def test_unmodified_components_match_numpy_reference(self):
        for i,r in enumerate(self.records):
            ref,_=evaluate_trial(r,JOINTS)
            for k in ('rotation','time','drop','instability','braking','joint_limit','penetration_proxy','action_change'):
                np.testing.assert_allclose(self.c[k][i].sum(),ref[k].sum(),atol=1e-5,err_msg=f'{i}: {k}')

    def test_no_credit_for_tiny_motion_jitter(self):
        self.assertEqual(self.c['participation'][2].sum(),0)

    def test_failed_thumb_cycles_have_no_discounted_credit(self):
        self.assertEqual(self.c['thumb'][3] @ (.995**np.arange(720)),0)
        self.assertEqual(self.c['thumb'][4].sum(),0)

    def test_drop_and_hold_stop_rewards(self):
        self.assertEqual(self.c['drop'][1].sum(),-40)
        for i in range(len(self.records)):
            self.assertTrue(all(np.all(v[i,self.stop[i]:]==0) for v in self.c.values()))
        self.assertEqual(self.c['hold'][5].sum(),0)

    def test_reset_isolated_to_requested_rows(self):
        model=PenReward(3,'cpu',JOINTS)
        for v in model.state.values():v.fill_(7)
        model.reset(torch.tensor([1]))
        for v in model.state.values():
            self.assertTrue(torch.all(v[1]==0))
            self.assertTrue(torch.all(v[[0,2]]==7))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_matches_cpu(self):
        c,stop=replay(self.records,JOINTS,'cuda:0')
        np.testing.assert_array_equal(stop,self.stop)
        for k in c:np.testing.assert_allclose(c[k],self.c[k],atol=2e-5)

if __name__=='__main__':unittest.main()
