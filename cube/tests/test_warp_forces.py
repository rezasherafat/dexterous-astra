"""Numerical checks for the custom CUDA forces, independent of MuJoCo-Warp's solver."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sim'))
try:
    import warp as wp
    from warp_backend import motors, detents, alignment, audit_contacts, audit_joints
except ImportError:
    wp = None


@unittest.skipIf(wp is None, 'Install mujoco-warp and warp-lang to test GPU forces')
class WarpForceTests(unittest.TestCase):
    def setUp(self):
        if not wp.is_cuda_available():
            self.skipTest('CUDA device required')
        self.scope = wp.ScopedDevice('cuda:0')
        self.scope.__enter__()
        self.addCleanup(self.scope.__exit__, None, None, None)

    def arr(self, value, dtype=float):
        return wp.array(np.asarray(value), dtype=dtype, device='cuda:0')

    def test_motor_filters_and_feedforward_match_cpu(self):
        from arms import JointFilter
        q, v, bias = np.array([[0.2, -0.3]]), np.array([[0.1, -0.1]]), np.array([[0.4, -0.4]])
        target, initial, velocity = np.array([0.6, -0.6]), np.array([0.2, -0.2]), np.array([0.1, -0.1])
        f = JointFilter(initial, 4.0, 60.0, frequency=40.0)
        f.v[:] = velocity
        expected_hand = np.clip(f.step(target, .001) + bias[0] / 2, -.4, .4)
        expected_arm = np.clip(10 * (target - q[0]) + 2 * (velocity - v[0]) + bias[0], -1, 1)
        ctrl, fq, fv = wp.zeros((1, 4)), self.arr(initial), self.arr(velocity)
        wp.launch(motors, dim=4, inputs=[self.arr(q), self.arr(v), self.arr(bias), ctrl,
                  self.arr([0, 1], int), self.arr([0, 1], int), self.arr([[-.4, .4]] * 2, wp.vec2),
                  self.arr([2, 2]), self.arr(target), fq, fv, self.arr([2, 3], int),
                  self.arr([0, 1], int), self.arr([0, 1], int), self.arr(target), self.arr(velocity),
                  self.arr([10, 10]), self.arr([2, 2]), self.arr([1, 1]), .001])
        np.testing.assert_allclose(ctrl.numpy()[0], np.r_[expected_hand, expected_arm], atol=1e-6)
        np.testing.assert_allclose(fq.numpy(), f.q, atol=1e-7)
        np.testing.assert_allclose(fv.numpy(), f.v, atol=1e-7)

    def test_cube_passive_forces_match_cpu(self):
        import mujoco
        import scene
        from physics import wells
        group = np.empty((len(scene.GROUP), 4))
        for i, rotation in enumerate(scene.GROUP):
            mujoco.mju_mat2Quat(group[i], rotation.astype(float).ravel())
        quats = np.array([[1, .02, -.03, .01], [-1, .01, -.04, .02]], float)
        quats /= np.linalg.norm(quats, axis=1, keepdims=True)
        vel = np.array([[.1, -.2, .3], [.4, -.1, .2]])
        force = wp.zeros((1, 6))
        wp.launch(alignment, dim=2, inputs=[self.arr(quats.reshape(1, -1)), self.arr(vel.reshape(1, -1)),
                  force, self.arr([0, 4], int), self.arr([0, 3], int), self.arr(group, wp.vec4),
                  .04, np.deg2rad(8), .0003])
        expected = wells(quats, vel, group, .04, np.deg2rad(8), .0003)
        np.testing.assert_allclose(force.numpy().reshape(2, 3), expected, atol=1e-7, rtol=1e-5)
        angle = np.array([[.1, 1.7, -1.4, -3.0]])
        force = wp.zeros((1, 4))
        wp.launch(detents, dim=4, inputs=[self.arr(angle), force, self.arr(range(4), int),
                  self.arr(range(4), int), .01])
        expected = -.01 * (angle - np.round(angle / (np.pi / 2)) * np.pi / 2)
        np.testing.assert_allclose(force.numpy(), expected, atol=1e-8)

    def test_substep_audit_ignores_unused_contacts_and_tracks_limits(self):
        worst = wp.zeros(2)
        wp.launch(audit_contacts, dim=3, inputs=[self.arr([2], int), self.arr([[0, 1], [1, 2], [0, 2]], wp.vec2i),
                  self.arr([-.002, -.1, -100]), self.arr([1, 0, 0], int), worst])
        wp.launch(audit_joints, dim=2, inputs=[self.arr([[.5, 1.2]]), self.arr([0, 1], int),
                  self.arr([[0, 1], [0, 1]], wp.vec2), worst])
        np.testing.assert_allclose(worst.numpy(), [.002, .2], atol=1e-7)


if __name__ == '__main__':
    unittest.main()
