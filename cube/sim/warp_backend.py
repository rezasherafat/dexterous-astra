"""CUDA physics with CPU planning/readback at controller boundaries.

The integration, contacts, constraints, motor control, and passive cube forces run
on MuJoCo-Warp. CPU MuJoCo is retained for the existing IK planner and rendering.
There is no silent CPU stepping fallback. This backend changes numerical/contact
behavior and must be evaluated separately from the CPU-trained baseline.
"""
import time

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from arms import KP, KD, TORQUE
import scene


@wp.kernel
def motors(q: wp.array2d[float], v: wp.array2d[float], bias: wp.array2d[float],
           ctrl: wp.array2d[float], hand_act: wp.array[int], hand_dof: wp.array[int],
           hand_range: wp.array[wp.vec2], hand_gain: wp.array[float],
           target: wp.array[float], fq: wp.array[float], fv: wp.array[float],
           arm_act: wp.array[int], arm_qa: wp.array[int], arm_va: wp.array[int],
           arm_target: wp.array[float], arm_velocity: wp.array[float],
           kp: wp.array[float], kd: wp.array[float], torque: wp.array[float], dt: float):
    i = wp.tid()
    if i < hand_act.shape[0]:
        acceleration = wp.clamp(1600.0 * (target[i] - fq[i]) - 80.0 * fv[i], -60.0, 60.0)
        velocity = wp.clamp(fv[i] + acceleration * dt, -4.0, 4.0)
        position = fq[i] + velocity * dt
        fv[i] = velocity
        fq[i] = position
        command = position + bias[0, hand_dof[i]] / hand_gain[i]
        ctrl[0, hand_act[i]] = wp.clamp(command, hand_range[i][0], hand_range[i][1])
    else:
        j = i - hand_act.shape[0]
        command = kp[j] * (arm_target[j] - q[0, arm_qa[j]])
        command += kd[j] * (arm_velocity[j] - v[0, arm_va[j]]) + bias[0, arm_va[j]]
        ctrl[0, arm_act[j]] = wp.clamp(command, -torque[j], torque[j])


@wp.kernel
def detents(q: wp.array2d[float], force: wp.array2d[float], qa: wp.array[int], va: wp.array[int], k: float):
    i = wp.tid()
    angle = q[0, qa[i]]
    quarter = 1.5707963267948966
    force[0, va[i]] = -k * (angle - wp.round(angle / quarter) * quarter)


@wp.kernel
def alignment(q: wp.array2d[float], v: wp.array2d[float], force: wp.array2d[float],
              qa: wp.array[int], va: wp.array[int], group: wp.array[wp.vec4],
              stiffness: float, width: float, damping: float):
    i = wp.tid()
    a, b = qa[i], va[i]
    quat = wp.vec4(q[0, a], q[0, a + 1], q[0, a + 2], q[0, a + 3])
    best = float(-1.0)
    goal = wp.vec4(1.0, 0.0, 0.0, 0.0)
    for j in range(group.shape[0]):
        dot = wp.dot(quat, group[j])
        if wp.abs(dot) > best:
            best = wp.abs(dot)
            goal = group[j]
    if wp.dot(quat, goal) < 0.0:
        goal = -goal
    scalar = wp.dot(quat, goal)
    qv = wp.vec3(quat[1], quat[2], quat[3])
    gv = wp.vec3(goal[1], goal[2], goal[3])
    vector = quat[0] * gv - goal[0] * qv - wp.cross(qv, gv)
    norm = wp.length(vector)
    angle = 2.0 * wp.atan2(norm, wp.clamp(scalar, 0.0, 1.0))
    error = vector * (angle / wp.max(norm, 1.0e-12))
    near = wp.exp(-0.5 * (angle / width) * (angle / width))
    for j in range(3):
        force[0, b + j] = near * (stiffness * error[j] - damping * v[0, b + j])


@wp.kernel
def audit_contacts(ncon: wp.array[int], geoms: wp.array[wp.vec2i], dist: wp.array[float],
                   robot: wp.array[int], worst: wp.array[float]):
    i = wp.tid()
    if i < ncon[0]:
        pair = geoms[i]
        if pair[0] >= 0 and pair[1] >= 0:
            if robot[pair[0]] != 0 or robot[pair[1]] != 0:
                wp.atomic_max(worst, 0, -dist[i])


@wp.kernel
def audit_joints(q: wp.array2d[float], qa: wp.array[int], limits: wp.array[wp.vec2], worst: wp.array[float]):
    i = wp.tid()
    value = q[0, qa[i]]
    wp.atomic_max(worst, 1, wp.max(limits[i][0] - value, value - limits[i][1]))


@wp.kernel
def collect_overflow(current: wp.array[int], accumulated: wp.array[int]):
    wp.atomic_or(accumulated, 0, current[0])


class WarpBackend:
    def __init__(self, sim, device='cuda:0', nconmax=2048, njmax=8192):
        self.sim = sim
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError('MuJoCo-Warp physics requires a CUDA device.')
        if sim.arms is None:
            raise ValueError('This GPU backend currently supports the policy arm scene only.')
        started = time.monotonic()
        self.steps = 0
        self.graphs = {}
        self.model_copies = {}
        self.nconmax, self.njmax = nconmax, njmax
        m, d = sim.m, sim.d
        with wp.ScopedDevice(self.device):
            self.model = mjw.put_model(m)
            # Warp 1.17 occupancy queries load kernels at the default 256-thread
            # block size; using the same launch size avoids stale symbol metadata.
            self.model.block_dim.convex_ccd = 256
            self.data = mjw.put_data(m, d, nconmax=nconmax, njmax=njmax)
            self._remember_model()
            arr = lambda x, dtype=float: wp.array(np.asarray(x), dtype=dtype, device=self.device)
            self.hand_act = arr(sim.hand_act, int)
            self.hand_dof = arr(sim.hand_dofs, int)
            self.hand_range = arr(m.actuator_ctrlrange[sim.hand_act], wp.vec2)
            self.hand_gain = arr(m.actuator_gainprm[sim.hand_act, 0])
            self.target = arr(sim.hand_target)
            self.fq, self.fv = arr(sim.fingers.q), arr(sim.fingers.v)
            self.arm_act = arr(np.concatenate([sim.arms.act[s] for s in 'lr']), int)
            self.arm_qa = arr(np.concatenate([sim.arms.qadr[s] for s in 'lr']), int)
            self.arm_va = arr(np.concatenate([sim.arms.dofs[s] for s in 'lr']), int)
            self.arm_target = arr(np.concatenate([sim.arms.target[s] for s in 'lr']))
            self.arm_velocity = arr(np.concatenate([sim.arms.filters[s].v for s in 'lr']))
            self.kp, self.kd, self.torque = [arr(np.tile(x, 2)) for x in (KP, KD, TORQUE)]
            self.centre_qa = arr([int(j.qposadr[0]) for j in sim.centres], int)
            self.centre_va = arr([int(j.dofadr[0]) for j in sim.centres], int)
            balls = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL
                     and m.body(m.jnt_bodyid[j]).name.startswith('cube/')]
            self.ball_qa, self.ball_va = arr(m.jnt_qposadr[balls], int), arr(m.jnt_dofadr[balls], int)
            group = np.empty((len(scene.GROUP), 4))
            for i, rotation in enumerate(scene.GROUP):
                mujoco.mju_mat2Quat(group[i], rotation.astype(float).ravel())
            self.group = arr(group, wp.vec4)
            self.robot = arr(sim.robot_geom, int)
            self.joint_qa = arr(m.jnt_qposadr[sim.hand_joints], int)
            self.joint_limits = arr(m.jnt_range[sim.hand_joints], wp.vec2)
            self.worst = wp.zeros(2, dtype=float, device=self.device)
            self.overflow = wp.zeros(1, dtype=int, device=self.device)
            from physics import PROFILE
            self.force_parameters = PROFILE['passive_cubie_alignment']
            self.detent = PROFILE['detent_stiffness_Nm_per_rad']
            print('GPU physics: compiling CUDA kernels (first startup can take several minutes)', flush=True)
            self._capture(1)
            self.forward()
        self.setup_seconds = time.monotonic() - started
        print(f'GPU physics ready on {self.device}: {self.setup_seconds:.1f}s setup', flush=True)

    def _remember_model(self):
        for name in ('geom_contype', 'geom_conaffinity', 'eq_data', 'dof_armature', 'dof_damping', 'dof_frictionloss'):
            self.model_copies[name] = getattr(self.sim.m, name).copy()

    def _sync_model(self):
        m = self.sim.m
        # Collision masks are compiled into candidate-pair tables: merely copying
        # geom_contype would silently leave stale collision pairs on the GPU.
        if any(not np.array_equal(getattr(m, name), self.model_copies[name])
               for name in ('geom_contype', 'geom_conaffinity')):
            self.model = mjw.put_model(m)
            # Warp 1.17 occupancy queries load kernels at the default 256-thread
            # block size; using the same launch size avoids stale symbol metadata.
            self.model.block_dim.convex_ccd = 256
            self.graphs.clear()
            self._remember_model()
        for name in ('eq_data', 'dof_armature', 'dof_damping', 'dof_frictionloss'):
            value = getattr(m, name)
            if not np.array_equal(value, self.model_copies[name]):
                getattr(self.model, name).assign(value[None].astype(np.float32))
                self.model_copies[name] = value.copy()
        self.data.eq_active.assign(self.sim.d.eq_active[None])

    def _forces(self):
        d = self.data
        wp.launch(motors, dim=self.sim.m.nu, inputs=[d.qpos, d.qvel, d.qfrc_bias, d.ctrl,
                  self.hand_act, self.hand_dof, self.hand_range, self.hand_gain,
                  self.target, self.fq, self.fv, self.arm_act, self.arm_qa, self.arm_va,
                  self.arm_target, self.arm_velocity, self.kp, self.kd, self.torque, self.sim.m.opt.timestep])
        wp.launch(detents, dim=6, inputs=[d.qpos, d.qfrc_applied, self.centre_qa, self.centre_va, self.detent])
        p = self.force_parameters
        wp.launch(alignment, dim=self.ball_qa.shape[0], inputs=[d.qpos, d.qvel, d.qfrc_applied,
                  self.ball_qa, self.ball_va, self.group, p['stiffness_Nm_per_rad'],
                  np.deg2rad(p['width_degrees']), p['damping_Nm_s_per_rad']])

    def _capture(self, nsub):
        with wp.ScopedCapture(device=self.device) as capture:
            for _ in range(nsub):
                self._forces()
                mjw.step(self.model, self.data)
                wp.launch(audit_contacts, dim=self.nconmax, inputs=[self.data.nacon, self.data.contact.geom,
                          self.data.contact.dist, self.robot, self.worst])
                wp.launch(audit_joints, dim=self.joint_qa.shape[0], inputs=[self.data.qpos,
                          self.joint_qa, self.joint_limits, self.worst])
                wp.launch(collect_overflow, dim=1, inputs=[self.data.overflow, self.overflow])
        self.graphs[nsub] = capture.graph

    def _readback(self):
        overflow = int(self.overflow.numpy()[0]) | int(self.data.overflow.numpy()[0])
        if overflow:
            raise RuntimeError(f'MuJoCo-Warp buffer overflow {overflow}; simulation stopped instead of dropping contacts.')
        mjw.get_data_into(self.sim.d, self.sim.m, self.data)
        if not np.isfinite(self.sim.d.qpos).all() or not np.isfinite(self.sim.d.qvel).all():
            raise RuntimeError('Non-finite GPU simulation state.')

    def forward(self):
        with wp.ScopedDevice(self.device):
            self._sync_model()
            if 'forward' not in self.graphs:
                with wp.ScopedCapture(device=self.device) as capture:
                    mjw.forward(self.model, self.data)
                self.graphs['forward'] = capture.graph
            wp.capture_launch(self.graphs['forward'])
            self._readback()

    def step(self, nsub):
        with wp.ScopedDevice(self.device):
            self._sync_model()
            self.target.assign(self.sim.hand_target.astype(np.float32))
            self.arm_target.assign(np.concatenate([self.sim.arms.target[s] for s in 'lr']).astype(np.float32))
            self.arm_velocity.assign(np.concatenate([self.sim.arms.filters[s].v for s in 'lr']).astype(np.float32))
            self.worst.zero_()
            self.overflow.zero_()
            if nsub not in self.graphs:
                self._capture(nsub)
            wp.capture_launch(self.graphs[nsub])
            self._readback()
            self.steps += nsub
            self.sim.fingers.q[:] = self.fq.numpy()
            self.sim.fingers.v[:] = self.fv.numpy()
            worst = self.worst.numpy()
            self.sim.substep_worst = dict(penetration_m=float(worst[0]), joint_violation_rad=float(worst[1]))

    def info(self):
        return dict(backend='warp', device=str(self.device), steps=self.steps,
                    setup_seconds=self.setup_seconds, nconmax=self.nconmax, njmax=self.njmax,
                    precision='float32', planner='CPU')
