"""Constructed signal tests, not physically validated simulated trajectories."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sim'))
from spec import JOINTS
from rewards import evaluate_trial


def fixture(kind='success'):
    t = np.arange(1, 721) / 60
    angle = np.minimum(t * np.pi, 6 * np.pi)
    speed = np.where(t < 6, np.pi, 0.)
    force = np.full((720, 5), .03)
    force[(t >= 1) & (t < 2), 0] = 0
    qd = np.full((720, 22), .2)
    position = np.tile([0, 0, .6], (720, 1)).astype(float)
    action = np.zeros((720, 22))
    if kind == 'static_jitter':
        angle[:] = 0; speed[:] = 0
        action[:] = np.where(np.arange(720)[:, None] % 2, 1, -1)
        qd *= 10
    elif kind == 'rocking':
        angle = .2 * (1 - np.cos(4 * np.pi * t)); speed = .8 * np.pi * np.sin(4 * np.pi * t)
    elif kind == 'drop_then_spin':
        position[t >= .5, 2] -= .04
    elif kind == 'unsupported_thumb':
        force[(t >= 1) & (t < 2)] = 0
    elif kind == 'no_thumb_release':
        force[:, 0] = .03
    elif kind == 'unstable_hold':
        speed[t >= 6] = np.radians(90)
    elif kind == 'overspin':
        angle = t * np.pi; speed[:] = np.pi
    elif kind == 'thumb_cycles':
        angle = .3 * t; speed[:] = .3
        force[:, 0] = np.where(np.arange(720) % 30 < 24, 0, .03)
    elif kind == 'near_zero_forward_jitter':
        angle = .00001 * t; speed[:] = .00001; qd *= 10
    elif kind == 'bad_penetration':
        pass
    elif kind != 'success':
        raise ValueError(kind)
    return dict(pen_axis=np.c_[np.cos(angle), np.sin(angle), np.zeros(720)],
        heading0=0., pen_angvel=np.c_[np.zeros((720, 2)), speed],
        finger_force=force, pen_pos=position, ref_pos=np.array([0, 0, .6]),
        q=np.zeros((720, 22)), q0=np.zeros(22), qd=qd,
        q_lo=np.full(22, -1.), q_hi=np.ones(22), action=action,
        physx_penetration=np.full(720, .004 if kind == 'bad_penetration' else .0005))


def evaluate_scenarios():
    result = {}
    for kind in ('success', 'static_jitter', 'rocking', 'drop_then_spin', 'unsupported_thumb',
                 'no_thumb_release', 'unstable_hold', 'overspin', 'thumb_cycles',
                 'near_zero_forward_jitter', 'bad_penetration'):
        components, diagnostics = evaluate_trial(fixture(kind), JOINTS)
        total = sum(components.values())
        result[kind] = dict(components={k: float(v.sum()) for k, v in components.items()},
            total=float(total.sum()), discounted_099=float(total @ (.99 ** np.arange(720))),
            thumb_shaping_discounted_099=float(components['thumb_shaping'] @ (.99 ** np.arange(720))),
            diagnostics=diagnostics)
    return result
