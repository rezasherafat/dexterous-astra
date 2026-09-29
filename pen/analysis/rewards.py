"""Offline candidate reward v0. Does not change the simulator or frozen gate.

Arrays include the settled state at index zero internally. Returned reward arrays
contain one entry per recorded action. Shaping uses plain potential differences;
its discounted cycling behavior is deliberately audited, not claimed invariant.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
A = json.loads((ROOT / 'policy/acceptance.json').read_text())
WEIGHTS = json.loads(Path(__file__).with_name('reward_config.json').read_text())
FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')


def hold_potential(progress, omega, touch, dropped, third):
    """Best currently eligible partial hold; latch only after a full valid window."""
    phi = np.zeros(len(progress))
    complete = None
    if third is None:
        return phi, complete
    duration = round(A['hold_s'] * A['control_hz'])
    deadline = round(A['max_stop_s'] * A['control_hz'])
    for start in range(third, min(third + deadline + 1, len(progress))):
        stop = min(start + duration, len(progress))
        length = np.arange(1, stop - start + 1)
        heading = progress[start:stop]
        stable = (np.maximum.accumulate(omega[start:stop]) < np.radians(A['hold_max_omega_deg_s']))
        stable &= (np.maximum.accumulate(heading) - np.minimum.accumulate(heading)
                   < np.radians(A['hold_max_heading_change_deg']))
        stable &= ~np.maximum.accumulate(dropped[start:stop])
        support = np.cumsum(touch[start:stop].any(axis=1)) / length
        partial = length / duration * np.minimum(support / A['hold_min_support_frac'], 1) * stable
        phi[start:stop] = np.maximum(phi[start:stop], partial)
        if len(length) == duration and stable[-1] and support[-1] >= A['hold_min_support_frac']:
            end = stop - 1
            complete = end if complete is None else min(complete, end)
    if complete is not None:
        phi[complete:] = 1
    return phi, complete


def thumb_potential(progress, touch, dropped, end):
    """Credit maximal thumb-free intervals, as the gate does, not arbitrary prefixes.

An interval is committed when the thumb recontacts or the third turn is reached.
The zero-contact settled sample matches the gate's synthetic initial sample.
"""
    dt = 1 / A['control_hz']
    phi = np.zeros(len(progress))
    complete = None
    start = None
    for j in range(end + 1):
        if not touch[j, 0]:
            if start is None:
                start = j
            duration = (j - start + 1) * dt
            gain = max(0., progress[j] - progress[start])
            support = (touch[start:j + 1, 1:].sum(axis=1) >= A['thumb_free_min_other_fingers']).mean()
            safe = not dropped[start:j + 1].any()
            qualifies = support >= A['thumb_free_support_frac'] and safe
            phi[j] = (min(duration / A['thumb_free_min_s'], 1)
                      * min(gain / np.radians(A['thumb_free_min_rotation_deg']), 1) * qualifies)
            finishes = j == end or touch[j + 1, 0]
            if finishes:
                if (duration >= A['thumb_free_min_s'] and gain >= np.radians(A['thumb_free_min_rotation_deg'])
                        and qualifies):
                    complete = j if j == end else j + 1
                    phi[complete:] = 1
                    return phi, complete
                start = None
    return phi, complete


def evaluate_trial(rec, joints, weights=None):
    """Return candidate reward traces and diagnostics from one saved trial.

rec contains the evaluator's per-trial trajectory fields and common joint limits.
Drop ends rewards immediately; otherwise the first qualifying hold ends rewards.
No post-termination rewards are counted. PhysX is a proxy for the independent
MuJoCo penetration check; initial PhysX contact penetration was not recorded.
"""
    w = WEIGHTS if weights is None else weights
    dt = 1 / A['control_hz']
    T = len(rec['q'])
    size = T + 1
    heading = np.unwrap(np.r_[float(rec['heading0']), np.arctan2(rec['pen_axis'][:, 1], rec['pen_axis'][:, 0])])
    progress = A['spin_sign'] * (heading - heading[0])
    best = np.clip(np.maximum.accumulate(progress), 0, A['turns'] * 2 * np.pi)
    hits = []
    for turn in range(1, A['turns'] + 1):
        indices = np.flatnonzero(progress >= turn * 2 * np.pi)
        hits.append(int(indices[0]) if len(indices) else None)
    touch = np.vstack([np.zeros((1, 5), dtype=bool), rec['finger_force'] > A['contact_n']])
    omega = np.r_[0., np.abs(rec['pen_angvel'][:, 2])]
    rel = np.vstack([np.zeros(3), rec['pen_pos'] - rec['ref_pos']])
    tilt = np.r_[0., np.degrees(np.arcsin(np.clip(np.abs(rec['pen_axis'][:, 2]), 0, 1)))]
    ratios = np.c_[np.maximum(0, -rel[:, 2]) / A['drop_z'],
                   np.linalg.norm(rel[:, :2], axis=1) / A['drop_xy'], tilt / A['max_tilt_deg']]
    dropped = (ratios > 1).any(axis=1)
    hold_phi, hold_step = hold_potential(progress, omega, touch, dropped, hits[-1])
    drop_indices = np.flatnonzero(dropped)
    drop_step = int(drop_indices[0]) if len(drop_indices) else None
    stop = min(T, hold_step if hold_step is not None else T, drop_step if drop_step is not None else T)
    did_drop = drop_step is not None and drop_step <= stop
    thumb_phi, thumb_step = thumb_potential(progress, touch, dropped,
                                          min(stop, hits[-1] if hits[-1] is not None else stop))
    # Rollback incomplete potentials at the terminal state. Completed ones latch.
    if hold_step is None or hold_step > stop:
        hold_phi[stop:] = 0
    if thumb_step is None or thumb_step > stop:
        thumb_phi[stop:] = 0
    components = {}
    def put(name, values):
        values = np.asarray(values, dtype=float).copy()
        values[0] = 0
        values[stop + 1:] = 0
        components[name] = values[1:]
    def delta(phi):
        return np.r_[0., np.diff(phi)]
    put('rotation', w['rotation'] * delta(best) / (A['turns'] * 2 * np.pi))
    timely = np.zeros(size)
    previous = 0
    for hit in hits:
        if hit is None:
            break
        if (hit - previous) * dt <= A['max_turn_s'] + 1e-9:
            timely[hit] = w['timely_turn']
        previous = hit
    put('timely_turn', timely)
    put('hold_shaping', w['hold_shaping'] * delta(hold_phi))
    put('thumb_shaping', w['thumb_shaping'] * delta(thumb_phi))
    for name, event in [('hold_bonus', hold_step), ('thumb_bonus', thumb_step)]:
        value = np.zeros(size)
        if event is not None:
            value[event] = w[name]
        put(name, value)
    # Cumulative active-contact quota per finger and revolution.
    qd = np.vstack([np.zeros(rec['qd'].shape[1]), rec['qd']])
    speed = np.stack([np.sqrt((qd[:, [i for i, j in enumerate(joints) if f'_{f}_' in j]] ** 2).mean(axis=1))
                      for f in FINGERS], axis=1)
    active = touch & (speed > A['active_joint_speed_rad_s']) & (delta(progress) > 0)[:, None]
    quotas = np.zeros((size, 3, 4))
    start = 0
    for k, hit in enumerate(hits):
        end = hit if hit is not None else T
        increment = np.zeros((size, 4))
        increment[start + 1:end + 1] = active[start + 1:end + 1, 1:] * dt
        quotas[:, k] = np.minimum(np.cumsum(increment, axis=0) / A['participation_min_s'], 1)
        if hit is None:
            break
        start = hit
    two_of_three = np.sort(quotas[:, :, 1:], axis=2)[:, :, -2:].mean(axis=(1, 2))
    each_two_turns = np.sort(quotas, axis=1)[:, -2:, :].mean(axis=(1, 2))
    participation = .5 * (two_of_three + each_two_turns)
    put('participation', w['participation'] * delta(participation))
    put('time', np.full(size, -w['time_rate'] * dt))
    put('instability', -w['instability_rate'] * dt * (np.clip((ratios - .5) / .5, 0, 1) ** 2).mean(axis=1))
    braking = np.zeros(size)
    if hits[-1] is not None:
        braking[hits[-1]:] = -w['braking_rate'] * dt * np.clip(omega[hits[-1]:] / np.radians(A['hold_max_omega_deg_s']) - 1, 0, 2) ** 2
    put('braking', braking)
    q = np.vstack([rec['q0'], rec['q']])
    violation = np.maximum(0, np.maximum(rec['q_lo'] - q, q - rec['q_hi']).max(axis=1))
    put('joint_limit', -w['joint_limit_rate'] * dt * np.clip(violation / A['joint_limit_tol_rad'], 0, 4) ** 2)
    penetration = np.r_[0., rec['physx_penetration']]
    put('penetration_proxy', -w['penetration_rate'] * dt * np.clip(penetration / (A['max_penetration_mm'] / 1000) - .5, 0, 2) ** 2)
    action = np.vstack([np.zeros(rec['action'].shape[1]), rec['action']])
    put('action_change', np.r_[0., -w['action_change_rate'] * dt * (np.diff(action, axis=0) ** 2).mean(axis=1)])
    drop = np.zeros(size)
    if did_drop:
        drop[stop] = -w['drop']
    put('drop', drop)
    complete_turns = all(h is not None and h <= stop for h in hits)
    proxy_success = bool(complete_turns and hold_step is not None and hold_step <= stop and not did_drop
                         and timely.sum() == 3 * w['timely_turn'] and thumb_step is not None and thumb_step <= stop
                         and participation[stop] >= 1 - 1e-10
                         and violation[:stop + 1].max() <= A['joint_limit_tol_rad']
                         and penetration[:stop + 1].max() <= A['max_penetration_mm'] / 1000
                         and not bool(rec.get('unsettled', False)))
    completion = np.zeros(size)
    if proxy_success:
        completion[stop] = w['completion_proxy']
    put('completion_proxy', completion)
    return components, dict(stop_step=stop, stop_seconds=stop * dt,
        terminal='drop' if did_drop else 'hold' if hold_step is not None and hold_step <= stop else 'timeout',
        hold_step=hold_step, thumb_step=thumb_step, proxy_success=proxy_success,
        turn_steps=hits, participation_fraction=float(participation[stop]))
