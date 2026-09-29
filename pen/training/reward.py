"""Streaming batched pen reward v1. Pure Torch; no simulator imports or CPU copies.

Hold/thumb rewards are one-shot events, not partial potentials. The independent
acceptance gate remains the success oracle; there is no proxy completion bonus.
"""
import json
import math
from pathlib import Path
import torch

ACCEPTANCE = json.loads((Path(__file__).resolve().parents[1] / 'policy/acceptance.json').read_text())
DEFAULTS = dict(rotation=20., timely_turn=3., hold=10., thumb=8., participation=6.,
                drop=40., time_rate=.1, instability_rate=.2, braking_rate=.2,
                joint_limit_rate=1., penetration_rate=.5, action_change_rate=.1,
                participation_min_speed=.2)  # rad/s pen progress, additional training-only condition


class PenReward:
    def __init__(self, n, device, joints, weights=None):
        self.n, self.device = n, device
        self.w = dict(DEFAULTS, **(weights or {}))
        self.finger_indices = [[i for i, name in enumerate(joints) if f'_{f}_' in name]
                               for f in ('thumb', 'index', 'middle', 'ring', 'pinky')]
        self.state = {}
        for key in ('step', 'progress', 'best', 'last_turn_step', 'turns', 'third_step',
                    'thumb_len', 'thumb_start', 'thumb_support', 'thumb_done', 'hold_done',
                    'participation', 'finished'):
            self.state[key] = torch.zeros(n, device=device)
        self.state['quota'] = torch.zeros(n, 3, 4, device=device)
        self.state['previous_action'] = torch.zeros(n, 22, device=device)
        # Up to 61 eligible hold starts, each accumulating its own causal prefix.
        for key in ('hold_len', 'hold_low', 'hold_high', 'hold_support', 'hold_invalid'):
            self.state[key] = torch.zeros(n, 61, device=device)
        self.slots = torch.arange(61, device=device)[None, :]

    def reset(self, ids):
        for value in self.state.values():
            value[ids] = 0

    @torch.no_grad()
    def step(self, *, progress, relative_pos, axis, omega_z, finger_force, qd,
             q, q_lo, q_hi, action, penetration, timeout):
        s, w, a = self.state, self.w, ACCEPTANCE
        dt = 1 / a['control_hz']
        alive = s['finished'] == 0
        s['step'] += alive
        old_turns = s['turns'].clone()
        best = progress.clamp(0, 6 * math.pi).maximum(s['best'])
        turns = torch.floor(best / (2 * math.pi)).clamp(max=3)
        crossed = turns > old_turns
        timely = crossed & ((s['step'] - s['last_turn_step']) * dt <= a['max_turn_s'])
        s['last_turn_step'] = torch.where(crossed, s['step'], s['last_turn_step'])
        third = crossed & (turns == 3)
        s['third_step'] = torch.where(third, s['step'], s['third_step'])
        ratios = torch.stack(((-relative_pos[:, 2]).clamp(min=0) / a['drop_z'],
                              relative_pos[:, :2].norm(dim=-1) / a['drop_xy'],
                              torch.asin(axis[:, 2].abs().clamp(max=1)) / math.radians(a['max_tilt_deg'])), -1)
        drop = (ratios > 1).any(-1)
        touch = finger_force > a['contact_n']
        # Finger quota credits require new high-water progress and >=0.2 rad/s.
        speeds = torch.stack([qd[:, ids].square().mean(-1).sqrt() for ids in self.finger_indices], -1)
        active = touch[:, 1:] & (speeds[:, 1:] > a['active_joint_speed_rad_s'])
        useful = ((best - s['best']) >= w['participation_min_speed'] * dt) & (old_turns < 3)
        rev = torch.nn.functional.one_hot(old_turns.long().clamp(max=2), 3).to(q.dtype)
        s['quota'] += rev[:, :, None] * active[:, None, :] * (useful & alive)[:, None, None] * dt
        quotas = (s['quota'] / a['participation_min_s']).clamp(max=1)
        participation = .5 * (quotas[:, :, 1:].topk(2, dim=2).values.mean((1, 2))
                              + quotas.topk(2, dim=1).values.mean((1, 2)))
        # Maximal thumb-free interval commits at recontact or turn three.
        eligible = (old_turns < 3) & (s['thumb_done'] == 0)
        free = ~touch[:, 0] & eligible
        starts = free & (s['thumb_len'] == 0)
        s['thumb_start'] = torch.where(starts, progress, s['thumb_start'])
        s['thumb_len'] += free
        s['thumb_support'] += free & (touch[:, 1:].sum(-1) >= a['thumb_free_min_other_fingers'])
        # Recontact uses previous heading: recontact sample is outside the free interval.
        end_progress = torch.where(free, progress, s['progress'])
        thumb_end = eligible & ((touch[:, 0] & (s['thumb_len'] > 0)) | third)
        thumb = (thumb_end & (s['thumb_len'] * dt >= a['thumb_free_min_s'])
                 & (end_progress - s['thumb_start'] >= math.radians(a['thumb_free_min_rotation_deg']))
                 & (s['thumb_support'] >= a['thumb_free_support_frac'] * s['thumb_len']) & ~drop)
        s['thumb_done'] = torch.maximum(s['thumb_done'], thumb.float())
        clear = touch[:, 0] | third
        s['thumb_len'] = torch.where(clear, 0., s['thumb_len'])
        s['thumb_support'] = torch.where(clear, 0., s['thumb_support'])
        # Gate-compatible hold windows: start within 1s of third turn; 60 samples.
        age = s['step'] - s['third_step']
        window = (s['third_step'] > 0)[:, None] & (age[:, None] >= self.slots) & (age[:, None] < self.slots + 60)
        first = window & (s['hold_len'] == 0)
        s['hold_low'] = torch.where(first, progress[:, None], s['hold_low'])
        s['hold_high'] = torch.where(first, progress[:, None], s['hold_high'])
        s['hold_low'] = torch.where(window, torch.minimum(s['hold_low'], progress[:, None]), s['hold_low'])
        s['hold_high'] = torch.where(window, torch.maximum(s['hold_high'], progress[:, None]), s['hold_high'])
        s['hold_len'] += window
        s['hold_support'] += window & touch.any(-1)[:, None]
        invalid = drop | (omega_z.abs() >= math.radians(a['hold_max_omega_deg_s']))
        s['hold_invalid'] += window & invalid[:, None]
        hold = ((s['hold_len'] == 60) & (s['hold_invalid'] == 0)
                & (s['hold_high'] - s['hold_low'] < math.radians(a['hold_max_heading_change_deg']))
                & (s['hold_support'] >= 60 * a['hold_min_support_frac'])).any(-1)
        hold &= (s['hold_done'] == 0) & ~drop
        s['hold_done'] = torch.maximum(s['hold_done'], hold.float())
        violation = torch.maximum(q_lo - q, q - q_hi).amax(-1).clamp(min=0)
        components = dict(
            rotation=w['rotation'] * (best - s['best']) / (6 * math.pi),
            timely_turn=w['timely_turn'] * timely,
            hold=w['hold'] * hold, thumb=w['thumb'] * thumb,
            participation=w['participation'] * (participation - s['participation']),
            drop=-w['drop'] * drop,
            time=torch.full_like(progress, -w['time_rate'] * dt),
            instability=-w['instability_rate'] * dt * ((ratios - .5) / .5).clamp(0, 1).square().mean(-1),
            braking=-w['braking_rate'] * dt * (turns == 3) * (omega_z.abs() / math.radians(30) - 1).clamp(0, 2).square(),
            joint_limit=-w['joint_limit_rate'] * dt * (violation / a['joint_limit_tol_rad']).clamp(0, 4).square(),
            penetration_proxy=-w['penetration_rate'] * dt * (penetration / .002 - .5).clamp(0, 2).square(),
            action_change=-w['action_change_rate'] * dt * (action - s['previous_action']).square().mean(-1))
        components = {k: torch.where(alive, v, 0.) for k, v in components.items()}
        done = drop | hold | timeout
        s['finished'] = torch.maximum(s['finished'], done.float())
        s['progress'], s['best'], s['turns'] = progress.clone(), best, turns
        s['participation'], s['previous_action'] = participation, action.clone()
        return sum(components.values()), drop | hold, components
