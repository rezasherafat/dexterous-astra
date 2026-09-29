"""Training-only environment; the released inference environment stays unchanged."""
import torch
from pen_env import PenSpinCfg, PenSpinEnv
from spec import JOINTS
from reward import PenReward
from isaaclab.utils import configclass


@configclass
class PenTrainingCfg(PenSpinCfg):
    episode_length_s = 12.0
    obs_episode_s = 12.0
    target_turns = 3.0
    target_box = .5
    is_finite_horizon = True  # 12 seconds is part of the task, not an arbitrary truncation


class PenTrainingEnv(PenSpinEnv):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.reward_model = PenReward(self.num_envs, self.device, JOINTS)
        self.component_sums = {}
        self.completed_episodes = 0
        self.completed_holds = 0
        self.completed_drops = 0

    def _get_dones(self):
        self.extras.pop('log', None)  # Do not re-log the previous reset on later steps.
        # Updates progress/best used by the original 130-dimensional observation.
        super()._get_dones()
        pos, _, axis = self._pen_state()
        per, _ = self._finger_forces()
        timeout = self.episode_length_buf >= self.max_episode_length
        reward, terminal, components = self.reward_model.step(
            progress=self.progress, relative_pos=pos-self.ref_pos, axis=axis,
            omega_z=self.pen.data.root_ang_vel_w[:, 2], finger_force=per,
            qd=self._qd(), q=self._q(), q_lo=self.q_lo, q_hi=self.q_hi,
            action=self.action, penetration=self.physx_penetration(), timeout=timeout)
        self._training_reward = reward
        for key, values in components.items():
            if key not in self.component_sums:
                self.component_sums[key] = torch.zeros_like(values)
            self.component_sums[key] += values
        return terminal, timeout & ~terminal

    def _get_rewards(self):
        return self._training_reward

    def _reset_idx(self, env_ids):
        # Only the requested rows reset: never advance all physics to settle one row.
        if self.component_sums and len(env_ids):
            self.extras['log'] = {f'Reward/{k}': v[env_ids].mean().clone()
                                  for k, v in self.component_sums.items()}
            self.extras['log']['Episode/hold_fraction'] = self.reward_model.state['hold_done'][env_ids].mean().clone()
            self.completed_episodes += len(env_ids)
            self.completed_holds += int(self.reward_model.state['hold_done'][env_ids].sum())
            self.completed_drops += int((self.component_sums['drop'][env_ids] < 0).sum())
            for value in self.component_sums.values():
                value[env_ids] = 0
        super()._reset_idx(env_ids)
        self.reward_model.reset(env_ids)
