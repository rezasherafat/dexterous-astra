"""Train pen manipulation with Isaac Lab GPU physics and standard RSL-RL PPO."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'pen/sim'))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--num-envs', type=int, default=256)
parser.add_argument('--iterations', type=int, default=1000)
parser.add_argument('--steps', type=int, help='Rollout steps per environment per update')
parser.add_argument('--seed', type=int, default=41001000)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--warm-start', type=Path, help='Import released actor/normalizer only; new critic and optimizer')
parser.add_argument('--resume', type=Path, help='Resume a training checkpoint including optimizer')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.warm_start and args.resume:
    parser.error('Choose warm-start or resume, not both')
if args.num_envs < 1 or args.iterations < 1 or (args.steps is not None and args.steps < 1):
    parser.error('Environment, iteration and step counts must be positive')
if any(41100000 <= args.seed+i <= 41100099 for i in range(args.num_envs)):
    parser.error('Reserved final evaluation seeds cannot be used for training')
args.out.mkdir(parents=True, exist_ok=True)
if (args.out / 'config.json').exists():
    parser.error('Output directory already contains a run; use a fresh directory')
args.headless = True
app = AppLauncher(args).app

import numpy as np
import torch
from isaaclab.utils.math import quat_mul
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from env import PenTrainingCfg, PenTrainingEnv
from reward import DEFAULTS
from spec import JOINTS, PEN, FRICTION


def prepare_bank(env):
    """Build one independently seeded, randomized, settled reset state per row.

Physics stepping only happens before training. Asynchronous resets restore these
states without disturbing other environments. Mass/friction stay fixed per row.
"""
    n, dev = env.num_envs, env.device
    gens = [torch.Generator().manual_seed(args.seed+i) for i in range(n)]
    view = env.pen.root_physx_view
    masses, mats = view.get_masses(), view.get_material_properties()
    for i,g in enumerate(gens):
        u = torch.rand(2, generator=g)
        masses[i] = PEN['mass'] * (.85 + .30 * u[0])
        mats[i, :, :2] = FRICTION * (.9 + .2 * u[1])
    view.set_masses(masses, torch.arange(n))
    view.set_material_properties(mats, torch.arange(n))
    bank = {k: torch.zeros(n, width, device=dev) for k,width in [('q',22),('targets',22),('pos',3),('quat',4)]}
    pending = torch.ones(n, dtype=torch.bool, device=dev)
    attempts = torch.zeros(n, dtype=torch.long, device=dev)
    ids = torch.arange(n, device=dev)
    for _ in range(20):
        qs, ps, rots = [], [], []
        for g in gens:
            q = (env.grasp_q.cpu() + env.cfg.joint_noise*(2*torch.rand(22,generator=g)-1)).clamp(env.q_lo.cpu(),env.q_hi.cpu())
            pos = env.grasp_pen_pos.cpu().clone()
            pos[:2] += env.cfg.pen_xy_noise*(2*torch.rand(2,generator=g)-1)
            pos[2] += .002
            yaw = torch.deg2rad(torch.tensor(env.cfg.pen_yaw_noise_deg))*(2*torch.rand((),generator=g)-1)
            qs.append(q);ps.append(pos);rots.append(torch.tensor([torch.cos(yaw/2),0,0,torch.sin(yaw/2)]))
        q,pos = torch.stack(qs).to(dev),torch.stack(ps).to(dev)
        quat = quat_mul(torch.stack(rots).to(dev),env.grasp_pen_quat.expand(n,4))
        valid,sq,sp,squat,_ = env.settle(ids,q,pos,quat)
        take = pending & valid
        for key,value in [('q',sq),('targets',q),('pos',sp),('quat',squat)]:bank[key][take] = value[take]
        attempts += pending
        pending &= ~valid
        if not pending.any():break
    if pending.any():raise RuntimeError(f'{int(pending.sum())} starts did not settle; refusing invalid reset states')
    env.bank = bank
    np.savez_compressed(args.out/'reset_bank.npz', **{k:v.cpu().numpy() for k,v in bank.items()},
                        masses=masses.cpu().numpy(), materials=mats.cpu().numpy(), attempts=attempts.cpu().numpy())


class CheckedRunner(OnPolicyRunner):
    def log(self, locs, *a, **kw):
        values = {k:float(v) for k,v in locs['loss_dict'].items()}
        if not all(np.isfinite(v) for v in values.values()):
            raise FloatingPointError(f'Nonfinite PPO loss: {values}')
        with (Path(self.log_dir)/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(dict(iteration=locs['it'], losses=values,
                 collection_seconds=locs['collection_time'], update_seconds=locs['learn_time']))+'\n')
        super().log(locs,*a,**kw)


def main():
    cfg = PenTrainingCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    # Preserve the released controller's action scale.
    cfg.action_scale = json.loads((ROOT/'pen/checkpoints/env_cfg.json').read_text())['action_scale']
    env = PenTrainingEnv(cfg)
    prepare_bank(env)
    wrapped = RslRlVecEnvWrapper(env)
    train_cfg = json.loads(Path(__file__).with_name('config.json').read_text())
    if args.steps:train_cfg['num_steps_per_env'] = args.steps
    if args.num_envs*train_cfg['num_steps_per_env'] < train_cfg['algorithm']['num_mini_batches']:
        raise ValueError('Rollout must contain at least one sample per minibatch')
    provenance = dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                      ppo=train_cfg,reward=DEFAULTS,action_scale=cfg.action_scale,target_box=cfg.target_box,
                      note='New reward v1; hold is not strict success; no proxy completion bonus. Fixed randomized reset bank.')
    (args.out/'config.json').write_text(json.dumps(provenance,indent=2)+'\n')
    runner = CheckedRunner(wrapped, train_cfg, log_dir=str(args.out),device=env.device)
    runner.add_git_repo_to_log(__file__)
    if args.resume:
        runner.load(str(args.resume))
    elif args.warm_start:
        ck = torch.load(args.warm_start,map_location=env.device,weights_only=False)['model_state_dict']
        current = runner.alg.policy.state_dict()
        selected = {k:v for k,v in ck.items() if k.startswith(('actor.', 'actor_obs_normalizer.'))}
        if not selected or any(k not in current or current[k].shape != v.shape for k,v in selected.items()):
            raise ValueError('Warm-start actor architecture does not match')
        current.update(selected)
        runner.alg.policy.load_state_dict(current)
    before = {k:v.detach().clone() for k,v in runner.alg.policy.named_parameters()}
    start = time.perf_counter()
    runner.learn(args.iterations)
    if runner.writer is not None:
        runner.writer.flush()
        runner.writer.close()
    changes = {k:float((v-before[k]).abs().max()) for k,v in runner.alg.policy.named_parameters()}
    if not all(torch.isfinite(v).all() for v in runner.alg.policy.parameters()):
        raise FloatingPointError('Nonfinite trained parameters')
    summary = dict(device=str(env.device),num_envs=env.num_envs,iterations=args.iterations,
                   wall_seconds=time.perf_counter()-start,completed_episodes=env.completed_episodes,
                   completed_holds=env.completed_holds,completed_drops=env.completed_drops,
                   actor_max_parameter_change=max(v for k,v in changes.items() if k.startswith('actor.')),
                   critic_max_parameter_change=max(v for k,v in changes.items() if k.startswith('critic.')))
    # Export the actor in the released evaluator's scalar-std checkpoint format.
    exported = {k:v.detach().cpu() for k,v in runner.alg.policy.state_dict().items()}
    exported['std'] = exported.pop('log_std').exp()
    torch.save(dict(model_state_dict=exported,iter=runner.current_learning_iteration),args.out/'policy.pt')
    (args.out/'env_cfg.json').write_text(json.dumps(dict(action_scale=cfg.action_scale),indent=2)+'\n')
    (args.out/'state.json').write_text(json.dumps(dict(history=[dict(iter=0,stage=3,target_box=cfg.target_box)]),indent=2)+'\n')
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('TRAINING_COMPLETE', json.dumps(summary),flush=True)
    env.close()

if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush();sys.stderr.flush()
        os._exit(1)
    # Isaac 5.1 can hang during app.close() on Thor. Checkpoints and logs are
    # already saved and the environment closed; terminate the standalone process.
    sys.stdout.flush();sys.stderr.flush()
    os._exit(0)
