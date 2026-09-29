# Pen PPO training on Thor

This is a new training integration, not the authors' original reward or trainer.
Isaac Lab runs GPU physics; batched Torch computes the task reward on CUDA;
RSL-RL 3.0.1 handles trajectory storage, GAE, PPO losses and optimization.
The original evaluator and frozen acceptance gate are unchanged.

## Run

Use the isolated runtime described in `pen/THOR.md`. Additional dependency:
`tensorboard==2.20.0` (installed in that runtime on this host).

From the repository root, fine-tune the released actor:

```bash
bash pen/run_thor_training.sh \
  --num-envs 256 --iterations 1000 \
  --warm-start pen/checkpoints/best_policy.pt \
  --out runs/pen-ppo-001
```

Omit `--warm-start` to train from scratch. Warm-start imports only the actor and
its observation normalizer: critic, exploration variance and optimizer start new.
It does not imply that the new reward matches the checkpoint's original reward.
The policy has the same 130 observations and 22 actions as the released model.

A short integration check:

```bash
bash pen/run_thor_training.sh --num-envs 16 --iterations 2 --steps 32 \
  --out runs/pen-ppo-smoke
```

Resume a **training** checkpoint, writing into a fresh directory:

```bash
bash pen/run_thor_training.sh --num-envs 256 --iterations 1000 \
  --resume runs/pen-ppo-001/model_999.pt --out runs/pen-ppo-002
```

Keep the same seed/environment/configuration when resuming. Model and optimizer
resume; physical trajectories and random-generator positions do not. Each launch
rebuilds the independently seeded settled reset bank. The seed defaults to
41001000; reserved final-evaluation seeds 41100000–41100099 are rejected.

Each environment gets its own randomized grasp, mass and friction before
training. Failed grasps are redrawn, with a hard failure after 20 attempts.
During training, asynchronous resets restore that environment's settled state;
they never step all other environments to settle a single one. This initial
implementation reuses one reset state and fixed physical parameters per row,
which limits reset diversity; it is not full per-episode domain randomization.

## Reward v1

The offline v0 audit found reward loopholes. V1 deliberately changes:

- Hold: +10 once the gate-compatible hold completes; no partial shaping.
- Thumb release: +8 once a qualifying maximal free interval ends, by recontact
  or reaching turn three; no partial shaping. Failed cycles earn zero credit.
- Participation: capped quotas still total +6, but require new high-water
  rotation at >=0.2 rad/s as well as finger contact and joint motion. Repeated
  rocking through old progress and near-zero forward jitter earn no new credit.
- No +40 proxy completion bonus. PhysX penetration is a penalty measurement,
  **not** a substitute for independent penetration evaluation. Removing this
  bonus avoids the observed false-positive payout, but does not solve penetration
  accuracy or guarantee physically valid learned behavior.

Other terms retain v0 weights: up to +20 new rotation, +3 per timely revolution,
−40 drop, and time, instability, braking, joint-limit, PhysX penetration and
executed-action-change penalties. `reward.py` contains all weights and equations.
A first drop or first valid hold ends the episode; the 12-second task deadline is
a true finite-horizon end with no value bootstrap. PPO rollout boundaries alone
bootstrap normally. A completed hold is **not** strict task success.

`config.json` uses gamma=0.995, lambda=0.95, 64 steps per environment, four PPO
epochs/four minibatches, KL-adaptive learning rate initially 3e-4, and entropy
coefficient 0.001. Value clipping is disabled because the old +/-0.2 absolute
value clip is poorly matched to the new reward scale. These are initial settings,
not tuned results. PPO's Gaussian uses log standard deviation for positivity;
executed joint increments remain clipped by the original environment.

Reward state (quotas and event history) is not fully observed by the feedforward
actor/critic. The original observation does expose rotation remaining, holding
phase and time. Adding history observations or a recurrent policy is a future
option if this partial observability limits learning.

Reward arithmetic stays on CUDA. Simulator reset dispatch, logging and the
existing PhysX contact extraction have host synchronization; this is not an
entirely asynchronous GPU pipeline. The live web process is independent.

## Outputs and evaluation

- `config.json`: run arguments, PPO configuration and reward weights.
- `reset_bank.npz`: actual settled reset states and randomized physical properties.
- `metrics.jsonl` and TensorBoard events: losses and component episode returns.
- `model_*.pt`: full RSL-RL checkpoints, including optimizer, for resuming.
- `policy.pt`, `env_cfg.json`, `state.json`: export compatible with the existing
  `pen/policy/evaluate.py` interface (use stage 3). The export is the final actor,
  not a selected best policy.
- `summary.json`: completion, episode counts and actor/critic parameter changes.

Use the existing evaluator with `--ckpt RUN/policy.pt --run_cfg RUN/env_cfg.json`
and fresh **development** seeds, then `pen/policy/score.py` for independent strict
metrics. Training returns or hold fractions must not be reported as gate success.
The standalone launcher exits explicitly after saving/closing the environment
because Isaac Sim 5.1 can hang in application shutdown on this Thor runtime.

## Verification

```bash
runs/pen-runtime/venv/bin/python -m unittest discover -s pen/tests -v
runs/pen-runtime/venv/bin/python pen/training/replay.py \
  runs/pen-scaling/256 --out runs/pen-reward-v1 --device cuda:0
```

Tests cover CPU/CUDA agreement, unchanged terms against the NumPy reference,
event budgets, termination, isolated row resets, and two known exploit patterns.
Replay of all 256 existing development trajectories matches the frozen gate's
hold and thumb labels exactly. Replay is a reward audit, not a training result.

See [VALIDATION.md](VALIDATION.md) for the recorded Thor integration results,
including the unsuccessful strict evaluation after the short training smoke run.

## Weights & Biases

Authenticate interactively on the host (do not put API keys in commands or Git):

```bash
runs/pen-runtime/venv/bin/wandb login
```

Enable W&B for a training run:

```bash
bash pen/run_thor_training.sh --num-envs 1 --iterations 20 \
  --warm-start pen/checkpoints/best_policy.pt --out runs/pen-ppo-wandb \
  --logger wandb --wandb-project dexterous-astra --wandb-entity YOUR_ENTITY
```

The entity flag is optional if the account's default entity is appropriate.
`--wandb-mode offline` explicitly saves W&B data locally without uploading.
Online failures are not silently converted to offline runs. The logger retains
TensorBoard files and logs actor/value losses, entropy, learning rate, action
noise, throughput, reward components, episode return/length, and cumulative
hold/drop/timeout counts. A hold is not strict success. Configuration, checkpoints
and code diffs are saved through the standard RSL-RL W&B integration. Run ID and
URL are written to `wandb-run.json`, and uploads are finished before process exit.

A completed TensorBoard run can be imported without rerunning physics:

```bash
runs/pen-runtime/venv/bin/python pen/training/log_existing.py \
  runs/pen-training-1env-001 --project dexterous-astra --entity YOUR_ENTITY
```

The importer uploads scalar history and configuration/summary, not policy files.
