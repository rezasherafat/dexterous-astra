# Thor integration validation — 2026-09-28

These checks establish that the training integration executes. They do **not**
establish a learned improvement or a successful trained policy.

| Check | Result |
|---|---|
| Offline + streaming reward unit tests | 16 passed, including CUDA/CPU agreement |
| GPU reward replay of 256 saved development trajectories | Zero hold or thumb label disagreements with frozen gate |
| Scratch PPO smoke: 16 envs × 32 steps × 2 updates | Finite losses; actor and critic updated; 33 drop/reset episodes |
| Checkpoint resume: 16 envs × 32 steps × 1 update | Loaded optimizer/model; finite updates; 13 drop/reset episodes |
| Warm-start PPO: 256 envs × 64 steps × 12 updates | 196,608 transitions; finite losses and parameters |
| Warm-start episode ends | 259 total: 117 holds, 3 drops, 139 task deadlines |
| Training-loop wall time | 78.55 seconds, excluding scene creation and reset-bank preparation |
| Maximum actor / critic parameter change | 0.003104 / 0.011072 |
| Checkpoint export | Strictly loads into released evaluator architecture; finite inference actions |
| Exported-policy independent evaluation | 0/16 strict passes on seeds 41002000–41002015 |

The strict evaluation completed with zero unsettled starts. All 16 completed the
three turns within the per-turn deadlines; 13/16 held, 14/16 passed penetration,
15/16 passed the all-fingers check, and **0/16 passed thumb release**. There were no
drops. These fresh development seeds were not compared with a matched baseline,
so no improvement or regression is inferred. No reserved final seeds were used.

The final logging changes (clear stale episode metrics and flush TensorBoard
before standalone process exit) were checked with another successful resumed
16-environment run. All 16 unit tests also passed on the final reward code.

Local artifacts (ignored by Git):

- `runs/pen-training-tests-final.log`
- `runs/pen-reward-v1/{summary.json,components.npz}`
- `runs/pen-training-smoke-16/`
- `runs/pen-training-resume-16/`
- `runs/pen-training-smoke-256/`
- `runs/pen-training-final-check/`
- `runs/pen-training-eval-16/{meta.json,score.json,trajectories.npz}`

The 256-environment loop averaged approximately 2,503 transitions/s including PPO
updates. This was a smoke test on a shared host, not an isolated capacity benchmark.
