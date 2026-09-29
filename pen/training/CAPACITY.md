# Thor PPO capacity checks — 2026-09-28

One NVIDIA Thor GPU (`cuda:0`), approximately 122 GiB total shared host memory. These are short integration/throughput probes, not an exhaustive hardware-limit or policy-quality study.

| Environments | PPO updates | Transitions/s, including PPO update | Minimum observed host memory available |
|---:|---:|---:|---:|
| 4,096 | 3 | 17,161 | Not sampled |
| 16,384 | 2 | 22,642 | 78.5 GiB |
| 32,768 | 1 | 24,768 | 52.4 GiB |

All probes used 64 rollout steps per environment and resumed the one-environment model checkpoint. Losses and resulting parameters were finite. The 16,384 and 32,768 probes also verified budget-triggered checkpoint/export saving. No simulator errors or contact-buffer overflow were reported in the 32,768 log.

Selected **32,768** for the subsequent 30-minute training run: the largest validated power-of-two batch, also the fastest measured probe. 65,536 was not attempted: extrapolating the observed shared-memory growth suggested insufficient headroom for the host and PPO allocations. This is a practical selection, not proof of the absolute maximum.

Initialization and grasp settling are excluded from throughput and took several minutes at the larger sizes. Per-row independently seeded grasp, mass and friction randomization was retained. Settling checked every accepted start. The validated bank can be reused with the same seed/count, without repeating settling. Checkpoint resume and cached bank reuse were separately verified with a one-environment timed test.

The main run is `runs/pen-ppo-32768-001`, using the saved 32,768-environment bank and continuing from `runs/pen-capacity-32768/model_20.pt` (itself resumed from the one-environment checkpoint). Physics, rewards, policy and PPO updates use CUDA. Checkpoints save every five updates and at the 30-minute budget boundary. Host synchronization remains in reset dispatch, logging and contact extraction.

Local details: `runs/pen-capacity-results.json`, `runs/pen-capacity-{4096,16384,32768}/`, and the corresponding resource JSONL files. Shared-memory figures include other host activity; do not add process RSS to CUDA allocations. The main run is monitored with a 16 GiB available-memory guard.
