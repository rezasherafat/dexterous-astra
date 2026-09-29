# Dexterous Astra

![Zero-shot and trained pen spinning and Rubik's cube](assets/cover.jpg)

> [!NOTE]
> This repository contains the final artifacts from my Astra experiments, rather than a task-specific harness or benchmark provided to Astra. This repo also provides the zero-shot prompt used for pen spinning for reproducibility.

| Task         | Robot, simulator            | Zero-shot                               | Trained                                          |
| ------------ | --------------------------- | --------------------------------------- | ------------------------------------------------ |
| Pen spinning | Sharpa Wave hand, Isaac Lab | sampled finger-keyframe planner         | PPO policy: 3 turns and a hold                   |
| Rubik's cube | Wuji hands, MuJoCo          | two-hand scripted controller: two moves | one-hand learned turns and rolls: solves `U F L` |

Each task has `sim/` (environment), `zero_shot/` (script), `policy/` (runs the checkpoints) and `checkpoints/`.

```bash
ISAAC_PY=... PY=... pen/run_zero_shot.sh runs/pen-zero
ISAAC_PY=... PY=... pen/run_policy.sh    runs/pen-policy
PY=... cube/run_zero_shot.sh runs/cube-zero
PY=... cube/run_policy.sh    runs/cube-policy
```

For cube policy inference on a GPU, pass `cuda` (or `cuda:0`) as the second argument to `cube/run_policy.sh`, e.g. `PY=... cube/run_policy.sh runs/cube-cuda cuda`. The default is `cpu`; physics uses CPU unless the independent physics option below is selected. `report.json` records the inference device, call count, and total inference time including CPU/GPU transfers. Device changes can affect trajectories through floating-point differences, so check `passed` in the report.

`ISAAC_PY`: Python 3.11 with Isaac Lab 2.3.2 and `rsl-rl-lib==3.0.1`. `PY`: `pen/requirements.txt` or `cube/requirements.txt`.

For the installed Thor pen runtime, run `bash pen/run_thor_policy.sh`. See [pen/THOR.md](pen/THOR.md) for runtime details, evaluation results, and headless-rendering limitations.

Zero-shot prompt (pen spinning):

```
There's a pen simulator here. A robot hand lies palm-up with a pen on its fingers.
Spin it one full turn without dropping it, ending steady. No training.
```

## Live cube viewer

Run `bash cube/run_web.sh` and open http://127.0.0.1:8088. The script uses the repository's `.venv`; set `PY` to use another Python environment. It needs the cube runtime (`mujoco`, `numpy`, `torch`, and `pillow`); the web server itself uses the Python standard library.

The browser starts a fresh `U F L` simulation, with CPU/CUDA policy selection, a seed, pause/resume, stop, and overview/close-up cameras. Frames are rendered directly from the running MuJoCo state at up to 10 Hz; no trajectory or video is replayed. Pause freezes physics, and rendering uses separate simulation data. Final audit checks and run artifacts appear in the page. Live rendering adds overhead, so simulated time may advance slower than real time.

For the cube, CPU / seed 5005 is the verified baseline. CUDA is experimental: it executes successfully but previously timed out during the second face turn. Artifacts for each UI run are stored under `runs/web/`. Stopping early retains the last live frame and logs, but does not produce a completed solve report. Choose **Pen** or **Cube** in the Simulation selector. Pen uses the installed `runs/pen-runtime` GPU PhysX/CUDA environment and shows all 256 independently seeded environments in a labeled 16×16 live grid; startup takes several seconds. Pose snapshots from the running evaluator are rendered independently, with no trajectory/video replay. Pause stops physics for the whole batch. Click a tile or use the environment selector for an enlarged live view; select All 256 to return to the grid. Final checks show per-seed results and the batch pass count. The starting seed selects 256 consecutive seeds; all pen physics and policy inference run on GPU. Zero-shot demos are not wired into the viewer.

Use `bash cube/run_web.sh --port 8088` to choose a port. The default listener is `0.0.0.0` (all IPv4 interfaces); from another machine, open `http://THOR_IP:8088`. Use `--host 127.0.0.1` for localhost-only access, optionally with `ssh -L 8088:127.0.0.1:8088 USER@THOR`. The viewer provides no authentication.

## GPU physics (experimental)

Install `mujoco-warp==3.13.0` and `warp-lang==1.17.0` in the runtime environment. Select **GPU · MuJoCo-Warp** under **Physics engine** in the live viewer, or run:

```bash
PY=... cube/run_policy.sh runs/cube-warp cpu warp
# Without video rendering:
PY=... # path to the runtime Python
"$PY" cube/policy/solve.py --out runs/cube-warp --seed 5005 --scramble "U F L" --physics warp --device cpu
```

`--physics warp` runs contact generation, constraint solving, integration, bounded motor control, and passive cube forces on CUDA. The existing IK/planner and observation processing stay on CPU; state is read back at controller boundaries for them and for live rendering. Policy inference is selected independently with `--device cpu` or `--device cuda`. CPU remains the default, and Warp errors never fall back silently to CPU stepping.

The first run compiles CUDA kernels and can take several minutes. Contact/constraint capacity is checked; overflow stops the run. Changing collision masks rebuilds GPU candidate-pair tables. The pinned Warp version uses a 256-thread convex-collision launch configuration to avoid a kernel-symbol issue when querying occupancy between graph captures.

GPU and CPU trajectories are not expected to match: MuJoCo-Warp uses float32 and some capsule collision pairs have fewer contact points. Existing checkpoints were developed against CPU MuJoCo. A single interactive scene may be slower on GPU; this backend is an initial step toward batched GPU simulation, not a guarantee of faster or successful solves. `report.json` includes the simulation backend, device, setup time, and physics step count.

GPU force/audit checks: `.venv/bin/python -m unittest discover -s cube/tests -v`.

Validated on AGX Thor with **Warp physics + CPU inference**, seed 5005, scramble `U F L`: all five maneuvers and seven audit checks passed (54,030 CUDA physics steps). The observed rollout took 369 seconds for 54 seconds of simulated motion, plus 133 seconds of backend setup; other validation work overlapped portions of that run, so this is not an isolated speed benchmark. The CPU baseline remained bit-for-bit unchanged.

## Pen PPO training

A new Isaac Lab / RSL-RL training integration is available on Thor:

```bash
bash pen/run_thor_training.sh --num-envs 256 --iterations 1000 \
  --warm-start pen/checkpoints/best_policy.pt --out runs/pen-ppo-001
```

This uses a new task reward developed from the offline rollout audit, not the
original authors' reward. It supports training from scratch, actor warm-start,
checkpoint resume, component logging, and export to the unchanged evaluator.
See [pen/training/README.md](pen/training/README.md) for reward definitions,
configuration, limitations, and validation commands.
