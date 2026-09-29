# Pen policy on AGX Thor

The pretrained pen policy runs with GPU PhysX and CUDA policy inference using
Isaac Sim 5.1.0 and Isaac Lab 2.3.2 in a separate Python 3.11 environment.
The existing `anymal` environment uses newer Isaac Lab with Newton; it is not
used or modified by this demo.

## Repeat the evaluation

From the repository root:

```bash
bash pen/run_thor_policy.sh
```

This runs the original 32 seeded, 12-second trials, scores their trajectories,
and renders trial 4 from two views. Output defaults to a timestamped directory
under `runs/`. An optional first argument selects a **new** output directory;
the second selects the CUDA device (default `0`). Inspect `score.json`: a
successful process exit does not mean every trial passed the acceptance gate.
The rendered videos are recordings. For live interaction, run
`bash cube/run_web.sh --port 8088` from the repository root, open
`http://localhost:8088`, and select **Pen**. The browser displays the current
PhysX states of all 256 environments in a 16×16 grid, with a distinct seed
per environment (starting seed through starting seed + 255). GPU PhysX and
batched CUDA inference run the entire simulation batch. Tile labels show seeds,
turn counts, and hold state. Click a tile or use the environment selector
to enlarge one simulation; All 256 restores the overview. The full grid targets
2 updates/second and enlarged views target 10; actual rates depend on load. Start, pause/resume, stop,
and top/close-up cameras are supported. Rendering runs in a separate process
and does not read recorded trajectories. Simulation time may run slower than
wall time. The UI shows every seed’s final pass/fail result and the batch pass count.
The command-line reproduction above retains its original 32-trial evaluation.

The launcher defaults to `runs/pen-runtime/`, with these components:

- `venv/`: Python 3.11.16, Isaac Sim 5.1.0, editable Isaac Lab v2.3.2,
  PyTorch 2.9.0+cu130, torchvision 0.24.0, torchaudio 2.9.0, RSL-RL 3.0.1.
- `render-venv/`: separate Python 3.12 environment with `pen/requirements.txt`
  (MuJoCo 3.3.5) for scoring and rendering.
- `sharpa/`: official Sharpa assets pinned to
  `0d447b6889e6d993758169dfc0aa75ee9f6ad8d7`.
- `requirements-freeze.txt` and `render-requirements-freeze.txt`: installed versions.

Set `PEN_RUNTIME` to relocate this layout, or override `ISAAC_PY`, `PY`, and
`SHARPA_ROOT` individually. Runtime directories and results are ignored by Git.

## Compatibility findings

- CUDA tensor computation passed on NVIDIA Thor.
- Isaac Lab's `isaaclab.python.headless.kit` launched and exited successfully.
  The original evaluator automatically selects this configuration.
- The general Isaac Sim application crashed in its RTX renderer on this host.
  Use the physics-only headless evaluator and separate MuJoCo rendering.
- Isaac Sim's ARM startup requires `/lib/aarch64-linux-gnu/libgomp.so.1` in
  `LD_PRELOAD`; the launcher supplies it. A CPU PyTorch build also required its
  bundled OpenMP library, but the final CUDA build does not have that extra file.
- Isaac Lab installation needed a `setuptools<81` build constraint for
  `flatdict==4.0.1`, plus `h5py` and `hydra-core` for environment imports.
- The evaluator can hang during application shutdown after saving results.
  The upstream `run_policy.sh` already handles this by stopping its subprocess
  after scoring/rendering; completed trajectories are preserved.

NVIDIA officially lists DGX Spark as the supported ARM platform for Isaac Sim
5.1; these results establish a local working headless configuration on Thor,
not general support for all Isaac Sim features.

## Validation on this host

The unchanged evaluator completed 32 parallel environments, 720 control steps
per environment (12 simulated seconds), seeds 41000000–41000031, with GPU
PhysX and CUDA inference. Results are in `runs/thor-pen-policy-32/score.json`.

- All 32 completed three turns within the timing requirement, with no drops.
- 29/32 passed the hold check; 10/32 passed the thumb-release check.
- 8/32 passed every check. All passed joint limits, penetration, finger
  participation, and the all-fingers check.
- The upstream designated demo, trial 4 / seed 41000004, passed every check:
  3.0448 net turns and 1.5591 mm maximum independently measured penetration.
- A preceding one-environment run of seed 41000004 completed 3.0485 turns and
  held without dropping, but failed thumb release. Thus batch size/runtime
  numerics affect results; these are observed outcomes, not guaranteed reruns.

No policy weights, rewards, physics parameters, or acceptance thresholds were
changed to achieve these results. This reproduces the spin-and-hold behavior,
with partial success under the full strict gate.
