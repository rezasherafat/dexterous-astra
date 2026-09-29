"""Supervise the live pen evaluator, independent renderer, and final scoring."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=41000000)
    args = parser.parse_args()
    runtime = ROOT / 'runs/pen-runtime'
    env = os.environ.copy()
    env.update(OMNI_KIT_ACCEPT_EULA='YES', LD_PRELOAD='/lib/aarch64-linux-gnu/libgomp.so.1',
               SHARPA_ROOT=str(runtime / 'sharpa'), OMP_NUM_THREADS='4', MUJOCO_GL='egl')
    render_py = str(runtime / 'render-venv/bin/python')
    renderer = subprocess.Popen([render_py, str(ROOT / 'pen/web/render_live.py'), str(args.out / 'live')], env=env)
    evaluator = subprocess.Popen([str(runtime / 'venv/bin/python'), '-u', str(ROOT / 'pen/policy/evaluate.py'),
        '--ckpt', str(ROOT / 'pen/checkpoints/best_policy.pt'), '--run_cfg', str(ROOT / 'pen/checkpoints/env_cfg.json'),
        '--out', str(args.out), '--seed0', str(args.seed), '--trials', '256', '--seconds', '12',
        '--device', 'cuda:0', '--live-dir', str(args.out / 'live')], env=env)
    try:
        while not (args.out / 'meta.json').exists():
            if renderer.poll() is not None:
                raise RuntimeError('Live renderer exited; see solve.log.')
            if evaluator.poll() is not None:
                raise RuntimeError('Pen evaluator exited before saving results; see solve.log.')
            time.sleep(.2)
        # Allow the final pose to be displayed, then stop Isaac's known shutdown hang.
        time.sleep(.5)
        if evaluator.poll() is None:
            evaluator.terminate()
        subprocess.run([render_py, str(ROOT / 'pen/policy/score.py'), str(args.out), '--workers', '4'], env=env, check=True)
        score = json.loads((args.out / 'score.json').read_text())
        checks = {key: all(t[key] for t in score['trials'])
                  for key in score['trials'][0] if key.endswith('_ok')}
        checks['no_drop'] = not any(t['drop'] for t in score['trials'])
        report = dict(passed=score['summary']['passed'] == 256, completed=True,
                      checks=checks, trials=score['trials'], summary=score['summary'])
        (args.out / 'report.json').write_text(json.dumps(report, indent=2))
    finally:
        for process in (evaluator, renderer):
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == '__main__':
    main()
