"""Local HTTP controls and live JPEG transport for cube and pen simulators."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
import uuid

ROOT = Path(__file__).resolve().parents[2]
WEB = Path(__file__).resolve().parent
PEN_POLICIES = {
    'released': ('Released actor', ROOT / 'pen/checkpoints/best_policy.pt', ROOT / 'pen/checkpoints/env_cfg.json'),
    'trained-1env': ('Fine-tuned actor · 1 env / 20 updates', ROOT / 'runs/pen-training-1env-001/policy.pt', ROOT / 'runs/pen-training-1env-001/env_cfg.json'),
}


class Runner:
    def __init__(self, output):
        self.output = output
        self.lock = threading.RLock()
        self.process = None
        self.directory = None
        self.device = 'cpu'
        self.physics = 'cpu'
        self.task = 'cube'
        self.seed = 5005
        self.trials = 256
        self.policy = 'released'
        self.control = dict(paused=False, camera='both')
        self.stopped = False

    def save_controls(self):
        path = self.directory / 'live' / 'controls.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.control))
        tmp.replace(path)

    def state(self):
        with self.lock:
            if self.directory is None:
                return dict(status='idle', frame=0)
            data = dict(status='starting', frame=0, sim_time=0)
            try:
                data.update(json.loads((self.directory / 'live/state.json').read_text()))
            except FileNotFoundError:
                pass
            data.update(run=self.directory.name, device=self.device, physics=self.physics, task=self.task, seed=self.seed,
                        output=str(self.directory), trials=self.trials, policy=self.policy,
                        policy_label=PEN_POLICIES[self.policy][0], checkpoint=str(PEN_POLICIES[self.policy][1]))
            data['pickup_done'] = (self.directory / 'plan.json').exists()
            data['maneuvers'] = []
            for path in sorted(self.directory.glob('maneuver-*.json')):
                try:
                    data['maneuvers'].append(json.loads(path.read_text())['report'])
                except json.JSONDecodeError:
                    pass
            if self.task == 'pen' and self.process.poll() is None and (self.directory / 'meta.json').exists():
                data.update(status='scoring', phase=f'Scoring all {self.trials} completed trajectories')
            if self.process.poll() is not None:
                data['status'] = 'stopped' if self.stopped else 'failed'
                report_path = self.directory / 'report.json'
                if report_path.exists() and not self.stopped:
                    report = json.loads(report_path.read_text())
                    data.update(status='completed' if report.get('completed', report['passed']) else 'failed', report=report)
                elif not self.stopped:
                    data['error'] = (self.directory / 'solve.log').read_text()[-3000:]
            return data

    def start(self, device, seed, physics="cpu", task="cube", trials=256, policy="released"):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                raise ValueError('A simulation is already running.')
            if task not in ('cube', 'pen'):
                raise ValueError('Unknown simulation task.')
            if type(trials) is not int or trials not in (2, 16, 64, 128, 256):
                raise ValueError('Choose 2, 16, 64, 128, or 256 environments.')
            if policy not in PEN_POLICIES:
                raise ValueError('Unknown pen policy.')
            if task == 'pen' and not all(p.is_file() for p in PEN_POLICIES[policy][1:]):
                raise ValueError('Selected policy files are unavailable on this host.')
            if type(seed) is not int or not 0 <= seed < 2**32:
                raise ValueError('Seed must be an integer from 0 to 4294967295.')
            if task == 'pen':
                device, physics = 'cuda', 'physx'
                if seed > 2**32 - trials:
                    raise ValueError(f'Pen seed must leave room for {trials} trials.')
            if physics not in ('cpu', 'warp', 'physx') or (task == 'cube' and physics == 'physx'):
                raise ValueError('Physics must be cpu or warp.')
            if device not in ('cpu', 'cuda'):
                raise ValueError('Device must be cpu or cuda.')
            if type(seed) is not int or not 0 <= seed < 2**32:
                raise ValueError('Seed must be an integer from 0 to 4294967295.')
            self.directory = self.output / (time.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6])
            (self.directory / 'live').mkdir(parents=True)
            self.device, self.seed, self.stopped = device, seed, False
            self.physics, self.task = physics, task
            self.trials, self.policy = trials, policy
            self.control['paused'] = False
            if task == 'pen':
                self.control['camera'] = 'close'
                self.control['focus'] = -1
            self.save_controls()
            env = os.environ.copy()
            env.update(MUJOCO_GL='egl', OMP_NUM_THREADS='1', PYTHONUNBUFFERED='1',
                       XDG_CACHE_HOME=str(ROOT / 'runs/asset-cache'))
            command = [sys.executable, 'policy/solve.py', '--out', str(self.directory),
                       '--seed', str(seed), '--device', device, '--physics', physics, '--scramble', 'U F L',
                       '--live-dir', str(self.directory / 'live')]
            if task == 'pen':
                command = [sys.executable, str(ROOT / 'pen/web/run.py'), '--out', str(self.directory), '--seed', str(seed),
                           '--trials', str(trials), '--ckpt', str(PEN_POLICIES[policy][1]),
                           '--run-cfg', str(PEN_POLICIES[policy][2])]
            with (self.directory / 'solve.log').open('w') as log:
                self.process = subprocess.Popen(command, cwd=ROOT / 'cube', env=env,
                                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            return self.state()

    def stop(self):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                self.stopped = True
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait()
                # The supervisor can exit before Isaac's shutdown finishes.
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            return self.state()

    def configure(self, values):
        with self.lock:
            if 'camera' in values and values['camera'] not in ('both', 'overview', 'close'):
                raise ValueError('Unknown camera.')
            if 'paused' in values and type(values['paused']) is not bool:
                raise ValueError('Paused must be a boolean.')
            if 'focus' in values and (type(values['focus']) is not int or not -1 <= values['focus'] < self.trials):
                raise ValueError(f'Focus must be -1 (all) or an environment index from 0 to {self.trials-1}.')
            self.control.update({k: values[k] for k in ('paused', 'camera', 'focus') if k in values})
            if self.directory is not None:
                self.save_controls()
            return self.state()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        if args and str(args[0]).startswith('POST'):
            super().log_message(format, *args)

    def respond(self, code, data, mime='application/json'):
        if mime == 'application/json':
            data = json.dumps(data).encode()
        self.send_response(code)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == '/':
            return self.respond(200, (WEB / 'index.html').read_bytes(), 'text/html; charset=utf-8')
        if path == '/api/state':
            return self.respond(200, self.server.runner.state())
        if path == '/api/frame':
            with self.server.runner.lock:
                directory = self.server.runner.directory
                try:
                    data = (directory / 'live/frame.jpg').read_bytes() if directory else None
                except FileNotFoundError:
                    data = None
            if data:
                return self.respond(200, data, 'image/jpeg')
        self.respond(404, {'error': 'Not found'})

    def do_POST(self):
        origin = self.headers.get('Origin')
        if origin and origin not in ('http://' + self.headers.get('Host', ''), 'https://' + self.headers.get('Host', '')):
            return self.respond(403, {'error': 'Cross-origin control requests are not allowed.'})
        if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
            return self.respond(415, {'error': 'Send application/json.'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 4096:
                raise ValueError('Invalid request size.')
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError('Expected a JSON object.')
            path = urlsplit(self.path).path
            runner = self.server.runner
            if path == '/api/start':
                result = runner.start(body.get('device', 'cpu'), body.get('seed', 5005), body.get('physics', 'cpu'), body.get('task', 'cube'), body.get('trials', 256), body.get('policy', 'released'))
            elif path == '/api/control':
                result = runner.configure(body)
            elif path == '/api/stop':
                result = runner.stop()
            else:
                return self.respond(404, {'error': 'Not found'})
            self.respond(200, result)
        except (ValueError, TypeError) as exc:
            self.respond(400, {'error': str(exc)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8088)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/web')
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.runner = Runner(args.output.resolve())
    def shutdown(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, shutdown)
    print(f'Live simulation viewer: http://{args.host}:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.runner.stop()
        server.server_close()


if __name__ == '__main__':
    main()
