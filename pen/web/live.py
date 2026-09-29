"""Publish current PhysX state for an independent live renderer."""
import json
import time
from pathlib import Path


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value))
    tmp.replace(path)


class Publisher:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.last = 0

    def wait(self):
        while True:
            try:
                control = json.loads((self.directory / 'controls.json').read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                control = {}
            if not control.get('paused', False):
                return
            time.sleep(.03)

    def publish(self, q, pos, quat, sim_time, turns, force=False, holding=None, seed0=0):
        now = time.monotonic()
        if force or now - self.last >= .1:
            write_json(self.directory / 'pose.json', dict(q=q.tolist(), pos=pos.tolist(),
                quat=quat.tolist(), sim_time=sim_time, turns=turns, holding=holding, seed0=seed0, updated=time.time()))
            self.last = now
