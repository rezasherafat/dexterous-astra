"""Optional live rendering hook. No trajectory files are read by this viewer."""
import io
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import scene


def atomic_write(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_bytes(data)
    tmp.replace(path)


class LiveView:
    def __init__(self, directory, sim, fps=10):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.renderer = mujoco.Renderer(sim.m, height=360, width=640)
        self.data = mujoco.MjData(sim.m)
        self.period = 1 / fps
        self.last_frame = 0
        self.frames = 0
        self.started = time.monotonic()
        self.paused_seconds = 0

    def controls(self):
        try:
            return json.loads((self.directory / 'controls.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def update(self, sim, force=False):
        paused_at = None
        while True:
            control = self.controls()
            paused = bool(control.get('paused', False))
            if paused and paused_at is None:
                paused_at = time.monotonic()
            now = time.monotonic()
            if force or now - self.last_frame >= self.period:
                self.publish(sim, control, now, paused)
                force = False
            if not paused:
                if paused_at is not None:
                    self.paused_seconds += time.monotonic() - paused_at
                return
            time.sleep(0.05)

    def publish(self, sim, control, now, paused):
        # Forward only a private MjData so visualization cannot change controller state.
        self.data.qpos[:] = sim.d.qpos
        self.data.mocap_pos[:] = sim.d.mocap_pos
        self.data.mocap_quat[:] = sim.d.mocap_quat
        mujoco.mj_forward(sim.m, self.data)
        camera = control.get('camera', 'both')
        panels = []
        if camera in ('both', 'overview'):
            self.renderer.update_scene(self.data, 'overview', scene.VISUAL)
            panels.append(self.renderer.render().copy())
        if camera in ('both', 'close'):
            cam = mujoco.MjvCamera()
            cam.distance, cam.azimuth, cam.elevation = 0.26, 240, -30
            cam.lookat[:] = self.data.xpos[sim.core]
            self.renderer.update_scene(self.data, cam, scene.VISUAL)
            panels.append(self.renderer.render().copy())
        frame = np.concatenate(panels, axis=1)
        buffer = io.BytesIO()
        Image.fromarray(frame).save(buffer, format='JPEG', quality=85)
        atomic_write(self.directory / 'frame.jpg', buffer.getvalue())
        self.frames += 1
        elapsed = now - self.started - self.paused_seconds
        facelets, misalignment = scene.facelets(sim.m, self.data)
        status = dict(status='paused' if paused else 'running', phase=sim.phase,
                      sim_time=float(sim.d.time), elapsed=elapsed, frame=self.frames,
                      updated=time.time(), camera=camera, misalignment_deg=float(misalignment),
                      facelets=facelets)
        status['simulation'] = sim.backend.info() if sim.backend else dict(backend='cpu', device='cpu')
        atomic_write(self.directory / 'state.json', json.dumps(status).encode())
        self.last_frame = now

    def close(self):
        self.renderer.close()
