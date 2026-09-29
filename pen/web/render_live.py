"""Render all current PhysX environments as a live grid, never recorded trajectories."""
import argparse
import io
import json
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sim'))
from mj_scene import build, Index
from live import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    model = build()
    data = mujoco.MjData(model)
    index = Index(model)
    renderer = mujoco.Renderer(model, height=120, width=160)
    detail_renderer = mujoco.Renderer(model, height=480, width=640)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 10)
    frame = 0
    try:
        while True:
            started = time.monotonic()
            try:
                pose = json.loads((args.directory / 'pose.json').read_text())
                control = json.loads((args.directory / 'controls.json').read_text())
            except FileNotFoundError:
                time.sleep(.1)
                continue
            camera = control.get('camera', 'close')
            views = ('overview', 'close') if camera == 'both' else (camera,)
            count = len(pose['q'])
            focus = control.get('focus', -1)
            selected = [focus] if 0 <= focus < count else list(range(count))
            columns = 1 if len(selected) == 1 else min(16, math.ceil(math.sqrt(len(selected))))
            tile_w, tile_h = (640, 480) if len(selected) <= 4 else (160, 120)
            active_renderer = detail_renderer if len(selected) <= 4 else renderer
            width = tile_w * len(views)
            grid = Image.new('RGB', (columns * width, math.ceil(len(selected) / columns) * (tile_h + 20)), '#101414')
            draw = ImageDraw.Draw(grid)
            environments = []
            for cell, i in enumerate(selected):
                index.set_state(data, pose['q'][i], pose['pos'][i], pose['quat'][i])
                mujoco.mj_forward(model, data)
                panels = []
                for name in views:
                    cam = mujoco.MjvCamera()
                    cam.lookat[:] = [.14, 0, .575]
                    cam.distance = .30 if name == 'overview' else .24
                    cam.azimuth = 180 if name == 'overview' else 215
                    cam.elevation = -89 if name == 'overview' else -28
                    active_renderer.update_scene(data, cam)
                    panels.append(active_renderer.render().copy())
                x, y = (cell % columns) * width, (cell // columns) * (tile_h + 20)
                grid.paste(Image.fromarray(np.concatenate(panels, axis=1)), (x, y + 20))
                holding = pose['holding'][i]
                seed = pose['seed0'] + i
                turns = pose['turns'][i]
                draw.text((x + 3, y + 4), f'{i+1:03d} {seed} {turns:.1f}' + (' H' if holding else ''),
                          font=font, fill='#b6ef94' if holding else '#e8efeb')
                environments.append(dict(seed=seed, turns=turns, holding=holding))
            environments = [dict(seed=pose['seed0'] + i, turns=pose['turns'][i], holding=pose['holding'][i]) for i in range(count)]
            buffer = io.BytesIO()
            grid.save(buffer, format='JPEG', quality=85)
            temp = args.directory / 'frame.tmp'
            temp.write_bytes(buffer.getvalue())
            temp.replace(args.directory / 'frame.jpg')
            frame += 1
            held = sum(pose['holding'])
            write_json(args.directory / 'state.json', dict(status='paused' if control.get('paused') else 'running',
                phase=f'{held}/{count} in hold phase · {count-held}/{count} spinning', frame=frame,
                sim_time=pose['sim_time'], turns=float(np.mean(pose['turns'])),
                updated=pose['updated'], camera=camera, environments=environments,
                simulation=dict(backend='physx', device='cuda:0', num_envs=count, policy_device='cuda:0'),
                trials=count, focus=focus, grid_columns=columns, visible_environments=selected))
            time.sleep(max(0, (.5 if len(selected) > 4 else .1) - (time.monotonic() - started)))
    finally:
        renderer.close()
        detail_renderer.close()


if __name__ == '__main__':
    main()
