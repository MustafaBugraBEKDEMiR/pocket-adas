"""Record docs/demo.gif from the simulator (headless).

    python tools/make_gif.py
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pygame  # noqa: E402
from PIL import Image  # noqa: E402

from pocket_adas.sim.app import App  # noqa: E402
from pocket_adas.sim.osm_world import OsmWorld  # noqa: E402


def main(out="docs/demo.gif", seconds=12, fps=7, start=8.0, scale=0.45):
    app = App(mount="random", seed=3, driver="aggressive", world=OsmWorld("maps/ankara_sihhiye.osm"))
    frames, step = [], int(app.sim.fs / fps)
    k = 0
    while app.sim.t < start + seconds:
        for _ in pygame.event.get():
            pass
        app.tick()
        k += 1
        if app.sim.t >= start and k % step == 0:
            app.draw()
            raw = pygame.image.tobytes(app.screen, "RGB")
            img = Image.frombytes("RGB", app.screen.get_size(), raw)
            w, h = img.size
            frames.append(img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                          .quantize(colors=48, method=Image.Quantize.MEDIANCUT))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)
    print(out, len(frames), "frames", os.path.getsize(out) // 1024, "KB")


if __name__ == "__main__":
    main()
