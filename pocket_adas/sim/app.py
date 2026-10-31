"""Interactive pygame front-end.

Left: top-down world. Right: what the phone app would show, plus live signals.
"""
import math
from collections import deque

import numpy as np
import pygame

from ..core import events as E
from ..core.geo import latlon_to_xy
from .autopilot import Autopilot
from .phone import MOUNTS
from .simulation import Simulation

W, H = 1280, 720
VIEW_W = 800
PPM = 6.0  # pixels per metre on the track surface
MINI_W = 220

C_BG = (24, 26, 31)
C_GRASS = (58, 92, 54)
C_ROAD = (62, 64, 70)
C_SCHOOL = (92, 66, 58)
C_PANEL = (32, 35, 42)
C_TEXT = (230, 232, 236)
C_DIM = (140, 146, 156)
LEVEL_COLOURS = {
    E.CALIBRATING: (90, 120, 200), E.SAFE: (46, 160, 90),
    E.WARNING: (230, 170, 40), E.DANGER: (215, 55, 55),
}
EVENT_COLOURS = {
    E.HARSH_BRAKE: (230, 70, 70), E.HARSH_ACCEL: (240, 150, 50), E.HARSH_TURN: (200, 90, 220),
    E.BUMP: (240, 220, 60), E.POTHOLE: (160, 120, 80), E.SPEEDING: (80, 170, 255),
    E.SUDDEN_MOVE: (250, 250, 250),
}
HELP = ("Arrows/WASD drive  SPACE hard brake  1 manual  2 calm  3 aggressive  "
        "M remount phone  R reset  P pause  ESC quit+save")


def _beep(freq, ms, vol=0.35):
    try:
        sr = pygame.mixer.get_init()[0]
        t = np.arange(int(sr * ms / 1000)) / sr
        wave = (np.sin(2 * np.pi * freq * t) * vol * 32767 * np.minimum(1, (len(t) - np.arange(len(t))) / 200))
        arr = wave.astype(np.int16)
        ch = pygame.mixer.get_init()[2]
        if ch == 2:
            arr = np.column_stack([arr, arr])
        return pygame.sndarray.make_sound(np.ascontiguousarray(arr))
    except Exception:
        return None


class App:
    def __init__(self, mount="dashboard", seed=None, driver="manual"):
        pygame.init()
        try:
            pygame.mixer.init(44100, -16, 1)
        except Exception:
            pass
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("Pocket ADAS - smartphone driver assistance simulator")
        self.clock = pygame.time.Clock()
        self.f_small = pygame.font.Font(None, 20)
        self.f_med = pygame.font.Font(None, 26)
        self.f_big = pygame.font.Font(None, 40)
        self.f_huge = pygame.font.Font(None, 84)
        self.sim = Simulation(seed=seed, mount=mount)
        self.mount_idx = MOUNTS.index(mount)
        self.driver = driver
        self.autopilot = None if driver == "manual" else Autopilot(driver, self.sim.rng)
        self.throttle = 0.0
        self.steer = 0.0
        self.paused = False
        self.hist = deque(maxlen=int(10 * self.sim.fs))
        self.snd_warn = _beep(880, 140)
        self.snd_danger = _beep(1300, 260)
        self._build_track_surface()

    # ------------------------------------------------------------ track art
    def _build_track_surface(self):
        r = self.sim.road
        pts = r.pts
        self.minx, self.maxx = pts[:, 0].min() - 60, pts[:, 0].max() + 60
        self.miny, self.maxy = pts[:, 1].min() - 60, pts[:, 1].max() + 60
        size = (int((self.maxx - self.minx) * PPM), int((self.maxy - self.miny) * PPM))
        s = pygame.Surface(size)
        s.fill(C_GRASS)
        P = lambda p: self._w2t(p[0], p[1])
        n = len(pts)
        rw = int(r.width * PPM)
        for i in range(n):
            col = C_SCHOOL if r.zone_at_s(r.s[i])[1] == "school" else C_ROAD
            pygame.draw.line(s, col, P(pts[i]), P(pts[(i + 1) % n]), rw + 3)
            pygame.draw.circle(s, col, P(pts[i]), rw // 2)
        for side in (-1, 1):
            edge = pts + side * (r.width / 2 - 0.3) * r.normal
            pygame.draw.lines(s, (225, 225, 225), True, [P(p) for p in edge], 2)
        for i in range(0, n, 8):  # dashed centre line
            pygame.draw.line(s, (235, 210, 90), P(pts[i]), P(pts[(i + 4) % n]), 2)
        for f in r.features:
            tx, ty = f.tx, f.ty
            nx, ny = -ty, tx
            hl, hw = f.length / 2, f.width / 2
            corners = [(f.x + a * tx * hl + b * nx * hw, f.y + a * ty * hl + b * ny * hw)
                       for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
            if f.kind == "hump":
                pygame.draw.polygon(s, (230, 200, 40), [P(c) for c in corners])
                for k in np.linspace(-hw, hw, 7):
                    a = (f.x + nx * k - tx * hl, f.y + ny * k - ty * hl)
                    b = (f.x + nx * (k + 0.5) + tx * hl, f.y + ny * (k + 0.5) + ty * hl)
                    pygame.draw.line(s, (30, 30, 30), P(a), P(b), 3)
            else:
                pygame.draw.circle(s, (35, 30, 28), P((f.x, f.y)), int(hw * PPM))
        # zone signs
        for a, b, lim, name in r.zones:
            i = int(a) % n
            x, y = pts[i] + (r.width / 2 + 5) * r.normal[i]
            self._sign(s, P((x, y)), lim, 22)
            if name == "school":
                txt = self.f_med.render("SCHOOL ZONE", True, (255, 240, 200))
                s.blit(txt, (P((x, y))[0] + 26, P((x, y))[1] - 8))
        self.track = s
        mh = int(MINI_W * size[1] / size[0])
        self.mini = pygame.transform.smoothscale(s, (MINI_W, mh))
        self.mini.set_alpha(215)

    def _w2t(self, x, y):
        return int((x - self.minx) * PPM), int((self.maxy - y) * PPM)

    def _sign(self, surf, c, lim, r):
        pygame.draw.circle(surf, (250, 250, 250), c, r)
        pygame.draw.circle(surf, (210, 40, 40), c, r, max(3, r // 5))
        t = self.f_med.render(str(lim), True, (20, 20, 20))
        surf.blit(t, t.get_rect(center=c))

    # ----------------------------------------------------------------- input
    def _handle_input(self, dt):
        keys = pygame.key.get_pressed()
        up = keys[pygame.K_UP] or keys[pygame.K_w]
        down = keys[pygame.K_DOWN] or keys[pygame.K_s]
        left = keys[pygame.K_LEFT] or keys[pygame.K_a]
        right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        # ramps make gentle driving possible with a keyboard: tap = soft, hold = hard
        if keys[pygame.K_SPACE]:
            self.throttle = -1.0
        elif up:
            self.throttle = max(0.0, self.throttle) + dt / 1.5
        elif down:
            self.throttle = min(0.0, self.throttle) - dt / 1.2
        else:
            self.throttle = 0.0
        self.throttle = max(-1.0, min(1.0, self.throttle))
        target = (1.0 if left else 0.0) - (1.0 if right else 0.0)
        rate = 1.6 if target else 3.0
        self.steer += max(-rate * dt, min(rate * dt, target - self.steer))
        v = self.sim.vehicle.v
        return self.throttle, self.steer / (1 + (v / 10) ** 2)

    def _set_driver(self, name):
        self.driver = name
        self.autopilot = None if name == "manual" else Autopilot(name, self.sim.rng)

    # ------------------------------------------------------------------ loop
    def run(self):
        running = True
        while running:
            dt = self.clock.tick(self.sim.fs) / 1000.0
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    k = ev.key
                    if k == pygame.K_ESCAPE:
                        running = False
                    elif k == pygame.K_p:
                        self.paused = not self.paused
                    elif k == pygame.K_r:
                        self.sim.reset_vehicle()
                        self.hist.clear()
                    elif k == pygame.K_m:
                        self.mount_idx = (self.mount_idx + 1) % len(MOUNTS)
                        self.sim.remount_phone(MOUNTS[self.mount_idx])
                    elif k in (pygame.K_1, pygame.K_2, pygame.K_3):
                        self._set_driver({pygame.K_1: "manual", pygame.K_2: "calm",
                                          pygame.K_3: "aggressive"}[k])
            if not self.paused:
                self.tick()
            self.draw()
            pygame.display.flip()
        path = self.sim.save_trip(tag=self.driver)
        pygame.quit()
        if path:
            print(f"Trip saved to {path}")

    def tick(self):
        dt = self.sim.dt
        if self.autopilot and self.sim.t > 3.0:
            th, st = self.autopilot.control(self.sim, dt)
        elif self.autopilot:
            th, st = 0.0, 0.0   # stand still first so gravity can be calibrated
        else:
            th, st = self._handle_input(dt)
        new = self.sim.step(th, st)
        p = self.sim.pipeline
        self.hist.append((p.signals["lon"], p.signals["lat"], p.signals["vert"], p.signals["legacy"]))
        if new:
            danger = any(e.severity >= p.cfg.danger_ratio for e in new)
            snd = self.snd_danger if danger else self.snd_warn
            if snd:
                snd.play()

    # ------------------------------------------------------------------ draw
    def draw(self):
        self.screen.fill(C_BG)
        self._draw_world()
        self._draw_panel()
        t = self.f_small.render(HELP, True, C_DIM)
        self.screen.blit(t, (10, H - 22))

    def _draw_world(self):
        veh = self.sim.vehicle
        cx, cy = self._w2t(veh.x, veh.y)
        area = pygame.Rect(cx - VIEW_W // 2, cy - (H - 30) // 2, VIEW_W, H - 30)
        self.screen.fill(C_GRASS, (0, 0, VIEW_W, H - 30))
        self.screen.blit(self.track, (0, 0), area)
        to_screen = lambda x, y: (self._w2t(x, y)[0] - area.x, self._w2t(x, y)[1] - area.y)
        # event markers at their (noisy GPS) positions
        for e in self.sim.pipeline.events:
            if e.lat is None:
                continue
            sx, sy = to_screen(*latlon_to_xy(e.lat, e.lon))
            if -10 < sx < VIEW_W + 10 and -10 < sy < H:
                pygame.draw.circle(self.screen, EVENT_COLOURS.get(e.kind, (255, 255, 255)), (sx, sy), 6)
                pygame.draw.circle(self.screen, (20, 20, 20), (sx, sy), 6, 1)
        # car
        c, s = math.cos(veh.heading), math.sin(veh.heading)
        hl, hw = veh.p.length / 2, veh.p.width / 2
        body = [(veh.x + c * a - s * b, veh.y + s * a + c * b)
                for a, b in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]
        pygame.draw.polygon(self.screen, (40, 120, 230), [to_screen(*p) for p in body])
        wind = [(veh.x + c * a - s * b, veh.y + s * a + c * b)
                for a, b in ((hl * 0.55, hw * 0.8), (hl * 0.55, -hw * 0.8), (hl * 0.1, -hw * 0.8), (hl * 0.1, hw * 0.8))]
        pygame.draw.polygon(self.screen, (170, 210, 255), [to_screen(*p) for p in wind])
        if veh.a_long < -1.0:  # brake lights
            for b in (hw * 0.7, -hw * 0.7):
                pygame.draw.circle(self.screen, (255, 40, 40), to_screen(veh.x - c * hl + -s * b, veh.y - s * hl + c * b), 3)
        # minimap
        mx, my = VIEW_W - MINI_W - 10, 10
        sc = MINI_W / self.track.get_width()
        self.screen.blit(self.mini, (mx, my))
        pygame.draw.rect(self.screen, (0, 0, 0), (mx, my, MINI_W, self.mini.get_height()), 2)
        for e in self.sim.pipeline.events:
            if e.lat is not None:
                tx, ty = self._w2t(*latlon_to_xy(e.lat, e.lon))
                pygame.draw.circle(self.screen, EVENT_COLOURS.get(e.kind), (mx + tx * sc, my + ty * sc), 2)
        pygame.draw.circle(self.screen, (255, 255, 255), (mx + cx * sc, my + cy * sc), 4)
        pygame.draw.rect(self.screen, (255, 255, 255), (mx + area.x * sc, my + area.y * sc,
                                                        area.w * sc, area.h * sc), 1)
        # mode badge
        tag = {"manual": "MANUAL", "calm": "AUTOPILOT: CALM", "aggressive": "AUTOPILOT: AGGRESSIVE"}[self.driver]
        t = self.f_med.render(tag + ("  (PAUSED)" if self.paused else ""), True, C_TEXT)
        pygame.draw.rect(self.screen, (0, 0, 0), (8, 8, t.get_width() + 16, 30), border_radius=6)
        self.screen.blit(t, (16, 14))

    def _draw_panel(self):
        x0 = VIEW_W + 10
        pw = W - x0 - 10
        p = self.sim.pipeline
        st = p.status
        pygame.draw.rect(self.screen, C_PANEL, (x0, 8, pw, H - 38), border_radius=14)
        y = 18
        self.screen.blit(self.f_big.render("Pocket ADAS", True, C_TEXT), (x0 + 16, y))
        self.screen.blit(self.f_small.render("what the phone shows", True, C_DIM), (x0 + 210, y + 12))
        y += 44
        # speed + limit
        spd = self.f_huge.render(f"{p.speed_kmh:3.0f}", True, C_TEXT)
        self.screen.blit(spd, (x0 + 20, y))
        self.screen.blit(self.f_med.render("km/h", True, C_DIM), (x0 + 30 + spd.get_width(), y + 46))
        if p.limit_kmh:
            self._sign(self.screen, (x0 + pw - 60, y + 36), p.limit_kmh, 32)
        self.screen.blit(self.f_med.render(f"Score {p.score}", True, C_TEXT), (x0 + 190, y + 8))
        self.screen.blit(self.f_small.render(f"{p.distance_m / 1000:.2f} km  {self.sim.t:5.0f} s",
                                             True, C_DIM), (x0 + 190, y + 34))
        y += 92
        # status banner
        col = LEVEL_COLOURS[st.level]
        pygame.draw.rect(self.screen, col, (x0 + 14, y, pw - 28, 64), border_radius=10)
        msg = st.message if st.level in (E.SAFE, E.CALIBRATING) else f"{st.level}: {st.message}"
        t = self.f_big.render(msg, True, (255, 255, 255))
        if t.get_width() > pw - 40:
            t = self.f_med.render(msg, True, (255, 255, 255))
        self.screen.blit(t, t.get_rect(center=(x0 + pw // 2, y + 32)))
        y += 76
        # calibration line
        stage = {"gravity": "finding gravity (stand still)", "forward": "finding forward axis (drive)",
                 "done": "done"}[p.cal.stage]
        self.screen.blit(self.f_small.render(f"Mount: {self.sim.phone.mount}   calibration: {stage}",
                                             True, C_DIM), (x0 + 16, y))
        y += 22
        y = self._draw_plot(x0 + 14, y, pw - 28, 150)
        # 2023 comparison
        y += 6
        on = p.signals["legacy"] >= 15
        pygame.draw.circle(self.screen, (215, 55, 55) if on else (70, 74, 82), (x0 + 24, y + 9), 7)
        self.screen.blit(self.f_small.render(
            f"2023 rule |x|+|y|+|z|>=15: {p.signals['legacy']:5.1f}   alerts so far: {p.legacy_alerts}",
            True, C_TEXT), (x0 + 38, y + 2))
        y += 26
        # event log
        self.screen.blit(self.f_med.render("Events", True, C_TEXT), (x0 + 16, y))
        y += 24
        for e in reversed(p.events[-7:]):
            pygame.draw.circle(self.screen, EVENT_COLOURS.get(e.kind), (x0 + 24, y + 8), 6)
            unit = "km/h" if e.kind == E.SPEEDING else "m/s2"
            line = f"{e.t:6.1f}s  {e.message:<20} {e.peak:5.1f} {unit} @ {e.speed_kmh:3.0f} km/h"
            self.screen.blit(self.f_small.render(line, True, C_TEXT), (x0 + 38, y + 1))
            y += 20

    def _draw_plot(self, x, y, w, h):
        pygame.draw.rect(self.screen, (22, 24, 29), (x, y, w, h), border_radius=6)
        rng = 8.0
        mid = y + h // 2
        sc = (h / 2 - 6) / rng
        thr = self.sim.pipeline.cfg.harsh_brake
        for v in (thr, -thr):
            yy = int(mid - v * sc)
            for xx in range(x + 4, x + w - 4, 10):
                pygame.draw.line(self.screen, (80, 80, 90), (xx, yy), (xx + 5, yy))
        pygame.draw.line(self.screen, (60, 62, 70), (x + 4, mid), (x + w - 4, mid))
        if len(self.hist) > 2:
            data = np.asarray(self.hist)
            n = self.hist.maxlen
            xs = x + 4 + (np.arange(len(data)) + n - len(data)) * (w - 8) / n
            for k, col in ((0, (90, 170, 255)), (1, (220, 120, 230)), (2, (240, 220, 80))):
                ys = np.clip(mid - data[:, k] * sc, y + 2, y + h - 2)
                pygame.draw.lines(self.screen, col, False, list(zip(xs, ys)), 2)
        lab = [("forward", (90, 170, 255)), ("left", (220, 120, 230)), ("vertical", (240, 220, 80))]
        lx = x + 8
        for name, col in lab:
            t = self.f_small.render(name, True, col)
            self.screen.blit(t, (lx, y + 4))
            lx += t.get_width() + 12
        self.screen.blit(self.f_small.render("m/s2, last 10 s, gravity removed", True, C_DIM), (x + w - 215, y + 4))
        return y + h
