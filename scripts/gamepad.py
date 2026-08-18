import numpy as np
import evdev
import select
from evdev import ecodes

class Pad:
    """Minimal analog DS4 reader."""

    def __init__(self, path=None):
        self.dev = self._find(path)
        self.abs = dict(self.dev.capabilities().get(ecodes.EV_ABS, []))
        self.axis, self.btn = {}, {}
        self.grip_closed = False
        self._prev_r1 = 0
        # end this episode (save)
        self.done = False
        # quit everything
        self.terminate = False
        # discard + reset this episode
        self.qk_reset = False 
        self.dead = 0.12

    def _find(self, path):
        if path:
            return evdev.InputDevice(path)
        for p in evdev.list_devices():
            d = evdev.InputDevice(p)
            caps = d.capabilities()
            abs_codes = [c for c, _ in caps.get(ecodes.EV_ABS, [])]
            if ecodes.ABS_X in abs_codes and ecodes.ABS_RX in abs_codes:
                print(f"gamepad: {d.name} ({p})")
                return d
        raise RuntimeError("No controller found, check!")

    def _norm(self, code):
        i = self.abs[code]
        mid = (i.min + i.max) / 2
        half = (i.max - i.min) / 2
        v = (self.axis.get(code, mid) - mid) / half
        return 0.0 if abs(v) < self.dead else v

    def _drain(self):
        while select.select([self.dev.fd], [], [], 0)[0]:
            for e in self.dev.read():
                if e.type == ecodes.EV_ABS:
                    self.axis[e.code] = e.value
                elif e.type == ecodes.EV_KEY:
                    self.btn[e.code] = e.value

    def reset(self):
        self.done = False
        self.qk_reset = False
        self.grip_closed = False
        self._prev_r1 = 0
        self.btn = {}

    def action(self):
        self._drain()
        # stick up  -> +x
        dx = -self._norm(ecodes.ABS_Y)
        # stick left -> +y
        dy = -self._norm(ecodes.ABS_X)
        # r-stick up -> +z
        dz = -self._norm(ecodes.ABS_RY)  
        r1 = self.btn.get(ecodes.BTN_TR, 0)
        if r1 and not self._prev_r1:
            self.grip_closed = not self.grip_closed
        self._prev_r1 = r1
        # X
        if self.btn.get(ecodes.BTN_SOUTH, 0):
            self.done = True
            self.terminate = True
        # Square
        if self.btn.get(ecodes.BTN_WEST, 0):
            self.done = True
        # Triangle
        if self.btn.get(ecodes.BTN_NORTH, 0):
            self.qk_reset = True
        grip = -1.0 if self.grip_closed else 1.0

        return np.array([dx, dy, dz, grip], dtype=np.float32)
