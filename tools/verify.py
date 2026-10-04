"""
Verification harness for the particle explosion demo.

Runs the real app for N frames under the real GL context and captures
diagnostics that answer the questions we care about:

  * did the window/context come up at all?
  * are there GL errors during the draw passes?
  * is anything actually rendered? (mean luminance of the back buffer,
    plus separate means for the particle region and the camera PIP corner)
  * does the explosion path run end to end?

Usage:  .venv/bin/python tools/verify.py [--frames 90] [--width 960] ...
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pygame
from OpenGL.GL import (GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_NO_ERROR,
                       glClear, glGetError, glReadPixels, GL_RGB,
                       GL_UNSIGNED_BYTE, glFinish)

import main as app_main
from hand_tracker import HandState


def read_back_buffer(width, height):
    glFinish()
    data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    arr = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
    return np.flipud(arr).copy()          # top-left origin


def luma(rgb):
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])


def report(tag, frame, width, height, pip_rect=None):
    L = luma(frame)
    h, w = L.shape
    center = L[h // 4: 3 * h // 4, w // 4: 3 * w // 4].mean()
    out = {
        "tag": tag,
        "full_mean": round(float(L.mean()), 2),
        "full_max": round(float(L.max()), 1),
        "center_mean": round(float(center), 2),
        "bright_px": int((L > 40).sum()),
    }
    if pip_rect:
        x, y, pw, ph = pip_rect
        sub = L[max(0, y):y + ph, max(0, x):x + pw]
        if sub.size:
            out["pip_mean"] = round(float(sub.mean()), 2)
            out["pip_max"] = round(float(sub.max()), 1)
    return out


def drain_gl_errors(label):
    errs = []
    for _ in range(16):
        e = glGetError()
        if e == GL_NO_ERROR:
            break
        errs.append(e)
    if errs:
        print(f"  !! GL errors after {label}: {errs}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--explode-at", type=int, default=45)
    ap.add_argument("--no-auto-demo", action="store_true")
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    ns = app_main.parse_args.__wrapped__() if hasattr(
        app_main.parse_args, "__wrapped__") else None

    argv = ["x", "--width", str(args.width), "--height", str(args.height),
            "--camera", str(args.camera)]
    if args.no_auto_demo:
        argv.append("--no-auto-demo")
    sys.argv = argv

    a = app_main.parse_args()
    print(f"[verify] building app {a.width}x{a.height} camera={a.camera}")
    app = app_main.ParticleExplosionApp(a)
    print(f"[verify] window created OK; tracking_ok={app.tracking_ok} "
          f"camera_fallback={app.camera.fallback} auto_demo={app.auto_demo}")

    # Does the renderer think it has a VAO / which context?
    from OpenGL.GL import GL_VERSION, GL_SHADING_LANGUAGE_VERSION, glGetString
    print(f"[verify] GL_VERSION  : {glGetString(GL_VERSION).decode()}")
    print(f"[verify] GLSL version: "
          f"{glGetString(GL_SHADING_LANGUAGE_VERSION).decode()}")
    print(f"[verify] renderer VAO : {app.renderer._has_vao}")

    drain_gl_errors("startup")

    dt = 1.0 / 60.0
    rows = []
    exploded_at = None
    for i in range(args.frames):
        if i == args.explode_at:
            app._manual_explosion()
            exploded_at = i
        app.step(dt)
        errs = drain_gl_errors(f"frame {i}")
        if errs:
            break
        if i in (0, 5, args.explode_at - 1, args.explode_at,
                 args.explode_at + 8, args.explode_at + 30,
                 args.explode_at + 90, args.frames - 1):
            pip_w, pip_h = app.renderer._pip_size()
            pip = (app.renderer.width - pip_w - app.renderer.PIP_MARGIN,
                   app.renderer.height - pip_h - app.renderer.PIP_MARGIN,
                   pip_w, pip_h)
            frame = read_back_buffer(app.width, app.height)
            r = report(f"f{i:03d}", frame, app.width, app.height, pip)
            r["state"] = app.state
            r["pstate"] = app.particles.state
            rows.append(r)
            if args.save and i in (args.explode_at - 1,
                                   args.explode_at + 30):
                import cv2
                p = f"{args.save}_f{i:03d}.png"
                cv2.imwrite(p, np.ascontiguousarray(frame[..., ::-1]))
                print(f"[verify] wrote {p}")

    print("\n[verify] frame captures (luma, 0-255):")
    hdr = f"{'tag':<8}{'state':<10}{'pstate':<11}{'full':>7}{'max':>7}" \
          f"{'center':>8}{'bright':>8}{'pip':>7}{'pipmax':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['tag']:<8}{r.get('state',''):<10}{r.get('pstate',''):<11}"
              f"{r['full_mean']:>7}{r['full_max']:>7}{r['center_mean']:>8}"
              f"{r['bright_px']:>8}{r.get('pip_mean','-'):>7}"
              f"{r.get('pip_max','-'):>8}")

    print(f"\n[verify] exploded_at frame: {exploded_at}  "
          f"final state: {app.state} / {app.particles.state}")
    app.shutdown()
    print("[verify] clean shutdown")


if __name__ == "__main__":
    main()
