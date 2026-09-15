#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimize_images.py - whiteicey.github.io image optimization pipeline

Generates WebP derivatives in img/optimized/{hero,card,body}:
  - hero (max 1920w, q90)  for intro-header backgrounds (full-viewport width)
  - card (max  800w, q88)  for index post-card thumbnails
  - body (max 2048w, q88)  for in-post images (keeps lightbox zoom crisp)

Idempotent: skips up-to-date derivatives. Never touches originals.
"""
import argparse
import sys
import time
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / 'img'
OUT_DIR = IMG_DIR / 'optimized'

VARIANTS = {
    'hero': dict(max_width=1920, quality=90),
    'card': dict(max_width=800,  quality=88),
    'body': dict(max_width=2048, quality=88),
}

RASTER_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}
SKIP_EXTS = {'.webp'}  # already webp: re-encoding would be lossy again
SKIP_BASENAMES = {
    'favicon.ico', 'apple-touch-icon.png', 'holo-touch-icon.jpg',
    'avatar_g.jpg', 'avatar_m.jpg',
}


def human(n):
    return f'{n/1024:.0f} KB' if n < 1024*1024 else f'{n/1024/1024:.1f} MB'


def save_webp(im, dest, quality):
    dest.parent.mkdir(parents=True, exist_ok=True)
    im = im.convert('RGBA') if im.mode in ('RGBA', 'LA', 'P') else im.convert('RGB')
    im.save(dest, 'WEBP', quality=quality, method=6)


def process(src, force=False):
    base = src.name
    if base in SKIP_BASENAMES or src.suffix.lower() not in RASTER_EXTS:
        return dict(action='skip')
    if src.suffix.lower() in SKIP_EXTS:
        return dict(action='skip-webp-source')
    try:
        im = Image.open(src); im.load()
    except Exception as e:
        return dict(action=f'error: {e}')
    w, h = im.size
    src_size = src.stat().st_size
    stem = src.stem
    # disambiguate stems shared by multiple extensions (hug.jpg / hug.png)
    if (IMG_DIR / (stem + '.jpg')).exists() and (IMG_DIR / (stem + '.png')).exists():
        stem = f"{src.stem}-{src.suffix.lstrip('.').lower()}"
    total, actions = 0, []
    for name, cfg in VARIANTS.items():
        dest = OUT_DIR / name / f'{stem}.webp'
        if dest.exists() and not force and dest.stat().st_mtime >= src.stat().st_mtime:
            continue
        if w <= cfg['max_width']:
            im2 = im.copy()
        else:
            r = cfg['max_width'] / w
            im2 = im.resize((cfg['max_width'], round(h*r)), Image.LANCZOS)
        save_webp(im2, dest, cfg['quality'])
        total += src_size - dest.stat().st_size
        actions.append(f"{name}:{human(dest.stat().st_size)}")
    return dict(action='ok', saved=total, detail=', '.join(actions))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    files = sorted(p for p in IMG_DIR.iterdir()
                   if p.is_file() and p.suffix.lower() in RASTER_EXTS)
    print(f'Scanning {len(files)} images in {IMG_DIR} ...')
    t0 = time.time(); before = after = done = 0
    for i, src in enumerate(files, 1):
        r = process(src, force=args.force)
        if r['action'] == 'ok':
            done += 1
            sz = src.stat().st_size
            before += sz; after += sz - r['saved']
            print(f'[{i:3d}/{len(files)}] {src.name:42s} {human(sz):>8s} -> {r["detail"]}')
        elif r['action'].startswith('error'):
            print(f'[{i:3d}/{len(files)}] {src.name:42s}  !! {r["action"]}')
    print('=' * 60)
    print(f'Done {done} files in {time.time()-t0:.1f}s')
    print(f'Source total            : {human(before)}')
    print(f'Optimized total (3 var) : {human(after)}')
    print(f'Output: {OUT_DIR}')
    return 0


if __name__ == '__main__':
    sys.exit(main())