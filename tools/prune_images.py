#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prune unreferenced optimized and original image files.

Default is a dry run. Pass --apply to delete files. Original files are expected
to be backed up outside the repository before applying.
"""
import argparse
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / 'img'
OPT = IMG / 'optimized'
IMG_RE = re.compile(r'\.(jpe?g|png|webp|gif|ico|svg)$', re.I)
MD_IMG = re.compile(r'!\[[^\]]*\]\(([^)\s]+)')
SRC = re.compile(r'''(?:src|href)\s*=\s*["']([^"']+)["']''')
YAML_IMG = re.compile(r'''(?m)^\s*(?:header-img|sidebar-avatar)\s*:\s*["']?([^\s"']+)''')
SPECIAL = {
    'favicon.ico',
    'apple-touch-icon.png',
    'holo-touch-icon.jpg',
    'avatar_g.jpg',
    'avatar_m.jpg',
}


def normalize(url: str) -> str | None:
    if url.startswith(('http://', 'https://', '//', 'data:')):
        return None
    path = unquote(url.split('?')[0].split('#')[0]).lstrip('/')
    return path if path.startswith('img/') else None


def collect_referenced() -> set[str]:
    referenced = {f'img/{name}' for name in SPECIAL}
    files = [*(
        ROOT / '_posts').glob('*.md'), *(
        ROOT / '_layouts').glob('*.html'), *(
        ROOT / '_includes').glob('*.html'), ROOT / '_config.yml', *ROOT.glob('*.html')
    ]
    for file in files:
        if not file.is_file():
            continue
        in_fence = False
        for line in file.read_text(encoding='utf-8', errors='replace').splitlines():
            if line.lstrip().startswith('```'):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            urls = [m.group(1) for m in MD_IMG.finditer(line)]
            urls += [m.group(1) for m in SRC.finditer(line)]
            urls += [m.group(1) for m in YAML_IMG.finditer(line)]
            for url in urls:
                norm = normalize(url)
                if norm:
                    referenced.add(norm)
                    if '/optimized/hero/' in norm:
                        referenced.add(norm.replace('/optimized/hero/', '/optimized/card/'))
    return referenced


def safe_unlink(path: Path) -> None:
    resolved = path.resolve()
    img_root = IMG.resolve()
    if img_root not in resolved.parents:
        raise RuntimeError(f'Refusing to delete outside img/: {resolved}')
    path.unlink()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true', help='Delete files instead of printing a dry run')
    args = ap.parse_args()

    referenced = collect_referenced()
    candidates = [p for p in IMG.rglob('*') if p.is_file() and IMG_RE.search(p.name)]
    unused = [p for p in candidates if p.relative_to(ROOT).as_posix() not in referenced]

    before = sum(p.stat().st_size for p in candidates)
    removed_size = sum(p.stat().st_size for p in unused)
    print(f'referenced images : {len(referenced)}')
    print(f'candidate files   : {len(candidates)} ({before / 1024 / 1024:.1f} MB)')
    print(f'unused files      : {len(unused)} ({removed_size / 1024 / 1024:.1f} MB)')
    for path in unused[:30]:
        print(f'  {path.relative_to(ROOT).as_posix()}')
    if len(unused) > 30:
        print(f'  ... and {len(unused) - 30} more')

    if not args.apply:
        print('\nDry run only. Re-run with --apply after confirming the external backup.')
        return 0

    for path in unused:
        safe_unlink(path)
    print(f'\nDeleted {len(unused)} files ({removed_size / 1024 / 1024:.1f} MB).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())