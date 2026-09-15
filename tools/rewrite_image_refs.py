#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rewrite original raster image references to optimized WebP derivatives."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / 'img'
VARIANTS = {'hero': 'hero', 'card': 'card', 'body': 'body'}
SKIP_BASENAMES = {
    'favicon.ico', 'apple-touch-icon.png', 'holo-touch-icon.jpg',
    'avatar_g.jpg', 'avatar_m.jpg',
}


def derivative_stem(source: Path) -> str:
    stem = source.stem
    if (IMG / f'{stem}.jpg').exists() and (IMG / f'{stem}.png').exists():
        return f"{stem}-{source.suffix.lstrip('.').lower()}"
    return stem


def build_mapping() -> dict[str, dict[str, str]]:
    mapping = {}
    for source in IMG.iterdir():
        if not source.is_file() or source.name in SKIP_BASENAMES:
            continue
        if source.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue
        stem = derivative_stem(source)
        mapping[f'img/{source.name}'] = {
            variant: f'img/optimized/{variant}/{stem}.webp'
            for variant in VARIANTS
        }
    return mapping


def replace_header_img(line: str, mapping: dict[str, dict[str, str]]) -> str:
    match = re.match(r'^(\s*header-img:\s*["\']?)([^"\'\s]+)(["\']?\s*)$', line)
    if not match:
        return line
    original = match.group(2).lstrip('/')
    if original in mapping:
        return match.group(1) + mapping[original]['hero'] + match.group(3)
    return line


def replace_config_avatar(line: str, mapping: dict[str, dict[str, str]]) -> str:
    match = re.match(r'^(\s*sidebar-avatar:\s*["\']?)([^"\'\s]+)(["\']?\s*)$', line)
    if not match:
        return line
    original = match.group(2).lstrip('/')
    if original in mapping:
        return match.group(1) + mapping[original]['body'] + match.group(3)
    return line


def rewrite_markdown(path: Path, mapping: dict[str, dict[str, str]]) -> int:
    lines = path.read_text(encoding='utf-8').splitlines()
    changed = 0
    in_fence = False
    for i, line in enumerate(lines):
        if line.lstrip().startswith('```'):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        original = line
        if re.match(r'^\s*header-img:', line):
            line = replace_header_img(line, mapping)
        else:
            for old, variants in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
                line = line.replace('/' + old, '/' + variants['body'])
        if line != original:
            lines[i] = line
            changed += 1
    text = '\n'.join(lines) + '\n'
    text = re.sub(r'\{: loading="lazy"(?: decoding="async")?\}', '{: loading="lazy" decoding="async"}', text)
    text = re.sub(r'(!\[[^\]]*\]\([^)]*\))(?!\{:\s*loading=)', r'\1{: loading="lazy" decoding="async"}', text)
    path.write_text(text, encoding='utf-8', newline='')
    return changed


def rewrite_simple_file(path: Path, mapping: dict[str, dict[str, str]]) -> bool:
    text = path.read_text(encoding='utf-8')
    original = text
    lines = []
    for line in text.splitlines():
        if re.match(r'^\s*header-img:', line):
            line = replace_header_img(line, mapping)
        elif re.match(r'^\s*sidebar-avatar:', line):
            line = replace_config_avatar(line, mapping)
        lines.append(line)
    text = '\n'.join(lines) + ('\n' if not text.endswith('\n') else '')
    if text != original:
        path.write_text(text, encoding='utf-8', newline='')
        return True
    return False


def main() -> int:
    mapping = build_mapping()
    changed_posts = 0
    for post in (ROOT / '_posts').glob('*.md'):
        if rewrite_markdown(post, mapping):
            changed_posts += 1

    changed_files = 0
    for path in [ROOT / '_config.yml', *ROOT.glob('*.html')]:
        if rewrite_simple_file(path, mapping):
            changed_files += 1

    print(f'updated posts: {changed_posts}')
    print(f'updated config/pages: {changed_files}')
    print(f'available mappings: {len(mapping)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())