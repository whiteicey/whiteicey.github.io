#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit local and remote image references used by the Jekyll source."""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent
IMG_EXT = re.compile(r'\.(jpe?g|png|webp|gif|ico|svg)(?:[?#].*)?$', re.I)
MD_IMG = re.compile(r'!\[[^\]]*\]\(([^)\s]+)')
SRC = re.compile(r'''(?:src|href)\s*=\s*["']([^"']+)["']''')
YAML_IMG = re.compile(r'''(?m)^\s*(?:header-img|sidebar-avatar)\s*:\s*["']?([^\s"']+)''')

SCAN = [
    ROOT / '_posts',
    ROOT / '_layouts',
    ROOT / '_includes',
]
SCAN_FILES = [ROOT / '_config.yml'] + list(ROOT.glob('*.html'))


def normalize(url: str) -> str | None:
    url = url.strip()
    if url.startswith(('http://', 'https://', '//', 'data:')):
        return url
    path = unquote(url.split('?')[0].split('#')[0]).lstrip('/')
    if not path.startswith('img/'):
        return None
    return path


def iter_source_files():
    for directory in SCAN:
        if directory.exists():
            yield from directory.glob('*')
    yield from SCAN_FILES


def collect_refs():
    refs = []
    for file in iter_source_files():
        if not file.is_file():
            continue
        try:
            lines = file.read_text(encoding='utf-8').splitlines()
        except UnicodeDecodeError:
            lines = file.read_text(encoding='utf-8', errors='replace').splitlines()
        in_fence = False
        for line_no, line in enumerate(lines, 1):
            if line.lstrip().startswith('```'):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            matches = []
            matches.extend((m.group(1), 'markdown') for m in MD_IMG.finditer(line))
            matches.extend((m.group(1), 'html') for m in SRC.finditer(line))
            if file.suffix.lower() in {'.yml', '.yaml', '.html', '.md'}:
                matches.extend((m.group(1), 'front-matter') for m in YAML_IMG.finditer(line))
            for url, kind in matches:
                norm = normalize(url)
                if norm and (IMG_EXT.search(norm) or norm.startswith('img/')):
                    refs.append({
                        'file': file.relative_to(ROOT).as_posix(),
                        'line': line_no,
                        'raw': url,
                        'normalized': norm,
                        'kind': kind,
                    })
    return refs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', type=Path, default=ROOT / 'tools' / 'image_refs.json')
    args = ap.parse_args()

    refs = collect_refs()
    by_url = defaultdict(list)
    for ref in refs:
        by_url[ref['normalized']].append(ref)

    missing = []
    for url, occurrences in sorted(by_url.items()):
        if url.startswith(('http://', 'https://', '//', 'data:')):
            continue
        if not (ROOT / url).exists():
            missing.append({'url': url, 'refs': [f"{r['file']}:{r['line']}" for r in occurrences]})

    report = {
        'total_refs': len(refs),
        'distinct_urls': len(by_url),
        'local_urls': sum(not u.startswith(('http', '//', 'data:')) for u in by_url),
        'remote_urls': sum(u.startswith(('http', '//')) for u in by_url),
        'missing': missing,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({'report': report, 'refs': refs}, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f"total refs     : {report['total_refs']}")
    print(f"distinct urls  : {report['distinct_urls']}")
    print(f"local / remote : {report['local_urls']} / {report['remote_urls']}")
    print(f"missing files  : {len(missing)}")
    for item in missing:
        print(f"  {item['url']} <- {item['refs'][0]}")
    print(f"report         : {args.json}")
    return 1 if missing else 0


if __name__ == '__main__':
    raise SystemExit(main())