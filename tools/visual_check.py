#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build-independent visual and layout verification against _site/."""
import json
import threading
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / '_site'
OUT = ROOT / 'tools' / 'visual-check'
OUT.mkdir(parents=True, exist_ok=True)


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


def serve():
    server = ThreadingHTTPServer(('127.0.0.1', 8092), partial(QuietHandler, directory=str(SITE)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main():
    server = serve()
    base = 'http://127.0.0.1:8092'
    pages = [
        ('home', '/'),
        ('post', '/2026/08/21/%E6%9C%AA%E6%9D%A5%E8%BF%98%E6%98%AF%E6%9C%AA%E6%9D%A5/'),
        ('about', '/about/'),
        ('tags', '/tags/'),
        ('not-found', '/404.html'),
    ]
    report = {'pages': [], 'console_errors': [], 'page_errors': [], 'failed_requests': [], 'broken_images': [], 'layout_issues': []}

    with sync_playwright() as p:
        browser = p.chromium.launch(channel='chrome', headless=True)
        context = browser.new_context(
            viewport={'width': 1440, 'height': 900},
            device_scale_factor=1,
            service_workers='block',
        )

        def bind(page, name):
            page.on('console', lambda msg: report['console_errors'].append({'page': name, 'type': msg.type, 'text': msg.text}) if msg.type == 'error' else None)
            page.on('pageerror', lambda err: report['page_errors'].append({'page': name, 'text': str(err)}))
            page.on('requestfailed', lambda req: report['failed_requests'].append({'page': name, 'url': req.url, 'failure': req.failure}))

        for name, path in pages:
            for theme in ('light', 'dark'):
                page = context.new_page()
                bind(page, name)
                page.add_init_script(f"localStorage.setItem('theme', '{theme}')")
                response = page.goto(base + path, wait_until='networkidle', timeout=30000)
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(650)
                broken = page.evaluate("""() => Array.from(document.images)
                    .filter(img => img.complete && img.naturalWidth === 0)
                    .map(img => img.currentSrc || img.src)""")
                report['broken_images'].extend({'page': name, 'theme': theme, 'src': src} for src in broken)
                layout = page.evaluate("""() => ({
                    title: document.title,
                    theme: document.documentElement.getAttribute('data-theme') || 'light',
                    horizontalOverflow: document.documentElement.scrollWidth > window.innerWidth,
                    hiddenCards: document.querySelectorAll('.post-card:not(.revealed)').length,
                    visibleCardImages: Array.from(document.querySelectorAll('.post-card__img')).filter(img => img.complete && img.naturalWidth > 0).length,
                    navHeight: document.querySelector('.navbar-custom') ? document.querySelector('.navbar-custom').getBoundingClientRect().height : null
                })""")
                if layout['horizontalOverflow']:
                    report['layout_issues'].append({'page': name, 'theme': theme, 'issue': 'horizontal_overflow'})
                if layout['hiddenCards']:
                    report['layout_issues'].append({'page': name, 'theme': theme, 'issue': 'hidden_cards', 'count': layout['hiddenCards']})
                report['pages'].append({'page': name, 'theme': theme, 'status': response.status if response else None, **layout})
                page.screenshot(path=str(OUT / f'{name}-{theme}-desktop.png'), full_page=False)
                page.close()

        mobile = browser.new_context(
            viewport={'width': 390, 'height': 844},
            device_scale_factor=2,
            is_mobile=True,
            has_touch=True,
            service_workers='block',
        )
        page = mobile.new_page()
        bind(page, 'home-mobile')
        page.add_init_script("localStorage.setItem('theme', 'light')")
        page.goto(base + '/', wait_until='networkidle', timeout=30000)
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(650)
        layout = page.evaluate("""() => ({
            title: document.title,
            theme: document.documentElement.getAttribute('data-theme') || 'light',
            horizontalOverflow: document.documentElement.scrollWidth > window.innerWidth,
            hiddenCards: document.querySelectorAll('.post-card:not(.revealed)').length,
            visibleCardImages: Array.from(document.querySelectorAll('.post-card__img')).filter(img => img.complete && img.naturalWidth > 0).length,
            navHeight: document.querySelector('.navbar-custom') ? document.querySelector('.navbar-custom').getBoundingClientRect().height : null
        })""")
        if layout['horizontalOverflow']:
            report['layout_issues'].append({'page': 'home', 'theme': 'mobile', 'issue': 'horizontal_overflow'})
        report['pages'].append({'page': 'home', 'theme': 'mobile-light', 'status': 200, **layout})
        page.screenshot(path=str(OUT / 'home-light-mobile.png'), full_page=False)
        page.close(); mobile.close(); browser.close()

    server.shutdown(); server.server_close()
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = {
        'pages_checked': len(report['pages']),
        'console_errors': len(report['console_errors']),
        'page_errors': len(report['page_errors']),
        'failed_requests': len(report['failed_requests']),
        'broken_images': len(report['broken_images']),
        'layout_issues': len(report['layout_issues']),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'output: {OUT}')
    return 0 if not any(report[k] for k in ('console_errors', 'page_errors', 'failed_requests', 'broken_images', 'layout_issues')) else 1

if __name__ == '__main__':
    raise SystemExit(main())