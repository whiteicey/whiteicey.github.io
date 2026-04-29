/**
 * Phase 2+3 — Blog Reading Experience Enhancements
 * Dark mode toggle | Progress bar | Back to top | Code copy | Image lightbox
 * Hux Blog Jekyll theme — vanilla JS, no dependencies
 */
(function() {
    'use strict';

    // ── Dark Mode Toggle ──────────────────────
    function initDarkMode() {
        var html = document.documentElement;
        var btn = document.getElementById('theme-toggle');
        var icon = btn ? btn.querySelector('.theme-icon') : null;
        if (!btn) return;

        var saved = localStorage.getItem('theme');
        if (saved === 'dark') {
            html.setAttribute('data-theme', 'dark');
            if (icon) icon.innerHTML = '&#9788;';
        }

        btn.addEventListener('click', function() {
            if (html.getAttribute('data-theme') === 'dark') {
                html.removeAttribute('data-theme');
                localStorage.setItem('theme', 'light');
                if (icon) icon.innerHTML = '&#9789;';
            } else {
                html.setAttribute('data-theme', 'dark');
                localStorage.setItem('theme', 'dark');
                if (icon) icon.innerHTML = '&#9788;';
            }
        });
    }

    // ── Utility: throttle ─────────────────────
    function throttle(fn, delay) {
        var last = 0;
        return function() {
            var now = Date.now();
            if (now - last >= delay) {
                last = now;
                fn.apply(null, arguments);
            }
        };
    }

    // ── Reading Progress Bar ──────────────────
    function initProgressBar() {
        var bar = document.getElementById('reading-progress');
        if (!bar) return;

        var ticking = false;

        function update() {
            var scrollTop = window.pageYOffset || document.documentElement.scrollTop;
            var docHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
            var pct = docHeight > 0 ? Math.min((scrollTop / docHeight) * 100, 100) : 0;
            bar.style.width = pct + '%';
            ticking = false;
        }

        window.addEventListener('scroll', function() {
            if (!ticking) {
                requestAnimationFrame(update);
                ticking = true;
            }
        }, { passive: true });
    }

    // ── Back to Top Button ────────────────────
    function initBackToTop() {
        var btn = document.getElementById('back-to-top');
        if (!btn) return;

        var onScroll = throttle(function() {
            var scrollTop = window.pageYOffset || document.documentElement.scrollTop;
            if (scrollTop > 300) {
                btn.classList.add('visible');
            } else {
                btn.classList.remove('visible');
            }
        }, 100);

        window.addEventListener('scroll', onScroll, { passive: true });

        btn.addEventListener('click', function() {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    }

    // ── Code Copy Buttons ─────────────────────
    function initCodeCopy() {
        var container = document.querySelector('.post-container');
        if (!container) return;

        var pres = container.querySelectorAll('pre');
        for (var i = 0; i < pres.length; i++) {
            wrapCodeBlock(pres[i]);
        }
    }

    function wrapCodeBlock(pre) {
        // Don't double-wrap
        if (pre.parentNode.classList.contains('code-block-wrapper')) return;

        var wrapper = document.createElement('div');
        wrapper.className = 'code-block-wrapper';

        var btn = document.createElement('button');
        btn.className = 'copy-btn';
        btn.textContent = 'Copy';

        pre.parentNode.insertBefore(wrapper, pre);
        wrapper.appendChild(pre);
        wrapper.appendChild(btn);

        btn.addEventListener('click', function() {
            copyCode(pre, btn);
        });
    }

    function copyCode(pre, btn) {
        var text = pre.textContent;

        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text).then(function() {
                showCopied(btn);
            }).catch(function() {
                fallbackCopy(text, btn);
            });
        } else {
            fallbackCopy(text, btn);
        }
    }

    function fallbackCopy(text, btn) {
        var textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        try {
            document.execCommand('copy');
            showCopied(btn);
        } catch (e) {
            btn.textContent = 'Error';
        }
        document.body.removeChild(textarea);
    }

    function showCopied(btn) {
        var originalText = btn.textContent;
        btn.textContent = 'Copied!';
        btn.classList.add('copied');
        setTimeout(function() {
            btn.textContent = originalText;
            btn.classList.remove('copied');
        }, 2000);
    }

    // ── Image Lightbox ─────────────────────────
    function initLightbox() {
        var container = document.querySelector('.post-container');
        if (!container) return;

        var images = container.querySelectorAll('img:not(.no-lightbox)');
        for (var i = 0; i < images.length; i++) {
            images[i].addEventListener('click', function() {
                openLightbox(this);
            });
        }
    }

    function openLightbox(img) {
        var overlay = document.createElement('div');
        overlay.className = 'lightbox-overlay';

        var clonedImg = img.cloneNode(true);
        clonedImg.style.cursor = 'default';
        overlay.appendChild(clonedImg);

        document.body.appendChild(overlay);

        // Trigger transition on next frame
        requestAnimationFrame(function() {
            overlay.classList.add('active');
        });

        // Close on overlay click (not on image click)
        overlay.addEventListener('click', function(e) {
            if (e.target === overlay) {
                closeLightbox(overlay);
            }
        });

        // Close on Escape
        function onKeyDown(e) {
            if (e.key === 'Escape') {
                closeLightbox(overlay);
            }
        }
        document.addEventListener('keydown', onKeyDown);

        function closeLightbox(ov) {
            ov.classList.remove('active');
            document.removeEventListener('keydown', onKeyDown);
            setTimeout(function() {
                if (ov.parentNode) {
                    ov.parentNode.removeChild(ov);
                }
            }, 300); // match CSS transition duration
        }
    }

    // ── Init all on DOM ready ─────────────────
    function initAll() {
        initDarkMode();
        initProgressBar();
        initBackToTop();
        initCodeCopy();
        initLightbox();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initAll);
    } else {
        initAll();
    }
})();
