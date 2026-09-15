/**
 * Phase 6 — Blog Frontend Enhancements & Interactions
 * Dark mode | Search (Ctrl+K) | Audio Player | Code Blocks | Lightbox | Mobile TOC | Tag Filter
 * whiteicey.github.io — Vanilla JS (Zero external runtime dependencies)
 */
(function() {
    'use strict';

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

    // ── 1. Dark Mode Toggle & System Sync ──────
    function initDarkMode() {
        var html = document.documentElement;
        var btn = document.getElementById('theme-toggle');
        var icon = btn ? btn.querySelector('.theme-icon') : null;

        function syncUtterances(isDark) {
            var iframe = document.querySelector('iframe.utterances-frame');
            if (iframe && iframe.contentWindow) {
                iframe.contentWindow.postMessage({
                    type: 'set-theme',
                    theme: isDark ? 'github-dark' : 'github-light'
                }, 'https://utteranc.es');
            }
        }

        function updateIcon() {
            var isDark = html.getAttribute('data-theme') === 'dark';
            if (icon) {
                icon.textContent = isDark ? '☀️' : '🌙';
            }
            syncUtterances(isDark);
        }

        updateIcon();

        // Also attempt sync once Utterances iframe loads
        window.addEventListener('message', function(e) {
            if (e.origin === 'https://utteranc.es' && e.data && e.data.type === 'resize') {
                var isDark = html.getAttribute('data-theme') === 'dark';
                syncUtterances(isDark);
            }
        });

        if (btn) {
            btn.addEventListener('click', function(e) {
                e.preventDefault();
                if (html.getAttribute('data-theme') === 'dark') {
                    html.removeAttribute('data-theme');
                    localStorage.setItem('theme', 'light');
                } else {
                    html.setAttribute('data-theme', 'dark');
                    localStorage.setItem('theme', 'dark');
                }
                updateIcon();
            });
        }

        // Listen for OS color scheme change
        if (window.matchMedia) {
            window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e) {
                if (!localStorage.getItem('theme')) {
                    if (e.matches) {
                        html.setAttribute('data-theme', 'dark');
                    } else {
                        html.removeAttribute('data-theme');
                    }
                    updateIcon();
                }
            });
        }
    }

    // ── 2. Reading Progress Bar ──────────────────
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

    // ── 3. Back to Top Button ────────────────────
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

    // ── 4. Global Search Modal (Ctrl+K) ─────────
    function initSearch() {
        var modal = document.getElementById('search-modal');
        var input = document.getElementById('search-input');
        var resultsContainer = document.getElementById('search-results');
        var closeBtn = document.getElementById('search-close-btn');
        var toggleBtns = document.querySelectorAll('.nav-search-btn');

        if (!modal || !input || !resultsContainer) return;

        var searchIndex = null;
        var selectedIndex = -1;

        function fetchSearchIndex() {
            if (searchIndex) return;
            fetch('/search.json')
                .then(function(res) { return res.json(); })
                .then(function(data) { searchIndex = data; })
                .catch(function(err) { console.error('Failed to load search index:', err); });
        }

        function openSearch() {
            fetchSearchIndex();
            modal.classList.add('is-active');
            modal.setAttribute('aria-hidden', 'false');
            input.value = '';
            renderResults([]);
            setTimeout(function() { input.focus(); }, 80);
        }

        function closeSearch() {
            modal.classList.remove('is-active');
            modal.setAttribute('aria-hidden', 'true');
            selectedIndex = -1;
        }

        // Trigger bindings
        toggleBtns.forEach(function(btn) {
            btn.addEventListener('click', function(e) {
                e.preventDefault();
                openSearch();
            });
        });

        if (closeBtn) closeBtn.addEventListener('click', closeSearch);

        modal.addEventListener('click', function(e) {
            if (e.target === modal || e.target.classList.contains('search-modal-backdrop')) {
                closeSearch();
            }
        });

        window.addEventListener('keydown', function(e) {
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                if (modal.classList.contains('is-active')) {
                    closeSearch();
                } else {
                    openSearch();
                }
            }
            if (e.key === 'Escape' && modal.classList.contains('is-active')) {
                closeSearch();
            }
        });

        // Search Input handling
        input.addEventListener('input', function() {
            var rawQuery = input.value.trim();
            if (!rawQuery) {
                renderResults([], '');
                return;
            }
            if (!searchIndex) return;

            var tokens = rawQuery.toLowerCase().split(/\s+/).filter(Boolean);
            var matches = searchIndex.filter(function(post) {
                var title = (post.title || '').toLowerCase();
                var subtitle = (post.subtitle || '').toLowerCase();
                var tags = (post.tags || []).join(' ').toLowerCase();
                var snippet = (post.snippet || '').toLowerCase();
                var full = title + ' ' + subtitle + ' ' + tags + ' ' + snippet;
                return tokens.every(function(token) { return full.includes(token); });
            });

            renderResults(matches.slice(0, 10), rawQuery);
        });

        // Keyboard navigation in search
        input.addEventListener('keydown', function(e) {
            var items = resultsContainer.querySelectorAll('.search-item');
            if (!items.length) return;

            if (e.key === 'ArrowDown') {
                e.preventDefault();
                selectedIndex = (selectedIndex + 1) % items.length;
                updateSelection(items);
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                selectedIndex = (selectedIndex - 1 + items.length) % items.length;
                updateSelection(items);
            } else if (e.key === 'Enter' && selectedIndex >= 0 && items[selectedIndex]) {
                e.preventDefault();
                items[selectedIndex].click();
            }
        });

        function updateSelection(items) {
            items.forEach(function(el, idx) {
                if (idx === selectedIndex) {
                    el.classList.add('is-selected');
                    el.scrollIntoView({ block: 'nearest' });
                } else {
                    el.classList.remove('is-selected');
                }
            });
        }

        function escapeRegExp(str) {
            return str.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
        }

        function highlightText(text, query) {
            if (!query || !text) return text || '';
            var tokens = query.trim().split(/\s+/).filter(Boolean).map(escapeRegExp);
            if (!tokens.length) return text;
            var regex = new RegExp('(' + tokens.join('|') + ')', 'gi');
            return text.replace(regex, '<mark class="search-highlight">$1</mark>');
        }

        function extractSmartSnippet(content, query, maxLen) {
            if (!content) return '';
            maxLen = maxLen || 130;
            if (!query) return content.slice(0, maxLen) + (content.length > maxLen ? '...' : '');

            var tokens = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
            var lower = content.toLowerCase();
            var firstIdx = -1;
            for (var i = 0; i < tokens.length; i++) {
                var idx = lower.indexOf(tokens[i]);
                if (idx !== -1 && (firstIdx === -1 || idx < firstIdx)) {
                    firstIdx = idx;
                }
            }

            if (firstIdx === -1) {
                return content.slice(0, maxLen) + (content.length > maxLen ? '...' : '');
            }

            var start = Math.max(0, firstIdx - 30);
            var end = Math.min(content.length, start + maxLen);
            var prefix = start > 0 ? '...' : '';
            var suffix = end < content.length ? '...' : '';
            return prefix + content.slice(start, end).trim() + suffix;
        }

        function renderResults(posts, query) {
            selectedIndex = -1;
            if (!query) {
                resultsContainer.innerHTML = '<div class="search-empty">输入关键词以开始搜索文章...</div>';
                return;
            }
            if (!posts.length) {
                resultsContainer.innerHTML = '<div class="search-empty">未找到与 "<strong>' + query + '</strong>" 相关的文章</div>';
                return;
            }

            var html = '';
            posts.forEach(function(post) {
                var title = highlightText(post.title, query);
                var rawSnippet = post.snippet || post.subtitle || '';
                var smartSnippet = extractSmartSnippet(rawSnippet, query);
                var snippet = highlightText(smartSnippet, query);
                var tagsHtml = (post.tags || []).map(function(t) {
                    return '<span class="search-item-tag">' + t + '</span>';
                }).join('');

                html += '<a href="' + post.url + '" class="search-item">' +
                    '<div class="search-item-title">' + title + '</div>' +
                    '<div class="search-item-snippet">' + snippet + '</div>' +
                    '<div class="search-item-meta">' +
                        '<span>' + post.date + '</span>' +
                        tagsHtml +
                    '</div>' +
                '</a>';
            });
            resultsContainer.innerHTML = html;
        }
    }

    // ── 5. Custom Audio Player Card ──────────────
    function initAudioPlayers() {
        var audios = document.querySelectorAll('.post-container audio');
        if (!audios.length) return;

        audios.forEach(function(audio) {
            // Check if already transformed
            if (audio.parentNode.classList.contains('custom-audio-player')) return;

            // Extract title
            var src = audio.currentSrc || (audio.querySelector('source') ? audio.querySelector('source').src : '');
            var trackName = '文章专属伴奏';
            if (src) {
                try {
                    var decoded = decodeURIComponent(src);
                    var filename = decoded.split('/').pop().replace(/\.[^/.]+$/, '');
                    if (filename) trackName = filename;
                } catch(e) {}
            }

            var player = document.createElement('div');
            player.className = 'custom-audio-player';
            player.innerHTML = 
                '<div class="audio-player-top">' +
                    '<div class="audio-player-info-group">' +
                        '<button type="button" class="audio-play-btn" aria-label="Play audio">' +
                            '<svg viewBox="0 0 24 24" class="icon-play"><path d="M8 5v14l11-7z"/></svg>' +
                        '</button>' +
                        '<div class="audio-track-details">' +
                            '<div class="audio-track-title">' +
                                trackName +
                                '<span class="audio-track-badge">BGM</span>' +
                            '</div>' +
                            '<span class="audio-track-sub">点击播放伴读音乐 · 沉浸阅读</span>' +
                        '</div>' +
                    '</div>' +
                    '<div class="audio-wave">' +
                        '<div class="audio-wave-bar"></div>' +
                        '<div class="audio-wave-bar"></div>' +
                        '<div class="audio-wave-bar"></div>' +
                        '<div class="audio-wave-bar"></div>' +
                        '<div class="audio-wave-bar"></div>' +
                    '</div>' +
                '</div>' +
                '<div class="audio-progress-wrap">' +
                    '<div class="audio-progress-bar">' +
                        '<div class="audio-progress-fill"></div>' +
                    '</div>' +
                    '<div class="audio-time-row">' +
                        '<span class="audio-curr-time">00:00</span>' +
                        '<span class="audio-dur-time">--:--</span>' +
                    '</div>' +
                '</div>';

            audio.parentNode.insertBefore(player, audio);
            player.appendChild(audio);
            audio.style.display = 'none'; // hide native controls

            var playBtn = player.querySelector('.audio-play-btn');
            var fill = player.querySelector('.audio-progress-fill');
            var pBar = player.querySelector('.audio-progress-bar');
            var currTime = player.querySelector('.audio-curr-time');
            var durTime = player.querySelector('.audio-dur-time');

            function formatTime(s) {
                if (isNaN(s)) return '00:00';
                var m = Math.floor(s / 60);
                var sec = Math.floor(s % 60);
                return (m < 10 ? '0' : '') + m + ':' + (sec < 10 ? '0' : '') + sec;
            }

            playBtn.addEventListener('click', function() {
                if (audio.paused) {
                    audio.play();
                } else {
                    audio.pause();
                }
            });

            audio.addEventListener('play', function() {
                player.classList.add('audio-playing');
                playBtn.innerHTML = '<svg viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>';
            });

            audio.addEventListener('pause', function() {
                player.classList.remove('audio-playing');
                playBtn.innerHTML = '<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>';
            });

            audio.addEventListener('loadedmetadata', function() {
                durTime.textContent = formatTime(audio.duration);
            });

            audio.addEventListener('timeupdate', function() {
                if (audio.duration) {
                    var pct = (audio.currentTime / audio.duration) * 100;
                    fill.style.width = pct + '%';
                    currTime.textContent = formatTime(audio.currentTime);
                    durTime.textContent = formatTime(audio.duration);
                }
            });

            pBar.addEventListener('click', function(e) {
                var rect = pBar.getBoundingClientRect();
                var clickPos = (e.clientX - rect.left) / rect.width;
                if (audio.duration) {
                    audio.currentTime = clickPos * audio.duration;
                }
            });
        });
    }

    // ── 6. macOS-style Code Blocks ───────────────
    function initCodeBlocks() {
        var container = document.querySelector('.post-container');
        if (!container) return;

        var pres = container.querySelectorAll('pre');
        pres.forEach(function(pre) {
            if (pre.parentNode.classList.contains('code-block-wrapper')) return;

            // Detect language from class
            var lang = 'Code';
            var codeEl = pre.querySelector('code');
            var checkStr = (codeEl ? codeEl.className : '') + ' ' + pre.className + ' ' + (pre.parentNode.className || '');
            var match = checkStr.match(/(?:language-|highlight-)([a-zA-Z0-9_+#-]+)/);
            if (match && match[1]) {
                lang = match[1];
            }

            var wrapper = document.createElement('div');
            wrapper.className = 'code-block-wrapper';

            var header = document.createElement('div');
            header.className = 'code-mac-header';
            header.innerHTML = 
                '<div class="code-mac-dots">' +
                    '<span class="code-mac-dot red"></span>' +
                    '<span class="code-mac-dot yellow"></span>' +
                    '<span class="code-mac-dot green"></span>' +
                '</div>' +
                '<div class="code-mac-right">' +
                    '<span class="code-lang-badge">' + lang + '</span>' +
                    '<button type="button" class="copy-btn"><span>📋</span> 复制</button>' +
                '</div>';

            pre.parentNode.insertBefore(wrapper, pre);
            wrapper.appendChild(header);
            wrapper.appendChild(pre);

            var copyBtn = header.querySelector('.copy-btn');
            copyBtn.addEventListener('click', function() {
                copyCode(pre, copyBtn);
            });
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
        var originalText = btn.innerHTML;
        btn.innerHTML = '<span>✓</span> 已复制!';
        btn.classList.add('copied');
        setTimeout(function() {
            btn.innerHTML = originalText;
            btn.classList.remove('copied');
        }, 2000);
    }

    // ── 7. Image Lightbox with Caption ───────────
    function initLightbox() {
        var container = document.querySelector('.post-container');
        if (!container) return;

        var images = container.querySelectorAll('img:not(.no-lightbox)');
        images.forEach(function(img) {
            img.style.cursor = 'zoom-in';
            img.addEventListener('click', function() {
                openLightbox(this);
            });
        });
    }

    function openLightbox(img) {
        var overlay = document.createElement('div');
        overlay.className = 'lightbox-overlay';

        var content = document.createElement('div');
        content.className = 'lightbox-content';
        content.style.position = 'relative';
        content.style.textAlign = 'center';

        var clonedImg = img.cloneNode(true);
        clonedImg.style.maxWidth = '90vw';
        clonedImg.style.maxHeight = '80vh';
        clonedImg.style.borderRadius = '8px';
        clonedImg.style.cursor = 'default';
        content.appendChild(clonedImg);

        var captionText = img.alt || img.getAttribute('title');
        if (captionText) {
            var caption = document.createElement('div');
            caption.className = 'lightbox-caption';
            caption.textContent = captionText;
            caption.style.color = '#ffffff';
            caption.style.marginTop = '10px';
            caption.style.fontSize = '0.9rem';
            caption.style.fontFamily = "'Source Serif 4', Georgia, serif";
            caption.style.fontStyle = 'italic';
            content.appendChild(caption);
        }

        overlay.appendChild(content);
        document.body.appendChild(overlay);

        requestAnimationFrame(function() {
            overlay.classList.add('active');
        });

        overlay.addEventListener('click', function(e) {
            if (e.target === overlay || e.target === content) {
                closeLightbox(overlay);
            }
        });

        function onKeyDown(e) {
            if (e.key === 'Escape') closeLightbox(overlay);
        }
        document.addEventListener('keydown', onKeyDown);

        function closeLightbox(ov) {
            ov.classList.remove('active');
            document.removeEventListener('keydown', onKeyDown);
            setTimeout(function() {
                if (ov.parentNode) ov.parentNode.removeChild(ov);
            }, 300);
        }
    }

    // ── 8. Mobile Floating TOC Drawer ────────────
    function initMobileTOC() {
        var toggle = document.getElementById('mobile-toc-toggle');
        var drawer = document.getElementById('mobile-toc-drawer');
        var backdrop = document.getElementById('mobile-toc-backdrop');
        var closeBtn = document.getElementById('mobile-toc-close');
        var body = drawer ? drawer.querySelector('.mobile-toc-body') : null;
        var container = document.querySelector('.post-container');

        if (!toggle || !drawer || !body || !container) return;

        // Populate headings
        var headings = container.querySelectorAll('h1, h2, h3, h4');
        if (!headings.length) {
            toggle.style.display = 'none';
            return;
        }

        headings.forEach(function(h) {
            if (!h.id) return;
            var li = document.createElement('li');
            li.className = h.tagName.toLowerCase() + '-toc-item';
            var a = document.createElement('a');
            a.href = '#' + h.id;
            a.textContent = h.textContent.replace(/^#+\s*/, '');
            a.addEventListener('click', function() {
                closeDrawer();
            });
            li.appendChild(a);
            body.appendChild(li);
        });

        function openDrawer() {
            drawer.classList.add('is-open');
            if (backdrop) backdrop.classList.add('is-open');
        }

        function closeDrawer() {
            drawer.classList.remove('is-open');
            if (backdrop) backdrop.classList.remove('is-open');
        }

        toggle.addEventListener('click', openDrawer);
        if (closeBtn) closeBtn.addEventListener('click', closeDrawer);
        if (backdrop) backdrop.addEventListener('click', closeDrawer);
    }

    // ── 9. Tags Dynamic Filtering & Sort by Count ────────────────
    function initTagsFilter() {
        var filterBar = document.getElementById('tag_filter_bar');
        if (!filterBar) return;

        var allPill = filterBar.querySelector('.tag-pill[data-tag="all"]');
        var otherPills = Array.prototype.slice.call(filterBar.querySelectorAll('.tag-pill:not([data-tag="all"])'));

        // Sort tag pills by post count descending
        otherPills.sort(function(a, b) {
            var countA = parseInt(a.querySelector('.tag-pill__count') ? a.querySelector('.tag-pill__count').textContent : '0', 10);
            var countB = parseInt(b.querySelector('.tag-pill__count') ? b.querySelector('.tag-pill__count').textContent : '0', 10);
            if (countB !== countA) return countB - countA;
            return a.getAttribute('data-tag').localeCompare(b.getAttribute('data-tag'));
        });

        // Re-append sorted pills to DOM
        if (allPill) filterBar.appendChild(allPill);
        otherPills.forEach(function(p) {
            filterBar.appendChild(p);
        });

        // Sort tag section containers in DOM by count descending
        var postsWrapper = document.querySelector('.tag-posts-wrapper');
        if (postsWrapper) {
            var sections = Array.prototype.slice.call(postsWrapper.querySelectorAll('.one-tag-list'));
            sections.sort(function(a, b) {
                var countA = a.querySelectorAll('.post-preview').length;
                var countB = b.querySelectorAll('.post-preview').length;
                if (countB !== countA) return countB - countA;
                return (a.getAttribute('data-tag-section') || '').localeCompare(b.getAttribute('data-tag-section') || '');
            });
            sections.forEach(function(s) {
                postsWrapper.appendChild(s);
            });
        }

        // Bind interactive filter click
        var allPills = filterBar.querySelectorAll('.tag-pill');
        var allSections = document.querySelectorAll('.one-tag-list');

        allPills.forEach(function(pill) {
            pill.addEventListener('click', function(e) {
                var targetTag = pill.getAttribute('data-tag');
                if (!targetTag) return;

                allPills.forEach(function(p) { p.classList.remove('active'); });
                pill.classList.add('active');

                if (targetTag === 'all') {
                    allSections.forEach(function(s) { s.classList.remove('is-hidden'); });
                } else {
                    allSections.forEach(function(s) {
                        if (s.getAttribute('data-tag-section') === targetTag) {
                            s.classList.remove('is-hidden');
                        } else {
                            s.classList.add('is-hidden');
                        }
                    });
                }
            });
        });
    }

    // ── 10. GitHub-style Callouts / Alerts ──────
    function initCallouts() {
        var container = document.querySelector('.post-container');
        if (!container) return;

        var typeMap = {
            'note': {
                title: 'Note',
                svg: '<svg class="callout-icon" viewBox="0 0 16 16" fill="currentColor"><path d="M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8Zm8-6.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13ZM6.5 7.75A.75.75 0 0 1 7.25 7h1a.75.75 0 0 1 .75.75v2.75h.25a.75.75 0 0 1 0 1.5h-2a.75.75 0 0 1 0-1.5h.25v-2h-.25a.75.75 0 0 1-.75-.75ZM8 6a1 1 0 1 1 0-2 1 1 0 0 1 0 2Z"/></svg>'
            },
            'tip': {
                title: 'Tip',
                svg: '<svg class="callout-icon" viewBox="0 0 16 16" fill="currentColor"><path d="M8 1.5c-2.363 0-4 1.69-4 3.75 0 .984.424 1.625.984 2.304l.214.253c.223.264.47.556.673.91.316.551.459 1.137.459 1.916a.75.75 0 0 0 1.5 0c0-.62-.11-1.02-.323-1.391a4.25 4.25 0 0 0-.583-.787l-.234-.275C6.17 7.26 5.5 6.438 5.5 5.25c0-1.285 1.077-2.25 2.5-2.25s2.5.965 2.5 2.25c0 1.188-.67 2.01-1.19 2.572l-.233.275a4.25 4.25 0 0 0-.583.787c-.213.371-.323.77-.323 1.391a.75.75 0 0 0 1.5 0c0-.779.143-1.365.459-1.916.203-.354.45-.646.673-.91l.214-.253c.56-.679.984-1.32.984-2.304 0-2.06-1.637-3.75-4-3.75ZM5.75 12.5a.75.75 0 0 0 0 1.5h4.5a.75.75 0 0 0 0-1.5h-4.5Zm1 2.5a.75.75 0 0 0 0 1.5h2.5a.75.75 0 0 0 0-1.5h-2.5Z"/></svg>'
            },
            'important': {
                title: 'Important',
                svg: '<svg class="callout-icon" viewBox="0 0 16 16" fill="currentColor"><path d="M0 1.75C0 .784.784 0 1.75 0h12.5C15.216 0 16 .784 16 1.75v9.5A1.75 1.75 0 0 1 14.25 13H9.06l-2.573 2.573A1.458 1.458 0 0 1 4 14.543V13H1.75A1.75 1.75 0 0 1 0 11.25v-9.5Zm1.75-.25a.25.25 0 0 0-.25.25v9.5c0 .138.112.25.25.25h2.5a.75.75 0 0 1 .75.75v2.19l2.72-2.72a.749.749 0 0 1 .53-.22h6a.25.25 0 0 0 .25-.25v-9.5a.25.25 0 0 0-.25-.25H1.75ZM8 3a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-1.5 0v-3.5A.75.75 0 0 1 8 3Zm0 6.5a1 1 0 1 1 0-2 1 1 0 0 1 0 2Z"/></svg>'
            },
            'warning': {
                title: 'Warning',
                svg: '<svg class="callout-icon" viewBox="0 0 16 16" fill="currentColor"><path d="M6.457 1.047c.659-1.234 2.427-1.234 3.086 0l6.082 11.378A1.75 1.75 0 0 1 14.082 15H1.918a1.75 1.75 0 0 1-1.543-2.575Zm1.763.707a.25.25 0 0 0-.44 0L1.698 13.132a.25.25 0 0 0 .22.368h12.164a.25.25 0 0 0 .22-.368Zm.53 3.996v2.5a.75.75 0 0 1-1.5 0v-2.5a.75.75 0 0 1 1.5 0ZM9 11a1 1 0 1 1-2 0 1 1 0 0 1 2 0Z"/></svg>'
            },
            'caution': {
                title: 'Caution',
                svg: '<svg class="callout-icon" viewBox="0 0 16 16" fill="currentColor"><path d="M4.47.047A.75.75 0 0 1 5 0h6a.75.75 0 0 1 .53.22l4.25 4.25c.141.14.22.331.22.53v6a.75.75 0 0 1-.22.53l-4.25 4.25A.75.75 0 0 1 11 16H5a.75.75 0 0 1-.53-.22L.22 11.53A.75.75 0 0 1 0 11V5a.75.75 0 0 1 .22-.53L4.47.047Zm.53 1.453L1.5 5.5v5l3.5 3.5h5l3.5-3.5v-5L10.5 1.5h-5ZM8 4a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-1.5 0v-3.5A.75.75 0 0 1 8 4Zm0 6.5a1 1 0 1 1 0-2 1 1 0 0 1 0 2Z"/></svg>'
            }
        };

        var blockquotes = container.querySelectorAll('blockquote');
        blockquotes.forEach(function(bq) {
            var firstP = bq.querySelector('p');
            if (!firstP) return;

            var text = firstP.innerHTML;
            var match = text.match(/^\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\](?:\s*<br\s*\/?>|\s*\n)?/i);
            if (!match) return;

            var rawType = match[1].toLowerCase();
            var conf = typeMap[rawType] || typeMap['note'];

            bq.classList.add('callout', 'callout-' + rawType);

            firstP.innerHTML = text.replace(/^\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\](?:\s*<br\s*\/?>|\s*\n)?/i, '').trim();
            if (!firstP.innerHTML) {
                firstP.remove();
            }

            var titleDiv = document.createElement('div');
            titleDiv.className = 'callout-title';
            titleDiv.innerHTML = conf.svg + '<span>' + conf.title + '</span>';
            bq.insertBefore(titleDiv, bq.firstChild);
        });
    }

    // ── 11. Nav Scroll Enhancement ───────────────
    function initNavScroll() {
        var nav = document.querySelector('.navbar-custom');
        if (!nav) return;
        window.addEventListener('scroll', throttle(function() {
            nav.classList.toggle('is-scrolled', window.scrollY > 20);
        }, 50), { passive: true });
    }

    // ── 12. Post Card Scroll Reveal ──────────────
    function initScrollReveal() {
        var cards = document.querySelectorAll('.post-card');
        if (!cards.length) return;

        if (!('IntersectionObserver' in window)) {
            cards.forEach(function(c) { c.classList.add('revealed'); });
            return;
        }

        var observer = new IntersectionObserver(function(entries) {
            entries.forEach(function(e) {
                if (e.isIntersecting) {
                    e.target.classList.add('revealed');
                    observer.unobserve(e.target);
                }
            });
        }, { threshold: 0.08 });

        cards.forEach(function(c) { observer.observe(c); });
    }

    // ── Initialize Everything ────────────────────
    function initAll() {
        initDarkMode();
        initProgressBar();
        initBackToTop();
        initSearch();
        initAudioPlayers();
        initCodeBlocks();
        initLightbox();
        initMobileTOC();
        initTagsFilter();
        initCallouts();
        initNavScroll();
        initScrollReveal();
    }

    window.initCallouts = initCallouts;

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initAll);
    } else {
        initAll();
    }
})();
