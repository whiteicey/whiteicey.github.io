/*!
 * Native theme runtime for whiteicey.github.io
 * Replaces the legacy jQuery/Bootstrap runtime while preserving site behavior.
 */
(function () {
  'use strict';

  function onReady(callback) {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', callback);
    } else {
      callback();
    }
  }

  // Make post tables responsive without jQuery.
  function enhanceTables() {
    document.querySelectorAll('.post-container table').forEach(function (table) {
      if (table.closest('.table-responsive')) return;
      var wrapper = document.createElement('div');
      wrapper.className = 'table-responsive';
      table.parentNode.insertBefore(wrapper, table);
      wrapper.appendChild(table);
      table.classList.add('table');
    });
  }

  // Constrain video embeds without Bootstrap JS.
  function enhanceEmbeds() {
    document.querySelectorAll('iframe[src*="youtube.com"], iframe[src*="vimeo.com"]').forEach(function (iframe) {
      if (iframe.closest('.embed-responsive')) return;
      var wrapper = document.createElement('div');
      wrapper.className = 'embed-responsive embed-responsive-16by9';
      iframe.parentNode.insertBefore(wrapper, iframe);
      wrapper.appendChild(iframe);
      iframe.classList.add('embed-responsive-item');
    });
  }

  // Hide the navbar while scrolling down and reveal it while scrolling up.
  function initNavbarScroll() {
    var navbar = document.querySelector('.navbar-custom');
    if (!navbar) return;

    var previousTop = 0;
    var ticking = false;

    function update() {
      var currentTop = window.scrollY;
      var headerHeight = navbar.offsetHeight;

      if (currentTop < previousTop) {
        if (currentTop > 0 && navbar.classList.contains('is-fixed')) {
          navbar.classList.add('is-visible');
        } else {
          navbar.classList.remove('is-visible', 'is-fixed');
        }
      } else {
        navbar.classList.remove('is-visible');
        if (currentTop > headerHeight && !navbar.classList.contains('is-fixed')) {
          navbar.classList.add('is-fixed');
        }
      }

      previousTop = currentTop;
      ticking = false;
    }

    window.addEventListener('scroll', function () {
      if (!ticking && window.innerWidth > 1170) {
        requestAnimationFrame(update);
        ticking = true;
      }
    }, { passive: true });
  }

  onReady(function () {
    enhanceTables();
    enhanceEmbeds();
    initNavbarScroll();
  });
})();
