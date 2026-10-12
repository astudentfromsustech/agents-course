/* Course contents sidebar: toggle, remember, and open a fold when you jump into it. */
(function () {
  var aside = document.getElementById('toc');
  var btn   = document.getElementById('tocBtn');
  if (!aside || !btn) return;

  var KEY = 'agents.toc.open';
  function set(open) {
    aside.setAttribute('data-open', open ? '1' : '0');
    btn.setAttribute('aria-expanded', open ? 'true' : 'false');
    try { localStorage.setItem(KEY, open ? '1' : '0'); } catch (e) {}
  }
  var saved = '0';
  try { saved = localStorage.getItem(KEY) || '0'; } catch (e) {}
  set(saved === '1');

  btn.addEventListener('click', function () {
    set(aside.getAttribute('data-open') !== '1');
  });

  // "c" toggles, Escape closes — but never while typing in a field.
  document.addEventListener('keydown', function (e) {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    var t = e.target.tagName;
    if (t === 'INPUT' || t === 'TEXTAREA' || e.target.isContentEditable) return;
    if (e.key === 'c') set(aside.getAttribute('data-open') !== '1');
    if (e.key === 'Escape') set(false);
  });

  // A target inside a closed <details> is display:none, so the browser cannot
  // scroll to it. Open every enclosing fold first, then scroll.
  function reveal(hash) {
    if (!hash || hash.length < 2) return;
    var el;
    try { el = document.getElementById(decodeURIComponent(hash.slice(1))); } catch (e) { return; }
    if (!el) return;
    var p = el.parentElement;
    while (p) {
      if (p.tagName === 'DETAILS') p.open = true;
      p = p.parentElement;
    }
    requestAnimationFrame(function () {
      el.scrollIntoView({ block: 'start', behavior: 'smooth' });
    });
  }
  window.addEventListener('hashchange', function () { reveal(location.hash); });
  if (location.hash) setTimeout(function () { reveal(location.hash); }, 0);

  // Mark the section you are reading.
  var links = [].slice.call(aside.querySelectorAll('.toc-sec a'));
  if (!links.length) return;
  var targets = links.map(function (a) {
    try { return document.getElementById(decodeURIComponent(a.hash.slice(1))); } catch (e) { return null; }
  });
  // Highlight the heading you are UNDER, not merely one that happens to be on screen.
  // A heading intersecting the band is best; otherwise fall back to the last one scrolled
  // past, so the marker never blanks out in the middle of a long section.
  var BAND = 90;           // px from the top that counts as "reading here"
  var current = -1;
  function paint() {
    var active = -1;
    for (var i = 0; i < targets.length; i++) {
      var t = targets[i];
      if (!t) continue;
      if (t.getBoundingClientRect().top <= BAND) active = i;   // last one passed
    }
    if (active < 0) active = 0;
    if (active === current) return;
    current = active;
    links.forEach(function (a, i) { a.classList.toggle('on', i === active); });
    // keep the marker visible when the list is longer than the panel
    var el = links[active];
    var panel = document.getElementById('tocPanel');
    if (el && panel && panel.scrollHeight > panel.clientHeight) {
      var r = el.getBoundingClientRect(), pr = panel.getBoundingClientRect();
      if (r.top < pr.top + 8 || r.bottom > pr.bottom - 8) {
        panel.scrollTop += (r.top - pr.top) - panel.clientHeight / 2 + r.height / 2;
      }
    }
  }
  // Throttle on the clock, not on requestAnimationFrame: rAF is throttled or paused
  // in a backgrounded tab, which freezes the marker. paint() measures ~10 elements,
  // so calling it directly is cheap enough.
  var last = 0, timer = null;
  function onScroll() {
    var now = Date.now();
    if (now - last >= 80) { last = now; paint(); return; }
    if (timer) return;
    timer = setTimeout(function () { timer = null; last = Date.now(); paint(); }, 80);
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll, { passive: true });
  // folds opening/closing move every heading below them
  document.addEventListener('toggle', onScroll, true);
  paint();
})();
