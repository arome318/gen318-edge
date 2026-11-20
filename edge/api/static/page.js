/* Keeps the local page current without reloading it.

   Every few seconds it asks the box for the page's content again
   (/?fragment=1) and swaps in only the parts that changed, so nothing
   flashes or jumps. New SMS are added to the end of the feed and slide in.
   No library, no network beyond the box itself. Without JavaScript the page
   falls back to a plain reload (see the <noscript> tag). */
(function () {
  "use strict";
  var root = document.documentElement;
  var every = Math.max(2, Number(root.dataset.refresh) || 3) * 1000;
  var ago = document.getElementById("ago");
  var last = Date.now();
  var ok = true;

  function tick() {
    var s = Math.max(0, Math.round((Date.now() - last) / 1000));
    if (ago) {
      ago.textContent = ok ? "Updated " + s + " s ago" : "No answer from the box for " + s + " s";
    }
    root.classList.toggle("offline", !ok);
  }

  function updateFeed(cur, next) {
    var have = Array.prototype.map.call(cur.querySelectorAll("[data-id]"), function (e) { return e.dataset.id; });
    var want = Array.prototype.slice.call(next.querySelectorAll("[data-id]"));
    var ids = want.map(function (e) { return e.dataset.id; });
    var anchor = have.length ? ids.indexOf(have[have.length - 1]) : -1;
    if (!have.length || anchor < 0) {
      // first messages ever, or the feed no longer overlaps what is shown
      if (cur.innerHTML !== next.innerHTML) {
        var first = !have.length && want.length;
        cur.innerHTML = next.innerHTML;
        if (first) {
          Array.prototype.forEach.call(cur.querySelectorAll("[data-id]"), function (e) { e.classList.add("new"); });
        }
        cur.scrollTop = cur.scrollHeight;
      }
      return;
    }
    Array.prototype.forEach.call(cur.querySelectorAll("[data-id]"), function (e) {
      if (ids.indexOf(e.dataset.id) < 0) { e.remove(); }      // dropped off the top
    });
    var added = want.slice(anchor + 1);
    added.forEach(function (e) {
      var copy = document.importNode(e, true);
      copy.classList.add("new");
      cur.appendChild(copy);
    });
    if (added.length) {
      cur.scrollTo({ top: cur.scrollHeight, behavior: "smooth" });
    }
  }

  function apply(doc) {
    Array.prototype.forEach.call(doc.querySelectorAll("[data-live]"), function (next) {
      var cur = document.getElementById(next.id);
      if (!cur) { return; }
      if (cur.className !== next.className) { cur.className = next.className; }
      if (next.id === "feed") { updateFeed(cur, next); return; }
      if (cur.innerHTML !== next.innerHTML) { cur.innerHTML = next.innerHTML; }
    });
  }

  function refresh() {
    var url = location.pathname + "?fragment=1";
    fetch(url, { cache: "no-store" })
      .then(function (r) { if (!r.ok) { throw new Error(r.status); } return r.text(); })
      .then(function (html) {
        apply(new DOMParser().parseFromString(html, "text/html"));
        last = Date.now();
        ok = true;
      })
      .catch(function () { ok = false; })
      .then(tick);
  }

  var feed = document.getElementById("feed");
  if (feed) { feed.scrollTop = feed.scrollHeight; }
  tick();
  setInterval(tick, 1000);
  setInterval(refresh, every);
})();
