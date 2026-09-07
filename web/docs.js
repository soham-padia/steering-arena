/* Two behaviours, no dependency: copy a command block, and the mobile nav drawer.
   Loaded with `defer`, so the DOM is parsed before this runs. */
(function () {
  "use strict";

  // Heading anchors. Built here rather than in the generator so the markup stays
  // clean and a reader with JS off loses only the click target, not the id.
  document.querySelectorAll(".prose h2[id], .prose h3[id]").forEach(function (h) {
    var a = document.createElement("a");
    a.className = "anchor";
    a.href = "#" + h.id;
    a.textContent = "#";
    a.setAttribute("aria-label", "Link to this section");
    h.appendChild(a);
  });

  document.querySelectorAll("figure.code button.copy").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var code = btn.closest("figure").querySelector("code");
      if (!code) return;
      var done = function () {
        btn.textContent = "copied";
        btn.classList.add("done");
        setTimeout(function () {
          btn.textContent = "copy";
          btn.classList.remove("done");
        }, 1400);
      };
      // navigator.clipboard needs a secure context; the fallback covers plain http.
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(code.innerText).then(done, function () {});
        return;
      }
      var ta = document.createElement("textarea");
      ta.value = code.innerText;
      ta.setAttribute("readonly", "");
      ta.style.position = "fixed";
      ta.style.left = "-9999px";
      document.body.appendChild(ta);
      ta.select();
      try { document.execCommand("copy"); done(); } catch (e) { /* no clipboard */ }
      document.body.removeChild(ta);
    });
  });

  var toggle = document.querySelector(".navtoggle");
  var nav = document.querySelector(".sidenav");
  if (toggle && nav) {
    toggle.addEventListener("click", function () {
      var open = nav.classList.toggle("open");
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
      toggle.textContent = open ? "Close" : "Menu";
    });
  }

  // Highlight the on-this-page entry for whichever section is in view.
  var links = Array.prototype.slice.call(
    document.querySelectorAll(".onthispage a")
  );
  if (links.length && "IntersectionObserver" in window) {
    var byId = {};
    links.forEach(function (a) { byId[a.getAttribute("href").slice(1)] = a; });
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        var a = byId[e.target.id];
        if (a) a.style.color = e.isIntersecting ? "#3ad0ff" : "";
      });
    }, { rootMargin: "-64px 0px -70% 0px" });
    Object.keys(byId).forEach(function (id) {
      var el = document.getElementById(id);
      if (el) obs.observe(el);
    });
  }
})();
