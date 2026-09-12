/* Confirm before a file download, and offer GitHub instead.
 *
 * Every dataset link on this site used to be a bare <a download>, so a click started a
 * save with no warning and no indication of what was arriving. That is the wrong default
 * for research artifacts: most people clicking "SEED PAIRS" want to LOOK at the data, and
 * the repo copy is better for that -- it renders in the browser, it is versioned, and it
 * can be cited by commit, which a file in someone's Downloads folder cannot.
 *
 * So: intercept, ask, and put GitHub first.
 *
 * Vanilla on purpose. Alpine happens to be on all six pages today, but this is a
 * site-wide behaviour and should not break if one page ever stops loading it.
 *
 * Usage -- mark the link and give it a repo counterpart:
 *   <a href="/seed-pairs.jsonl" download
 *      data-gh="https://github.com/.../blob/main/data/seed_pairs.jsonl"
 *      data-size="41 KB">...</a>
 *
 * data-gh is optional; without it the dialog just offers download or cancel.
 */
(function () {
  "use strict";

  var dlg = null, lastFocus = null;

  function build() {
    if (dlg) return dlg;
    dlg = document.createElement("div");
    dlg.className = "dlc-veil";
    dlg.setAttribute("hidden", "");
    dlg.innerHTML =
      '<div class="dlc-box" role="dialog" aria-modal="true" aria-labelledby="dlc-h">' +
        '<h2 id="dlc-h">DOWNLOAD THIS FILE?</h2>' +
        '<p class="dlc-what"></p>' +
        '<p class="dlc-why">The copy in the repository is versioned and can be cited by ' +
          'commit. It also renders in the browser, so you can read it without saving it.</p>' +
        '<div class="dlc-row">' +
          '<a class="btn dlc-gh" href="#" target="_blank" rel="noopener">&#9656; VIEW ON GITHUB</a>' +
          // data-dl-confirmed, or the delegated handler below catches this link too --
          // it is an a[download] like any other -- and the dialog reopens forever.
          '<a class="btn dlc-go" href="#" data-dl-confirmed>&#8681; DOWNLOAD ANYWAY</a>' +
          '<button type="button" class="mini dlc-no">CANCEL</button>' +
        '</div>' +
      '</div>';
    document.body.appendChild(dlg);

    dlg.addEventListener("click", function (e) {
      // Click the backdrop, not the box, to dismiss.
      if (e.target === dlg) close();
    });
    dlg.querySelector(".dlc-no").addEventListener("click", close);
    // Both links are real navigations; just close behind them.
    dlg.querySelector(".dlc-gh").addEventListener("click", close);
    dlg.querySelector(".dlc-go").addEventListener("click", close);
    return dlg;
  }

  function close() {
    if (!dlg) return;
    dlg.setAttribute("hidden", "");
    if (lastFocus && lastFocus.focus) lastFocus.focus();
    lastFocus = null;
  }

  function open(link) {
    var d = build();
    var name = (link.getAttribute("href") || "").split("/").pop().split("?")[0];
    var size = link.getAttribute("data-size");
    var gh = link.getAttribute("data-gh");
    var label = (link.textContent || "").replace(/^[\s↓⇩⬇️]+/, "").trim();

    d.querySelector(".dlc-what").textContent =
      label ? label + " — " + name + (size ? " (" + size + ")" : "")
            : name + (size ? " (" + size + ")" : "");

    var ghEl = d.querySelector(".dlc-gh");
    if (gh) { ghEl.href = gh; ghEl.removeAttribute("hidden"); }
    else { ghEl.setAttribute("hidden", ""); }

    var go = d.querySelector(".dlc-go");
    go.href = link.getAttribute("href");
    // Carry the original download filename through, if the link set one.
    var dn = link.getAttribute("download");
    if (dn) go.setAttribute("download", dn); else go.setAttribute("download", "");

    lastFocus = document.activeElement;
    d.removeAttribute("hidden");
    (gh ? ghEl : go).focus();
  }

  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && dlg && !dlg.hasAttribute("hidden")) close();
  });

  // Delegated, so links rendered later by Alpine are covered too.
  document.addEventListener("click", function (e) {
    var a = e.target.closest ? e.target.closest("a[download]") : null;
    if (!a) return;
    if (a.hasAttribute("data-dl-confirmed")) return;   // escape hatch
    e.preventDefault();
    open(a);
  });
})();
