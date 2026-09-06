// Steering Arena frontend. Plain Alpine.js over the JSON API.
// SECURITY: all user-supplied content (handles, sequences) is rendered with
// x-text in index.html — never x-html/innerHTML — so stored values can't inject
// markup. Keep it that way (security audit H2). No browser storage is used.

function arena() {
  return {
    season: null,
    tab: "pro",
    boards: { pro: [], anti: [] },
    collapsed: true,
    collapseN: 5,
    expandedSeq: {},   // per-row sequence expand state (keyed by sequence text)
    handle: "",
    sequence: "",
    consent: true,   // research-use opt-out: ticked by default, user can untick
    tokenCount: 0,
    result: null,
    error: null,
    submitting: false,

    // Season switcher. `viewSeasonId === null` means "the active season", which is also
    // what /leaderboard returns with no ?season param. Archived seasons are read-only:
    // /submit always targets whatever is active server-side, so the form is hidden while
    // viewing one rather than silently posting to a different board than you are looking at.
    seasons: [],
    viewSeasonId: null,

    get viewSeason() {
      if (this.viewSeasonId == null) return this.season;
      return this.seasons.find((s) => s.id === this.viewSeasonId) || this.season;
    },
    get viewingArchive() {
      return this.viewSeasonId != null && this.viewSeasonId !== this.season?.id;
    },

    // "Inactive" is not the same as "archived": a season with a HIGHER id than the live
    // one has not opened yet. Calling an unopened season archived would be wrong in the
    // one place a visitor is most likely to be confused.
    seasonState(s) {
      if (s.active) return "";
      return s.id > (this.season?.id ?? 0) ? "preview" : "closed";
    },
    get viewStateLabel() {
      const s = this.viewSeason;
      if (!s || s.active) return "";
      return this.seasonState(s) === "preview"
        ? "NOT YET OPEN — PREVIEW"
        : "CLOSED — READ ONLY";
    },

    async selectSeason(id) {
      this.viewSeasonId = id;
      this.collapsed = true;
      this.syncUrl();
      await this.loadBoards();
    },

    // Reflect the viewed season in the URL so a board is linkable and survives a reload.
    // replaceState, not pushState: switching boards is not navigation, and stacking
    // history entries would make Back feel broken. No storage is used (see the rule at
    // the top of this file) — the URL is the only place this lives.
    syncUrl() {
      const u = new URL(window.location);
      if (this.viewSeasonId == null || this.viewSeasonId === this.season?.id) {
        u.searchParams.delete("season");
      } else {
        u.searchParams.set("season", this.viewSeasonId);
      }
      window.history.replaceState({}, "", u);
    },

    async init() {
      await this.loadSeason();
      await this.loadSeasons();
      // ?season=N deep link — what the season bar on every other page links to.
      // Ignored unless it names a real season, so a stale or hand-edited link falls
      // back to the live board instead of showing an empty one.
      const want = parseInt(new URLSearchParams(window.location.search).get("season"), 10);
      if (!Number.isNaN(want) && this.seasons.some((s) => s.id === want)) {
        this.viewSeasonId = want;
      }
      await this.loadBoards();
    },

    async loadSeasons() {
      try {
        const r = await fetch("/seasons");
        const j = await r.json();
        // Seasons with no submissions (the id=1 scaffold) would be dead tabs.
        this.seasons = (j.seasons || []).filter((s) => s.d_version !== "v0-stub");
      } catch (_) { this.seasons = []; }
    },

    // Rows for the currently selected board.
    get entries() {
      return this.boards[this.tab] || [];
    },

    // Collapsed view shows only the top N; toggle expands to the full board.
    get visibleEntries() {
      return this.collapsed ? this.entries.slice(0, this.collapseN) : this.entries;
    },

    // Long sequences are clamped per row; clicking the cell expands/collapses it.
    toggleSeq(seq) {
      this.expandedSeq[seq] = !this.expandedSeq[seq];
    },

    // Real token count from the model tokenizer (matches server enforcement).
    async updateTokens() {
      if (!this.sequence.trim()) { this.tokenCount = 0; return; }
      try {
        const r = await fetch("/tokenize?text=" + encodeURIComponent(this.sequence));
        const j = await r.json();
        if (j.tokens != null) this.tokenCount = j.tokens;
      } catch (_) { /* keep previous count */ }
    },

    get overBudget() {
      return this.tokenCount > (this.season?.token_budget ?? 20);
    },

    fmtScore(v) {
      return v == null ? "—" : Number(v).toFixed(4);
    },

    // Direction-specificity z (closed-form; |z| ≤ √hidden ≈ 71.6). "—" = scored
    // before the metric existed and not yet backfilled.
    fmtSpec(v) {
      return v == null ? "—" : Number(v).toFixed(1);
    },

    // Copy a citation block to the clipboard; flash the button label.
    copyCite(id, ev) {
      const text = document.getElementById(id)?.textContent || "";
      const btn = ev?.currentTarget;
      const restore = btn ? btn.textContent : "";
      const done = (ok) => {
        if (btn) { btn.textContent = ok ? "✓ COPIED" : "COPY FAILED"; setTimeout(() => { btn.textContent = restore; }, 1500); }
      };
      navigator.clipboard?.writeText(text).then(() => done(true), () => done(false));
    },

    async loadSeason() {
      try {
        const r = await fetch("/season");
        this.season = await r.json();
        // Load Cloudflare Turnstile only when CAPTCHA is enabled for the season.
        if (this.season?.captcha_sitekey && !document.getElementById("cf-turnstile-js")) {
          const s = document.createElement("script");
          s.id = "cf-turnstile-js";
          s.src = "https://challenges.cloudflare.com/turnstile/v0/api.js";
          s.async = true;
          s.defer = true;
          document.head.appendChild(s);
        }
      } catch (_) {
        this.season = null;
      }
    },

    async fetchBoard(board) {
      try {
        const q = this.viewSeasonId == null ? "" : `&season=${encodeURIComponent(this.viewSeasonId)}`;
        const r = await fetch(`/leaderboard?board=${board}&limit=1000${q}`);
        const j = await r.json();
        return j.entries || [];
      } catch (_) {
        return [];
      }
    },

    async loadBoards() {
      const [pro, anti] = await Promise.all([this.fetchBoard("pro"), this.fetchBoard("anti")]);
      this.boards = { pro, anti };
    },

    async submit() {
      this.error = null;
      this.result = null;
      // Re-check the real token count right before sending (the live counter is
      // debounced, so a fast click could otherwise slip an over-budget sequence
      // through to the server). Block locally — no harsh "GAME OVER" round-trip.
      await this.updateTokens();
      if (this.overBudget) return;  // red counter + disabled button already convey it
      this.submitting = true;
      try {
        const tok = document.querySelector('[name="cf-turnstile-response"]')?.value || "";
        const r = await fetch("/submit", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ handle: this.handle, sequence: this.sequence, turnstile_token: tok, consent: this.consent }),
        });
        const j = await r.json();
        if (!r.ok) {
          let msg = j.error || "Submission failed.";
          if (j.score != null) {
            msg += ` (existing score ${this.fmtScore(j.score)}, rank ${j.rank}).`;
          }
          this.error = msg;
        } else {
          this.result = j;
          await this.loadBoards();
        }
      } catch (_) {
        this.error = "Network error — please try again.";
      } finally {
        this.submitting = false;
        // Turnstile tokens are single-use — refresh for the next submission
        // (managed mode re-issues silently, so no page refresh / re-click needed).
        if (window.turnstile) { try { window.turnstile.reset(); } catch (_) {} }
      }
    },
  };
}
