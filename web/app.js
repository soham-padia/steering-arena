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
    tokenCount: 0,
    result: null,
    error: null,
    submitting: false,

    async init() {
      await this.loadSeason();
      await this.loadBoards();
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
        const r = await fetch(`/leaderboard?board=${board}&limit=50`);
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
          body: JSON.stringify({ handle: this.handle, sequence: this.sequence, turnstile_token: tok }),
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
