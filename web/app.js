// Steering Arena frontend. Plain Alpine.js over the JSON API.
// SECURITY: all user-supplied content (handles, sequences) is rendered with
// x-text in index.html — never x-html/innerHTML — so stored values can't inject
// markup. Keep it that way (security audit H2). No browser storage is used.

function arena() {
  return {
    season: null,
    board: [],
    handle: "",
    sequence: "",
    result: null,
    error: null,
    submitting: false,

    async init() {
      await this.loadSeason();
      await this.loadBoard();
    },

    // Rough client-side token estimate; the server is authoritative.
    get tokenEstimate() {
      const s = this.sequence.trim();
      if (!s) return 0;
      const words = s.split(/\s+/).length;
      return Math.max(words, Math.ceil(s.length / 4));
    },

    fmtScore(v) {
      return v == null ? "—" : Number(v).toFixed(4);
    },

    async loadSeason() {
      try {
        const r = await fetch("/season");
        this.season = await r.json();
      } catch (_) {
        this.season = null;
      }
    },

    async loadBoard() {
      try {
        const r = await fetch("/leaderboard?limit=50");
        const j = await r.json();
        this.board = j.entries || [];
      } catch (_) {
        this.board = [];
      }
    },

    async submit() {
      this.error = null;
      this.result = null;
      this.submitting = true;
      try {
        const r = await fetch("/submit", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ handle: this.handle, sequence: this.sequence }),
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
          await this.loadBoard();
        }
      } catch (_) {
        this.error = "Network error — please try again.";
      } finally {
        this.submitting = false;
      }
    },
  };
}
