// Steering Arena frontend. Plain Alpine.js over the JSON API.
// SECURITY: all user-supplied content (handles, sequences) is rendered with
// x-text in index.html — never x-html/innerHTML — so stored values can't inject
// markup. Keep it that way (security audit H2). No browser storage is used.

function arena() {
  return {
    season: null,
    tab: "pro",
    boards: { pro: [], anti: [] },
    handle: "",
    sequence: "",
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
          await this.loadBoards();
        }
      } catch (_) {
        this.error = "Network error — please try again.";
      } finally {
        this.submitting = false;
      }
    },
  };
}
