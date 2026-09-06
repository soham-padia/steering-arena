// Shared season context bar for the non-arena pages.
//
// Deliberately NOT arena() — that component carries submit, tokenize, captcha and
// consent state which a methodology page has no use for. This fetches one endpoint and
// renders one strip.
//
// SECURITY: same rule as app.js — every value renders through x-text, never x-html.
// No browser storage.

function seasonBar() {
  return {
    seasons: [],
    live: null,

    async init() {
      try {
        const r = await fetch("/seasons");
        const j = await r.json();
        // The v0-stub scaffold has no submissions; it would render as a dead link.
        this.seasons = (j.seasons || []).filter((s) => s.d_version !== "v0-stub");
        this.live = this.seasons.find((s) => s.active) || null;
      } catch (_) {
        this.seasons = [];
      }
    },

    // "Inactive" is not "archived": a season with a higher id than the live one has not
    // opened yet. Mislabelling an unopened season as archived is wrong in exactly the
    // place a reader is most likely to be confused.
    state(s) {
      if (s.active) return "live";
      return s.id > (this.live?.id ?? 0) ? "preview" : "archived";
    },

    // Layer(s) for a season: the band when it has one, the single layer otherwise.
    depth(s) {
      return s.layers ? "layers " + s.layers : "layer " + s.layer;
    },
  };
}
