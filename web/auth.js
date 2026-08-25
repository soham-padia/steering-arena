/* Shared Supabase sign-in for the arena.
 *
 * Talks to Supabase Auth's REST endpoints directly rather than pulling in the SDK: no
 * third-party script, and — the reason that matters here — no localStorage session. The
 * access token lives in the Alpine component and dies on reload, which keeps the
 * project's no-browser-storage rule intact.
 *
 * Two ways in, because Supabase's default OTP email is a magic LINK, not a code:
 * paste the 6-digit token, or click the link and land back here with the session in the
 * URL fragment (which we wipe immediately so it is not left in history).
 */
function arenaAuth() {
  return {
    cfg: { enabled: false },
    authStep: "email",
    email: "",
    code: "",
    token: "",
    authBusy: false,
    authErr: "",

    async initAuth() {
      try { this.cfg = await (await fetch("/admin/config")).json(); }
      catch (_) { this.cfg = { enabled: false }; }
      const h = new URLSearchParams((location.hash || "").replace(/^#/, ""));
      const tok = h.get("access_token");
      if (tok) {
        history.replaceState(null, "", location.pathname);
        this.token = tok;
        if (this.onSignedIn) await this.onSignedIn();
      }
      return this.cfg;
    },

    async _auth(path, body) {
      const base = (this.cfg.supabase_url || "").replace(/\/$/, "");
      const r = await fetch(`${base}/auth/v1/${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json", apikey: this.cfg.anon_key },
        body: JSON.stringify(body),
      });
      const j = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(j.msg || j.error_description || j.error || "Sign-in failed.");
      return j;
    },

    async sendCode() {
      this.authBusy = true; this.authErr = "";
      try {
        await this._auth("otp", {
          email: this.email,
          create_user: true,
          redirect_to: location.origin + location.pathname,
        });
        this.authStep = "code";
      } catch (e) { this.authErr = String(e.message || e); }
      finally { this.authBusy = false; }
    },

    async verifyCode() {
      this.authBusy = true; this.authErr = "";
      try {
        const j = await this._auth("verify", { email: this.email, token: this.code, type: "email" });
        this.token = j.access_token || "";
        if (!this.token) throw new Error("No session returned.");
        this.authStep = "in";
        if (this.onSignedIn) await this.onSignedIn();
      } catch (e) { this.authErr = String(e.message || e); }
      finally { this.authBusy = false; }
    },

    signOut() {
      this.token = ""; this.code = ""; this.authStep = "email"; this.authErr = "";
      if (this.onSignedOut) this.onSignedOut();
    },
  };
}
