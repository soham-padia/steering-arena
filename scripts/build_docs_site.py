#!/usr/bin/env python3
"""Render the docs/ markdown set into web/docs/ as a static documentation site.

Run from the repo root with the `steering-arena` env active:

    python scripts/build_docs_site.py

It writes web/docs/**/*.html, web/docs.css, web/docs.js, and refreshes the docs
entries in web/sitemap.xml. Output is COMMITTED and served by the existing
`StaticFiles(directory=web, html=True)` catch-all mount in app/main.py, so there
is no new server route, no template engine in the request path, and no markdown
dependency in the served image. markdown-it-py is a build dependency only; it is
pinned in requirements-research.txt, which the Dockerfile never installs.

Three things this does that a generic markdown renderer does not, each because of
how the docs set is actually written:

1. **Backticked paths become links.** The set contains zero markdown links — every
   cross-reference is a bare backticked path like `data/analysis/prefix_eval_s3.md`
   or `app/scoring.py:cosine`. Rendered naively the published site is a dead end.
   A path is linked only if `git ls-files` tracks it, so gitignored CLAUDE.md and
   STRUCTURE.md stay unlinked, which is what docs/README.md promises the reader.
2. **Commands and output are styled apart.** The house pattern is a ```bash fence
   followed by an unlabelled fence holding real output. Only the command gets a
   copy button; output is dimmer and marked as such, so a reader can tell at a
   glance what to type and what to expect.
3. **The mode is a badge.** "Pick one mode per page and never mix modes" is the
   set's organising rule, and the mode lives in the path. The badge surfaces it.

Failure modes: the script exits non-zero if a nav entry names a missing file, if a
page has no H1, or if two pages resolve to the same output path. It does not fail
on an unresolvable backticked path — that is normal prose, not an error.
"""
from __future__ import annotations

import filecmp
import hashlib
import html
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "docs"
SITE = "https://sohampadianeu-steering-arena.hf.space"
GITHUB = "https://github.com/soham-padia/steering-arena/blob/main"

# Reading order, not alphabetical. Mirrors docs/README.md's own routing: the
# repr trap comes first under how-to because it has corrupted five live rows.
NAV: list[tuple[str, str, str, list[str]]] = [
    ("Start here", "", "", ["README.md"]),
    ("Tutorials", "tutorial", "learn by doing", [
        "tutorials/score-a-sequence-yourself.md",
        "tutorials/rebuild-a-published-number.md",
    ]),
    ("How-to", "howto", "one task each", [
        "how-to/move-a-searched-string-without-corrupting-it.md",
        "how-to/search-a-banded-objective-with-gcg.md",
        "how-to/run-a-blind-behavioural-eval.md",
        "how-to/run-a-job-on-the-aicr-cluster.md",
        "how-to/open-a-new-season.md",
        "how-to/verify-the-public-surface.md",
    ]),
    ("Reference", "reference", "lookup", [
        "reference/scoring.md",
        "reference/seasons.md",
        "reference/artifacts.md",
        "reference/withdrawn-claims.md",
        "reference/environments.md",
    ]),
    ("Explanation", "explanation", "the why", [
        "explanation/what-the-behavioural-tests-establish.md",
        "explanation/why-the-metric-is-shaped-this-way.md",
        "explanation/the-case-for-the-metric.md",
    ]),
    ("The dated record", "record", "frozen, not maintained", [
        "EXTRACTION.md",
        "DEPLOY.md",
        "HANDOFF_BEHAVIORAL_S3.md",
        "PREFIX_BLIND_RUBRIC.md",
    ]),
]

MODE_LABEL = {
    "tutorial": "TUTORIAL",
    "howto": "HOW-TO",
    "reference": "REFERENCE",
    "explanation": "EXPLANATION",
    "record": "DATED RECORD",
    "": "OVERVIEW",
}

EXT = "py|md|json|sql|npz|txt|html|css|js|jsonl|sh|sbatch|yml|yaml|png|svg|toml|cfg|ini"


def tracked_files() -> tuple[dict[str, str], dict[str, str]]:
    """Every path that will exist in a clone, plus unambiguous basenames.

    Excluding gitignored paths is the point: it keeps a reference to CLAUDE.md or
    STRUCTURE.md from being linked to a 404, which is what docs/README.md promises
    the reader, and it doubles as the link check.

    `--cached --others --exclude-standard` rather than plain `ls-files` so the
    output does not depend on what happens to be staged. With the bare form,
    `git add` of a new script silently changes which paths render as links, and a
    rebuild after staging produces different HTML than one before it.
    """
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    paths = {p: p for p in out if p}
    base: dict[str, str | None] = {}
    for p in paths:
        name = p.rsplit("/", 1)[-1]
        base[name] = None if name in base else p
    return paths, {k: v for k, v in base.items() if v}


PATHS, BASENAMES = tracked_files()


def docs_href(path: str) -> str:
    rel = path[len("docs/"):]
    return "/docs/" if rel == "README.md" else "/docs/" + rel[:-3] + ".html"


def resolve(text: str) -> tuple[str, bool] | None:
    """Map a backticked string to (href, is_external), or None if it is not a path."""
    t = text.strip()
    t = re.sub(r"\s+§.*$", "", t)          # `docs/PREFIX_BLIND_RUBRIC.md §3`
    t = re.sub(r"^[(\[]|[)\]:,.]$", "", t)
    line = None
    m = re.match(rf"^(.+?\.(?:{EXT})):(\d+)[-\u2013](\d+)$", t)
    if m:                                  # `extract_direction.py:124\u2013128`
        t, line = m.group(1), f"{m.group(2)}-L{m.group(3)}"
    else:
        m = re.match(rf"^(.+?\.(?:{EXT})):(\d+)$", t)
        if m:
            t, line = m.group(1), m.group(2)   # `_falsifier/verify.py:31` -> #L31
        else:
            m = re.match(rf"^(.+?\.(?:{EXT})):[A-Za-z_][\w.]*$", t)
            if m:
                t = m.group(1)                 # `app/scoring.py:cosine`
    if not re.search(rf"\.(?:{EXT})$", t):
        return None
    # docs/README.md refers to its siblings docs-relatively ("reference/scoring.md"),
    # so try that prefix before giving up on an otherwise unknown path.
    p = PATHS.get(t) or PATHS.get("docs/" + t) or BASENAMES.get(t)
    if not p:
        return None
    if p.startswith("docs/") and p.endswith(".md"):
        return docs_href(p), False
    return f"{GITHUB}/{p}" + (f"#L{line}" if line else ""), True


def slug(text: str) -> str:
    s = re.sub(r"`|\*\*|\*", "", text).strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    return re.sub(r"[\s_]+", "-", s).strip("-") or "section"


def render(md_text: str) -> tuple[str, list[tuple[int, str, str]]]:
    """Return (html, toc) where toc is a list of (level, slug, text)."""
    md = MarkdownIt("commonmark").enable("table").enable("strikethrough")
    tokens = md.parse(md_text)
    toc: list[tuple[int, str, str]] = []
    seen: dict[str, int] = {}

    # --- heading anchors, recorded for the on-this-page rail -------------------
    for i, tok in enumerate(tokens):
        if tok.type == "heading_open":
            raw = tokens[i + 1].content
            s = slug(raw)
            if s in seen:
                seen[s] += 1
                s = f"{s}-{seen[s]}"
            else:
                seen[s] = 1
            tok.attrSet("id", s)
            lvl = int(tok.tag[1])
            if lvl in (2, 3):
                toc.append((lvl, s, re.sub(r"`|\*\*", "", raw)))

    # --- callout class from the blockquote's first line ------------------------
    for i, tok in enumerate(tokens):
        if tok.type != "blockquote_open":
            continue
        first = next((t.content for t in tokens[i + 1:i + 5] if t.type == "inline"), "")
        kind = ("note" if first.startswith("**I did not run this.")
                else "corrected" if first.startswith("**CORRECTED")
                else "trap" if first.startswith("**TRAP")
                else "")
        if kind:
            tok.attrJoin("class", f"callout callout-{kind}")

    rules = md.renderer.rules

    def code_inline(tokens_, idx, opts, env):
        content = tokens_[idx].content
        esc = html.escape(content, quote=False)
        hit = resolve(content)
        if not hit:
            return f"<code>{esc}</code>"
        href, external = hit
        extra = ' target="_blank" rel="noopener"' if external else ""
        cls = "xref xref-out" if external else "xref"
        return f'<a class="{cls}" href="{href}"{extra}><code>{esc}</code></a>'

    def fence(tokens_, idx, opts, env):
        tok = tokens_[idx]
        lang = (tok.info or "").strip().split()[0] if tok.info.strip() else ""
        body = html.escape(tok.content, quote=False)
        if lang:
            return (
                f'<figure class="code" data-lang="{html.escape(lang)}">'
                f'<figcaption><span class="lang">{html.escape(lang)}</span>'
                f'<button class="copy" type="button" aria-label="Copy this block">copy</button>'
                f'</figcaption>'
                f'<pre><code class="language-{html.escape(lang)}">{body}</code></pre></figure>'
            )
        return ('<figure class="code out"><figcaption><span class="lang">output</span>'
                f'</figcaption><pre><code>{body}</code></pre></figure>')

    def table_open(tokens_, idx, opts, env):
        return '<div class="table-scroll"><table>'

    def table_close(tokens_, idx, opts, env):
        return "</table></div>"

    rules["code_inline"] = code_inline
    rules["fence"] = fence
    rules["table_open"] = table_open
    rules["table_close"] = table_close
    return md.renderer.render(tokens, md.options, {}), toc


def title_and_lede(md_text: str) -> tuple[str, str]:
    title, lede = "", ""
    lines = md_text.split("\n")
    for i, ln in enumerate(lines):
        if ln.startswith("# "):
            title = ln[2:].strip()
            for nxt in lines[i + 1:]:
                s = nxt.strip()
                if not s or s.startswith(("#", "```", "|", ">", "-", "*", "**Prereq")):
                    continue
                lede = re.sub(r"`|\*\*|\*", "", s)
                break
            break
    return title, lede


def sidebar(current: str) -> str:
    parts = ['<nav class="sidenav" aria-label="Documentation">']
    for group, mode, hint, files in NAV:
        parts.append(f'<p class="navgroup"><span>{group}</span>'
                     + (f'<i>{hint}</i>' if hint else "") + "</p><ul>")
        for f in files:
            src = ROOT / "docs" / f
            t, _ = title_and_lede(src.read_text())
            on = " class=\"on\"" if f == current else ""
            label = "Overview" if f == "README.md" else t
            parts.append(f'<li{on}><a href="{docs_href("docs/" + f)}">{html.escape(label)}</a></li>')
        parts.append("</ul>")
    parts.append('<p class="navgroup"><span>Elsewhere</span></p><ul>'
                 '<li><a href="/">The arena ↗</a></li>'
                 '<li><a href="/limitations.html">Limitations ↗</a></li>'
                 f'<li><a href="{GITHUB.replace("/blob/main", "")}" target="_blank" '
                 'rel="noopener">GitHub ↗</a></li></ul>')
    parts.append("</nav>")
    return "\n".join(parts)


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title} — Steering Arena docs</title>
<meta name="description" content="{desc}" />
<link rel="canonical" href="{canon}" />
<meta name="robots" content="index, follow" />
<meta name="author" content="Soham Padia" />
<meta property="og:type" content="article" />
<meta property="og:site_name" content="Steering Arena" />
<meta property="og:title" content="{title} — Steering Arena docs" />
<meta property="og:description" content="{desc}" />
<meta property="og:url" content="{canon}" />
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap" rel="stylesheet" />
<link rel="stylesheet" href="/docs.css?v={ver}" />
<script defer src="/docs.js?v={ver}"></script>
</head>
<body class="docs">
<a class="skip" href="#main">Skip to content</a>

<header class="docbar">
  <div class="docbar-in">
    <a class="brand" href="/"><span class="coin" aria-hidden="true">●</span> STEERING ARENA</a>
    <span class="crumb">/ docs</span>
    <button class="navtoggle" type="button" aria-expanded="false">Menu</button>
  </div>
</header>

<div class="shell">
  {sidebar}
  <main id="main" class="prose">
    <p class="eyebrow"><span class="badge badge-{mode}">{modelabel}</span>{edit}</p>
    {body}
    <hr class="foot-rule" />
    <footer class="pagefoot">
      <p>Source: <a href="{ghsrc}" target="_blank" rel="noopener"><code>{srcpath}</code></a>.
      If this page and a committed <code>.json</code> disagree, the JSON wins and the page is
      a bug.</p>
      <p class="muted">Steering Arena · <a href="/">the board</a> ·
      <a href="/docs/">docs index</a></p>
    </footer>
  </main>
  {toc}
</div>
</body>
</html>
"""


def toc_html(toc: list[tuple[int, str, str]]) -> str:
    if len(toc) < 3:
        return '<div class="onthispage"></div>'
    items = "".join(
        f'<li class="l{lvl}"><a href="#{s}">{html.escape(t)}</a></li>' for lvl, s, t in toc
    )
    return ('<aside class="onthispage"><p class="navgroup"><span>On this page</span></p>'
            f"<ul>{items}</ul></aside>")

def last_modified() -> str:
    """Date of the most recent commit, for sitemap <lastmod>.

    Taken from git rather than the clock so two people building the same commit
    publish the same sitemap.
    """
    r = subprocess.run(["git", "log", "-1", "--format=%cs"], cwd=ROOT,
                       capture_output=True, text=True)
    return r.stdout.strip() or "1970-01-01"


def update_sitemap(pages: list[str]) -> int:
    """Replace the /docs/ URLs in web/sitemap.xml, leaving the hand-written ones alone.

    Rewritten rather than appended so a renamed or deleted page cannot leave a
    stale URL advertised to crawlers.
    """
    sm = ROOT / "web" / "sitemap.xml"
    text = re.sub(r"[ \t]*<url>\s*<loc>[^<]*/docs/[^<]*</loc>.*?</url>\n", "",
                  sm.read_text(), flags=re.S)
    mod = last_modified()
    urls = []
    for href in pages:
        prio = "0.9" if href == "/docs/" else "0.6"
        urls.append(f"  <url>\n    <loc>{SITE}{href}</loc>\n"
                    f"    <lastmod>{mod}</lastmod>\n"
                    f"    <changefreq>monthly</changefreq>\n"
                    f"    <priority>{prio}</priority>\n  </url>\n")
    sm.write_text(text.replace("</urlset>", "".join(urls) + "</urlset>"))
    return len(urls)


def asset_version() -> str:
    """Cache-busting stamp derived from content, not from the clock.

    A date stamp would make every rebuild rewrite all 20 pages, which buries a
    real docs change in churn and makes `--check` impossible. A content hash
    changes exactly when something a browser caches changes.
    """
    h = hashlib.sha256()
    for rel in ("web/docs.css", "web/docs.js", "scripts/build_docs_site.py"):
        h.update((ROOT / rel).read_bytes())
    return h.hexdigest()[:8]


def build(target: Path) -> tuple[int, list[str]]:
    """Render every nav page into `target`. Returns (exit code, hrefs)."""
    ver = asset_version()
    seen: dict[Path, str] = {}
    hrefs: list[str] = []

    for _group, mode, _hint, files in NAV:
        for f in files:
            src = ROOT / "docs" / f
            if not src.exists():
                print(f"FAIL nav names a missing file: docs/{f}", file=sys.stderr)
                return 1, hrefs
            text = src.read_text()
            title, lede = title_and_lede(text)
            if not title:
                print(f"FAIL no H1 in docs/{f}", file=sys.stderr)
                return 1, hrefs
            body, toc = render(text)
            href = docs_href("docs/" + f)
            rel = "index.html" if f == "README.md" else f[:-3] + ".html"
            dest = target / rel
            if dest in seen:
                print(f"FAIL two pages collide at {rel}: {seen[dest]} and docs/{f}",
                      file=sys.stderr)
                return 1, hrefs
            seen[dest] = f"docs/{f}"
            hrefs.append(href)
            page = PAGE.format(
                title=html.escape(title), desc=html.escape((lede or title)[:200], quote=True),
                canon=SITE + href, ver=ver, sidebar=sidebar(f),
                mode=mode or "overview", modelabel=MODE_LABEL[mode],
                body=body, toc=toc_html(toc),
                srcpath=f"docs/{f}", ghsrc=f"{GITHUB}/docs/{f}",
                edit="" if mode != "record" else
                '<span class="frozen">frozen at its date \u2014 kept quotable, '
                'not maintained</span>',
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(page)
    return 0, hrefs


def check() -> int:
    """Exit non-zero if web/docs/ is not what the current markdown would produce.

    Catches the failure this setup invites: someone edits a page under docs/,
    commits, deploys, and the published site still serves the old HTML because
    nothing rebuilt it. Run it in the same breath as the tests.
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rc, _ = build(tmp)
        if rc:
            return rc
        fresh = {p.relative_to(tmp) for p in tmp.rglob("*.html")}
        on_disk = {p.relative_to(OUT) for p in OUT.rglob("*.html")} if OUT.exists() else set()
        stale = sorted(str(x) for x in on_disk - fresh)
        missing = sorted(str(x) for x in fresh - on_disk)
        differ = sorted(str(r) for r in fresh & on_disk
                        if not filecmp.cmp(tmp / r, OUT / r, shallow=False))
        for label, items in (("no longer generated", stale),
                             ("never built", missing),
                             ("out of date", differ)):
            for i in items:
                print(f"STALE ({label}): web/docs/{i}")
        if stale or missing or differ:
            sys.stdout.flush()
            print("\nweb/docs/ is stale. Run: python scripts/build_docs_site.py",
                  file=sys.stderr)
            return 1
    print(f"web/docs/ is up to date with docs/ ({len(fresh)} pages)")
    return 0


if __name__ == "__main__":
    if "--check" in sys.argv[1:]:
        sys.exit(check())
    OUT.mkdir(parents=True, exist_ok=True)
    rc, pages = build(OUT)
    if rc:
        sys.exit(rc)
    n = update_sitemap(pages)
    print(f"wrote {len(pages)} pages under web/docs/, {n} docs URLs in web/sitemap.xml")
