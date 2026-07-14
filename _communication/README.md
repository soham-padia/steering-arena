# _communication — append-only agent/human collaboration log

A lightweight channel for collaborators (and their coding agents, e.g. Claude Code) to
exchange findings on this project asynchronously, with full provenance, in the repo.

## Protocol

1. **One file per message.** Name: `NNN-YYYY-MM-DD-author-topic.md` (NNN = next number).
2. **Append-only, non-rewritable.** Never edit or delete an existing message file.
   Corrections go in a NEW message that references the old one. A PR that modifies an
   existing file in this folder will be rejected; since each message is its own file,
   "append-only" is trivially auditable in any diff.
3. **Header required** at the top of each message:
   `author:` (human name), `agent:` (agent/model used, if any), `date:`, `re:` (message
   number it replies to, if any).
4. **Humans review before merge.** Agents draft; a human commits/opens the PR. External
   collaborators send messages as PRs from a fork touching ONLY this folder.
5. Keep data small and inline (tables, token ids, short JSON). Big artifacts go in the
   repo proper or a linked gist; reference them here.

Started 2026-07-14 by Soham Padia (with Claude Code) to coordinate the score-reproduction
investigation with Jesse Li (also using a Claude-based workflow).
