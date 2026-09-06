# weekend/live-fix-equities

**What and why.** `origin/main`'s `public/index.html` (the exact live tree, verified by L1
via sha256 + git blob id + Vercel's own `list_deployments` record for `66d12a6`) claims in
its footer and an inline JS comment to be bound to real data with an "honest 503 if
missing... never fabricated" fallback — but the page issues zero `fetch()`/XHR/`import()`
calls (grep-confirmed): the canvas map is a client-side seeded LCG generator over 5
hardcoded cluster centers, and the 12 roster tiles' per-ticker AR/stretch and either/depth
numbers are hardcoded JS literals with no backing file. Separately, the orphaned `/game`
page's stylesheet link pointed at a sibling-repo path that never existed on any ref of this
repo and 404s live.

**Measured evidence (before/after, local repro of the live tree).**
- `curl / | grep -c "never fabricated"` — before: 2, after: 0.
- `curl / | grep -c "illustrative placeholder"` — before: 0, after: 2 (footer + roster
  caption).
- `curl /game -L | grep tokens.css href` — before:
  `../../vector-hub/packages/vector-tokens/tokens.css` (a sibling-repo path; `git log --all
  --diff-filter=A -- '*vector-tokens*'` shows only the commit that introduced `/game`
  itself ever referenced it); after: `/assets/tokens.css`.
- `curl <old resolved path>` — 404 both before and after (still correctly dead); `curl
  /assets/tokens.css` — after: 200, 1479 bytes, matching the sha of the repo's own
  `assets/tokens.css`.

**Fixes.**
- `public/index.html` footer + JS comment: replaced the false real-data-binding claim with
  an accurate description (deterministic seeded generative visualization; names the two
  real, git-tracked files it evokes without reading).
- `public/index.html` roster caption: added "illustrative placeholder values, not sourced
  from a data file" for the per-tile AR/stretch and either/depth numbers.
- `public/game/index.html` + `public/assets/tokens.css`: repointed the dead cross-repo href
  to a real file already committed to this repo (`assets/tokens.css`, defines the
  `--void`/`--ink` vars the page's inline style consumes) that was simply never copied
  into `public/`.

No number, metric, or data value was added anywhere that a real file did not already
produce.

**Verified, and how.**
- Tests: 29/29 passed both before and after (2 unrelated test files failed to collect
  before and after too — `ModuleNotFoundError: vector_bench` / `vector_core`, pre-existing
  missing packages not present anywhere in this checkout, unrelated to this diff).
- Worktree branched off `origin/main@66d12a6a4678242ff0f7f1c84c7dc46ef96e343b` (the commit
  L1 verified produced the live bytes). Home checkout `C:\Users\jcdav\vector-equities` was
  never written to (`git status --porcelain` empty before and after; only
  status/log/show/ls-tree/worktree-add run there).

**Explicitly NOT done** (`docs/LIVE_FIX_FINDINGS_equities_2026-09-06.md`):
- `vercel.json`'s `/model`,`/play`,`/players`,`/trends`,`/lab`,`/dfs` rewrites are not
  honored by the live deployment even though the committed config has had them since
  2026-08-21, 9 days before the 2026-08-30 production build — a dashboard/platform routing
  issue, not something a commit can fix.
- `/game`'s `Math.random()` ticker tape and fake "PASS SOTA CQS0.72" gate — a labeling/
  product decision on an orphaned, unlinked toy page, not a broken fetch or dead file ref.

**Merge target and blocker.** Base: `origin/main` (`66d12a6a`) — **not** `master`; this is
the branch that actually matches what's live. 1 commit ahead, clean. This IS the branch
that reaches equities.dumbmodel.com if merged. No git-level blocker. Contrast
`weekend/fix-model-page-assets`, which is based on `master` (a different, currently
non-production tree) and does not reach the live site even after merging.
