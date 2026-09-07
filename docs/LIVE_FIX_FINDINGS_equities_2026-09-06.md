# Live-site fix findings — equities.dumbmodel.com — 2026-09-06

Lane: L2-equities (`weekend/live-fix-equities`, worktree off `origin/main@66d12a6a4678242ff0f7f1c84c7dc46ef96e343b`,
the commit L1 verified produced the live bytes by sha256 + git blob id + Vercel's own
`list_deployments` record).

This lane fixed the defects that a branch/commit can actually fix (D1, D2, D4 below — see the
commit on `weekend/live-fix-equities`). Two items from L1's audit are **not** fixed here because
they are not code defects a repo patch can close:

## D3 — vercel.json rewrites not honored by the live deployment (dashboard, not code)

**Evidence (from L1, re-confirmed):** `origin/main:vercel.json` declares
`/play`, `/players`, `/model`, `/trends`, `/lab`, `/dfs` → `/index.html` rewrites plus a catch-all
`/(.*)→/index.html`. All six named routes 404 live with the identical Vercel platform `NOT_FOUND`
body/headers (79 bytes, `x-vercel-error: NOT_FOUND`) as a genuinely nonexistent path
(`/totally-nonexistent-xyz123` — same body, same headers). `git log --format='%h %ci %s' origin/main
-- vercel.json` shows the file's rewrites have been in this exact form since `a2f768b`
(2026-08-21), nine days before the live production deploy (`66d12a6`, 2026-08-30) — so this is not
a stale build predating the config; the production build that is live today was built from a tree
that already carried these rewrites, and the platform is not applying them.

**Why this lane did not touch it:** there is nothing wrong in the committed `vercel.json` to patch
— the config already says what should happen. A `git push` cannot change how a live Vercel
project's routing layer is currently resolving requests; that is a property of the deployed
project/dashboard (a routing override in Project Settings, or some other platform-side
precedence issue), not of the tree. Per the L2 brief: "If the only fix is a product decision...
write docs/…FINDINGS.md and stop, as the pitch lane did" — this is the equivalent for a
dashboard-only defect. Operator action needed: check the Vercel dashboard's Project Settings for
this project (`prj_kPPvKr01iEEXXn2ObZdkmLcWwdiM`) for a routing/rewrites override, or force a clean
redeploy from `66d12a6` to see if the rewrites take effect once actually rebuilt fresh.

## D5 — `/game` fabricates a live-looking ticker tape and a fake "PASS" gate (product decision)

**Evidence (from L1):** `/game` (orphaned — not linked from the homepage nav) drives a scrolling
ticker of real tickers with `Math.random()`-generated up/down percentages every 1.1s, and a
"Rotate factors" button that increments a fake `ic` variable and displays `PASS SOTA CQS0.72` once
a threshold is crossed, with no computation behind either number. The page's own title ("Factor
Rotation Mini") signals it is a toy, but nothing on the page labels the tape or the "PASS" gate as
non-real.

**Why this lane did not touch it:** this is not a broken fetch or a dead file reference (this
lane's mandate) — it is a self-contained toy whose entire mechanic is client-side randomness by
design. The honest fix is a labeling/product decision (mark it "demo"/"illustrative", or retire the
page) that this lane is not positioned to make unilaterally, matching the L2 brief's "if a defect
is really a product decision, write findings and stop" instruction. The page is not linked from the
site's own nav, so it does not affect what a visitor following the live site's own links would see;
flagging it here so the operator can decide whether to label or retire it.

## What this lane did fix (for cross-reference — see the commit for full evidence)

- **D1** (critical, footer + inline JS comment falsely claiming a live `fetch()`-backed, "honest
  503... never fabricated" data path when the page issues zero `fetch()` calls): rewrote both to
  accurately describe the page as a deterministic, seeded generative visualization, and to name the
  two real (non-fabricated, git-tracked) files it evokes without reading.
- **D2** (roster tiles: hardcoded per-ticker numeric fields with real tickers, no backing file, no
  "illustrative" label): added an explicit "illustrative placeholder values, not sourced from a
  data file" note to the section caption.
- **D4** (`/game`'s stylesheet `<link>` pointed at `../../vector-hub/packages/vector-tokens/tokens.css`,
  a monorepo-relative path that 404s live and never existed on any ref of this repo — confirmed via
  `git log --all --diff-filter=A -- '*vector-tokens*'`): repointed it to `/assets/tokens.css`, a
  real file already committed to this repo at `assets/tokens.css` (defines the `--void`/`--ink`
  variables `/game`'s inline styles actually consume) that was simply never copied into the served
  `public/` tree; copied it there and updated the href.

No number, metric, or data value was added anywhere that a real file did not already produce.
