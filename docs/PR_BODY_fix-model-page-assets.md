# weekend/fix-model-page-assets (vector-equities)

**What and why.** `model.html`'s `loadManifest()` fetched `assets/manifest.json`, but that
path is the PWA app manifest (icons/display/start_url) — a filename collision with the
real model-metadata file, which has always lived at `assets/manifest_v6_real.json` (model,
dim, rows, tickers, tower_list, real_features, etc). The fetch succeeded (200) so no error
ever surfaced; it just silently rendered blank/undefined fields in the "Current deployed"
panel.

**Measured evidence.**
- Also removed `<script src="assets/mtnn.js">` and `<script src="assets/network-viz.js">` —
  verified via `git log --all --diff-filter=A` that neither file has ever existed in any
  branch of this repo (copy-pasted from vector-hoops, which does have a real MTNN onnx
  viewer; vector-equities never built one). Both 404'd on every page load.
- After the fix, `manifest_v6_real.json` returns
  `.model=equities_mtnn_v6_real_d64_towers20_ic0.5066_calibrated_isotonic_bias<1%`,
  `rows=2741`, `tickers=283`. `eval_scoreboard.json` (path was already correct) confirms
  `purity@10=0.7057`, `ic_rank_6m=0.007`, `computed_at=2026-08-05T03:13:45Z`, matching the
  operator's cited 2026-08-05 measurement exactly.

**Verified, and how.**
- Served the exact tree that would ship (no `outputDirectory` in this branch's
  `vercel.json`, so served root = repo root) without deploying:
  - `python -m http.server --directory <home checkout> 8902` (before, read-only):
    `/assets/mtnn.js` 404, `/assets/network-viz.js` 404, `/assets/manifest.json` 200 but
    `.model=None`/`.tower_list=None`.
  - `python -m http.server --directory <worktree> 8903` (after): every `<script src>` and
    `fetch()` target in the served `model.html` now resolves 200 from that same origin —
    error-boundary.js, keyboard-a11y.js, site-nav.js, pwa-install.js,
    eval_scoreboard.json, feature_manifest_v6_real.json, manifest_v6_real.json,
    real_data.json, real_pca.json.
- Full test suite: `python -m pytest -q` → 32 passed, exit 0.

**Explicitly NOT done** (pre-existing, separate issues, flagged not touched):
- `assets/mtnn_meta.json` (referenced only in `sw.js`'s dead `FULL_MTNN` array, never
  fetched by any page): no real artifact by that name was ever produced for this repo
  either — an aspirational reference copied from hoops, nothing to relocate.
- `model.html:576` and static prose at `:331` fall back to hardcoded `full_history`
  numbers (7370/7310/60/59/21/21) that do not match
  `assets/universe_full_history_manifest.json`'s real values (active 6253, defunct 1117,
  not 7310/60). Neither manifest file carries a `full_history` key, so this fix does not
  change whether those numbers render — a separate, pre-existing standing-rule violation
  needing its own fix.
- **Production (equities.dumbmodel.com) is actually served from `origin/main` with
  `vercel.json outputDirectory: "public"`, not from this `master`-based branch.** `main`'s
  `public/` tree has no model page and no data JSONs at all — a different,
  self-contained single-canvas "japandi v4" site. **Merging this branch to master will not
  change what is live** — that gap is separate and larger, and is the operator's to
  resolve (which branch/design should actually be in production).

**Merge target and blocker.** Base: `origin/master` (`e8caf9c9`), 1 commit ahead, clean —
touches only `model.html` (1 line). No git-level blocker to merging into `master`. The
blocker is architectural, not mergeability: `master` is not what production serves, so this
fix (like all of `master`'s content) does not reach equities.dumbmodel.com until the
operator decides whether `master` or `main` is the intended production tree. See
`weekend/live-fix-equities` (based on `origin/main`) for the fix lane that actually reaches
the live site.

Note: `vector-pitch` has a *different* branch that happens to share this exact name,
`weekend/fix-model-page-assets` — same branch name, unrelated repo, unrelated content.
