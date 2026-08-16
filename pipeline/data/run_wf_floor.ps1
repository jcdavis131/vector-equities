# THIS SCRIPT DOES NOT RUN AGAINST HEAD. It passes `--split temporal`, and
# train_career_mtnn_v6.py has no --split argument -- not today, and not at any
# commit in its git history (`git log -S '"--split"'` is empty). argparse exits
# 2 before a single epoch. Kept, unmodified, because it is the ONLY record of
# how wf_floor.log's numbers were produced.
#
# Why that matters more than a broken script: wf_floor.log holds the only
# TEMPORAL-split evaluation this repo has (best IC 0.0434-0.1724, train
# FY<=2021 / val FY>2021). HEAD's split is ticker-disjoint but NOT
# time-disjoint -- train_career_mtnn_v6.py:80-90 shuffles unique tickers 70/15
# with train and val spanning the SAME fiscal years, so market-wide regime
# information is shared across the split. That is why HEAD reports IC ~0.50
# (commit 1bf0406's own message records "v6 IC 0.549") against a temporal floor
# of ~0.04-0.17. The ~3x gap is the leak, not alpha.
#
# So: a climb baseline measured on HEAD is a valid A/B ANCHOR -- every arm
# trains on the identical split -- but it is a "ticker-split IC", not a forward
# IC, and it must never be quoted as predictive performance. Restoring a
# temporal split is a NEW DATA REGIME and requires re-measuring the baseline in
# the same commit.
#
# Found 2026-08-15 while unblocking this repo for the harness.

Set-Location "C:\Users\jcdav\vector-equities"
foreach ($s in 42,43,44,45,46) {
  "=== temporal seed $s start $(Get-Date -Format HH:mm:ss) ===" | Add-Content pipeline\data\wf_floor.log
  python pipeline/train_career_mtnn_v6.py --epochs 15 --val-every 5 --split temporal --seed $s --out pipeline/data/mtnn_v6_wf_s$s.pt *>> pipeline\data\wf_floor.log
  "=== temporal seed $s exit $LASTEXITCODE $(Get-Date -Format HH:mm:ss) ===" | Add-Content pipeline\data\wf_floor.log
}
"WF-FLOOR-DONE" | Add-Content pipeline\data\wf_floor.log
