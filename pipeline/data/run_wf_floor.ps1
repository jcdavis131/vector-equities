Set-Location "C:\Users\jcdav\vector-equities"
foreach ($s in 42,43,44,45,46) {
  "=== temporal seed $s start $(Get-Date -Format HH:mm:ss) ===" | Add-Content pipeline\data\wf_floor.log
  python pipeline/train_career_mtnn_v6.py --epochs 15 --val-every 5 --split temporal --seed $s --out pipeline/data/mtnn_v6_wf_s$s.pt *>> pipeline\data\wf_floor.log
  "=== temporal seed $s exit $LASTEXITCODE $(Get-Date -Format HH:mm:ss) ===" | Add-Content pipeline\data\wf_floor.log
}
"WF-FLOOR-DONE" | Add-Content pipeline\data\wf_floor.log
