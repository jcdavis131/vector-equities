"""
Rebuild all pipeline — like vector-hoops rebuild_all.py
"""
import subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def run(cmd):
    print(f"$ {cmd}")
    subprocess.check_call(cmd, shell=True, cwd=ROOT)

if __name__=="__main__":
    run("python3 pipeline/build_demo_v3.py --companies 1200 --years 12 --continuity 0.72 --out pipeline/data")
    run("python3 pipeline/build_skills.py")
    run("python3 pipeline/build_archetypes.py")
    run("python3 pipeline/train_mtnn.py --epochs 50 --dim 48 --fusion gated --tower-blocks 2 --val-every 5")
    print("Rebuild done")
