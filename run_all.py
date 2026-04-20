"""Run every step in order."""
import subprocess
import sys

STEPS = [
    "step01_load.py",
    "steps_02_04.py",
    "step05_plots.py",
    "step06_imbalance.py",
    "step07_forward.py",
    "step08_deciles.py",
    "step09_regression.py",
    "step10_tradable.py",
    "step11_robustness.py",
]

if __name__ == "__main__":
    for script in STEPS:
        print(f"\n\n{'#' * 78}\n# {script}\n{'#' * 78}")
        if subprocess.run([sys.executable, script]).returncode:
            sys.exit(f"{script} failed")
    print("\nall steps completed")
