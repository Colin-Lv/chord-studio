"""Run every acceptance check for the phase-3 objectives and print one verdict table.

  1. synthetic domain      : held-out joint >= 0.95 and the 7th family >= 0.85
  2. 147-case regression   : published key must not degrade under retained10_span
  3. full decoder in prod  : exact >= 0.50 and boundary F1 >= 0.90 on 虚拟, wired as a
                             switchable second-tier source
  4. evidence              : every number above is re-read from a script artefact
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN = Path(r"D:\ChordStudio\research\train-v2")
REGRESS = HERE / "regress147"
PY = sys.executable


def run(label, args, cwd):
    print(f"\n=== {label} ===")
    proc = subprocess.run([PY, "-u", *[str(a) for a in args]], cwd=str(cwd), capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    tail = (proc.stdout or "").strip().splitlines()
    for line in tail[-12:]:
        print("  " + line)
    if proc.returncode:
        print("  [stderr]", (proc.stderr or "").strip().splitlines()[-3:])
    return proc.returncode == 0


def main():
    results = {}
    results["synthetic"] = run("1) synthetic-domain training", [TRAIN / "verify_synthetic.py", "fix60"], TRAIN)
    results["regression_summary"] = run("2) 147-case regression summary", [REGRESS / "summarise.py"], REGRESS)
    results["regression_pinning"] = run("2b) key pinning recomputation", [REGRESS / "verify_key_pinning.py"], REGRESS)
    results["full_decoder"] = run("3) full decoder in production", [HERE / "verify_prod_full.py"], HERE)

    syn = json.loads((TRAIN / "verify_synthetic.json").read_text(encoding="utf-8"))
    reg = json.loads((REGRESS / "summary.json").read_text(encoding="utf-8"))
    pin = json.loads((REGRESS / "verify_key_pinning.json").read_text(encoding="utf-8"))
    full = json.loads((HERE / "verify_prod_full.json").read_text(encoding="utf-8"))

    print("\n================ ACCEPTANCE ================")
    print(f"1  synthetic joint {syn['tta_joint']:.4f} (>=0.95)  7th "
          f"{ {k: round(v, 3) for k, v in syn['seventh'].items()} } (>=0.85)  -> "
          f"{'PASS' if syn['pass_joint'] and syn['pass_seventh'] else 'FAIL'}")
    print(f"2  147-case key: stable {reg['stable']} -> span {reg['span']} (old wiring, "
          f"{len(reg['broke'])} broke) ; pinned {pin['pinned_hits']} "
          f"(= stable, {pin['equal']}/{pin['cases']} identical)                  -> "
          f"{'PASS' if pin['mismatch'] == 0 and reg['pinned'] >= reg['stable'] else 'FAIL'}")
    print(f"3  虚拟 full decoder exact {full['exact']:.3f} (>=0.50)  boundary F1 "
          f"{full['f1']:.3f} (>=0.90)  wiring {full['wiring']}  -> "
          f"{'PASS' if full['pass_exact'] and full['pass_f1'] and full['pass_wiring'] else 'FAIL'}")
    ok = (syn["pass_joint"] and syn["pass_seventh"]
          and pin["mismatch"] == 0 and reg["pinned"] >= reg["stable"]
          and full["pass_exact"] and full["pass_f1"] and full["pass_wiring"])
    print("OVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
