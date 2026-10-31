"""Compare keys by pitch class + mode, not by spelling (G# vs Ab is the same key)."""
from pathlib import Path

PATH = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\frontend\app.js")

OLD = '''  const same=block.key===state.key;'''
NEW = '''  const keyIdentity=value=>{const m=/^([A-G])([#b]?)(.*)$/.exec(String(value||"").trim());if(!m)return null;const pc={C:0,D:2,E:4,F:5,G:7,A:9,B:11}[m[1]]+({"#":1,"b":-1,"":0}[m[2]]||0);const rest=m[3]||"";const minor=/^m(?!aj)/.test(rest)||/^:m(in(or)?)?$/.test(rest)||/minor$/i.test(rest);return((pc%12)+12)%12+(minor?"m":"");};
  const same=keyIdentity(block.key)===keyIdentity(state.key)&&state.key!=="";'''

text = PATH.read_text(encoding="utf-8")
count = text.count(OLD)
print(f"occurrences: {count}")
if count != 1:
    raise SystemExit("expected exactly one")
PATH.write_text(text.replace(OLD, NEW), encoding="utf-8")
print("patched")
