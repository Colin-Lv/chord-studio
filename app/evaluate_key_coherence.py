"""Freeze code and inputs, then replay exposed key regressions in a new directory.
This entry point never overwrites the historical v4 replay.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from app.audit_key_revision import freeze, compare

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    ap.add_argument('--output',type=Path,default=Path('D:/ChordStudio/research')/('key-replay-'+stamp))
    args=ap.parse_args()
    freeze(args.output)
    compare(args.output)

if __name__=='__main__':main()
