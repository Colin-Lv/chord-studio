"""Freeze grouped external pilot splits BEFORE inference/threshold selection.

GiantSteps+ record 1095691 (not Beatport/MTG record 1101082). Audio is optional
at preparation time. Input archives are separately downloaded and SHA recorded.
No reference values are supplied to the acoustic worker.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile
import zipfile

import pandas as pd
from key_fusion import key_identity
from evaluate_global_key_fusion import signature

SEED = 20261006


def digest(text):
    return hashlib.sha256(f"{SEED}:{text}".encode()).hexdigest()


def grouped_split(rows):
    parent={r["id"]:r["id"] for r in rows}
    def root(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    seen={}
    for row in rows:
        for token in row["group_tokens"]:
            if token in seen:parent[root(row["id"])]=root(seen[token])
            seen[token]=row["id"]
    groups={}
    for row in rows:groups.setdefault(root(row["id"]),[]).append(row["id"])
    for row in rows:
        group=digest("|".join(sorted(groups[root(row["id"])])))
        fraction=int(group[:8],16)/2**32
        row.update(group=group,split="development" if fraction<.6 else "calibration" if fraction<.8 else "sealed_test")
    return rows


def freeze(path, data):
    content=json.dumps(data,ensure_ascii=False,indent=2)+"\n"
    if path.exists() and path.read_text(encoding="utf-8")!=content:
        raise ValueError(f"Refusing to overwrite frozen split: {path}")
    path.write_text(content,encoding="utf-8")


def prepare(root, output, limit=180):
    root,output=Path(root).resolve(),Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    metadata={}
    with zipfile.ZipFile(root/"original_metadata.zip") as archive:
        for name in archive.namelist():
            if name.startswith("meta/") and name.endswith(".json"):
                try:
                    value=json.loads(archive.read(name));metadata[str(value["id"])]=value
                except json.JSONDecodeError:
                    # Original archive contains malformed metadata. Excel artist
                    # names remain a grouping fallback, never an inference input.
                    continue
    rows=[]
    for _,entry in pd.read_excel(root/"GiantSteps+.xlsx").iterrows():
        if pd.isna(entry["id"]):continue
        identifier=str(int(entry["id"]));meta=metadata.get(identifier,{})
        label=str(entry["global-key"]).strip()
        match=re.fullmatch(r"([A-G][b#]?) (major|minor)(?: [a-z-]+)?",label)
        confidence=None if pd.isna(entry["confidence"]) else float(entry["confidence"])
        # Single labelled key only; retain exclusions rather than invent mode.
        eligible=bool(match and confidence is not None and confidence>=1)
        artists=meta.get("artists",[])+meta.get("remixers",[])
        title=str(meta.get("name") or entry["title"]).strip().casefold()
        tokens=["artist:"+str(a["id"]) for a in artists]+["work:"+title,"track:"+identifier]
        if not artists:tokens.append("artist_text:"+str(entry["artists"]).strip().casefold())
        rows.append(dict(id=identifier,dataset="giantsteps_plus",genre=str(entry["genres"]),
                         group_tokens=tokens,original_key=label,annotation_confidence=confidence,
                         reference=key_identity(match[1]+("m" if match[2]=="minor" else "")) if eligible else None,
                         eligible=eligible,exclusion=None if eligible else "not a reliable single 24-class major/minor label",
                         audio=str(root/"audio"/(identifier+".mp3"))))
    if len({r['id'] for r in rows})!=len(rows):raise ValueError("Duplicate track IDs")
    grouped_split(rows)
    selected={r['id'] for r in sorted((r for r in rows if r['eligible']),key=lambda r:digest(r['id']))[:limit]}
    for row in rows:row['selected']=row['id'] in selected
    freeze(output/"giantsteps_split.json",dict(schema="resolver_split_v1",seed=SEED,
        selection="hash order before any inference, pilot cap",limit=limit,grouping="connected artist/remixer IDs + work title + track ID",
        reference_semantics="Single stated major/minor family; modal subtype retained in original_key; multi-key/detuned/unknown excluded",
        upstream_pretraining_overlap="not verified; resolver split independence does not prove CNN pretraining independence",
        source=signature(root/"GiantSteps+.xlsx"),rows=rows))
    # Extract only safe numeric ID paths; no archive filenames become shell code.
    for filename,leaf in [("billboard-salami.tar.gz","salami_chords.txt"),("billboard-lab.tar.gz","full.lab")]:
        with tarfile.open(root/filename) as archive:
            for member in archive.getmembers():
                parts=member.name.split("/")
                if member.isfile() and len(parts)==3 and parts[1].isdigit() and parts[2]==leaf:
                    target=root/"billboard"/parts[1]/leaf;target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(archive.extractfile(member).read())
    billboard=[]
    for path in sorted((root/"billboard").glob("*/salami_chords.txt")):
        text=path.read_text(encoding="utf-8-sig")
        fields=dict(re.findall(r"^#\s*(title|artist):\s*(.*)$",text,re.M))
        title=fields.get("title","").casefold().strip();artist=fields.get("artist","").casefold().strip()
        billboard.append(dict(id=path.parent.name,dataset="billboard",group_tokens=["artist:"+artist,"work:"+title],
                              title=title,artist=artist,salami=str(path),lab=str(path.with_name("full.lab"))))
    grouped_split(billboard)
    # Duplicate chart slots are not separate independent songs.
    unique={}
    for row in billboard:
        identity=(row['artist'],row['title']);row['duplicate_of']=unique.get(identity);unique.setdefault(identity,row['id'])
    freeze(output/"billboard_split.json",dict(schema="billboard_symbolic_split_v1",seed=SEED,rows=billboard,
        audio_available=False,mode_reference_available=False,reference_semantics="tonic pitch class only; local changes retained",
        license="CC0",source=[signature(root/f) for f in ("billboard-salami.tar.gz","billboard-lab.tar.gz","billboard-index.csv")]))
    print({s:sum(r['selected'] and r['split']==s for r in rows) for s in ['development','calibration','sealed_test']})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--root",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);parser.add_argument("--limit",type=int,default=180)
    args=parser.parse_args();prepare(args.root,args.output,args.limit)
