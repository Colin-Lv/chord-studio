from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"app"))
from diagnose_candidate_pool_v2 import diagnostic_reference


def test_offline_ulp_repair_preserves_masks_and_original_file(tmp_path):
    path=tmp_path/"ref.json"
    data=[dict(start=0,end=1.0000000000000002,chord="C/E",field_mask={"bass":False}),
          dict(start=1,end=2,chord="C9",valid_fields=["ext9"])]
    path.write_text(json.dumps(data),encoding="utf-8")
    before=path.read_bytes()
    rows,metadata=diagnostic_reference(path)
    assert rows[0].end==1
    assert rows[0].field_mask=={"bass":False}
    assert rows[1].valid_fields==("ext9",)
    assert metadata["diagnostic_float_overlap_repairs"]==1
    assert path.read_bytes()==before


def test_offline_json_repair_rejects_real_millisecond_overlap(tmp_path):
    path=tmp_path/"ref.json"
    path.write_text(json.dumps([dict(start=0,end=1.001,chord="C"),dict(start=1,end=2,chord="D")]),encoding="utf-8")
    with pytest.raises(ValueError,match="exceeds offline"):
        diagnostic_reference(path)
