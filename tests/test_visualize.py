import json
import re

import pytest
import torch

from lcrl.config import FWY_LANE_DROP, SIG_SHARED
from lcrl.visualize import F_ABORT, F_LC, F_START, PLACEHOLDER, build, write_html

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_recording_is_consistent_and_embeds_in_viewer(tmp_path):
    data = build({"rule": "rule"}, [FWY_LANE_DROP, SIG_SHARED], warmup=100, steps=60)
    runs = data["runs"]["rule"]
    assert [r["static"]["scenario"] for r in runs] == ["freeway_lane_drop", "signal_shared_turn"]
    for r in runs:
        assert len(r["frames"]) == 60 and len(r["signal"]) == 60
        last_x = {}
        for f, row in enumerate(r["frames"]):
            assert len(row) % 5 == 0
            for i in range(0, len(row), 5):
                uid, x, y, v, fl = row[i:i + 5]
                veh = r["vehicles"][uid]
                assert veh["spawn"] <= f and (veh["end"] is None or f <= veh["end"])
                assert v >= 0 and -0.01 <= y <= 5.01
                assert x >= last_x.get(uid, -1e9) - 1e-6          # vehicles never move backwards
                last_x[uid] = x
                if fl & (F_START | F_ABORT):
                    assert not (fl & F_START and fl & F_LC)      # a start is flagged on the last in-lane frame
    out = tmp_path / "viewer.html"
    write_html(data, str(out))
    html = out.read_text(encoding="utf-8")
    assert PLACEHOLDER not in html
    blob = re.search(r"const EMBEDDED = (.*?);\n", html, re.S).group(1)
    assert json.loads(blob.replace("<\\/", "</"))["meta"]["policies"] == ["rule"]
