"""Contact-run regression: short runs and invalid/time gaps cannot create stance."""
import numpy as np
import json
from pathlib import Path
from contact_diagnostics import retain_runs, bbox_quality

mask = np.zeros((9, 2), bool)
mask[1:6, 0] = True
mask[0:3, 1] = True
mask[7:9, 1] = True
links = np.ones(9, bool)
links[0] = False
links[3] = False
out, run_id, records = retain_runs(mask, links, 3)
expected = np.zeros_like(mask)
expected[3:6, 0] = True
expected[0:3, 1] = True
np.testing.assert_array_equal(out, expected)
assert len(records) == 2
assert np.all(run_id[~out] == -1)
assert [r["samples"] for r in records] == [3, 3]
empty, _, record_empty = retain_runs(np.zeros((4, 2), bool), np.array([False, True, True, True]), 3)
assert not empty.any() and not record_empty
print("PASS: stance candidates require >=3 consecutive samples; gaps/short runs are excluded")

folder = Path(__file__).resolve().parent / "qa_synthetic"
folder.mkdir(exist_ok=True)
tracks_path = folder / "bbox_tracks_test.json"
camera_path = folder / "bbox_camera_test.json"
camera_path.write_text(json.dumps({"width": 100, "height": 100}), encoding="utf-8")
tracks_path.write_text(json.dumps([
    {"frame_index": 0, "bbox_xyxy": [12, 30, 70, 80], "valid": True},
    {"frame_index": 1, "bbox_xyxy": [30, 30, 89, 80], "valid": True},
    {"frame_index": 2, "bbox_xyxy": [13, 30, 70, 80], "valid": True},
]), encoding="utf-8")
_, _, near, available, _, _ = bbox_quality(np.arange(4), tracks_path, camera_path, 12.)
np.testing.assert_array_equal(near, [True, True, False, False])
np.testing.assert_array_equal(available, [True, True, True, False])
print("PASS: 12-pixel image-border screen includes threshold and tracks missing bbox separately")
