"""Compare raw/temporal IK references including the same contact-candidate samples."""
import argparse
import json
from pathlib import Path
import numpy as np

def p95(x):
    x = np.asarray(x, float)
    return float(np.nanpercentile(x, 95)) if np.isfinite(x).any() else None

def run(raw, temporal, output):
    raw, temporal = Path(raw), Path(temporal)
    with np.load(raw, allow_pickle=False) as a, np.load(temporal, allow_pickle=False) as b, \
         np.load(raw.with_name(raw.stem + ".contact_candidates.npz"), allow_pickle=False) as ac, \
         np.load(temporal.with_name(temporal.stem + ".contact_candidates.npz"), allow_pickle=False) as bc:
        if not np.array_equal(a["frame_indices"], b["frame_indices"]) or not np.array_equal(a["timestamps_s"], b["timestamps_s"]):
            raise ValueError("Variant comparison requires identical timelines")
        am = json.loads(str(a["metadata"].item())); bm = json.loads(str(b["metadata"].item()))
        ar = json.loads(str(ac["metadata"].item())); br = json.loads(str(bc["metadata"].item()))
        valid = a["valid"] & b["valid"]
        delta = b["qpos"][valid, :3] - a["qpos"][valid, :3]
        common_stance = ac["candidate_stance"] & bc["candidate_stance"]
        raw_stance = ac["candidate_stance"]
        result = {
            "raw": str(raw.resolve()), "temporal": str(temporal.resolve()), "common_valid_frames": int(valid.sum()),
            "robot_root_position_change_rms_m": float(np.sqrt(np.mean(np.sum(delta ** 2, axis=1)))),
            "robot_root_position_change_max_m": float(np.linalg.norm(delta, axis=1).max()),
            "robot_joint_angle_change_rms_rad": float(np.sqrt(np.mean((b["qpos"][valid, 7:] - a["qpos"][valid, 7:]) ** 2))),
            "raw_vs_temporal": {k: [am[k], bm[k]] for k in
                ("mean_target_fit_m", "p95_target_fit_m", "max_sole_penetration_m", "max_joint_limit_excess_rad",
                 "p95_near_ground_foot_speed_xy_mps", "max_joint_speed_rad_s", "max_root_yaw_speed_deg_s", "robot_endpoint_displacement_xy_m")},
            "contact_candidate_counts_left_right": [ar["candidate_samples_left_right"], br["candidate_samples_left_right"]],
            "own_candidate_robot_foot_xy_speed_p95_mps": [ar["candidate_stance_robot_xy_speed_mps"]["p95"], br["candidate_stance_robot_xy_speed_mps"]["p95"]],
            "common_candidate_samples": int(common_stance.sum()),
            "common_candidate_robot_foot_xy_speed_p95_mps": [p95(ac["robot_foot_speed_xy_mps"][common_stance]), p95(bc["robot_foot_speed_xy_mps"][common_stance])],
            "raw_candidate_same_samples_robot_foot_xy_speed_p95_mps": [p95(ac["robot_foot_speed_xy_mps"][raw_stance]), p95(bc["robot_foot_speed_xy_mps"][raw_stance])],
            "bbox_border_frames": [ar["bbox_border_screen"].get("near_border_frames"), br["bbox_border_screen"].get("near_border_frames")],
            "combined_review_frames": [ar["combined_review_frames"], br["combined_review_frames"]],
            "notes": ["Raw input and raw output are unchanged.", "Temporal source applies a recorded root translation filter; no endpoint loop constraint.",
                "Own-candidate metrics select different populations; common and raw-mask results provide controlled comparisons.",
                "A smoother reference still has no dynamic-feasibility or policy validation."],
        }
    Path(output).write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("raw", type=Path); ap.add_argument("temporal", type=Path); ap.add_argument("output", type=Path)
    args = ap.parse_args()
    run(args.raw, args.temporal, args.output)
