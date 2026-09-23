"""Quantitative audit of real retargeted motion, without altering any samples."""
from pathlib import Path
import argparse
import json
import numpy as np

def percentiles(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return dict(zip(("min", "p05", "median", "p95", "max"), np.percentile(a, [0, 5, 50, 95, 100]).tolist())) if len(a) else None

def audit(source, robot, output=None):
    with np.load(source, allow_pickle=False) as s, np.load(robot, allow_pickle=False) as r:
        p = s["world_joints"] if "world_joints" in s else s["joints_world"]
        names = [str(n).replace("-", "_") for n in s["joint_names"]]
        times, frames, valid = r["timestamps_s"], r["frame_indices"], r["valid"]
        qp, qvel, soles = r["qpos"], r["qvel"], r["sole_contact_positions"]
        metadata = json.loads(str(r["metadata"].item()))
        hips = p[:, [names.index("left_hip"), names.index("right_hip")]].mean(axis=1)
        source_foot_names = [side + part for side in ("left", "right") for part in ("_big_toe_tip", "_small_toe_tip", "_heel")]
        source_foot_ids = [names.index(n) for n in source_foot_names if n in names]
        yaw_speed = r["root_yaw_speed_rad_s"]
        joint_speed = np.full(len(qvel), np.nan)
        velocity_valid = np.isfinite(qvel[:, 6:]).all(axis=1)
        joint_speed[velocity_valid] = np.max(np.abs(qvel[velocity_valid, 6:]), axis=1)
        finite_yaw = np.flatnonzero(np.isfinite(yaw_speed))
        largest_yaw = finite_yaw[np.argsort(-np.abs(yaw_speed[finite_yaw]))[:10]]
        finite_speed = np.flatnonzero(np.isfinite(joint_speed))
        largest_joint = finite_speed[np.argsort(-joint_speed[finite_speed])[:10]]
        links = r["temporal_link_from_previous"]
        raw_hip_yaw = np.arctan2((p[:, names.index("left_hip")] - p[:, names.index("right_hip")])[:, 1],
                                (p[:, names.index("left_hip")] - p[:, names.index("right_hip")])[:, 0]) - np.pi / 2
        raw_step = (np.diff(raw_hip_yaw) + np.pi) % (2 * np.pi) - np.pi
        result = {
            "source": str(Path(source).resolve()), "robot": str(Path(robot).resolve()),
            "source_metadata": metadata["source_metadata"],
            "metric_scale_status": metadata["metric_scale_status"],
            "frames": len(valid), "valid_frames": int(valid.sum()),
            "xy_preservation_max_abs_m": float(np.max(np.abs(qp[valid, :2] - hips[valid, :2]))),
            "human_pelvis_z_m": percentiles(hips[valid, 2]), "robot_root_z_m": percentiles(qp[valid, 2]),
            "robot_lowest_sole_height_m": percentiles(soles[valid, :, 2].min(axis=1)),
            "frames_all_soles_over_5cm": int((soles[valid, :, 2].min(axis=1) > .05).sum()),
            "frames_any_sole_below_minus_1cm": int((soles[valid, :, 2].min(axis=1) < -.01).sum()),
            "source_lowest_toe_heel_height_m": percentiles(p[valid][:, source_foot_ids, 2].min(axis=1)) if source_foot_ids else None,
            "root_yaw_speed_abs_deg_s": percentiles(np.degrees(abs(yaw_speed))),
            "source_hip_heading_step_abs_deg": percentiles(np.degrees(abs(raw_step[links[1:]]))),
            "max_abs_joint_velocity_per_frame_rad_s": percentiles(joint_speed),
            "largest_yaw_speed_samples": [{"frame": int(frames[i]), "time_s": float(times[i]), "deg_s": float(np.degrees(yaw_speed[i]))} for i in largest_yaw],
            "largest_joint_speed_samples": [{"frame": int(frames[i]), "time_s": float(times[i]), "rad_s": float(joint_speed[i]),
                 "joint": str(r["joint_names"][np.nanargmax(abs(qvel[i, 6:]))])} for i in largest_joint],
            "retarget_summary": {k: v for k, v in metadata.items() if k not in ("source_metadata",)},
            "interpretation": "Diagnostic only; floor comes from source gauge and potential hovering/skating is not independent calibration evidence.",
        }
    output = Path(output) if output else Path(robot).with_suffix(".audit.json")
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("source_metadata", "retarget_summary")}, ensure_ascii=False, indent=2))
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path); parser.add_argument("robot", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit(args.source, args.robot, args.output)
