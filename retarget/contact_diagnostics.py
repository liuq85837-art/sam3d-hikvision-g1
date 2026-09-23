"""Contact candidates and speed screening without modifying raw human/robot motion.

Candidates are kinematic heuristics in model-scale coordinates, not measured contact.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

def stats(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    return {"samples": len(x), **dict(zip(("median", "p90", "p95", "max"), np.percentile(x, [50, 90, 95, 100]).tolist()))} if len(x) else {"samples": 0, "median": None, "p90": None, "p95": None, "max": None}

def retain_runs(mask, links, min_samples=3):
    out = np.zeros_like(mask, bool)
    run_ids = np.full(mask.shape, -1, int)
    records = []
    for side in range(mask.shape[1]):
        start = None
        for i in range(len(mask) + 1):
            present = i < len(mask) and mask[i, side]
            continuous = i > 0 and i < len(mask) and links[i]
            if start is not None and (not present or not continuous):
                if i - start >= min_samples:
                    rid = len(records)
                    out[start:i, side] = True
                    run_ids[start:i, side] = rid
                    records.append({"id": rid, "side": ("left", "right")[side], "start": start, "end": i - 1, "samples": i - start})
                start = None
            if present and start is None:
                start = i
    return out, run_ids, records

def bbox_quality(frames, tracks_path, camera_input_path, border_threshold_px=12.):
    available = np.zeros(len(frames), bool)
    near_border = np.zeros(len(frames), bool)
    boxes = np.full((len(frames), 4), np.nan)
    margin = np.full(len(frames), np.nan)
    valid = np.zeros(len(frames), bool)
    if not tracks_path.exists() or not camera_input_path.exists():
        return boxes, margin, near_border, available, valid, {"available": False, "reason": "tracks.json or camera/input.json missing"}
    camera = json.loads(camera_input_path.read_text(encoding="utf-8"))
    width, height = float(camera["width"]), float(camera["height"])
    records = {int(x["frame_index"]): x for x in json.loads(tracks_path.read_text(encoding="utf-8"))}
    for i, frame in enumerate(frames):
        record = records.get(int(frame))
        if record is None:
            continue
        valid[i] = bool(record.get("valid", False))
        box = np.asarray(record.get("bbox_xyxy", [np.nan] * 4), float)
        if box.shape != (4,) or not np.isfinite(box).all():
            continue
        available[i] = True
        boxes[i] = box
        margin[i] = min(box[0], box[1], width - box[2], height - box[3])
        near_border[i] = margin[i] <= border_threshold_px
    meta = {"available": True, "tracks": str(tracks_path.resolve()),
            "tracks_sha256": hashlib.sha256(tracks_path.read_bytes()).hexdigest(),
            "camera_input": str(camera_input_path.resolve()), "image_width": int(width), "image_height": int(height),
            "threshold_px": border_threshold_px, "heuristic": True,
            "definition": "min(x1,y1,image_width-x2,image_height-y2) <= threshold; possible truncation, not proven keypoint failure",
            "source_valid_altered": False, "missing_bbox_frames": int((~available).sum()), "near_border_frames": int(near_border.sum())}
    return boxes, margin, near_border, available, valid, meta

def run(source, robot, source_speed_threshold=.25, low_height_threshold=.025, min_stance_samples=3, joint_speed_review_threshold=6., tracks=None, camera_input=None, bbox_border_px=12.):
    with np.load(source, allow_pickle=False) as s, np.load(robot, allow_pickle=False) as r:
        p = np.asarray(s["world_joints"] if "world_joints" in s else s["joints_world"], float)
        names = [str(n).replace("-", "_") for n in s["joint_names"]]
        times, valid, frames = r["timestamps_s"], r["valid"], r["frame_indices"]
        if len(p) != len(times) or not np.array_equal(s["frame_indices"], frames) or not np.allclose(s["timestamps_s"], times, rtol=0, atol=1e-9):
            raise ValueError("Human and robot must share the exact selected frames and timestamps")
        links = r["temporal_link_from_previous"]
        source_pos = np.full((len(times), 2, 3), np.nan)
        source_low = np.full((len(times), 2), np.nan)
        for side_index, side in enumerate(("left", "right")):
            foot_names = [side + suffix for suffix in ("_big_toe_tip", "_small_toe_tip", "_heel")]
            if not all(n in names for n in foot_names):
                raise ValueError("MHR toe/heel joints are required for contact candidates")
            feet = p[:, [names.index(n) for n in foot_names]]
            source_pos[:, side_index] = feet.mean(axis=1)
            source_low[:, side_index] = feet[:, :, 2].min(axis=1)
        robot_pos = np.asarray(r["foot_positions"], float)
        source_vel = np.full_like(source_pos, np.nan)
        robot_vel = np.full_like(robot_pos, np.nan)
        pelvis_vel = np.full((len(times), 3), np.nan)
        pelvis = p[:, [names.index("left_hip"), names.index("right_hip")]].mean(axis=1)
        for i in np.flatnonzero(links):
            dt = times[i] - times[i - 1]
            source_vel[i] = (source_pos[i] - source_pos[i - 1]) / dt
            robot_vel[i] = (robot_pos[i] - robot_pos[i - 1]) / dt
            pelvis_vel[i] = (pelvis[i] - pelvis[i - 1]) / dt
        source_speed = np.linalg.norm(source_vel, axis=2)
        source_speed_xy = np.linalg.norm(source_vel[:, :, :2], axis=2)
        robot_speed = np.linalg.norm(robot_vel, axis=2)
        robot_speed_xy = np.linalg.norm(robot_vel[:, :, :2], axis=2)
        low_source = (source_low <= low_height_threshold) & valid[:, None]
        low_robot = np.asarray(r["foot_near_floor"], bool)
        candidate, run_ids, runs = retain_runs(low_source & (source_speed < source_speed_threshold), links, min_stance_samples)
        for record in runs:
            si = 0 if record["side"] == "left" else 1
            a, b = record["start"], record["end"]
            record.update(start_frame=int(frames[a]), end_frame=int(frames[b]), start_s=float(times[a]), end_s=float(times[b]),
                          duration_s=float(times[b] - times[a]), source_speed_xyz_mps=stats(source_speed[a:b + 1, si]),
                          robot_speed_xy_mps=stats(robot_speed_xy[a:b + 1, si]), robot_speed_xyz_mps=stats(robot_speed[a:b + 1, si]))
        qvel = np.asarray(r["qvel"])
        joint_max = np.full(len(times), np.nan)
        vv = np.isfinite(qvel[:, 6:]).all(axis=1)
        joint_max[vv] = np.max(abs(qvel[vv, 6:]), axis=1)
        robot_speed_review = candidate & (robot_speed_xy > source_speed_threshold)
        joint_speed_review = joint_max > joint_speed_review_threshold
        camera_dir = Path(source).resolve().parent.parent / "camera"
        tracks_path = Path(tracks) if tracks else camera_dir / "tracks.json"
        camera_input_path = Path(camera_input) if camera_input else camera_dir / "input.json"
        boxes, border_margin, bbox_near_border, bbox_available, track_valid, bbox_meta = bbox_quality(frames, tracks_path, camera_input_path, bbox_border_px)
        quality_review = robot_speed_review.any(axis=1) | joint_speed_review | bbox_near_border
        if bbox_meta["available"]:
            quality_review |= ~bbox_available | ~track_valid
        source_metadata = json.loads(str(r["metadata"].item()))
        report = {
            "source": str(Path(source).resolve()), "robot": str(Path(robot).resolve()),
            "metric_scale_status": source_metadata["metric_scale_status"], "method": "unsmoothed backward differences on adjacent valid samples",
            "foot_points": {"human": "mean of big toe, small toe and heel", "robot": "mean of four official sole-contact sphere bottom points"},
            "candidate_criteria": {"source_lowest_toe_or_heel_height_lte_m": low_height_threshold,
                "source_foot_world_speed_xyz_lt_mps": source_speed_threshold, "min_consecutive_samples": min_stance_samples,
                "no_bridging_invalid_or_time_gaps": True, "measured_contact": False},
            "candidate_samples_left_right": candidate.sum(axis=0).tolist(), "candidate_runs": len(runs),
            "candidate_frames_any_foot": int(candidate.any(axis=1).sum()),
            "near_ground_robot_xy_speed_mps": stats(robot_speed_xy[low_robot]),
            "same_samples_source_xy_speed_mps": stats(source_speed_xy[low_robot]),
            "low_source_foot_xyz_speed_mps": stats(source_speed[low_source]),
            "candidate_stance_source_xyz_speed_mps": stats(source_speed[candidate]),
            "candidate_stance_source_xy_speed_mps": stats(source_speed_xy[candidate]),
            "candidate_stance_robot_xy_speed_mps": stats(robot_speed_xy[candidate]),
            "candidate_stance_robot_xyz_speed_mps": stats(robot_speed[candidate]),
            "candidate_stance_xy_velocity_mismatch_mps": stats(np.linalg.norm(robot_vel[:, :, :2] - source_vel[:, :, :2], axis=2)[candidate]),
            "candidate_stance_robot_speed_gt_threshold_samples_left_right": robot_speed_review.sum(axis=0).tolist(),
            "candidate_stance_robot_speed_gt_threshold_frames": int(robot_speed_review.any(axis=1).sum()),
            "joint_speed_screen": {"review_threshold_rad_s": joint_speed_review_threshold, "hardware_limit": False,
                "frames_over_threshold": int(joint_speed_review.sum()), "max_abs_joint_speed_rad_s": stats(joint_max)},
            "bbox_border_screen": bbox_meta,
            "combined_review_frames": int(quality_review.sum()),
            "review_pass_frames": int((valid & ~quality_review).sum()),
            "review_pass_meaning": "only passes configured kinematic and image-border screens; not a training-ready or dynamics-valid label",
            "pelvis_speed_xyz_mps": stats(np.linalg.norm(pelvis_vel, axis=1)),
            "candidate_stance_pelvis_speed_xyz_mps": stats(np.linalg.norm(pelvis_vel, axis=1)[candidate.any(axis=1)]),
            "runs": runs,
            "limitations": ["Thresholds use uncalibrated model meters when metric_scale_status is model_prior_only.",
                "Source speed gating can reject noisy true contacts; it does not establish contact ground truth.",
                "A reference passing these screens is not dynamically validated, trained, torque-feasible, or safe for hardware.",
                "No source or robot positions, endpoints, headings or timestamps were altered."],
        }
        stem = Path(robot).with_suffix("")
        out = stem.with_name(stem.name + ".contact_candidates.npz")
        np.savez_compressed(out, timestamps_s=times, frame_indices=frames, valid=valid,
            source_foot_positions=source_pos, robot_foot_positions=robot_pos,
            source_foot_velocity_mps=source_vel, robot_foot_velocity_mps=robot_vel,
            source_foot_speed_xyz_mps=source_speed, source_foot_speed_xy_mps=source_speed_xy,
            robot_foot_speed_xyz_mps=robot_speed, robot_foot_speed_xy_mps=robot_speed_xy,
            source_foot_low_height=low_source, source_foot_min_height_m=source_low,
            candidate_stance=candidate, candidate_stance_run_id=run_ids,
            candidate_robot_slip_review=robot_speed_review, max_abs_joint_speed_rad_s=joint_max,
            joint_speed_review=joint_speed_review, source_bbox_xyxy=boxes,
            source_bbox_min_border_margin_px=border_margin, source_bbox_near_border=bbox_near_border,
            source_bbox_metadata_available=bbox_available, source_track_valid=track_valid,
            reference_quality_review=quality_review, reference_review_pass=valid & ~quality_review,
            metadata=json.dumps(report))
        out.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=2, ensure_ascii=False))
        return report

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path); ap.add_argument("robot", type=Path)
    ap.add_argument("--source-speed-threshold", type=float, default=.25)
    ap.add_argument("--low-height-threshold", type=float, default=.025)
    ap.add_argument("--min-stance-samples", type=int, default=3)
    ap.add_argument("--joint-speed-review-threshold", type=float, default=6.)
    ap.add_argument("--tracks", type=Path); ap.add_argument("--camera-input", type=Path)
    ap.add_argument("--bbox-border-px", type=float, default=12.)
    args = ap.parse_args()
    run(args.source, args.robot, args.source_speed_threshold, args.low_height_threshold, args.min_stance_samples, args.joint_speed_review_threshold, args.tracks, args.camera_input, args.bbox_border_px)
