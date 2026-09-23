"""Named Z-up world joints to joint-limited G1 kinematic reference motion.

This is an offline IK reference, not a controller or proof of dynamic feasibility.
XY translation is retained exactly; no endpoint, loop or drift correction is made.
"""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

MODEL = Path(__file__).resolve().parents[1] / "assets/unitree_g1/scene.xml"
CORE = [s + "_" + j for s in ("left", "right") for j in
        ("hip", "knee", "ankle", "shoulder", "elbow", "wrist")]
BODY_MAP = {s + "_" + j: s + "_" + b for s in ("left", "right") for j, b in
            (("hip", "hip_pitch_link"), ("knee", "knee_link"),
             ("ankle", "ankle_pitch_link"), ("shoulder", "shoulder_pitch_link"),
             ("elbow", "elbow_link"), ("wrist", "wrist_yaw_link"))}
TARGET_NAMES = list(BODY_MAP.values())

def load_model(path=MODEL):
    """MuJoCo's Windows file API cannot reliably open non-ASCII project paths."""
    path = Path(path).resolve()
    assets = {p.relative_to(path.parent).as_posix(): p.read_bytes()
              for p in path.parent.rglob("*") if p.is_file() and p != path
              and p.suffix.lower() in (".xml", ".stl", ".obj", ".png", ".jpg")}
    return mujoco.MjModel.from_xml_string(path.read_text(encoding="utf-8"), assets=assets)

def normalize(v):
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm < 1e-7:
        raise ValueError("degenerate_anatomical_segment")
    return v / norm

def basis(left, up):
    y = normalize(left)
    z = normalize(up - np.dot(up, y) * y)
    return np.column_stack((normalize(np.cross(y, z)), y, z))

def get_alias(z, *names):
    for name in names:
        if name in z:
            return np.asarray(z[name])
    raise ValueError("Missing field: " + "/".join(names))

def load_source(path, min_confidence=.1):
    with np.load(path, allow_pickle=False) as z:
        points = get_alias(z, "world_joints", "joints_world").astype(float)
        names = [str(n).replace("-", "_").lower() for n in z["joint_names"]]
        times = get_alias(z, "timestamps_s", "timestamps").astype(float)
        conf = np.asarray(z["confidence"], float) if "confidence" in z else np.ones(points.shape[:2])
        valid = np.asarray(z["valid"], bool) if "valid" in z else np.ones(len(points), bool)
        indices = np.asarray(z["frame_indices"], int) if "frame_indices" in z else np.arange(len(points))
        metadata = json.loads(str(z["metadata"].item())) if "metadata" in z else {}
        for key in ("metric_scale_status", "source_type", "world_frame_id"):
            if key in z and z[key].ndim == 0:
                metadata[key] = str(z[key].item())
    if points.shape != (len(times), len(names), 3) or len(times) == 0:
        raise ValueError("Expected world_joints[T,J,3], timestamps_s[T], joint_names[J]")
    if conf.shape != points.shape[:2] or valid.shape != (len(times),) or indices.shape != (len(times),):
        raise ValueError("confidence, valid or frame_indices shape mismatch")
    if len(set(names)) != len(names) or set(CORE) - set(names):
        raise ValueError("Missing or duplicate named core joints: " + str(sorted(set(CORE) - set(names))))
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("timestamps_s must be finite and strictly increasing")
    if metadata.get("units", "m") != "m" or metadata.get("world_up", "z").lower() != "z":
        raise ValueError("Source must use meters and Z-up world coordinates")
    ids = [names.index(n) for n in CORE]
    valid &= np.isfinite(points[:, ids]).all(axis=(1, 2))
    valid &= np.isfinite(conf[:, ids]).all(axis=1) & (conf[:, ids].min(axis=1) >= min_confidence)
    return points, names, times, conf, valid, indices, metadata

def links_between_frames(times, valid, max_gap_s):
    links = np.zeros(len(times), bool)
    links[1:] = valid[1:] & valid[:-1] & (np.diff(times) <= max_gap_s)
    return links

def estimate_leg_scale(points, names, valid, rest):
    src = {n: points[:, names.index(n)] for n in CORE}
    lengths, robot_lengths = [], []
    for side in ("left", "right"):
        h, k, a = [side + "_" + n for n in ("hip", "knee", "ankle")]
        lengths.append(np.linalg.norm(src[h] - src[k], axis=1) + np.linalg.norm(src[k] - src[a], axis=1))
        robot_lengths.append(np.linalg.norm(rest[BODY_MAP[h]] - rest[BODY_MAP[k]]) +
                             np.linalg.norm(rest[BODY_MAP[k]] - rest[BODY_MAP[a]]))
    median = float(np.median(np.asarray(lengths)[:, valid]))
    if not np.isfinite(median) or median < .1:
        raise ValueError("Cannot estimate human leg length")
    return float(np.mean(robot_lengths) / median), median

def estimate_source_ankle_clearance(points, names, valid):
    """One anatomical offset, not a per-frame support/root correction."""
    samples = []
    for side in ("left", "right"):
        foot_names = [side + suffix for suffix in ("_big_toe_tip", "_small_toe_tip", "_heel")]
        if all(n in names for n in foot_names):
            foot = points[valid][:, [names.index(n) for n in foot_names], 2]
            ankle = points[valid, names.index(side + "_ankle"), 2]
            heights = ankle - np.min(foot, axis=1)
            samples.extend(heights[np.isfinite(heights) & (heights >= 0) & (heights < .3)].tolist())
    return (float(np.median(samples)), "constant median ankle-to-lowest-toe/heel height") if samples else (0., "unavailable; ankle treated as source foot endpoint")

def targets_for_frame(points, names, rest, leg_scale, ankle_clearance, floor_z=0.0, source_ankle_clearance=0.):
    src = dict(zip(names, points))
    pelvis = (src["left_hip"] + src["right_hip"]) / 2
    shoulders = (src["left_shoulder"] + src["right_shoulder"]) / 2
    # Only heading is observable reliably from a hip pair; pelvis tilt uses gravity.
    y = src["left_hip"] - src["right_hip"]
    y[2] = 0
    root_r = basis(y, np.array([0., 0., 1.]))
    chest_r = basis(src["left_shoulder"] - src["right_shoulder"], shoulders - pelvis)
    hip_offset = (rest[BODY_MAP["left_hip"]] + rest[BODY_MAP["right_hip"]]) / 2 - rest["pelvis"]
    robot_hip_mid = pelvis.copy()
    robot_hip_mid[2] = floor_z + (pelvis[2] - floor_z - source_ankle_clearance) * leg_scale + ankle_clearance
    root = robot_hip_mid - root_r @ hip_offset
    targets = {}
    torso_pos = root + root_r @ (rest["torso_link"] - rest["pelvis"])
    orientations = [("torso_link", chest_r, .24)]
    for side in ("left", "right"):
        h, k, a, s, e, w = [side + "_" + n for n in ("hip", "knee", "ankle", "shoulder", "elbow", "wrist")]
        targets[BODY_MAP[h]] = root + root_r @ (rest[BODY_MAP[h]] - rest["pelvis"])
        targets[BODY_MAP[k]] = targets[BODY_MAP[h]] + normalize(src[k] - src[h]) * np.linalg.norm(rest[BODY_MAP[k]] - rest[BODY_MAP[h]])
        targets[BODY_MAP[a]] = targets[BODY_MAP[k]] + normalize(src[a] - src[k]) * np.linalg.norm(rest[BODY_MAP[a]] - rest[BODY_MAP[k]])
        targets[BODY_MAP[s]] = torso_pos + chest_r @ (rest[BODY_MAP[s]] - rest["torso_link"])
        targets[BODY_MAP[e]] = targets[BODY_MAP[s]] + normalize(src[e] - src[s]) * np.linalg.norm(rest[BODY_MAP[e]] - rest[BODY_MAP[s]])
        targets[BODY_MAP[w]] = targets[BODY_MAP[e]] + normalize(src[w] - src[e]) * np.linalg.norm(rest[BODY_MAP[w]] - rest[BODY_MAP[e]])
        # Flat-foot/heading is a weak prior; toe landmarks are not a calibrated sole frame.
        orientations.append((side + "_ankle_roll_link", root_r, .10))
    return root, root_r, np.asarray([targets[n] for n in TARGET_NAMES]), orientations, pelvis

def run(source, output, model=MODEL, min_confidence=.1, iterations=20, max_gap_s=.2, floor_z=0.):
    points, names, times, conf, valid, indices, source_meta = load_source(source, min_confidence)
    if not valid.any():
        raise ValueError("No valid source frames")
    m = load_model(model)
    d = mujoco.MjData(m)
    if (m.nq, m.nv, m.nu) != (36, 35, 29):
        raise ValueError("Expected official G1 29-DOF floating-base model (nq=36,nv=35,nu=29)")
    mujoco.mj_forward(m, d)
    rest = {m.body(i).name: d.xpos[i].copy() for i in range(1, m.nbody)}
    leg_scale, human_leg = estimate_leg_scale(points, names, valid, rest)
    source_ankle_clearance, source_ankle_clearance_method = estimate_source_ankle_clearance(points, names, valid)
    # Four official spherical sole-contact geoms have radius .005 and z=-.03.
    ankle_clearance = .052558
    target_ids = [m.body(n).id for n in TARGET_NAMES]
    weight = np.tile([.4, 1.5, 2.8, .8, 1.2, 1.7], 2)
    joint_ids = np.arange(1, m.njnt)
    qadr = m.jnt_qposadr[joint_ids]
    vadr = m.jnt_dofadr[joint_ids]
    limits = m.jnt_range[joint_ids]
    sole_geoms = [i for i in range(m.ngeom) if m.body(m.geom_bodyid[i]).name in
                  ("left_ankle_roll_link", "right_ankle_roll_link") and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE]
    qp = np.full((len(times), m.nq), np.nan)
    xpos = np.full((len(times), m.nbody, 3), np.nan)
    xquat = np.full((len(times), m.nbody, 4), np.nan)
    targets = np.full((len(times), len(TARGET_NAMES), 3), np.nan)
    errors = np.full((len(times), len(TARGET_NAMES)), np.nan)
    soles = np.full((len(times), len(sole_geoms), 3), np.nan)
    human_pelvis = (points[:, names.index("left_hip")] + points[:, names.index("right_hip")]) / 2
    reason = np.where(valid, "", "nonfinite_or_low_confidence_core_joint").astype("<U80")
    links = links_between_frames(times, valid, max_gap_s)
    jp = np.zeros((3, m.nv)); jr = np.zeros_like(jp)
    identity = np.eye(29)
    neutral = np.zeros(29)
    for s in ("left", "right"):
        neutral[m.joint(s + "_elbow_joint").qposadr[0] - 7] = .35
    previous = None
    for f in range(len(times)):
        if not links[f]:
            previous = None
        if not valid[f]:
            continue
        try:
            root, root_r, target, orientations, _ = targets_for_frame(points[f], names, rest, leg_scale, ankle_clearance, floor_z, source_ankle_clearance)
        except ValueError as e:
            valid[f] = False; reason[f] = str(e); previous = None
            continue
        d.qpos[:] = m.qpos0 if previous is None else previous
        d.qpos[:3] = root
        d.qpos[3:7] = np.roll(Rotation.from_matrix(root_r).as_quat(), 1)
        for _ in range(iterations * 3 if previous is None else iterations):
            mujoco.mj_forward(m, d)
            rows, rhs = [], []
            for j, bid in enumerate(target_ids):
                mujoco.mj_jacBody(m, d, jp, jr, bid)
                rows.append(jp[:, vadr].copy() * weight[j])
                rhs.append((target[j] - d.xpos[bid]) * weight[j])
            for body_name, desired_r, w in orientations:
                bid = m.body(body_name).id
                mujoco.mj_jacBody(m, d, jp, jr, bid)
                rows.append(jr[:, vadr].copy() * w)
                rhs.append(Rotation.from_matrix(desired_r @ d.xmat[bid].reshape(3, 3).T).as_rotvec() * w)
            # Soft ground inequality can trade limb fit for collision reduction; root stays fixed.
            for gid in sole_geoms:
                penetration = floor_z + m.geom_size[gid, 0] - d.geom_xpos[gid, 2]
                if penetration > 0:
                    mujoco.mj_jac(m, d, jp, jr, d.geom_xpos[gid], int(m.geom_bodyid[gid]))
                    rows.append(jp[2:3, vadr].reshape(1, 29).copy() * 4.)
                    rhs.append(np.array([penetration * 4.]))
            rows.append(identity * .018); rhs.append((neutral - d.qpos[qadr]) * .018)
            if previous is not None:
                rows.append(identity * .025); rhs.append((previous[qadr] - d.qpos[qadr]) * .025)
            a, e = np.concatenate(rows), np.concatenate(rhs)
            dq = np.linalg.solve(a.T @ a + identity * .0015, a.T @ e)
            d.qpos[qadr] = np.clip(d.qpos[qadr] + np.clip(dq, -.18, .18), limits[:, 0], limits[:, 1])
            if np.linalg.norm(dq) < 1e-5:
                break
        mujoco.mj_forward(m, d)
        qp[f], xpos[f], xquat[f] = d.qpos, d.xpos, d.xquat
        if previous is not None and np.dot(qp[f, 3:7], previous[3:7]) < 0:
            qp[f, 3:7] *= -1
        targets[f] = target
        errors[f] = np.linalg.norm(d.xpos[target_ids] - target, axis=1)
        soles[f] = d.geom_xpos[sole_geoms]
        soles[f, :, 2] -= m.geom_size[sole_geoms, 0]
        previous = qp[f].copy()
        if f % 100 == 0:
            print(json.dumps({"frame": f, "total": len(times), "mean_target_fit_m": float(errors[f].mean())}), flush=True)
    qvel = np.full((len(times), m.nv), np.nan)
    links = links_between_frames(times, valid, max_gap_s)
    for f in np.flatnonzero(valid):
        a = f - 1 if f and links[f] else f
        b = f + 1 if f + 1 < len(times) and links[f + 1] else f
        if b > a:
            mujoco.mj_differentiatePos(m, qvel[f], times[b] - times[a], qp[a], qp[b])
    if not valid.any():
        raise ValueError("All frames rejected due to degenerate anatomical geometry")
    foot_positions = np.full((len(times), 2, 3), np.nan)
    foot_near_floor = np.zeros((len(times), 2), bool)
    foot_speed_xy = np.full((len(times), 2), np.nan)
    for side_index, side in enumerate(("left", "right")):
        members = [j for j, gid in enumerate(sole_geoms) if m.body(m.geom_bodyid[gid]).name.startswith(side)]
        foot_positions[valid, side_index] = soles[valid][:, members].mean(axis=1)
        foot_near_floor[valid, side_index] = soles[valid][:, members, 2].min(axis=1) <= floor_z + .025
    for f in np.flatnonzero(links):
        foot_speed_xy[f] = np.linalg.norm(foot_positions[f, :, :2] - foot_positions[f - 1, :, :2], axis=1) / (times[f] - times[f - 1])
    near_ground_speeds = foot_speed_xy[foot_near_floor & np.isfinite(foot_speed_xy)]
    yaw_step = np.full(len(times), np.nan)
    yaw_speed = np.full(len(times), np.nan)
    yaw = 2 * np.arctan2(qp[:, 6], qp[:, 3])
    for f in np.flatnonzero(links):
        yaw_step[f] = (yaw[f] - yaw[f - 1] + np.pi) % (2 * np.pi) - np.pi
        yaw_speed[f] = yaw_step[f] / (times[f] - times[f - 1])
    max_limit_excess = float(np.maximum(np.maximum(limits[:, 0] - qp[valid, 7:], qp[valid, 7:] - limits[:, 1]), 0).max())
    start, end = np.flatnonzero(valid)[[0, -1]]
    provenance_path = Path(model).parent / "SOURCE.json"
    model_provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else {}
    meta = {
        "source": str(Path(source).resolve()), "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "source_metadata": source_meta, "metric_scale_status": source_meta.get("metric_scale_status", "unspecified_unverified"),
        "source_type": source_meta.get("source_type", "unspecified"),
        "model": str(Path(model).resolve()), "model_sha256": hashlib.sha256(Path(model).read_bytes()).hexdigest(),
        "model_repository": model_provenance.get("repository"), "model_commit": model_provenance.get("commit"),
        "official_model_sha256": model_provenance.get("model_sha256"),
        "units": "m", "world_up": "z", "world_frame_id": source_meta.get("world_frame_id", "source_fixed_world"),
        "qpos_layout": "xyz + quaternion wxyz + 29 hinge angles rad in joint_names order",
        "model_local_axes": "+X forward, +Y left, +Z up", "method": "bounded damped body-joint IK with posture and temporal priors",
        "root_xy_transform": "identity: human hip midpoint world XY retained; no drift or loop correction",
        "root_z_transform": "floor + (human hip z-floor-source_ankle_clearance)*leg_scale + sole_clearance - rotated model hip offset z",
        "human_median_leg_length_m": human_leg, "leg_scale": leg_scale, "sole_clearance_m": ankle_clearance,
        "source_ankle_clearance_m": source_ankle_clearance, "source_ankle_clearance_method": source_ankle_clearance_method,
        "root_orientation": "hip-line heading plus gravity upright prior; pelvis tilt unobserved",
        "foot_orientation": "weak flat-foot heading prior; not a measured foot orientation",
        "ground_constraint": "soft sole nonpenetration penalty; no root adjustment",
        "unobserved_dofs": "wrist orientation, twist distribution and pelvis tilt depend on priors",
        "simulation": "kinematic reference replay; dynamics, balance, tracking policy and torque feasibility not validated",
        "frames": len(times), "valid_frames": int(valid.sum()), "invalid_frames": int((~valid).sum()),
        "max_gap_s": max_gap_s, "mean_target_fit_m": float(np.nanmean(errors)),
        "p95_target_fit_m": float(np.nanpercentile(errors, 95)), "max_joint_limit_excess_rad": max_limit_excess,
        "max_sole_penetration_m": float(np.maximum(floor_z - soles[valid, :, 2], 0).max()),
        "p95_near_ground_foot_speed_xy_mps": float(np.percentile(near_ground_speeds, 95)) if near_ground_speeds.size else None,
        "near_ground_foot_speed_interpretation": "potential foot skating diagnostic for sole height <= 0.025 m; proximity is not measured contact",
        "max_joint_speed_rad_s": float(np.nanmax(np.abs(qvel[:, 6:]))) if np.isfinite(qvel[:, 6:]).any() else None,
        "max_root_yaw_speed_deg_s": float(np.nanmax(np.abs(np.degrees(yaw_speed)))) if np.isfinite(yaw_speed).any() else None,
        "root_yaw_steps_over_90deg": indices[np.abs(yaw_step) > np.pi / 2].tolist(),
        "root_heading_smoothing": "none; abrupt heading diagnostics exported for review",
        "human_endpoint_displacement_xy_m": float(np.linalg.norm(human_pelvis[end, :2] - human_pelvis[start, :2])),
        "robot_endpoint_displacement_xy_m": float(np.linalg.norm(qp[end, :2] - qp[start, :2])),
        "quality_interpretation": "target fit is IK residual, not world reconstruction accuracy; endpoints are measurements, not a loop claim",
    }
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.stem + ".partial.npz")
    np.savez_compressed(temp, qpos=qp, qvel=qvel, xpos=xpos, xquat=xquat, pelvis_world=qp[:, :3],
        human_pelvis_world=human_pelvis, timestamps_s=times, timestamps=times, frame_indices=indices,
        valid=valid, invalid_reason=reason, temporal_link_from_previous=links, joint_names=np.array([m.joint(j).name for j in joint_ids]),
        body_names=np.array([m.body(j).name for j in range(m.nbody)]), target_body_names=np.array(TARGET_NAMES),
        target_positions=targets, ik_residual_m=errors, sole_contact_positions=soles,
        foot_positions=foot_positions, foot_near_floor=foot_near_floor, foot_speed_xy_mps=foot_speed_xy,
        root_yaw_step_rad=yaw_step, root_yaw_speed_rad_s=yaw_speed,
        metric_scale_status=meta["metric_scale_status"], metadata=json.dumps(meta))
    temp.replace(output)
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return meta

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--min-confidence", type=float, default=.1)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--max-gap-s", type=float, default=.2)
    parser.add_argument("--floor-z", type=float, default=0.)
    args = parser.parse_args()
    run(args.source, args.output, args.model, args.min_confidence, args.iterations, args.max_gap_s, args.floor_z)
