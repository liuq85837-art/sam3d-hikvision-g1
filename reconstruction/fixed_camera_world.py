#!/usr/bin/env python3
"""One fixed gauge for monocular SAM 3D Body geometry; never closes a trajectory.

Input: camera_sequence.npz or its directory (MHR70, optical x-right/y-down/z-forward).
Explicit calibration: T_world_from_camera or R_world_from_camera and
t_world_from_camera_m. A scale multiplies all camera points before the rigid
transform, preserving their image projection. No per-frame foot/root correction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

MHR_NAMES = [
    'nose', 'left-eye', 'right-eye', 'left-ear', 'right-ear',
    'left-shoulder', 'right-shoulder', 'left-elbow', 'right-elbow',
    'left-hip', 'right-hip', 'left-knee', 'right-knee', 'left-ankle', 'right-ankle',
    'left-big-toe-tip', 'left-small-toe-tip', 'left-heel',
    'right-big-toe-tip', 'right-small-toe-tip', 'right-heel',
]
FOOT_IDS = [15, 16, 17, 18, 19, 20]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def json_write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def unit(v):
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm < 1e-10:
        raise ValueError('Cannot normalize a degenerate direction')
    return v / norm


def validate_rigid(T):
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError('Camera-to-world transform must be finite 4x4')
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError('Invalid homogeneous transform bottom row')
    R = T[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R), 1, atol=1e-5):
        raise ValueError('Camera-to-world rotation must be proper SO(3), without reflection or scale')
    return T


def load_sequence(source):
    source = Path(source)
    path = source / 'camera_sequence.npz' if source.is_dir() else source
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    key = 'joints_camera' if 'joints_camera' in data else 'camera_joints_m'
    joints = np.asarray(data[key], dtype=float)
    if joints.ndim != 3 or joints.shape[1:] != (70, 3):
        raise ValueError('Expected MHR70 camera joints [T,70,3]')
    n = len(joints)
    if n < 2:
        raise ValueError('At least two selected frames are required')
    if 'joint_names' in data:
        names = [str(x).replace('_', '-') for x in data['joint_names']]
        if names[:21] != MHR_NAMES:
            raise ValueError('Joint order is not the official MHR70 order')
    else:
        raise ValueError('joint_names is required to verify the MHR70 mapping')
    if 'timestamps_ns' in data:
        raw_ts = np.asarray(data['timestamps_ns'])
        if raw_ts.dtype.kind not in 'iu':
            raise ValueError('timestamps_ns must retain an exact integer dtype')
        ts = raw_ts.astype(np.int64)
    elif 'timestamps_s' in data:
        seconds = np.asarray(data['timestamps_s'], dtype=float)
        if not np.isfinite(seconds).all():
            raise ValueError('Nonfinite timestamps_s')
        ts = np.rint((seconds - seconds[0]) * 1e9).astype(np.int64)
    else:
        raise ValueError('timestamps_ns or timestamps_s is required; FPS is not guessed')
    frames = np.asarray(data.get('frame_indices', np.arange(n)), dtype=np.int64)
    if ts.shape != (n,) or frames.shape != (n,) or np.any(np.diff(ts) <= 0) or np.any(np.diff(frames) <= 0):
        raise ValueError('Frame indices and timestamps must strictly increase and match T')
    if 'timestamps_s' in data:
        seconds = np.asarray(data['timestamps_s'], dtype=float)
        if seconds.shape != (n,) or not np.isfinite(seconds).all() or np.any(np.diff(seconds) <= 0):
            raise ValueError('timestamps_s must be finite, increasing, and match T')
        if not np.allclose(seconds-seconds[0], (ts-ts[0])/1e9, atol=2e-6, rtol=1e-7):
            raise ValueError('timestamps_s and timestamps_ns have different relative timing')
    valid = np.asarray(data.get('valid', np.ones(n, dtype=bool)), dtype=bool).copy()
    if valid.shape != (n,):
        raise ValueError('valid must have shape [T]')
    finite = np.isfinite(joints).all(axis=(1, 2)) & (joints[:, :, 2] > 0).all(axis=1)
    valid &= finite
    if valid.sum() < 2:
        raise ValueError('Fewer than two valid positive-depth MHR70 frames')
    return path, data, joints, ts, frames, valid


def infer_ground_gauge(joints, valid, scale=1.0, seed=42):
    """Infer one plane/origin from model geometry, keeping every motion sample intact.

    The body-up fallback is important: straight walking or a short standing clip
    can make the foot cloud poorly conditioned for unconstrained plane fitting.
    This is a display gauge, not a metric calibration or observed contact label.
    """
    p = joints[valid] * scale
    pelvis = p[:, [9, 10]].mean(axis=1)
    body = p[:, [5, 6]].mean(axis=1) - pelvis
    lengths = np.linalg.norm(body, axis=1)
    usable = lengths > .08 * scale
    if usable.sum() < 2:
        raise ValueError('Insufficient nondegenerate body-up observations')
    directions = body[usable] / lengths[usable, None]
    up = unit(np.median(directions, axis=0))
    angle = np.degrees(np.arccos(np.clip(directions @ up, -1, 1)))
    feet = p[:, FOOT_IDS]
    heights = feet @ up
    # Low foot points are candidate supports only; no contact truth is inferred.
    low = heights <= heights.min(axis=1, keepdims=True) + .035 * scale
    candidates = feet[low]
    if len(candidates) > 10000:
        candidates = candidates[np.linspace(0, len(candidates)-1, 10000).astype(int)]
    threshold = .035 * scale
    rng = np.random.default_rng(seed)
    best_mask = np.zeros(len(candidates), dtype=bool)
    best_error = float('inf')
    best_normal = up.copy()
    for _ in range(512 if len(candidates) >= 3 else 0):
        a, b, c = candidates[rng.choice(len(candidates), 3, replace=False)]
        cross = np.cross(b-a, c-a)
        if np.linalg.norm(cross) < 1e-7 * scale * scale:
            continue
        normal = unit(cross)
        if normal @ up < 0:
            normal = -normal
        if normal @ up < np.cos(np.radians(35)):
            continue
        residual = np.abs((candidates-a) @ normal)
        mask = residual < threshold
        err = float(np.median(residual[mask]))
        if mask.sum() > best_mask.sum() or (mask.sum() == best_mask.sum() and err < best_error):
            best_mask, best_error, best_normal = mask, err, normal
    method = 'body_up_and_median_low_foot_height'
    normal = up.copy()
    offset = float(np.median(candidates @ normal))
    spread = [0., 0., 0.]
    fit_fraction = float(best_mask.mean())
    if best_mask.sum() >= 12:
        chosen = candidates[best_mask]
        center = np.median(chosen, axis=0)
        _, singular, vh = np.linalg.svd(chosen-center, full_matrices=False)
        spread = (singular / np.sqrt(len(chosen))).tolist()
        candidate_n = vh[-1]
        if candidate_n @ up < 0:
            candidate_n = -candidate_n
        # Require support in two horizontal directions, not just a line.
        if fit_fraction >= .35 and singular[1] / np.sqrt(len(chosen)) > .035 * scale and candidate_n @ up >= np.cos(np.radians(35)):
            normal = unit(candidate_n)
            offset = float(np.median(chosen @ normal))
            method = 'robust_low_foot_cloud_plane_with_body_up_prior'
    horizontal = np.array([1., 0., 0.])
    horizontal -= normal * (normal @ horizontal)
    if np.linalg.norm(horizontal) < .1:
        horizontal = np.array([0., 0., 1.])
        horizontal -= normal * (normal @ horizontal)
    x = unit(horizontal)
    y = unit(np.cross(normal, x))
    R = np.stack([x, y, normal])
    initial_pelvis = np.median(pelvis[:min(15, len(pelvis))], axis=0)
    origin = initial_pelvis - normal * (initial_pelvis @ normal - offset)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ origin
    residuals = candidates @ normal - offset
    diagnostics = {
        'method': method, 'measured_ground': False, 'contact_labels_available': False,
        'normal_in_scaled_camera_coordinates': normal.tolist(),
        'plane_equation': 'normal dot (metric_scale * camera_point) = offset_m',
        'offset_m': offset, 'body_up_direction': up.tolist(),
        'body_up_angle_p90_deg': float(np.percentile(angle, 90)),
        'candidate_count': len(candidates), 'ransac_best_inlier_fraction': fit_fraction,
        'inlier_cloud_principal_spread_m': spread,
        'candidate_plane_abs_residual_median_m': float(np.median(abs(residuals))),
        'candidate_plane_abs_residual_p90_m': float(np.percentile(abs(residuals), 90)),
        'origin': 'Initial valid pelvis median projected to the inferred plane',
        'yaw': 'Camera optical right projected onto inferred ground defines world +x',
        'limitations': [
            'Human pose, body lean, foot occlusion and monocular depth errors bias this inferred plane.',
            'Stationary camera is assumed, not established by this geometry module.',
            'No independently measured floor, gravity, scene yaw or metric scale is established.',
            'A single constant transform is used; foot heights and endpoint displacement are not corrected.',
        ],
    }
    return validate_rigid(T), diagnostics


def closure_metrics(pelvis, timestamps_ns, valid, endpoint_window_s=.25, frame_indices=None):
    """Descriptive endpoint closure only; no imposed start/end correspondence."""
    p = np.asarray(pelvis, dtype=float)
    t = (np.asarray(timestamps_ns, dtype=np.int64) - timestamps_ns[0]) / 1e9
    good = np.asarray(valid, dtype=bool) & np.isfinite(p).all(axis=1)
    ids = np.flatnonzero(good)
    if len(ids) < 2:
        return {'status': 'insufficient_valid_frames', 'trajectory_corrected_for_closure': False}
    first, last = int(ids[0]), int(ids[-1])
    # Very short clips must not use overlapping start/end windows: that would
    # manufacture a zero endpoint distance by averaging the same observations.
    effective_window = min(float(endpoint_window_s), float(t[last]-t[first]) * .25)
    start_ids = ids[t[ids] <= t[first] + effective_window]
    end_ids = ids[t[ids] >= t[last] - effective_window]
    start = np.median(p[start_ids], axis=0)
    end = np.median(p[end_ids], axis=0)
    delta = end-start
    dt = np.diff(t)
    median_dt = float(np.median(dt))
    max_gap = max(.1, median_dt * 2.5)
    pair = good[:-1] & good[1:] & (dt <= max_gap)
    speed = np.linalg.norm(np.diff(p, axis=0), axis=1)
    path = float(speed[pair].sum())
    observed_duration = float(dt[pair].sum())
    result = {
        'status': 'descriptive_only_no_ground_truth_return_correspondence',
        'trajectory_corrected_for_closure': False,
        'requested_endpoint_window_s': endpoint_window_s,
        'endpoint_window_s': effective_window,
        'start_selected_index': first, 'end_selected_index': last,
        'start_timestamp_s': float(t[first]), 'end_timestamp_s': float(t[last]),
        'valid_frames': int(good.sum()), 'selected_frames': len(good),
        'start_valid_window_frames': len(start_ids), 'end_valid_window_frames': len(end_ids),
        'start_pelvis_m': start.tolist(), 'end_pelvis_m': end.tolist(),
        'endpoint_displacement_xyz_m': delta.tolist(),
        'endpoint_displacement_3d_m': float(np.linalg.norm(delta)),
        'endpoint_displacement_xy_m': float(np.linalg.norm(delta[:2])),
        'endpoint_height_difference_m': float(delta[2]),
        'raw_contiguous_path_length_m': path,
        'endpoint_to_path_ratio': float(np.linalg.norm(delta) / path) if path > 1e-8 else None,
        'observed_contiguous_duration_s': observed_duration,
        'omitted_invalid_or_gap_pairs': int((~pair).sum()),
        'raw_speed_p95_m_s': float(np.percentile(speed[pair] / dt[pair], 95)) if pair.any() else None,
        'notes': [
            'Endpoint windows are fixed at the first and last valid observations, never selected to minimize closure error.',
            'A small return distance alone does not establish trajectory accuracy or that the subject performed a loop.',
            'Body pose at the start and end may differ; raw path length includes model jitter.',
            'If scale is model-prior-only, quantities labelled m are nominal model metres, not verified physical metres.',
        ],
    }
    if frame_indices is not None:
        result['start_frame_index'] = int(frame_indices[first])
        result['end_frame_index'] = int(frame_indices[last])
    return result


def reconstruct(source, output, calibration=None, metric_scale=None, endpoint_window_s=.25):
    path, data, joints, ts, frames, valid = load_sequence(source)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    calibration_path = Path(calibration) if calibration else None
    cal = json.loads(calibration_path.read_text(encoding='utf-8-sig')) if calibration_path else {}
    if metric_scale is not None and 'metric_scale' in cal and not np.isclose(float(metric_scale), float(cal['metric_scale'])):
        raise ValueError('CLI and calibration metric_scale disagree')
    scale = float(metric_scale if metric_scale is not None else cal.get('metric_scale', 1.))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('metric_scale must be finite and positive')
    if endpoint_window_s <= 0:
        raise ValueError('endpoint_window_s must be positive')
    scale_supplied = metric_scale is not None or 'metric_scale' in cal
    scale_kind = 'user_supplied_constant_scale' if scale_supplied else 'model_prior_only'
    if cal:
        if cal.get('world_up', 'z') != 'z':
            raise ValueError('This export requires an explicit z-up world convention')
        if 'T_world_from_camera' in cal:
            T = validate_rigid(cal['T_world_from_camera'])
        elif 'R_world_from_camera' in cal and 't_world_from_camera_m' in cal:
            T = np.eye(4)
            T[:3, :3] = np.asarray(cal['R_world_from_camera'], dtype=float)
            T[:3, 3] = np.asarray(cal['t_world_from_camera_m'], dtype=float)
            T = validate_rigid(T)
        else:
            raise ValueError('Calibration must explicitly specify camera-to-world R/t or T; world-to-camera is not silently accepted')
        diagnostics = {'method': 'explicit_camera_to_world_calibration',
                       'provenance': cal.get('provenance', 'user supplied; measurement provenance unspecified')}
        gauge_kind = 'explicit_fixed_camera_world'
    else:
        T, diagnostics = infer_ground_gauge(joints, valid, scale)
        gauge_kind = 'inferred_display_ground_world'
    world = (joints * scale) @ T[:3, :3].T + T[:3, 3]
    world[~valid] = np.nan
    pelvis = world[:, [9, 10]].mean(axis=1)
    identity = hashlib.sha256(json.dumps({'T': T.tolist(), 'scale': scale, 'source': digest(path)}, sort_keys=True).encode()).hexdigest()[:16]
    gauge = {
        'schema_version': 1, 'world_frame_id': cal.get('world_frame_id', 'hik_fixed_' + identity),
        'gauge_kind': gauge_kind, 'world_up': 'z', 'handedness': 'right',
        'source_coordinate_frame': 'camera_optical_x_right_y_down_z_forward',
        'units': 'm', 'physical_metric_accuracy_verified': False,
        'scale_status': scale_kind, 'metric_scale_status': scale_kind, 'metric_scale': scale,
        'scale_provenance': cal.get('scale_provenance', 'explicit constant without supplied measurement evidence' if scale_supplied else 'SAM 3D Body model prior'),
        'T_world_from_camera': T.tolist(),
        'transform_formula': 'world_point = R_world_from_camera @ (metric_scale * camera_point) + t_world_from_camera_m',
        'pelvis_definition': 'midpoint of MHR70 left-hip 9 and right-hip 10; not an original MHR joint',
        'source_path': str(path.resolve()), 'source_sha256': digest(path),
        'calibration_path': str(calibration_path.resolve()) if calibration_path else None,
        'calibration_sha256': digest(calibration_path) if calibration_path else None,
        'per_frame_correction_applied': False, 'loop_constraint_applied': False,
        'diagnostics': diagnostics,
    }
    metrics = closure_metrics(pelvis, ts, valid, endpoint_window_s, frames)
    metrics['world_frame_id'] = gauge['world_frame_id']
    metrics['scale_status'] = scale_kind
    metrics['gauge_kind'] = gauge_kind
    output_seconds = np.asarray(data['timestamps_s'], dtype=float) if 'timestamps_s' in data else (ts-ts[0])/1e9
    metrics['source_video_start_time_s'] = float(output_seconds[metrics['start_selected_index']])
    metrics['source_video_end_time_s'] = float(output_seconds[metrics['end_selected_index']])
    exported = {k: v for k, v in data.items() if k not in ('metadata', 'joints_world', 'joints_world_raw', 'pelvis_world')}
    exported.update(joints_world=world.astype(np.float32), world_joints=world.astype(np.float32), joints_world_raw=world.astype(np.float32),
                    pelvis_world=pelvis.astype(np.float32), joints_camera=joints.astype(np.float32),
                    valid=valid, source_person_valid=np.asarray(data.get('valid', valid), dtype=bool),
                    frame_indices=frames, timestamps_ns=ts, timestamp_ns=ts,
                    timestamps_s=output_seconds, T_world_from_camera=T,
                    constant_body_scale=np.float64(scale), world_frame_id=np.array(gauge['world_frame_id']),
                    metric_scale_status=np.array(scale_kind),
                    metadata=np.array(json.dumps(gauge, ensure_ascii=False)))
    if 'metadata' in data:
        exported['source_metadata'] = data['metadata']
    if ('keypoints_2d_calibrated' in data and 'keypoints_2d' not in data
            and str(data.get('keypoints_2d_image_space', '')) == 'source_video_pixels'):
        exported['keypoints_2d'] = data['keypoints_2d_calibrated']
    temporary = output / 'world_joints.tmp.npz'
    np.savez_compressed(temporary, **exported)
    temporary.replace(output / 'world_joints.npz')
    json_write(output / 'gauge.json', gauge)
    json_write(output / 'closure_metrics.json', metrics)
    json_write(output / 'status.json', {'status': 'complete', 'valid_frames': int(valid.sum()),
                                      'selected_frames': len(valid), 'gauge_kind': gauge_kind,
                                      'scale_status': scale_kind, 'loop_constraint_applied': False})
    return gauge, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--metric-scale', type=float)
    parser.add_argument('--endpoint-window-s', type=float, default=.25)
    args = parser.parse_args()
    gauge, metrics = reconstruct(args.input, args.output, args.calibration, args.metric_scale, args.endpoint_window_s)
    print(json.dumps({'world_frame_id': gauge['world_frame_id'], 'gauge_kind': gauge['gauge_kind'],
                      'scale_status': gauge['scale_status'],
                      'endpoint_displacement_3d_m': metrics['endpoint_displacement_3d_m']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
