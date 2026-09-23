#!/usr/bin/env python3
"""Render synchronized source video / world skeleton and actual MuJoCo G1 qpos.

All coordinates are preserved. Rendering does not close a loop, align endpoints,
remove drift, or certify that a kinematic motion is dynamically feasible.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
    import mujoco
except ImportError:
    # Numerical clock/trajectory checks remain usable before graphics installation.
    cv2 = mujoco = None

HUMAN = (43, 208, 221)
ROBOT = (255, 174, 73)
BG = (13, 21, 31)
CARD = (23, 35, 48)
TEXT = (235, 240, 245)
MUTED = (163, 178, 195)

# Aliases include COCO, SMPL, and the MHR names exported by the supplied project.
BONES = [
    ('nose', 'left_eye'), ('nose', 'right_eye'), ('left_eye', 'left_ear'),
    ('right_eye', 'right_ear'), ('left_ear', 'left_shoulder'), ('right_ear', 'right_shoulder'),
    ('left_shoulder', 'right_shoulder'), ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'), ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'), ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'), ('left_hip', 'right_hip'),
    ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
    ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
    ('left_ankle', 'left_big_toe'), ('right_ankle', 'right_big_toe'),
    ('left_ankle', 'left_heel'), ('right_ankle', 'right_heel'),
    ('neck', 'head'), ('pelvis', 'spine'), ('spine', 'neck'),
    ('neck', 'left_shoulder'), ('neck', 'right_shoulder'),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(name):
    name = str(name).lower().replace('-', '_').replace(' ', '_')
    aliases = {'lhip': 'left_hip', 'rhip': 'right_hip', 'lknee': 'left_knee',
               'rknee': 'right_knee', 'lankle': 'left_ankle', 'rankle': 'right_ankle',
               'lshoulder': 'left_shoulder', 'rshoulder': 'right_shoulder',
               'lelbow': 'left_elbow', 'relbow': 'right_elbow',
               'lwrist': 'left_wrist', 'rwrist': 'right_wrist',
               'root': 'pelvis', 'hip': 'pelvis', 'spine1': 'spine',
               'left_foot': 'left_big_toe', 'right_foot': 'right_big_toe',
               'left_big_toe_tip': 'left_big_toe', 'right_big_toe_tip': 'right_big_toe'}
    return aliases.get(name, name)


def load_sequence(path, kind):
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    if 'timestamps_s' in a:
        times = np.asarray(a['timestamps_s'], float)
    elif 'timestamps_ns' in a:
        # Subtract in integer space to retain sub-millisecond precision.
        origin = int(a['timestamps_ns'][0])
        times = (a['timestamps_ns'] - origin).astype(float) / 1e9
        a['_timestamp_origin_ns'] = origin
    else:
        raise ValueError(f'{kind}: timestamps_s or timestamps_ns is required')
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f'{kind}: expected >=2 strictly increasing finite timestamps')
    a['times'] = times
    if kind == 'human':
        a['points'] = np.asarray(a.get('world_joints', a.get('joints_world')), float)
        if a['points'].ndim != 3 or a['points'].shape[0] != len(times) or a['points'].shape[-1] != 3:
            raise ValueError('human: world_joints/joints_world must have shape [N,J,3]')
        a['names'] = [canonical(s) for s in a['joint_names']]
        if len(a['names']) != a['points'].shape[1]:
            raise ValueError('human: joint_names length does not match joint dimension')
        indices = {s: i for i, s in enumerate(a['names'])}
        a['bones'] = [(indices[x], indices[y]) for x, y in BONES if x in indices and y in indices]
        a['pelvis_2d_indices'] = ([indices['pelvis']] if 'pelvis' in indices else
                                  [indices['left_hip'], indices['right_hip']] if 'left_hip' in indices and 'right_hip' in indices else None)
        if 'pelvis_world' in a:
            a['pelvis'] = np.asarray(a['pelvis_world'], float)
        elif 'pelvis' in indices:
            a['pelvis'] = a['points'][:, indices['pelvis']]
        elif 'left_hip' in indices and 'right_hip' in indices:
            a['pelvis'] = (a['points'][:, indices['left_hip']] + a['points'][:, indices['right_hip']]) / 2
        else:
            raise ValueError('human: pelvis_world or named pelvis / left_hip+right_hip required')
        finite = np.isfinite(a['pelvis']).all(axis=1)
    else:
        a['qpos'] = np.asarray(a['qpos'], float)
        if a['qpos'].ndim != 2 or len(a['qpos']) != len(times) or a['qpos'].shape[1] < 7:
            raise ValueError('robot: expected qpos [N,nq], beginning xyz+wxyz')
        finite = np.isfinite(a['qpos']).all(axis=1)
        norm = np.linalg.norm(a['qpos'][:, 3:7], axis=1)
        if np.any(finite & (np.abs(norm - 1) > .01)):
            raise ValueError('robot: root quaternion must be unit wxyz')
        a['pelvis'] = a['qpos'][:, :3].copy()
    valid = np.asarray(a.get('valid', np.ones(len(times), bool)), bool)
    if valid.shape != (len(times),):
        raise ValueError(f'{kind}: valid must be shape [N]')
    a['valid'] = valid & finite
    if not a['valid'].any():
        raise ValueError(f'{kind}: no finite valid poses')
    a['path'] = str(Path(path).resolve())
    return a


def align_clock_origins(human, robot):
    ho, ro = human.get('_timestamp_origin_ns'), robot.get('_timestamp_origin_ns')
    if (ho is None) != (ro is None):
        raise ValueError('Both inputs must use timestamps_s, or both timestamps_ns, to establish a shared clock')
    if ho is not None:
        robot['times'] = robot['times'] + (ro - ho) / 1e9


def sample(seq, ts, max_gap_s):
    t = seq['times']
    hi = int(np.searchsorted(t, ts, side='left'))
    if hi < len(t) and abs(t[hi] - ts) < 1e-7:
        lo = hi
    else:
        lo = hi - 1
    if lo < 0 or hi >= len(t):
        return None
    if not seq['valid'][lo] or not seq['valid'][hi] or t[hi] - t[lo] > max_gap_s:
        return None
    alpha = 0. if lo == hi else float((ts - t[lo]) / (t[hi] - t[lo]))
    return lo, hi, alpha


def interpolate(a, support):
    lo, hi, alpha = support
    return a[lo] * (1 - alpha) + a[hi] * alpha


def sample_qpos(seq, support):
    q = interpolate(seq['qpos'], support)
    lo, hi, alpha = support
    qa, qb = seq['qpos'][lo, 3:7], seq['qpos'][hi, 3:7].copy()
    dot = float(qa @ qb)
    if dot < 0:
        qb = -qb
        dot = -dot
    if dot > .9995:
        quat = qa * (1 - alpha) + qb * alpha
    else:
        theta = math.acos(np.clip(dot, -1, 1))
        quat = (math.sin((1 - alpha) * theta) * qa + math.sin(alpha * theta) * qb) / math.sin(theta)
    q[3:7] = quat / np.linalg.norm(quat)
    return q


def path_metrics(seq, max_gap_s):
    indices = np.flatnonzero(seq['valid'])
    p = seq['pelvis']
    valid_links = seq['valid'][:-1] & seq['valid'][1:] & (np.diff(seq['times']) <= max_gap_s)
    delta = p[indices[-1]] - p[indices[0]]
    length = float(np.linalg.norm(np.diff(p[:, :2], axis=0)[valid_links], axis=1).sum())
    return dict(start_index=int(indices[0]), end_index=int(indices[-1]),
                start_timestamp_s=float(seq['times'][indices[0]]), end_timestamp_s=float(seq['times'][indices[-1]]),
                endpoint_displacement_xyz_m=delta.tolist(), endpoint_distance_3d_m=float(np.linalg.norm(delta)),
                endpoint_distance_xy_m=float(np.linalg.norm(delta[:2])), path_length_xy_m=length,
                endpoint_distance_over_path_length=float(np.linalg.norm(delta[:2]) / length) if length > 1e-8 else None,
                valid_frame_count=int(seq['valid'].sum()), invalid_frame_count=int((~seq['valid']).sum()),
                unsupported_link_count=int((~valid_links).sum()),
                closure_interpretation='Endpoint displacement only; a loop is not assumed or enforced')


def line(scene, a, b, color, radius=.007):
    if scene.ngeom >= scene.maxgeom or not np.isfinite([a, b]).all() or np.linalg.norm(np.asarray(b) - a) < 1e-7:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array(color, float))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, np.asarray(a, float), np.asarray(b, float))
    scene.ngeom += 1


def sphere(scene, point, color, radius=.027):
    if scene.ngeom >= scene.maxgeom or not np.isfinite(point).all():
        return
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                       np.array([radius, 0, 0]), np.asarray(point, float), np.eye(3).ravel(), np.array(color, float))
    scene.ngeom += 1


def draw_world(scene, seq, ts, color, max_gap_s, ground_z, bounds, current=None, grid=True):
    rgba = (*np.array(color) / 255., 1.)
    idx = np.flatnonzero(seq['valid'] & (seq['times'] <= ts))
    # Preserve gap boundaries while limiting the number of scene geoms.
    if len(idx) > 1:
        pieces = np.split(idx, np.flatnonzero((np.diff(idx) > 1) | (np.diff(seq['times'][idx]) > max_gap_s)) + 1)
        for piece in pieces:
            stride = max(1, math.ceil(len(piece) / 350))
            selected = np.unique(np.r_[piece[::stride], piece[-1]])
            for i, j in zip(selected[:-1], selected[1:]):
                line(scene, seq['pelvis'][i], seq['pelvis'][j], rgba, .010)
                a, b = seq['pelvis'][i].copy(), seq['pelvis'][j].copy()
                a[2] = b[2] = ground_z + .012
                line(scene, a, b, (*rgba[:3], .34), .008)
    if current is not None:
        sphere(scene, current, rgba, .043)
        foot = np.array([current[0], current[1], ground_z + .015])
        line(scene, foot, current, (*rgba[:3], .35), .005)
    if not grid:
        return
    lo, hi = bounds
    step = max(.5, math.ceil(max(hi - lo) / 12 / .5) * .5)
    for x in np.arange(math.floor(lo[0] / step) * step, hi[0] + step, step):
        line(scene, [x, lo[1], ground_z + .002], [x, hi[1], ground_z + .002], [.39, .46, .54, .34], .002)
    for y in np.arange(math.floor(lo[1] / step) * step, hi[1] + step, step):
        line(scene, [lo[0], y, ground_z + .002], [hi[0], y, ground_z + .002], [.39, .46, .54, .34], .002)
    origin = [0, 0, ground_z + .02]
    for delta, c in [([.5, 0, 0], [1, .3, .3, 1]), ([0, .5, 0], [.3, .9, .4, 1]), ([0, 0, .5], [.3, .55, 1, 1])]:
        line(scene, origin, np.asarray(origin) + delta, c, .006)


@lru_cache(maxsize=32)
def font(size, bold=False):
    paths = [Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts' / ('msyhbd.ttc' if bold else 'msyh.ttc'),
             Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')]
    for path in paths:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def fit_image(rgb, width, height):
    canvas = np.full((height, width, 3), BG, np.uint8)
    scale = min(width / rgb.shape[1], height / rgb.shape[0])
    w, h = max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale))
    resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    left, top = (width - w) // 2, (height - h) // 2
    canvas[top:top + h, left:left + w] = resized
    return canvas, scale, (left, top)


class SourceVideo:
    def __init__(self, path, human, camera_path=None, offset_s=0.):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise ValueError(f'Cannot open source video: {path}')
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.fps <= 0:
            raise ValueError('Source video FPS must be available')
        self.human = human
        self.frame = -1
        self.image = None
        self.offset_s = offset_s
        self.camera = json.loads(Path(camera_path).read_text(encoding='utf-8-sig')) if camera_path else None

    def read(self, ts, support):
        human = self.human
        if support is not None and 'frame_indices' in human:
            lo, hi, alpha = support
            index = int(human['frame_indices'][hi if alpha > .5 else lo])
        else:
            index = int(round((ts - self.offset_s) * self.fps))
        if index < 0:
            return None, index, None
        if index != self.frame:
            if index != self.frame + 1:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = self.cap.read()
            self.image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None
            self.frame = index
        points = None
        if support is not None:
            if 'keypoints_2d' in human:
                # Source-video overlays use the same source frame, never a blended detection.
                lo, hi, alpha = support
                points = np.asarray(human['keypoints_2d'][hi if alpha > .5 else lo, :, :2], float)
                if points.shape != (len(human['names']), 2):
                    raise ValueError('keypoints_2d must use the same joint order as joint_names')
                if human['keypoints_2d'].shape[-1] >= 3:
                    conf = human['keypoints_2d'][hi if alpha > .5 else lo, :, 2]
                    points = points.copy()
                    points[conf < .25] = np.nan
            elif self.camera is not None:
                lo, hi, alpha = support
                points = project_world(human['points'][hi if alpha > .5 else lo], self.camera)
        return None if self.image is None else self.image.copy(), index, points


def project_world(points, camera):
    k = np.asarray(camera['K'], float)
    twc = np.asarray(camera['T_world_camera'], float)
    pc = (points - twc[:3, 3]) @ twc[:3, :3]
    projected, _ = cv2.projectPoints(pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), k,
                                    np.asarray(camera.get('distortion', [0, 0, 0, 0, 0]), float))
    projected = projected.reshape(-1, 2)
    projected[pc[:, 2] <= 0] = np.nan
    return projected


def overlay_pose(rgb, points, bones, pelvis_indices=None):
    if points is None:
        return rgb
    h, w = rgb.shape[:2]
    good = np.isfinite(points).all(axis=1) & (np.abs(points[:, 0]) < 3 * w) & (np.abs(points[:, 1]) < 3 * h)
    stroke = max(3, round(w / 240))
    joint_radius = max(4, round(w / 260))
    for a, b in bones:
        if good[a] and good[b]:
            cv2.line(rgb, tuple(np.rint(points[a]).astype(int)), tuple(np.rint(points[b]).astype(int)), (6, 25, 31), stroke + 4, cv2.LINE_AA)
            cv2.line(rgb, tuple(np.rint(points[a]).astype(int)), tuple(np.rint(points[b]).astype(int)), HUMAN, stroke, cv2.LINE_AA)
    # Display only body landmarks used by the skeleton; 40 finger dots obscure hands.
    displayed = sorted(set(i for bone in bones for i in bone))
    for j in displayed:
        if not good[j]:
            continue
        p = points[j]
        cv2.circle(rgb, tuple(np.rint(p).astype(int)), joint_radius + 3, (4, 23, 30), -1, cv2.LINE_AA)
        cv2.circle(rgb, tuple(np.rint(p).astype(int)), joint_radius, (220, 255, 255), -1, cv2.LINE_AA)
    if pelvis_indices and good[pelvis_indices].all():
        pelvis = np.rint(np.mean(points[pelvis_indices], axis=0)).astype(int)
        radius = joint_radius * 2 + 1
        cv2.circle(rgb, tuple(pelvis), radius + 4, (4, 23, 30), -1, cv2.LINE_AA)
        cv2.circle(rgb, tuple(pelvis), radius, HUMAN, -1, cv2.LINE_AA)
        cv2.circle(rgb, tuple(pelvis), max(3, radius // 3), (245, 255, 255), -1, cv2.LINE_AA)
        tx, ty = min(w - 145, max(0, int(pelvis[0]) + 24)), min(h - 42, max(0, int(pelvis[1]) - 19))
        cv2.rectangle(rgb, (tx, ty), (tx + 140, ty + 40), (6, 25, 31), -1)
        cv2.putText(rgb, 'PELVIS', (tx + 9, ty + 29), cv2.FONT_HERSHEY_SIMPLEX, .85, HUMAN, 2, cv2.LINE_AA)
    return rgb


def path_panel(draw, rect, human, robot, ts, bounds, max_gap_s, smallfont):
    x, y, w, h = rect
    draw.rounded_rectangle((x, y, x + w, y + h), 14, fill=CARD)
    draw.text((x + 18, y + 12), 'PELVIS TRAJECTORY  /  fixed world XY [m]', fill=TEXT, font=smallfont)
    lo, hi = bounds
    margin = 40
    area = (x + margin, y + 49, w - 2 * margin, h - 76)
    ax, ay, aw, ah = area
    scale = min(aw / max(hi[0] - lo[0], .1), ah / max(hi[1] - lo[1], .1))
    center = (lo + hi) / 2
    def xy(p):
        return (ax + aw / 2 + (p[0] - center[0]) * scale, ay + ah / 2 - (p[1] - center[1]) * scale)
    step = max(.5, math.ceil(max(hi - lo) / 8 / .5) * .5)
    for v in np.arange(math.ceil(lo[0] / step) * step, hi[0] + 1e-6, step):
        a, b = xy([v, lo[1]]), xy([v, hi[1]])
        draw.line([a, b], fill=(47, 61, 76), width=1)
        draw.text((a[0] - 11, ay + ah + 5), f'{v:g}', fill=MUTED, font=font(11))
    for v in np.arange(math.ceil(lo[1] / step) * step, hi[1] + 1e-6, step):
        a, b = xy([lo[0], v]), xy([hi[0], v])
        draw.line([a, b], fill=(47, 61, 76), width=1)
        draw.text((ax - 31, a[1] - 7), f'{v:g}', fill=MUTED, font=font(11))
    for seq, color, width in [(human, HUMAN, 7), (robot, ROBOT, 3)]:
        idx = np.flatnonzero(seq['valid'] & (seq['times'] <= ts))
        if not len(idx):
            continue
        pieces = np.split(idx, np.flatnonzero((np.diff(idx) > 1) | (np.diff(seq['times'][idx]) > max_gap_s)) + 1)
        for piece in pieces:
            coords = [xy(p) for p in seq['pelvis'][piece]]
            if len(coords) > 1:
                # Coincident XY paths remain visible without changing either coordinate.
                draw.line(coords, fill=color, width=width)
        start = xy(seq['pelvis'][idx[0]])
        radius = 9 if seq is human else 5
        draw.ellipse((start[0] - radius, start[1] - radius, start[0] + radius, start[1] + radius), outline=color, width=2)
        support = sample(seq, ts, max_gap_s)
        if support:
            end = xy(interpolate(seq['pelvis'], support))
            radius = 6 if seq is human else 3
            draw.ellipse((end[0] - radius, end[1] - radius, end[0] + radius, end[1] + radius), fill=color)


def human_model(ground_z):
    return mujoco.MjModel.from_xml_string(f'''<mujoco model="world human diagnostic">
      <visual><global offwidth="1600" offheight="1000"/><quality shadowsize="2048"/>
      <headlight ambient=".42 .42 .42" diffuse=".7 .7 .7" specular=".15 .15 .15"/>
      <rgba haze=".10 .14 .19 1"/></visual>
      <asset><texture name="sky" type="skybox" builtin="gradient" rgb1=".10 .14 .19" rgb2=".25 .31 .38" width="512" height="2048"/>
      <material name="floor" rgba=".26 .32 .39 1" reflectance=".05" shininess=".15"/></asset>
      <worldbody><light pos="2 -3 5" dir="-.3 .3 -1" diffuse=".9 .9 .9"/>
      <geom name="floor" type="plane" size="0 0 .05" pos="0 0 {ground_z}" material="floor"/>
      </worldbody></mujoco>''')


def load_model(path):
    """Use MuJoCo's VFS so Windows non-ASCII paths do not reach C fopen."""
    path = Path(path).resolve()
    assets = {p.relative_to(path.parent).as_posix(): p.read_bytes()
              for p in path.parent.rglob('*') if p.is_file() and p != path
              and p.suffix.lower() in ('.xml', '.stl', '.obj', '.png', '.jpg')}
    return mujoco.MjModel.from_xml_string(path.read_text(encoding='utf-8'), assets=assets)


def fit_fixed_camera(renderer, data, camera, human, robot, aspect, ground_z, source_video):
    """Fit all original motion points once; the resulting camera never follows."""
    renderer.update_scene(data, camera)
    gl = renderer.scene.camera[0]
    forward = np.array(gl.forward, float)
    forward /= np.linalg.norm(forward)
    up = np.array(gl.up, float)
    up /= np.linalg.norm(up)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    root = robot['pelvis'][robot['valid']]
    floor = root.copy()
    floor[:, 2] = ground_z
    clouds = [root + [0, 0, .65], floor, human['pelvis'][human['valid']]]
    if 'xpos' in robot:
        clouds.append(robot['xpos'][robot['valid'], 1:].reshape(-1, 3))
    else:
        clouds.extend([root + [.4, 0, 0], root + [-.4, 0, 0], root + [0, .4, 0], root + [0, -.4, 0]])
    if not source_video:
        clouds.append(human['points'][human['valid']].reshape(-1, 3))
    cloud = np.concatenate(clouds)
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    # Small conservative mesh margin around sampled body frames.
    cloud = np.concatenate([cloud + offset for offset in np.r_[np.zeros((1, 3)), np.eye(3) * .12, -np.eye(3) * .12]])
    delta = cloud - np.asarray(camera.lookat)
    u, v, depth = delta @ right, delta @ up, delta @ forward
    tan_y = float(gl.frustum_top / gl.frustum_near)
    tan_x = tan_y * aspect
    distance = max(2.5, float(np.max(np.abs(u) / (tan_x * .92) - depth)),
                   float(np.max(np.abs(v) / (tan_y * .84) - depth)))
    camera.distance = distance
    return dict(distance=distance, azimuth=float(camera.azimuth), elevation=float(camera.elevation),
                lookat=np.asarray(camera.lookat).tolist(), fit_points=len(cloud),
                horizontal_ndc_extent=float(np.max(np.abs(u) / ((depth + distance) * tan_x))),
                vertical_ndc_extent=float(np.max(np.abs(v) / ((depth + distance) * tan_y))),
                method='One camera fitted to full-sequence body positions and both pelvis paths; no following')


def source_camera_from_world(human, video):
    if video is None or 'K' not in human or 'T_world_from_camera' not in human:
        raise ValueError('--match-source-camera requires source video and human NPZ K/T_world_from_camera')
    k = np.asarray(human['K'], float)
    transform = np.asarray(human['T_world_from_camera'], float)
    if k.shape != (3, 3) or transform.shape != (4, 4) or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError('Malformed source camera K/T')
    return dict(position=transform[:3, 3].copy(), forward=transform[:3, 2].copy(),
                up=-transform[:3, 1].copy(), K=k, width=video.width, height=video.height)


def apply_source_camera(scene, source):
    """Set both GL cameras exactly, retaining roll and the source principal point."""
    k = source['K']
    for cam in scene.camera:
        cam.pos[:] = source['position']
        cam.forward[:] = source['forward']
        cam.up[:] = source['up']
        near = cam.frustum_near
        cam.frustum_top = near * k[1, 2] / k[1, 1]
        cam.frustum_bottom = -near * (source['height'] - k[1, 2]) / k[1, 1]
        cam.frustum_center = near * (source['width'] / 2 - k[0, 2]) / k[0, 0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--human', type=Path, required=True)
    ap.add_argument('--robot', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--source-video', type=Path)
    ap.add_argument('--camera', type=Path, help='JSON K, T_world_camera, optional distortion')
    ap.add_argument('--video-time-offset-s', type=float, default=0.)
    ap.add_argument('--title', default='Human motion to Unitree G1')
    ap.add_argument('--source-label', default='SOURCE VIDEO')
    ap.add_argument('--robot-label', default='UNITREE G1 / MuJoCo')
    ap.add_argument('--quality-label', default='MODEL-SCALE / UNCALIBRATED')
    ap.add_argument('--pose-label', default='raw SAM 3D projection', help='Description of source-video overlay provenance')
    ap.add_argument('--match-source-camera', action='store_true', help='Use fixed K/T from human NPZ, including camera roll')
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--width', type=int, default=1600)
    ap.add_argument('--height', type=int, default=1000)
    ap.add_argument('--start-s', type=float, default=0.)
    ap.add_argument('--duration-s', type=float)
    ap.add_argument('--max-gap-s', type=float, default=.1)
    ap.add_argument('--ground-z', type=float, default=0.)
    ap.add_argument('--azimuth', type=float, default=135.)
    ap.add_argument('--elevation', type=float, default=-22.)
    ap.add_argument('--ffmpeg', default=shutil.which('ffmpeg'))
    args = ap.parse_args(argv)
    if cv2 is None or mujoco is None:
        raise RuntimeError('Rendering requires opencv-python and mujoco in the selected Python environment')
    if not args.ffmpeg:
        raise ValueError('ffmpeg not found; pass --ffmpeg')
    if args.width < 1400 or args.height < 900 or args.width % 2 or args.height % 2 or args.fps < 1:
        raise ValueError('Use even width >=1400, even height >=900, FPS >=1')
    h = load_sequence(args.human, 'human')
    r = load_sequence(args.robot, 'robot')
    align_clock_origins(h, r)
    video = SourceVideo(args.source_video, h, args.camera, args.video_time_offset_s) if args.source_video else None
    source_camera = source_camera_from_world(h, video) if args.match_source_camera else None
    start = max(h['times'][0], r['times'][0]) + args.start_s
    end = min(h['times'][-1], r['times'][-1])
    if args.duration_s is not None:
        end = min(end, start + args.duration_s)
    if end <= start:
        raise ValueError('Human and robot timestamps have no requested overlap')
    display_times = start + np.arange(int(math.floor((end - start) * args.fps)) + 1) / args.fps
    allp = np.concatenate([h['pelvis'][h['valid']], r['pelvis'][r['valid']]])
    lo, hi = np.min(allp[:, :2], axis=0) - .6, np.max(allp[:, :2], axis=0) + .6
    bounds = (lo, hi)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [*((lo + hi) / 2), args.ground_z + .75]
    camera.distance = max(2.8, float(np.linalg.norm(hi - lo)) * .72 + 1.0)
    camera.azimuth, camera.elevation = args.azimuth, args.elevation
    margin, gap, header = 26, 18, 130
    panel_w = (args.width - 2 * margin - gap) // 2
    panel_h = round(args.height * .51)
    model = load_model(args.model)
    if model.nq != r['qpos'].shape[1] or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or model.jnt_qposadr[0] != 0:
        raise ValueError('G1 model/qpos mismatch or root joint is not first free joint')
    for name in ['offwidth', 'offheight']:
        setattr(model.vis.global_, name, max(getattr(model.vis.global_, name), panel_w if name == 'offwidth' else panel_h))
    model.vis.headlight.ambient[:] = [.45, .45, .45]
    model.vis.headlight.diffuse[:] = [.75, .75, .75]
    md = mujoco.MjData(model)
    hm = human_model(args.ground_z)
    hd = mujoco.MjData(hm)
    mujoco.mj_forward(hm, hd)
    viewport_h = panel_h - 33
    rr = mujoco.Renderer(model, viewport_h, panel_w, max_geom=12000)
    hr = mujoco.Renderer(hm, viewport_h, panel_w, max_geom=12000)
    md.qpos[:] = r['qpos'][np.flatnonzero(r['valid'])[0]]
    mujoco.mj_forward(model, md)
    camera_framing = fit_fixed_camera(rr, md, camera, h, r, panel_w / viewport_h, args.ground_z, bool(args.source_video))
    if source_camera:
        camera_framing = dict(method='Fixed source optical camera pose and nominal K; no following or dynamic framing',
                              position=source_camera['position'].tolist(), forward=source_camera['forward'].tolist(),
                              up=source_camera['up'].tolist(), K=source_camera['K'].tolist(),
                              original_dimensions=[source_camera['width'], source_camera['height']],
                              independently_calibrated=False)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.stem + '.partial.mp4')
    command = [args.ffmpeg, '-y', '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24',
               '-video_size', f'{args.width}x{args.height}', '-framerate', str(args.fps), '-i', '-',
               '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(partial)]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    fonts = {size: font(size, size >= 28) for size in [13, 15, 16, 18, 20, 23, 30, 34]}
    metrics_h, metrics_r = path_metrics(h, args.max_gap_s), path_metrics(r, args.max_gap_s)
    snapshot_ids = set(np.linspace(0, len(display_times) - 1, min(6, len(display_times)), dtype=int))
    snapshots, sync = [], []
    try:
        for n, ts in enumerate(display_times):
            hs, rs = sample(h, ts, args.max_gap_s), sample(r, ts, args.max_gap_s)
            robot_im = np.full((viewport_h, panel_w, 3), CARD, np.uint8)
            if rs is not None:
                q = sample_qpos(r, rs)
                md.qpos[:] = q
                mujoco.mj_forward(model, md)
                rr.update_scene(md, camera)
                if source_camera:
                    apply_source_camera(rr.scene, source_camera)
                draw_world(rr.scene, r, ts, ROBOT, args.max_gap_s, args.ground_z, bounds, q[:3])
                draw_world(rr.scene, h, ts, HUMAN, args.max_gap_s, args.ground_z, bounds,
                           interpolate(h['pelvis'], hs) if hs else None, grid=False)
                robot_im = rr.render().copy()
            source_index = -1
            overlay_available = False
            if video:
                raw, source_index, xy = video.read(ts, hs)
                overlay_available = xy is not None
                if raw is not None:
                    raw = overlay_pose(raw, xy, h['bones'], h['pelvis_2d_indices'])
                    human_im, _, _ = fit_image(raw, panel_w, viewport_h)
                else:
                    human_im = np.full((panel_h, panel_w, 3), CARD, np.uint8)
            else:
                hr.update_scene(hd, camera)
                if source_camera:
                    apply_source_camera(hr.scene, source_camera)
                if hs:
                    p = interpolate(h['points'], hs)
                    for a, b in h['bones']:
                        line(hr.scene, p[a], p[b], [*np.array(HUMAN) / 255, 1], .017)
                    for j in sorted(set(i for bone in h['bones'] for i in bone)):
                        sphere(hr.scene, p[j], [.84, .99, 1., 1.], .025)
                    draw_world(hr.scene, h, ts, HUMAN, args.max_gap_s, args.ground_z, bounds, interpolate(h['pelvis'], hs))
                human_im = hr.render().copy()
            frame = Image.new('RGB', (args.width, args.height), BG)
            frame.paste(Image.fromarray(human_im), (margin, header))
            frame.paste(Image.fromarray(robot_im), (margin + panel_w + gap, header))
            d = ImageDraw.Draw(frame)
            d.text((margin, 19), args.title, font=fonts[34], fill=TEXT)
            d.text((margin, 67), args.quality_label + '  |  kinematic retarget / physics not validated', font=fonts[16], fill=MUTED)
            d.text((args.width - 230, 27), f'{ts - start:06.2f} / {display_times[-1] - start:.2f} s', font=fonts[23], fill=TEXT)
            d.rectangle((margin, 103, margin + 4, 123), fill=HUMAN)
            d.text((margin + 14, 101), (args.source_label + '  +  POSE') if video else 'WORLD SKELETON  /  DIAGNOSTIC', font=fonts[18], fill=HUMAN)
            right_x = margin + panel_w + gap
            d.rectangle((right_x, 103, right_x + 4, 123), fill=ROBOT)
            d.text((right_x + 14, 101), args.robot_label, font=fonts[18], fill=ROBOT)
            strip_y = header + panel_h - 33
            for px in [margin, right_x]:
                d.rectangle((px, strip_y, px + panel_w, strip_y + 33), fill=CARD)
            pose_kind = args.pose_label if 'keypoints_2d' in h else 'world reprojection'
            left_note = f'Source frame {source_index}  |  {pose_kind}' if video and overlay_available else ('VIDEO ONLY  /  no registered 2D projection' if video else 'NO SOURCE VIDEO  /  world coordinates retained')
            d.text((margin + 10, strip_y + 6), left_note, font=fonts[13], fill=MUTED)
            view_note = 'Fixed source-camera pose' if source_camera else 'Fixed world overview'
            d.text((right_x + 10, strip_y + 6), view_note + '  |  cyan: human  |  amber: G1 pelvis', font=fonts[13], fill=MUTED)
            if hs is None:
                d.text((margin + 24, header + 24), 'NO SUPPORTED HUMAN POSE', font=fonts[20], fill=(255, 139, 139))
            if rs is None:
                d.text((right_x + 24, header + 24), 'NO SUPPORTED ROBOT POSE', font=fonts[20], fill=(255, 139, 139))
            bottom_y = header + panel_h + 18
            bottom_h = args.height - bottom_y - 48
            path_w = int((args.width - 2 * margin) * .54)
            path_panel(d, (margin, bottom_y, path_w, bottom_h), h, r, ts, bounds, args.max_gap_s, fonts[16])
            stats_x = margin + path_w + gap
            d.rounded_rectangle((stats_x, bottom_y, args.width - margin, bottom_y + bottom_h), 14, fill=CARD)
            d.text((stats_x + 18, bottom_y + 12), 'TRAJECTORY ENDPOINT DISTANCE', font=fonts[16], fill=TEXT)
            d.text((stats_x + 18, bottom_y + 42), 'Full input sequence; zero endpoint correction', font=fonts[13], fill=MUTED)
            for j, (name, seq, metrics, color, support) in enumerate([('HUMAN', h, metrics_h, HUMAN, hs), ('G1', r, metrics_r, ROBOT, rs)]):
                yy = bottom_y + 73 + j * 73
                d.text((stats_x + 18, yy), name, font=fonts[18], fill=color)
                d.text((stats_x + 121, yy), f'XY  {metrics["endpoint_distance_xy_m"]:.3f} m    |    XYZ  {metrics["endpoint_distance_3d_m"]:.3f} m', font=fonts[18], fill=TEXT)
                cur = interpolate(seq['pelvis'], support) if support else None
                position_text = f'pelvis  [{cur[0]:+.2f}, {cur[1]:+.2f}, {cur[2]:+.2f}] m' if cur is not None else 'pelvis unavailable / unsupported frame'
                d.text((stats_x + 121, yy + 30), position_text, font=fonts[13], fill=MUTED)
            d.text((stats_x + 18, bottom_y + bottom_h - 31), 'Open circle = start   |   solid dot = current   |   no forced loop', font=fonts[13], fill=MUTED)
            footer = 'Cyan: human pelvis    Amber: G1 pelvis    World Z-up    Source gaps stay visible    Render interpolation adds no observations'
            d.text((margin, args.height - 30), footer, font=fonts[13], fill=MUTED)
            proc.stdin.write(np.asarray(frame).tobytes())
            if n in snapshot_ids:
                snapshots.append(frame.copy())
            sync.append((float(ts), source_index, hs[0] if hs else -1, hs[1] if hs else -1, rs[0] if rs else -1, rs[1] if rs else -1))
            if n % max(1, args.fps * 2) == 0:
                print(json.dumps({'rendered_frames': n + 1, 'total_frames': len(display_times)}), flush=True)
        proc.stdin.close()
        err = proc.stderr.read().decode('utf-8', errors='replace')
        code = proc.wait()
        if code:
            raise RuntimeError(f'ffmpeg exited {code}: {err}')
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    finally:
        rr.close()
        hr.close()
        if video:
            video.cap.release()
    partial.replace(output)
    poster = snapshots[len(snapshots) // 2]
    poster.save(output.with_suffix('.preview.png'))
    thumb_w = 800
    thumb_h = round(args.height * thumb_w / args.width)
    sheet = Image.new('RGB', (thumb_w * 2, thumb_h * math.ceil(len(snapshots) / 2)), BG)
    for i, frame in enumerate(snapshots):
        sheet.paste(frame.resize((thumb_w, thumb_h)), ((i % 2) * thumb_w, (i // 2) * thumb_h))
    sheet.save(output.with_suffix('.contact_sheet.jpg'), quality=92)
    np.savez_compressed(output.with_suffix('.timeline.npz'), columns=np.array(['timestamps_s', 'source_frame', 'human_left', 'human_right', 'robot_left', 'robot_right']), samples=np.asarray(sync))
    report = dict(status='complete', output=str(output), frames=len(display_times), fps=args.fps,
                  duration_s=len(display_times) / args.fps, dimensions=[args.width, args.height],
                  human_source=h['path'], robot_source=r['path'], model=str(args.model.resolve()),
                  source_video=str(args.source_video.resolve()) if args.source_video else None,
                  source_label=args.source_label, quality_label=args.quality_label,
                  robot_label=args.robot_label,
                  source_overlay_label=args.pose_label,
                  fixed_world_camera=True, endpoint_correction_applied=False,
                  camera_framing=camera_framing,
                  input_coordinate_transform_applied=False, dynamically_validated=False,
                  kinematic_playback_only=True, human_metrics=metrics_h, robot_metrics=metrics_r,
                  max_interpolation_gap_s=args.max_gap_s,
                  model_sha256=sha256(args.model), human_sha256=sha256(args.human), robot_sha256=sha256(args.robot),
                  video_sha256=sha256(output), renderer_sha256=sha256(__file__))
    output.with_suffix('.render.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    main()
