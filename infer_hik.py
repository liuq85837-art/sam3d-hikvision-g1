"""Offline Hikvision video -> official SAM 3D Body camera coordinates.

Original upstream code and checkpoints are read-only. No ZED metadata is reused.
Default focal length is the upstream image-diagonal prior, explicitly uncalibrated.
"""
from __future__ import annotations
import argparse
import contextlib
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
SAM = ROOT.parent / 'pose_world_sam3d_repro_20260918/pose_project/sam3d'


@contextlib.contextmanager
def official_backbone_only():
    """Call the same pinned official hub factory without unrelated eval imports."""
    import torch
    sys.path.insert(0, str(SAM / 'dinov3_official'))
    from dinov3.hub import backbones
    original = torch.hub.load
    original_jit_load = torch.jit.load
    def load(repo, name, *args, **kwargs):
        if str(repo).split(':')[0] == 'facebookresearch/dinov3':
            if name != 'dinov3_vith16plus' or kwargs.get('pretrained') is not False:
                raise ValueError('Unexpected DINO backbone or external weight request')
            kwargs.pop('source', None)
            return getattr(backbones, name)(*args, **kwargs)
        return original(repo, name, *args, **kwargs)
    torch.hub.load = load
    def load_jit_file(path, *args, **kwargs):
        # libtorch fopen does not support these Windows Unicode paths.
        # Passing a Python file object uses the identical TorchScript bytes.
        if isinstance(path, (str, Path)):
            with open(path, 'rb') as stream:
                return original_jit_load(stream, *args, **kwargs)
        return original_jit_load(path, *args, **kwargs)
    torch.jit.load = load_jit_file
    try:
        yield
    finally:
        torch.hub.load = original
        torch.jit.load = original_jit_load


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def video_info(path):
    p = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,avg_frame_rate,nb_frames:frame=best_effort_timestamp_time',
        '-of', 'json', str(path)], capture_output=True, check=True, text=True)
    info = json.loads(p.stdout)
    stream = info['streams'][0]
    a, b = map(float, stream['avg_frame_rate'].split('/'))
    times = np.array([float(f['best_effort_timestamp_time']) for f in info['frames']])
    if not len(times) or not np.all(np.diff(times) > 0):
        raise ValueError('Video presentation timestamps are missing or not increasing')
    return int(stream['width']), int(stream['height']), a / b, times


def select_box(boxes, scores, previous, width):
    if not len(boxes):
        return None
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # Choose the full-body foreground person, then track continuity. No identity inference.
    if previous is None:
        return int(np.argmax(area * scores))
    center = (boxes[:, :2] + boxes[:, 2:]) / 2
    prev_center = (previous[:2] + previous[2:]) / 2
    dist = np.linalg.norm(center - prev_center, axis=1) / max(previous[3] - previous[1], 1)
    area_ratio = np.abs(np.log(np.maximum(area, 1) / max(np.prod(previous[2:] - previous[:2]), 1)))
    cost = dist + 0.15 * area_ratio - 0.1 * scores
    i = int(np.argmin(cost))
    return i if dist[i] < 1.5 else None


def prepare(args):
    os.environ['YOLO_OFFLINE'] = 'true'
    # Ultralytics sanitizes non-ASCII config paths; use an app-local ASCII cache.
    config_dir = Path(os.environ.get('LOCALAPPDATA', str(ROOT))) / 'hik_g1_runtime'
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(config_dir)
    from ultralytics import YOLO
    from ultralytics import settings
    settings.update({'sync': False})
    video = args.video.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'images').mkdir(exist_ok=True)
    w, h, fps, timestamps = video_info(video)
    indices = np.arange(0, len(timestamps), args.stride, dtype=np.int64)
    if args.max_frames:
        indices = indices[:args.max_frames]
    model = YOLO(str(args.detector))
    cap = cv2.VideoCapture(str(video))
    selected = set(indices.tolist())
    tracks = []
    previous = None
    for index in range(int(indices[-1]) + 1):
        ok, img = cap.read()
        if not ok:
            raise RuntimeError(f'Video decode failed at frame {index}')
        if index not in selected:
            continue
        pred = model.predict(img, classes=[0], conf=.25, imgsz=960, device='cpu', verbose=False)[0]
        boxes = pred.boxes.xyxy.cpu().numpy()
        scores = pred.boxes.conf.cpu().numpy()
        chosen = select_box(boxes, scores, previous, w)
        row = dict(frame_index=index, timestamp_s=float(timestamps[index]), valid=chosen is not None)
        if chosen is not None:
            previous = boxes[chosen].copy()
            row.update(bbox_xyxy=previous.tolist(), score=float(scores[chosen]), track_id='primary_person')
        image_path = out / 'images' / f'{index:06d}.jpg'
        ok, encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 96])
        if not ok:
            raise RuntimeError('JPEG encoding failed')
        encoded.tofile(image_path)
        tracks.append(row)
        if len(tracks) % 50 == 0:
            print(f'Detected {len(tracks)}/{len(indices)} frames', flush=True)
    cap.release()
    if args.calibration:
        calibration = json.loads(args.calibration.read_text(encoding='utf-8-sig'))
        K = np.array(calibration['K'], dtype=float)
        if calibration.get('distortion', []) and np.any(calibration['distortion']):
            raise ValueError('Undistort input first; this wrapper refuses nonzero distortion silently ignored')
        camera_status = 'user_supplied_intrinsics'
    else:
        focal = args.focal if args.focal else float(np.hypot(w, h))
        K = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]])
        camera_status = 'assumed_focal_uncalibrated'
    dump(out / 'tracks.json', tracks)
    dump(out / 'input.json', dict(source_video=str(video), source_sha256=sha256(video),
        width=w, height=h, source_fps=fps, source_frames=len(timestamps), selected_frames=len(indices),
        inference_stride=args.stride, K=K.tolist(), camera_status=camera_status,
        scale_status='model_prior_unverified', distortion_status='unknown_not_corrected',
        timestamp_source='ffprobe_best_effort_timestamp_time', detector=str(args.detector),
        detector_sha256=sha256(args.detector), note='No measured intrinsics or ground scale unless explicitly supplied'))
    print(f'Prepared {len(tracks)} frames; valid={sum(x["valid"] for x in tracks)}', flush=True)


def infer(args):
    import torch
    sys.path.insert(0, str(SAM / 'repo'))
    sys.path.insert(0, str(SAM))
    os.environ['MOMENTUM_ENABLED'] = '0'
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    out = args.output.resolve()
    tracks = json.loads((out / 'tracks.json').read_text(encoding='utf-8'))
    if args.max_frames:
        tracks = tracks[:args.max_frames]
    info = json.loads((out / 'input.json').read_text(encoding='utf-8'))
    contract = dict(source_sha256=info['source_sha256'], K=info['K'],
        tracks_sha256=sha256(out/'tracks.json'), inference_type='body',
        model='user_dinov3_20260917_official')
    contract_path = out/'cache_contract.json'
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding='utf-8')) != contract:
            raise ValueError('Cached poses belong to different source, tracking or intrinsics; choose a new output directory')
    elif (out/'inference_provenance.json').exists():
        old = json.loads((out/'inference_provenance.json').read_text(encoding='utf-8'))['input']
        if old['source_sha256'] != info['source_sha256'] or old['K'] != info['K']:
            raise ValueError('Cached inference intrinsics/source changed; choose a new output directory')
    dump(contract_path, contract)
    K = np.array(info['K'], np.float32)
    names = runpy.run_path(str(SAM / 'repo/sam_3d_body/metadata/mhr70.py'))['mhr_names']
    checkpoint = SAM / 'checkpoints/user_dinov3_20260917/model.ckpt'
    mhr = checkpoint.parent / 'assets/mhr_model.pt'
    torch.set_num_threads(4)
    with official_backbone_only():
        model, cfg = load_sam_3d_body(str(checkpoint), device=args.device, mhr_path=str(mhr))
    model.eval()
    estimator = SAM3DBodyEstimator(model, cfg, human_detector=None, fov_estimator=None)
    camera_tensor = torch.from_numpy(K).unsqueeze(0)
    (out / 'frames').mkdir(exist_ok=True)
    t0 = time.monotonic()
    log = []
    for row in tracks:
        idx = row['frame_index']
        p = out / 'frames' / f'{idx:06d}.npz'
        if not row['valid'] or p.exists():
            continue
        # Unicode Windows paths cannot be loaded by upstream cv2.imread reliably.
        img = cv2.imdecode(np.fromfile(out / 'images' / f'{idx:06d}.jpg', np.uint8), cv2.IMREAD_COLOR)
        st = time.monotonic()
        with torch.inference_mode(), contextlib.redirect_stdout(io.StringIO()):
            predictions = estimator.process_one_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                bboxes=np.array(row['bbox_xyxy'], np.float32)[None], cam_int=camera_tensor,
                inference_type='body')
        if len(predictions) != 1:
            raise RuntimeError(f'Expected one body at frame {idx}')
        pred = predictions[0]
        joints = np.asarray(pred['pred_keypoints_3d']) + np.asarray(pred['pred_cam_t'])[None]
        if joints.shape != (70, 3) or not np.isfinite(joints).all() or np.any(joints[:, 2] <= 0):
            raise RuntimeError(f'Invalid joints at frame {idx}')
        projected = joints @ K.T
        uv = projected[:, :2] / projected[:, 2:3]
        temporary = p.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, camera_joints_m=joints.astype(np.float32),
            keypoints_2d=uv.astype(np.float32), frame_index=idx, timestamp_s=row['timestamp_s'],
            joint_names=names, bbox=row['bbox_xyxy'], pred_cam_t=pred['pred_cam_t'])
        temporary.replace(p)
        log.append(dict(frame=idx, seconds=time.monotonic()-st,
            peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9 if args.device.startswith('cuda') else None))
        if len(log) % 10 == 0 or len(log) == 1:
            dump(out / 'status.json', dict(status='running', newly_inferred=len(log),
                cached_frames=len(list((out/'frames').glob('*.npz'))), total=len(tracks), last=log[-1]))
            print(json.dumps(log[-1]), flush=True)
    joints = np.full((len(tracks),70,3), np.nan, np.float32)
    uv = np.full((len(tracks),70,2), np.nan, np.float32)
    valid = np.zeros(len(tracks), bool)
    for i, row in enumerate(tracks):
        p = out / 'frames' / f'{row["frame_index"]:06d}.npz'
        if p.exists() and row['valid']:
            with np.load(p, allow_pickle=False) as a:
                joints[i], uv[i] = a['camera_joints_m'], a['keypoints_2d']
            valid[i] = True
    np.savez_compressed(out / 'camera_sequence.npz', joints_camera=joints, keypoints_2d=uv,
        keypoints_2d_calibrated=uv, joint_names=names, valid=valid, K=K,
        timestamps_s=np.array([r['timestamp_s'] for r in tracks]),
        frame_indices=np.array([r['frame_index'] for r in tracks]),
        bbox_xyxy=np.array([r.get('bbox_xyxy',[np.nan]*4) for r in tracks],np.float32),
        detector_score=np.array([r.get('score',np.nan) for r in tracks],np.float32),
        metric_scale_status='model_prior_unverified', metadata=json.dumps(info,ensure_ascii=False))
    dump(out / 'inference_provenance.json', dict(model='official_sam_3d_body_dinov3',
        checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint), mhr_sha256=sha256(mhr),
        inference_type='body', device=args.device, torch_version=torch.__version__,
        wrapper_sha256=sha256(__file__), elapsed_s=time.monotonic()-t0, input=info))
    dump(out / 'status.json', dict(status='complete', selected=len(tracks), valid=int(valid.sum()),
        newly_inferred=len(log), elapsed_s=time.monotonic()-t0))
    print(f'Completed {int(valid.sum())}/{len(tracks)} camera poses', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare','infer','all'])
    p.add_argument('--video', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stride', type=int, default=4)
    p.add_argument('--max-frames', type=int)
    p.add_argument('--focal', type=float)
    p.add_argument('--calibration', type=Path)
    p.add_argument('--detector', type=Path, default=ROOT/'assets/yolo11n.pt')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.stride < 1: p.error('--stride must be >= 1')
    if args.stage in ('prepare','all'):
        if not args.video: p.error('--video required for prepare')
        prepare(args)
    if args.stage in ('infer','all'): infer(args)


if __name__ == '__main__':
    main()
