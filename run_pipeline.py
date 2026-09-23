"""Reproducible local video -> official SAM3D -> fixed world -> G1 -> video."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(*args):
    command = [sys.executable, *map(str, args)]
    print('RUN', subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--stride', type=int, default=4)
    p.add_argument('--max-frames', type=int)
    p.add_argument('--skip-prepare', action='store_true')
    p.add_argument('--skip-infer', action='store_true')
    p.add_argument('--intrinsics', type=Path)
    p.add_argument('--world-calibration', type=Path)
    p.add_argument('--metric-scale', type=float)
    p.add_argument('--raw-only', action='store_true', help='Produce the raw overview without temporal refinement')
    p.add_argument('--overview', action='store_true', help='Use a fixed global overview instead of the source camera')
    a = p.parse_args()
    video = a.video.resolve()
    out = a.output.resolve() if a.output else ROOT / 'outputs' / video.stem
    out.mkdir(parents=True, exist_ok=True)
    camera, world = out / 'camera', out / 'world'
    if not a.skip_prepare:
        options = ['--max-frames', a.max_frames] if a.max_frames else []
        if a.intrinsics: options += ['--calibration', a.intrinsics.resolve()]
        run(ROOT/'infer_hik.py', 'prepare', '--video', video, '--output', camera,
            '--stride', a.stride, *options)
    if not a.skip_infer:
        run(ROOT/'infer_hik.py', 'infer', '--output', camera)
    options = ['--calibration', a.world_calibration.resolve()] if a.world_calibration else []
    if a.metric_scale: options += ['--metric-scale', a.metric_scale]
    run(ROOT/'reconstruction/fixed_camera_world.py', '--input', camera/'camera_sequence.npz',
        '--output', world, *options)
    run(ROOT/'retarget/world_to_g1.py', world/'world_joints.npz', out/'g1_motion.npz')
    label = 'USER CALIBRATION / REVIEW PROVENANCE' if a.intrinsics and a.world_calibration else 'MODEL-SCALE / UNCALIBRATED'
    human, robot = world/'world_joints.npz', out/'g1_motion.npz'
    if not a.raw_only:
        temporal = out/'world_temporal'
        run(ROOT/'reconstruction/temporal_root.py', '--input', human, '--output', temporal)
        report = json.loads((temporal/'temporal_report.json').read_text(encoding='utf-8'))
        if report['status'] == 'complete_candidate':
            human, robot = temporal/'world_joints.npz', out/'g1_motion_temporal.npz'
            run(ROOT/'retarget/world_to_g1.py', human, robot)
            label = 'ROOT-SMOOTHED / ' + label
    camera_options = ['--azimuth','100','--elevation','-18'] if a.overview or a.raw_only else ['--match-source-camera']
    run(ROOT/'visualization/render_comparison.py', '--human', human,
        '--robot', robot, '--model', ROOT/'assets/unitree_g1/scene.xml',
        '--source-video', video, '--output', out/('comparison_raw.mp4' if a.raw_only else 'comparison.mp4'),
        '--quality-label', label, '--pose-label', 'source 2D / raw SAM 3D projection', *camera_options,
        '--title', 'HIKVISION  /  Human motion to Unitree G1')


if __name__ == '__main__': main()
