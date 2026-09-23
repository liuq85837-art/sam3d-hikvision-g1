"""Check actual delivered sequence contracts, without claiming physical validity."""
from pathlib import Path
import json
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parent
SEQUENCES = ['Video_20260922162126269', 'Video_20260922161535257']


def main():
    result = {'passed': True, 'physical_metric_accuracy_verified': False,
              'dynamic_feasibility_verified': False, 'sequences': []}
    for name in SEQUENCES:
        out = ROOT / 'outputs' / name
        row = {'sequence': name, 'checks': []}
        camera = np.load(out/'camera/camera_sequence.npz', allow_pickle=False)
        raw = np.load(out/'world/world_joints.npz', allow_pickle=False)
        smooth = np.load(out/'world_temporal/world_joints.npz', allow_pickle=False)
        n, valid = len(raw['valid']), raw['valid']
        for world, suffix in [(raw, ''), (smooth, '_temporal')]:
            robot = np.load(out/f'g1_motion{suffix}.npz', allow_pickle=False)
            assert robot['qpos'].shape == (n,36) and robot['qvel'].shape == (n,35)
            np.testing.assert_array_equal(camera['frame_indices'], world['frame_indices'])
            np.testing.assert_allclose(camera['timestamps_s'], world['timestamps_s'], atol=1e-8)
            np.testing.assert_array_equal(robot['timestamps_s'], world['timestamps_s'])
            np.testing.assert_array_equal(robot['valid'], valid)
            np.testing.assert_allclose(robot['qpos'][valid,:2], world['pelvis_world'][valid,:2], atol=5e-7)
            np.testing.assert_allclose(np.linalg.norm(robot['qpos'][valid,3:7],axis=1),1,atol=1e-6)
            assert np.isfinite(robot['qpos'][valid]).all()
            assert np.isnan(robot['qpos'][~valid]).all()
            meta = json.loads(str(robot['metadata']))
            assert meta['max_joint_limit_excess_rad'] == 0
            assert meta['metric_scale_status'] == 'model_prior_only'
            row['checks'].append(f'{suffix or "raw"}: frame/time/valid/XY/quaternion/joint-limits/scale')
        np.testing.assert_array_equal(raw['valid'], smooth['valid'])
        np.testing.assert_array_equal(raw['T_world_from_camera'], smooth['T_world_from_camera'])
        rel0 = raw['world_joints'] - raw['pelvis_world'][:,None]
        rel1 = smooth['world_joints'] - smooth['pelvis_world'][:,None]
        np.testing.assert_allclose(rel0[valid], rel1[valid], atol=2e-6)
        np.testing.assert_array_equal(raw['keypoints_2d'], smooth['keypoints_2d'])
        row['checks'].append('temporal: same fixed gauge, relative poses, source 2D, and validity')
        row['selected_frames'], row['valid_frames'] = n, int(valid.sum())
        row['videos'] = []
        for stem in ['comparison','comparison_raw']:
            video = out/f'{stem}.mp4'
            report = json.loads((out/f'{stem}.render.json').read_text(encoding='utf-8'))
            validation = json.loads((out/f'{stem}.validation.json').read_text(encoding='utf-8'))
            assert video.is_file() and validation['status'] == 'passed'
            assert report['fixed_world_camera'] and not report['endpoint_correction_applied']
            assert hashlib.sha256(video.read_bytes()).hexdigest() == report['video_sha256']
            row['videos'].append({'file':str(video.relative_to(ROOT)), 'frames':report['frames'],
                                  'duration_s':report['duration_s'], 'dimensions':report['dimensions'],
                                  'sha256':report['video_sha256']})
        result['sequences'].append(row)
    (ROOT/'delivery').mkdir(exist_ok=True)
    (ROOT/'delivery/verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
