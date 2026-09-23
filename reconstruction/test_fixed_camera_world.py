"""Synthetic interface/geometry checks; these are not Hikrobot inference results."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from fixed_camera_world import MHR_NAMES, closure_metrics, infer_ground_gauge, reconstruct, validate_rigid


def motion(n=120):
    """Straight motion over a known floor, expressed in optical camera axes."""
    names = MHR_NAMES + [f'unused-{i}' for i in range(21, 70)]
    w = np.zeros((n, 70, 3))
    w[:, :, 0] = np.linspace(0, 1.2, n)[:, None]
    w[:, :, 2] = 1.
    for idx, offset in [(9, [.0, .1, 1.]), (10, [.0, -.1, 1.]),
                        (5, [0., .2, 1.5]), (6, [0., -.2, 1.5]),
                        (15, [.12, .1, 0.]), (16, [.10, .16, 0.]), (17, [-.1, .1, 0.]),
                        (18, [.12, -.1, 0.]), (19, [.1, -.16, 0.]), (20, [-.1, -.1, 0.])]:
        w[:, idx] = np.array(offset) + np.column_stack([np.linspace(0, 1.2, n), np.zeros(n), np.zeros(n)])
    R = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
    t = np.array([0., -4., 2.])
    c = (w-t) @ R
    ts = np.arange(n, dtype=np.int64) * 16_666_667
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return c, w, ts, names, T


class GeometryContract(unittest.TestCase):
    def test_exact_extrinsic_and_scale(self):
        c, w, ts, names, T = motion()
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            np.savez(p/'camera.npz', joints_camera=c/1.2, joint_names=names,
                     timestamps_ns=ts, frame_indices=np.arange(len(c)), valid=np.ones(len(c), bool))
            (p/'cal.json').write_text(json.dumps({'T_world_from_camera': T.tolist(), 'metric_scale': 1.2}))
            gauge, metrics = reconstruct(p/'camera.npz', p/'out', p/'cal.json')
            with np.load(p/'out/world_joints.npz') as z:
                np.testing.assert_allclose(z['joints_world'], w, atol=2e-7)
            self.assertEqual(gauge['gauge_kind'], 'explicit_fixed_camera_world')
            self.assertGreater(metrics['endpoint_displacement_3d_m'], 1.)
            self.assertFalse(metrics['trajectory_corrected_for_closure'])

    def test_inferred_plane_rigid_and_distance_preserved(self):
        c, w, _, _, _ = motion()
        T, info = infer_ground_gauge(c, np.ones(len(c), bool))
        result = c @ T[:3, :3].T + T[:3, 3]
        np.testing.assert_allclose(np.linalg.norm(result[-1]-result[0], axis=1),
                                   np.linalg.norm(w[-1]-w[0], axis=1), atol=1e-8)
        self.assertLess(np.max(abs(result[:, [15, 16, 17, 18, 19, 20], 2])), 1e-6)
        self.assertFalse(info['measured_ground'])

    def test_reflection_rejected(self):
        T = np.eye(4)
        T[0, 0] = -1
        with self.assertRaises(ValueError):
            validate_rigid(T)

    def test_gap_not_integrated_and_original_valid_retained(self):
        c, _, ts, names, _ = motion()
        c[30] = np.nan
        valid = np.ones(len(c), bool)
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            np.savez(p/'camera.npz', joints_camera=c, joint_names=names,
                     timestamps_ns=ts, frame_indices=np.arange(len(c)), valid=valid)
            _, metrics = reconstruct(p/'camera.npz', p/'out')
            with np.load(p/'out/world_joints.npz') as z:
                self.assertFalse(z['valid'][30])
                self.assertTrue(z['source_person_valid'][30])
                self.assertTrue(np.isnan(z['joints_world'][30]).all())
            self.assertEqual(metrics['omitted_invalid_or_gap_pairs'], 2)

    def test_return_is_measured_not_enforced(self):
        t = np.arange(100, dtype=np.int64) * 10_000_000
        angle = np.linspace(0, 2*np.pi, 100)
        p = np.column_stack([np.cos(angle), np.sin(angle), np.ones(100)])
        a = closure_metrics(p, t, np.ones(100, bool), .001)
        p[-1, 0] += .4
        b = closure_metrics(p, t, np.ones(100, bool), .001)
        self.assertLess(a['endpoint_displacement_3d_m'], 1e-10)
        self.assertAlmostEqual(b['endpoint_displacement_3d_m'], .4)

    def test_short_clip_windows_do_not_overlap(self):
        p = np.array([[0., 0., 1.], [1., 0., 1.]])
        metrics = closure_metrics(p, np.array([0, 16_666_667], np.int64), np.ones(2, bool))
        self.assertAlmostEqual(metrics['endpoint_displacement_3d_m'], 1.)
        self.assertEqual(metrics['start_valid_window_frames'], 1)
        self.assertEqual(metrics['end_valid_window_frames'], 1)

    def test_video_time_offset_and_scale_provenance(self):
        c, _, ts, names, _ = motion()
        seconds = ts/1e9 + 12.5
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            np.savez(p/'camera.npz', joints_camera=c, joint_names=names,
                     timestamps_ns=ts, timestamps_s=seconds, frame_indices=np.arange(len(c))+750,
                     valid=np.ones(len(c), bool), metric_scale_status=np.array('stale_calibrated_claim'))
            gauge, metrics = reconstruct(p/'camera.npz', p/'out')
            with np.load(p/'out/world_joints.npz') as z:
                np.testing.assert_array_equal(z['timestamps_s'], seconds)
                np.testing.assert_array_equal(z['world_joints'], z['joints_world'])
                self.assertEqual(str(z['metric_scale_status']), 'model_prior_only')
            self.assertEqual(gauge['metric_scale_status'], 'model_prior_only')
            self.assertAlmostEqual(metrics['source_video_start_time_s'], 12.5)


if __name__ == '__main__':
    unittest.main()
