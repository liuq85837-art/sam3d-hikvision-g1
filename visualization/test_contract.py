"""Tests for honest timing/trajectory handling in a visualization deliverable."""
import unittest
import numpy as np

try:
    from .render_comparison import align_clock_origins, path_metrics, sample, sample_qpos
except ImportError:
    from render_comparison import align_clock_origins, path_metrics, sample, sample_qpos


class RenderingContract(unittest.TestCase):
    def test_absolute_clock_offset_survives_normalization(self):
        human = dict(times=np.array([0., .02]), _timestamp_origin_ns=1700000000000000000)
        robot = dict(times=np.array([0., .02]), _timestamp_origin_ns=1700000000010000000)
        align_clock_origins(human, robot)
        np.testing.assert_allclose(robot['times'], [.01, .03])

    def test_does_not_bridge_invalid_frame_or_long_gap(self):
        seq = dict(times=np.array([0., .02, .04, .3]), valid=np.array([True, False, True, True]))
        self.assertIsNone(sample(seq, .01, .1))
        self.assertIsNone(sample(seq, .2, .1))
        self.assertEqual(sample(seq, .04, .1), (2, 2, 0.))

    def test_known_open_path_is_not_closed_or_centered(self):
        seq = dict(times=np.array([0., .1, .2]), valid=np.ones(3, bool),
                   pelvis=np.array([[12., -4., 1.], [12.3, -4., 1.], [12.3, -3.6, 1.2]]))
        original = seq['pelvis'].copy()
        m = path_metrics(seq, .11)
        self.assertAlmostEqual(m['endpoint_distance_xy_m'], .5)
        self.assertAlmostEqual(m['path_length_xy_m'], .7)
        self.assertAlmostEqual(m['endpoint_distance_3d_m'], np.sqrt(.29))
        np.testing.assert_array_equal(seq['pelvis'], original)

    def test_gap_length_is_not_fabricated(self):
        seq = dict(times=np.array([0., .02, 1., 1.02]), valid=np.ones(4, bool),
                   pelvis=np.array([[0., 0., 1.], [1., 0., 1.], [100., 0., 1.], [101., 0., 1.]]))
        m = path_metrics(seq, .1)
        self.assertAlmostEqual(m['path_length_xy_m'], 2.)
        self.assertEqual(m['unsupported_link_count'], 1)
        self.assertAlmostEqual(m['endpoint_distance_xy_m'], 101.)

    def test_antipodal_quaternions_do_not_spin(self):
        seq = dict(qpos=np.array([[3., 4., 1., 1., 0., 0., 0.], [5., 6., 1., -1., 0., 0., 0.]]))
        midpoint = sample_qpos(seq, (0, 1, .5))
        np.testing.assert_allclose(midpoint, [4., 5., 1., 1., 0., 0., 0.])


if __name__ == '__main__':
    unittest.main()
