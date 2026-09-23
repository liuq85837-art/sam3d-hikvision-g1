"""Meaningful synthetic regression checks; outputs are explicitly synthetic QA."""
from pathlib import Path
import json
import numpy as np
import mujoco
from world_to_g1 import MODEL, CORE, BODY_MAP, load_source, load_model, run

HERE = Path(__file__).resolve().parent

def main():
    dest = HERE / "qa_synthetic"
    dest.mkdir(exist_ok=True)
    m = load_model(MODEL); d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    baseline_ankle_z = np.mean([d.xpos[m.body(s + "_ankle_pitch_link").id, 2] for s in ("left", "right")])
    times = np.arange(81) / 20.
    points = []
    for t in times:
        d.qpos[:] = m.qpos0
        d.qpos[0] = .2 * t
        d.qpos[1] = .12 * np.sin(t)
        for s, phase in (("left", 0.), ("right", np.pi)):
            value = np.sin(t * 3 + phase)
            for joint, angle in (("hip_pitch_joint", .10 * value), ("knee_joint", .2 + .15 * value),
                                 ("shoulder_pitch_joint", -.2 * value), ("elbow_joint", .4)):
                d.qpos[m.joint(s + "_" + joint).qposadr] = angle
        mujoco.mj_forward(m, d)
        p = np.asarray([d.xpos[m.body(BODY_MAP[n]).id].copy() for n in CORE])
        p[:, 2] -= baseline_ankle_z
        points.append(p)
    points = np.asarray(points)
    valid = np.ones(len(times), bool)
    valid[40] = False
    points[40] = np.nan
    source = dest / "human_synthetic.npz"
    output = dest / "robot_synthetic.npz"
    np.savez_compressed(source, world_joints=points, timestamps_s=times, joint_names=CORE,
        pelvis_world=(points[:, CORE.index("left_hip")] + points[:, CORE.index("right_hip")]) / 2,
        valid=valid, metadata=json.dumps({"units": "m", "world_up": "z",
        "source_type": "synthetic_regression_fixture", "metric_scale_status": "synthetic_metric_ground_truth"}))
    meta = run(source, output, iterations=18)
    with np.load(output, allow_pickle=False) as z:
        qp, good = z["qpos"], z["valid"]
        assert qp.shape == (81, 36)
        assert not good[40] and np.isnan(qp[40]).all(), "Invalid gap must remain explicit"
        assert not z["temporal_link_from_previous"][41], "Do not link velocities across an invalid frame"
        expected_xy = (points[:, CORE.index("left_hip"), :2] + points[:, CORE.index("right_hip"), :2]) / 2
        np.testing.assert_allclose(qp[good, :2], expected_xy[good], atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(qp[good, 3:7], axis=1), 1., atol=1e-10)
        assert meta["max_joint_limit_excess_rad"] == 0.
        assert meta["robot_endpoint_displacement_xy_m"] > .79, "Open path must not be forced to close"
        assert meta["metric_scale_status"] == "synthetic_metric_ground_truth"
        assert np.isfinite(z["qvel"][good]).all(), "Within-segment qvel should be finite"
    bad = dest / "bad_timestamps.npz"
    np.savez(bad, world_joints=points[:2], timestamps_s=[0., 0.], joint_names=CORE)
    try:
        load_source(bad)
        raise AssertionError("Duplicate timestamps accepted")
    except ValueError as e:
        assert "increasing" in str(e)
    result = {"checks": ["model_loads", "joint_bounds", "normalized_quaternion", "world_xy_preserved", "open_path_not_closed",
        "invalid_gap_not_fabricated", "no_velocity_bridge", "scale_status_preserved", "duplicate_timestamps_rejected"], "passed": True,
        "mean_target_fit_m": meta["mean_target_fit_m"], "max_sole_penetration_m": meta["max_sole_penetration_m"]}
    (dest / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
