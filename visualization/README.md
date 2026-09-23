# Human / Unitree G1 comparison renderer

`render_comparison.py` renders the actual supplied MuJoCo model and qpos with a fixed
world camera. It preserves the input coordinates and never aligns endpoints or
forces a trajectory to close. The video includes raw source frames with a pose
overlay, a G1 view, both pelvis paths, an XY trajectory plot, and endpoint distances.

## Input contract

- Human NPZ: `timestamps_s [N]`, `world_joints [N,J,3]`, `joint_names [J]`.
  `joints_world` is accepted as an alias. Optional `pelvis_world [N,3]`;
  otherwise named pelvis or hip midpoint is used. Optional `valid [N]`,
  `frame_indices [N]`, `keypoints_2d [N,J,2]` in original video pixels.
- Robot NPZ: `timestamps_s [M]`, `qpos [M,nq]`, optional `valid [M]`.
  The first seven qpos values must be free-root XYZ and quaternion WXYZ.
- Both sequences must use the same world coordinates and clock. `timestamps_ns`
  is also supported when used by **both** inputs. No implicit independent clock
  alignment is allowed. Coordinates are displayed in meters, Z up.
- Optional camera JSON: `K [3,3]`, `T_world_camera [4,4]`, `distortion [5]`.
  This supports world-to-image projection when image keypoints are unavailable.
- The supplied model should be a scene XML with floor and lights. The renderer
  changes only in-memory light and framebuffer settings; source assets are untouched.
- `--match-source-camera` uses human NPZ `K` and `T_world_from_camera` for a fixed
  optical-camera view, including camera roll. These may be inferred / nominal values;
  the option does not turn them into independently measured calibration. Without
  this flag, one camera is fitted to the entire trajectory and held fixed.

Run from the project directory:

```powershell
.\.venv\Scripts\python.exe visualization\render_comparison.py `
  --human outputs\human_world.npz --robot outputs\g1_motion.npz `
  --model assets\unitree_g1\scene.xml --source-video C:\path\capture.avi `
  --source-label "HIKVISION / FIXED CAMERA" `
  --quality-label "MODEL-SCALE / UNCALIBRATED" `
  --output outputs\comparison.mp4
```

For the primary presentation with root-smoothed world motion, retain the raw image
projections stored in that NPZ, and make the distinction visible:

```powershell
.\.venv\Scripts\python.exe visualization\render_comparison.py `
  --human outputs\SESSION\world_temporal\world_joints.npz `
  --robot outputs\SESSION\g1_motion_temporal.npz `
  --model assets\unitree_g1\scene.xml --source-video C:\path\capture.avi `
  --match-source-camera --source-label "HIKVISION / SOURCE 2D / RAW" `
  --robot-label "UNITREE G1 / ROOT-SMOOTHED" `
  --pose-label "raw SAM 3D projection / source pixels" `
  --quality-label "ROOT-SMOOTHED / MODEL SCALE / UNCALIBRATED" `
  --fps 30 --output outputs\SESSION\comparison.mp4
```

The source video frame and 2D overlay always come from the same inference sample.
At 30 FPS output with 15 Hz inference, these matched left-panel frames repeat;
the robot qpos is interpolated for display. This preserves overlay alignment and
does not claim 30 Hz independent pose observations.

Without `--source-video`, the left panel is an explicitly labeled world-skeleton
diagnostic. It is not presented as a real Hikvision recording. Use the quality label
to describe whether world extrinsics and metric scale have actually been calibrated.

Outputs: MP4 (H.264), preview PNG, contact sheet JPG, synchronization NPZ, and a JSON
report with input/model hashes and full-sequence endpoint metrics. Endpoint distance
is only evidence of loop closure when the recording itself contains a known return
to the starting location. It is not a drift metric for an arbitrary non-loop motion.

Interpolation is for display only. Invalid samples and gaps larger than 0.1 s are
not bridged. Neither resampling nor a plausible-looking render certifies contact,
balance, collision safety, actuator limits, or suitability for locomotion training.
