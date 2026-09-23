# 海康固定单目 → SAM 3D Body → 固定世界 → MuJoCo G1

此目录是原 ZED 复现包之外的新适配工程。真实源视频在 `C:\Users\liuq8\MVS\Data\MV-CS020-10GC(DB2677346)`；没有用旧 ZED 成片替代海康推理结果。`retarget/qa_synthetic` 和 `validation` 下的 synthetic 文件仅是合成测试。

## 运行

```powershell
cd C:\YXW\视觉\hik_g1
.\.venv\Scripts\python.exe run_pipeline.py --video "C:\Users\liuq8\MVS\Data\MV-CS020-10GC(DB2677346)\Video_20260922162126269.avi"
```

默认每4个源帧推理1帧，约15Hz的真实SAM3D结果渲染为30FPS。插值不增加真实观测；原始帧号和 ffprobe 时间戳保留。首次运行需要已安装依赖、官方原包权重及公开 YOLO 检测权重。所有视频解码、推理和渲染均在本机，不上传视频。

`infer_hik.py prepare`：检测并跟踪主要前景人物、抽取帧、记录源文件hash和内参来源。

`infer_hik.py infer`：使用原包官方 DINOv3 SAM 3D Body 和 MHR，body 模式输出70关节。帧结果可断点续跑。

`reconstruction/fixed_camera_world.py`：整段共用一个相机到世界刚体变换。未提供标定时，从人体上方向与候选低足点推断展示用地面。它不是实测地面，不逐帧锁脚，也不施加回环约束。

`retarget/world_to_g1.py`：使用官方 G1 29DOF 模型做关节范围内的雅可比IK；人体骨盆XY保持，人体身高按恒定腿长比例适配G1，记录高度转换。保存 qpos、qvel、机器人各刚体轨迹、关节顺序及质量指标。

`visualization/render_comparison.py`：真实视频上叠加投影骨架，右侧固定世界相机渲染G1，显示人/机器人骨盆3D拖尾、地面投影、XY图与首尾间距。

默认流水线另生成 `world_temporal` 和 `g1_motion_temporal.npz`，仅对人体整体平移做有位移/地面门限的时序降抖，并从原相机 K/T 渲染 `comparison.mp4`。`--raw-only` 输出不经平移降抖的 `comparison_raw.mp4` 全景版；原始人体/G1数据始终保留。降抖可能扰动部分支撑足，不能将展示更平滑理解为动力学更可行。实际结果与局限见 `RESULTS_CN.md`。

## 结果的正确解释

- 未给内参时，使用上游默认焦距 `sqrt(width²+height²)`、图像中心主点；畸变未知。字幕必须保留 MODEL-SCALE / UNCALIBRATED。标为m的数值是模型名义米，不是经实测校准的物理精度。
- 固定相机允许整段共享同一参考坐标，但单目视频本身不能提供独立的绝对尺度保证。需相机内参/畸变、地面外参，以及人体身高或场景已知尺度来校验。
- 三段视频分别估计世界原点和地面，不经公共场景标定不能拼成同一世界坐标。
- 首尾间距只描述这两个时刻；只有人在实际场景中回到同一位置，这个间距才有闭合误差意义。不同身体姿态也会改变骨盆位置。
- 人与G1的XY轨迹重合来自重定向设计，不能作为人体世界轨迹准确性的独立证据。
- 运动学回放不是已训练的行走控制器。后续模仿学习要筛查缺帧、速度、接触、滑脚、自碰撞并训练动力学跟踪策略；当前不宣称机器人可直接稳定行走。

## 标定输入

`--intrinsics` 接收JSON：`K`为3×3矩阵；如视频有非零畸变，应先按同一标定去畸变，并使用去畸变后的视频及K。当前入口会拒绝非零畸变直接略过。

`--world-calibration` 接收JSON：`T_world_from_camera`为4×4、Z向上、右手世界坐标的相机到世界变换；可附`metric_scale`与`scale_provenance`。公式为 `P_world = R @ (metric_scale * P_camera) + t`。尺度必须作用所有相机点，不能只缩放人的骨架而遗漏根轨迹。

详见 `docs/world_reconstruction_audit.md`、`retarget/README.md`、`visualization/README.md`。

## 来源

- SAM 3D Body：[官方仓库](https://github.com/facebookresearch/sam-3d-body)，代码与权重沿用用户复现包。
- G1：[Unitree 官方 MuJoCo模型](https://github.com/unitreerobotics/unitree_mujoco/tree/main/unitree_robots/g1)，固定commit和逐文件hash保存在`assets/unitree_g1/SOURCE.json`。
- 检测器：[Ultralytics官方模型资产](https://github.com/ultralytics/assets/releases/tag/v8.3.0)，YOLO仅用于人物框，人体3D来自SAM3D。
