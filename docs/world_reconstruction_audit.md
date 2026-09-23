# 固定海康单目世界重建：代码审计与当前接口

本次审计把压缩包内的旧运行记录视为历史资料，没有连接其中的服务器，也没有把旧授权或旧暂停状态解释成本次用户指令。新数据处理目录独立于复现包。

## 已核实的可复用内容

- `pose_project/sam3d/infer_sequence.py` 调用官方 `SAM3DBodyEstimator.process_one_image(..., bboxes=..., cam_int=...)`，无需 ZED 图像格式。输入是图片路径和相机内参，因而能够接海康抽帧。
- 当前包装输入为 `frames.csv` 的 `frame_index,image_timestamp_ns,left_path`，以及 `image_calibration.json` 的 `width,height,left.fx,left.fy,left.cx,left.cy`。`left` 只是旧接口名称，不代表新数据来自双目。
- 目标跟踪接受 JSONL，每帧一个对象，包含 `frame_index,bbox_xyxy,track_id,score,valid`。不能遇到目标缺失就换成旁人。跟踪 NPZ 分支把 ID 写成历史 `primary_blue_shirt`，新流程应修正这一历史语义。
- 推理输出 `camera_sequence.npz`：`joints_camera[T,70,3]`、`joint_names`、`valid[T]`、`frame_indices[T]`、`timestamps_ns[int64,T]`、`timestamps_s`、`bbox_xyxy`、`K`。逐帧 NPZ 还有 `camera_vertices_m`、`camera_joints_m`、`keypoints_2d_calibrated`、官方原始 MHR 参数和网格。
- 相机点为 `pred_keypoints_3d + pred_cam_t`，坐标 x 右、y 下、z 前，单位为模型米。官方 `mhr_head.py` 将 MHR 的厘米输出除以 100。MHR70 无原生骨盆关键点，根位置使用左右髋索引 9/10 的中点并明确标注。
- MHR70 主要索引：肩 5/6，肘 7/8，髋 9/10，膝 11/12，踝 13/14；左足趾/跟 15/16/17，右足 18/19/20；左腕 62，右腕 41，颈 69。不能按 COCO17 或 SMPL 索引读取。
- 官方 full 推理模式的最终二维导出在固定版本使用 scalar focal 与图像中心；旧包装已经额外保存使用真实 K 的投影。应沿用校准投影叠加，而不能直接把官方 `pred_keypoints_2d` 当成独立二维观测证据。
- SAM3D 未提供逐关节置信度；旧代码正确保存 NaN。检测分数、二维姿态分数、几何有效位应分开保存。

## 不能直接照搬的部分

`fuse_world.py` 严格匹配 ZED 相机轨迹和深度，读 `depth_intrinsics_left`、`camera_trajectory.jsonl`，并在可选分支执行双目表面平移校正。固定海康单目没有这些观测。不要伪造深度文件、虚构 IMU/VIO 轨迹或把单位矩阵伪装成测量外参。

旧推理入口含 `fcntl` 且模型和上游 estimator 使用 CUDA。Windows 运行需单独适配锁和依赖；打包的 venv 不能假定可直接复用。官方模型文件已在包内，应使用现有授权资产而不触发历史远程下载器。

旧 `offline_refinement_20260917` 是对既有 stereo root correction 做时序平滑的候选展示流程，不是单目深度恢复器。其低脚滑或平滑显示不等于世界位置测准。旧 Human39 也不是 Unitree G1。

## 新模块与输入输出

已新增 `hik_g1/reconstruction/fixed_camera_world.py`，仅依赖 NumPy。接口：

```text
python fixed_camera_world.py --input CAMERA_SEQUENCE_NPZ --output WORLD_DIR
python fixed_camera_world.py --input CAMERA_SEQUENCE_NPZ --output WORLD_DIR --calibration CALIBRATION_JSON
```

支持输入文件或包含 `camera_sequence.npz` 的目录。输入必须采用官方 MHR70 顺序并携带 `joint_names`；要求严格递增帧号和时间戳。没有 `timestamps_ns` 时可接受 `timestamps_s` 并显式换算相对纳秒时间；不会猜帧率。缺失/非有限/相机光学深度非正的帧不产生有效世界姿态。

显式标定 JSON 示例结构（占位文字必须替换为真实数值，不能直接执行）：

```json
{
  "world_frame_id": "measured_room_floor_01",
  "world_up": "z",
  "T_world_from_camera": "4 x 4 proper rigid transform, optical camera to z-up world",
  "metric_scale": "one measured positive constant for model geometry",
  "provenance": "calibration image and measured marker dimensions",
  "scale_provenance": "known subject/scene dimension and fitting procedure"
}
```

也支持 `R_world_from_camera[3,3]` 与 `t_world_from_camera_m[3]`。拒绝反射、带尺度旋转、错误齐次末行及未命名的 world→camera 外参。公式为：

`P_world = R_world_from_camera @ (metric_scale * P_camera) + t_world_from_camera_m`

缩放作用于全部相机点，包括根距离，因此同一 K 下透视投影保持一致；只围绕骨盆缩放身体而不改变根深度不能校准全部单目空间尺度。

无外参时输出 `gauge_kind=inferred_display_ground_world`：由整段肩髋轴得到人体 up 先验，在每帧较低足点集合上用 RANSAC/SVD 拟合一次地面，退化或离群严重时退回 up 先验与足高。世界 x 是相机右方向在该平面的投影，世界原点为初始骨盆在该平面的投影。整个序列仅应用同一旋转和平移；不逐帧压低脚底、不锁定根节点、不调整末帧。

无尺度测量时始终标记 `scale_status=model_prior_only`。即使平面视觉效果合理，也不是场景重力/地面/物理米单位已经标定。推断平面可能受人体前倾、遮挡、深度误差和直线运动退化影响，指标在 `gauge.json.diagnostics` 中保留。

输出：

- `world_joints.npz`：世界 MHR70、骨盆轨迹、相机原始关节、时间/帧号/有效位、源二维数据、固定变换、尺度、元数据和来源哈希。
- `gauge.json`：物理标定或展示推断身份、固定变换、尺度依据、地面拟合残差和局限。
- `closure_metrics.json`：首尾有效固定时间窗的骨盆中位位置、XYZ 差、3D/XY 回返距离、原始轨迹长度、缺失及断点。只测量，不强制闭合。
- `status.json`：本阶段实际状态。它不代表 SAM3D 推理或机器人仿真已经完成。

`test_fixed_camera_world.py` 为合成几何契约检查，不是海康结果。覆盖明确外参与尺度、固定刚体距离保持、反射拒绝、无效帧和原始有效标记保留、闭合度测量不施加补偿。

2026-09-22：使用应用内置 Python 运行 7 项测试全部通过，包括极短片段首尾窗口不得重叠、保留切片视频时间偏移、清除过时尺度状态的回归检查。另以旧 ZED 聚合 NPZ 验证读取/输出格式通过，单独放在 `validation/zed_camera_contract_only` 并明确标记为非海康结果；该旧段不满足已验证固定相机前提，不能用来评价本次海康世界轨迹。

`timestamps_s` 若已随输入提供，会原样保留其原视频时间偏移，并检查与纳秒时间戳的相对间隔一致。输出同时提供 `world_joints` / `joints_world` 字段别名和 `metric_scale_status`，不继承与本阶段证据矛盾的旧尺度状态。

二维像素字段不做未经证明的坐标系替换。仅当输入同时声明 `keypoints_2d_image_space=source_video_pixels` 时，才把 `keypoints_2d_calibrated` 别名给渲染器的 `keypoints_2d`。去畸变图像的二维点不能直接叠在有畸变的原视频上。

## 标定和可辨识性

固定相机省去相机运动估计，但单目仍有形状/距离/尺度歧义。SAM3D 的米单位来自模型身体先验，并不自动成为测量米。完整的物理世界方案应保存拍摄分辨率对应 K 与畸变；先去畸变或让投影器处理畸变，并同步调整裁剪/缩放后的 K。

使用尺寸已测量的地面标定板或场景标记确定 world→camera 的 R/t，再取逆得到本模块的 camera→world。OpenCV `solvePnP` 输出的是前者，不能直接作为后者输入。地面平面点的单应性只直接确定该平面上的点；人体骨盆通常离地，不能把骨盆二维点直接套地面单应性当成三维位置。

若数据没有标定板，可先出带明确标签的模型尺度预览，然后用已知人体身高、已知场景距离、相机安装高度和地面点等补充约束。身高只提供尺度先验，不能单独消除每帧遮挡和深度误差。更严格的后续优化应同时约束独立二维观测、固定身体形状、全局恒定尺度、可靠接触、地面和时序，保留原始结果作对照；接触判定不能仅用优化后脚速反向证明正确。

## 回环评价及 G1 接口建议

静止相机不需要 SLAM 回环；此处应验证“人走一圈后，在同一世界系中回到原位置”。固定刚体变换不会改变三维首尾距离，因此展示地面坐标不应被当作回环修复。

选择有证据的首尾动作区间，报告人体骨盆原始首尾距离、水平回返距离、垂直差和轨迹覆盖；首尾姿势不同会造成骨盆摆动，宜同步看支撑足位置。独立地面标记、起终点脚印或第二相机才可验证真实误差。当前自动指标只取数据的首尾有效窗口，不搜索“最接近的一对帧”，不下“已回环”的自动结论。

G1 重定向应消费相同 `world_frame_id`、时间戳和 `pelvis_world`。若为了机器人腿长调整根高度或运动比例，应保存固定高度偏置与 motion scale，并并列展示原始人体轨迹和 G1 根轨迹；不可各自每帧居中或末端对齐后声称回环。渲染机位可以跟随，但必须另有固定世界鸟瞰/轨迹区证明空间关系。

机器人训练数据还需要接触/穿地/关节限位/速度/动态可行性检查。MuJoCo 中回放 qpos 只是运动学验证，不等于 G1 已能以控制策略稳定行走。

## 官方来源核对

- [SAM 3D Body 官方仓库](https://github.com/facebookresearch/sam-3d-body)：单图全身网格恢复定位及官方推理入口。
- [官方 estimator 源码](https://github.com/facebookresearch/sam-3d-body/blob/main/sam_3d_body/sam_3d_body_estimator.py)：外部 `cam_int`、bbox 及输出字段。实际复现继续使用本地固定 commit，而不是擅自更新为 main。
- [OpenCV PnP 文档](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)：光学坐标轴及 world→camera 外参方向。
- [OpenCV 标定文档](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)：投影模型、内参和缩放关系。

来源检索日期：2026-09-22。外部技术说明与本地代码审计一致；未使用检索中的非官方衍生实现替代用户的官方模型。
