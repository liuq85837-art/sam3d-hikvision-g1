# G1 世界坐标重定向

`world_to_g1.py` 将具名人体世界关节转换为宇树官方 29 自由度 G1 的有限位运动学参考轨迹。
不修改源世界轨迹端点，不做回环强制闭合。机器人 XY 轨迹保留输入人体髋关节中点的米坐标；Z 高度使用全片固定腿长比例匹配机器人身高，因此机器人和人体的骨盆 Z 不会相等。此变换及单目尺度可信度写入输出 metadata。

```powershell
.\.venv\Scripts\python.exe retarget\world_to_g1.py outputs\human_world.npz outputs\g1_motion.npz
.\.venv\Scripts\python.exe retarget\verify_retarget.py
```

输入 NPZ：`world_joints[T,J,3]`（也接受 `joints_world`）、`joint_names[J]`、`timestamps_s[T]`（也接受 `timestamps`）。必须为 Z 向上、米；若 metadata 提供 units/world_up，程序验证它们。左右 hip/knee/ankle/shoulder/elbow/wrist 共 12 个关节为必需字段。可选 `confidence[T,J]`、`valid[T]`、`frame_indices[T]`、JSON 字符串 `metadata`。低置信度与 NaN 帧保留为显式无效间隙，不在重定向阶段伪造。

输出 NPZ：`qpos[T,36]` = 根平移 xyz、四元数 wxyz、29 个弧度制关节角；`joint_names[29]` 给出顺序。`qvel[T,35]` 为 MuJoCo tangent velocity；其根旋转部分与 qpos 四元数不是同一个维度。另有 `xpos`、`xquat`、`body_names`、`pelvis_world`（机器人浮动根）、`human_pelvis_world`（人体髋中点）、`valid`、`timestamps_s`、`frame_indices`、`ik_residual_m`、`sole_contact_positions`。metadata 与同名 JSON 文件记录模型路径、源 hash、尺度标签和质量摘要。

IK 保留人体各肢段方向，按 G1 自身骨长构造目标；使用 MuJoCo body Jacobian、阻尼最小二乘、关节限位、弱中性姿势和时序约束。肩线与躯干方向约束 torso；根方向来自髋横轴和重力。脚底采用弱平放方向先验，地面穿透用软约束降低。髋的俯仰、肢段扭转和手腕方向无法从 12 个点唯一确定，依赖先验。

质量指标中的 target-fit 是 IK 对构造目标的残差，不是单目世界坐标误差。`max_sole_penetration_m` 为官方脚底接触球的穿地量；未证明零滑脚、自碰撞安全、力矩可行、动态平衡或闭环控制稳定。输出可以作为后续模仿学习候选参考，但须再做接触/速度筛查、动力学跟踪策略训练与仿真验收。

`qa_synthetic` 为明确标注的合成回归夹具，仅用于接口/几何验证，不是海康视频结果。

模型来自 [Unitree 官方 unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco/tree/main/unitree_robots/g1)。`fetch_model.py` 记录下载 commit 和逐文件 SHA-256 到 `assets/unitree_g1/SOURCE.json`，保留官方许可。`g1_29dof.xml` 未修改；`scene.xml` 是本项目的灯光地面配置。

Windows 中文目录通过 `load_model()` 的内存 XML 和资源字节加载，规避 MuJoCo 原生文件路径接口的编码问题。合成测试已在当前环境用真实 G1 模型实际通过。根朝向暂不平滑；输出相邻有效帧 yaw 跳变量/角速度，以及超过 90 度的帧号，以便审计转身与左右翻转歧义。

`contact_diagnostics.py source.npz robot.npz` 输出独立的 `.contact_candidates.npz/.json`，不改变 raw 运动。默认 source toe/heel 最低高度 ≤ 2.5 cm、同脚世界 XYZ 速度 < 0.25 m/s、连续至少 3 帧定义候选支撑段；人脚取三点中心，机器人脚取四个官方脚底接触球底部的中心。输出 source/robot 对应脚速度、candidate_stance、run_id、潜在滑脚 review mask 和关节速度 review mask。候选 contact 和 6 rad/s review 阈值均不是实测接触或硬件界限；未标定单目数据中的米也只是模型尺度。`verify_contacts.py` 验证短段及无效/时间间隔不会拼成支撑段。

同一审计会读取对应片段 `camera/tracks.json` 与 `camera/input.json`，增加 `source_bbox_near_border`（12 像素内触边启发式）、边距、检测框来源有效性及缺失标记。`reference_quality_review` 汇总触边、配置关节速度阈值、候选支撑段异常脚速；`reference_review_pass` 仅表示通过这些筛查，不是训练已就绪标签。不会覆盖源 valid。可通过 `--tracks`、`--camera-input`、`--bbox-border-px` 指定其它输入。

后续可对比作者发布的 [GMR](https://github.com/YanjieZe/GMR)，其支持 G1 和更丰富的具名骨骼旋转输入。本模块未复制 GMR，也不将本模块称作 GMR。
