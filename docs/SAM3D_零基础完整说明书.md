# SAM 3D Body 零基础完整说明书

**从打开项目，到海康人体姿态、固定世界坐标与宇树 G1 重定向**

版本：2026-09-23 · 适用系统：你当前的 Windows / PowerShell · 依据：本机实际源码、运行参数和已生成数据。

这份手册面向第一次接触 Python、三维姿态和机器人仿真的读者。你不需要先学会神经网络，先按“观看结果 → 小规模运行 → 读懂数据 → 修改一个参数”的顺序操作即可。

**目前已经完成：** 两段真实海康视频的人体三维推理、固定参考世界坐标、G1 运动学重定向与对比视频。**尚未完成：** 实测尺度验证、真实回环误差验收、机器人动态平衡和行走策略训练。后文会解释这些区别。

本手册的命令以现有工程为准，不是任意版本 SAM 3D 的通用安装教程。新增的中文源码注释不改变模型计算；既有视频也不会因为修改源码而自动更新。

## 阅读路线与目录

第一次使用，先读第 1～7 章；准备修改程序，继续读第 8～16 章；准备做标定、回环验收和机器人训练，再读第 17～20 章。

| 你想完成的事 | 对应章节 |
|---|---|
| 理解这个项目到底做什么 | [1. 项目全貌](#s01)、[2. 基础词汇](#s02) |
| 找源码、在 VS Code 中操作 | [3. 文件地图](#s03)、[4. 编辑器与终端](#s04) |
| 看视频、第一次运行 | [5. 观看结果](#s05)、[6. 环境检查](#s06)、[7. 小规模试跑](#s07) |
| 看懂核心 Python 文件 | [8. 全流程](#s08)、[9. 外层接口](#s09)、[10. 核心模型](#s10) |
| 理解相机与世界坐标 | [11. 坐标和单位](#s11)、[12. 世界重建与降抖](#s12) |
| 理解 G1 与输出数据 | [13. 重定向](#s13)、[14. 文件说明](#s14)、[15. 读取数据](#s15) |
| 修改参数、重新生成结果 | [16. 修改与重跑](#s16) |
| 做严谨的精度与训练验证 | [17. 标定](#s17)、[18. 回环](#s18)、[19. 训练准备](#s19) |
| 解决问题、迁移工程 | [20. 排错](#s20)、[21. 备份与复现](#s21)、[22. 学习练习](#s22) |

<a id="s01"></a>

## 1. 这个项目到底在做什么

你录制的是人走路的视频。视频中每一帧只有颜色和像素位置，没有直接记录人的膝盖离相机几米，也没有记录机器人每个电机应转多少角度。项目分几步补上这些信息。

```text
海康原始视频 AVI
    ↓ 检测人在哪里，选择主要人物，按时间抽取图像
SAM 3D Body
    ↓ 从单张图像估计人体形状、三维关节点与相机相对位置
相机坐标中的人体序列
    ↓ 用整段固定的旋转、平移与尺度建立参考世界坐标
固定世界坐标中的人体序列
    ↓ 可选：只对整体平移降抖，保留原始版
人体运动 → G1 运动学重定向
    ↓ 按机器人骨长、关节结构和限位求关节角
MuJoCo 渲染
    ↓ 同步真实画面、人体骨架、机器人动作与骨盆轨迹
MP4 视频 + NPZ 动作数据 + JSON 质量记录
```

**SAM 3D Body 是其中的人体模型，不是整套项目的名称。** 检测、相机到世界变换、时序处理、G1 重定向和视频排版由外部程序组织。

原复现包针对 ZED 双目相机和 IMU，能够利用那套设备的额外观测。当前海康方案使用固定的单目视频，不能直接继承 ZED 的深度、IMU 或外参，也不能把旧视频的坐标当作新场景的坐标。

相机固定后，同一段视频可以使用同一个相机到世界的变换。但单目深度仍依赖人体模型先验，相机固定本身不会消除尺度不确定性。

### 1.1 你现在看到的“界面”是什么

当前工程通过命令行执行计算。VS Code 是编辑代码的工具；PowerShell 是输入命令、看日志的窗口；播放出的 MP4 是已完成结果的展示。工程目前没有独立的“点击按钮实时识别人”的桌面应用，也没有接入海康实时视频流的运行界面。

启动视频播放不会重新推理。修改代码后，需要执行对应流程，生成新的数据或视频，才能看到变化。

<a id="s02"></a>

## 2. 先认识这些基础词汇

| 名词 | 用简单的话解释 | 在本项目中的例子 |
|---|---|---|
| Python | 执行 `.py` 文件的程序语言 | `infer_hik.py` |
| 源代码 | 人写的处理规则 | 加了中文注释的两个核心文件 |
| 模型权重 / checkpoint | 模型学习得到的一大组数值 | `model.ckpt` |
| 推理 | 使用已有权重计算新图像的结果 | 从海康图片恢复人体 |
| 训练 | 用数据调整模型或控制策略的参数 | 后续训练机器人跟踪参考动作 |
| 虚拟环境 | 一套单独存放的 Python 与依赖库 | `hik_g1\.venv` |
| 依赖库 | 程序调用的现成工具 | PyTorch、NumPy、MuJoCo |
| CPU / GPU | 通用计算处理器 / 擅长大规模并行计算的处理器 | YOLO 检测用 CPU，当前人体推理用 CUDA GPU |
| CUDA | NVIDIA GPU 的计算接口 | PyTorch 将张量放到 `cuda` |
| 帧 | 视频中的一张图片 | 原片约每秒 60 帧 |
| FPS / Hz | 每秒画面数 / 每秒采样或计算次数 | 输出 30 FPS，人体推理约 15 Hz |
| 关键点 | 描述人体某个位置的三维点或二维点 | 左髋、右膝、左脚踝 |
| 网格 mesh | 用大量三角形组成的身体表面 | SAM 3D 生成的人体表面 |
| 骨盆 pelvis | 本项目用左右髋关键点的中点表示 | 绘制人体整体运动轨迹 |
| 坐标系 | 规定原点和三个轴的参考规则 | 相机坐标、世界坐标 |
| 内参 K | 像素与相机射线之间的关系 | 焦距 fx、fy 和主点 cx、cy |
| 外参 R、t | 两个坐标系之间的旋转和平移 | 相机到固定世界的变换 |
| IK | 逆运动学：给定末端位置，反求关节角 | 让 G1 的脚和手接近构造的目标 |
| DOF | 自由度：可以独立变化的运动量 | 当前 G1 有 29 个可动关节 |
| root / 根 | 整个模型的整体位置和朝向 | G1 在房间里走到哪里、面向哪里 |
| NaN | 缺失或不能确定的数值 | 无有效人体时保留缺测 |
| JSON | 可阅读的结构化文本 | 内参、运行记录、质量报告 |
| NPZ | 保存多个数值数组的压缩文件 | 人体坐标、机器人关节角 |

### 2.1 怎样读 Python 中的几个符号

```python
# 以 # 开头的是注释，用来解释代码，不参与计算。
name = "left_hip"       # 把字符串保存到变量 name。
indices = [9, 10]        # 列表，包含两个整数。
result = {"valid": True} # 字典，用名字查找内容。

def midpoint(a, b):      # 定义一个函数。
    return (a + b) / 2   # 缩进表示这一行属于函数。
```

`.` 常用于访问对象的成员；例如 `model.run_inference(...)` 是调用模型的方法。`self` 表示当前对象，`self.cfg` 是它保存的配置。`class` 把相关数据和函数组织成一个对象；`SAM3DBodyEstimator` 和 `SAM3DBody` 是两个职责不同的类。

Python 数组通常从 **0** 开始编号。因此 `a[0]` 是第一项；`a[3:7]` 取第 3、4、5、6 号位置，不包含 7。`*` 常表示乘法，`@` 在 NumPy/PyTorch 中用于矩阵乘法。

### 2.2 张量 shape 是什么

张量可以先理解为多维表格。`[461, 70, 3]` 表示 461 个时间样本，每个样本有 70 个关键点，每个关键点有 x、y、z 三个数。

```python
point = joints[0, 9]       # 第一个时间样本中，第9号关键点的xyz。
all_left_hips = joints[:, 9, :]  # 所有时间样本中的左髋。
```

这里的 461 是抽样后的人体数据数量，不是原视频总帧数。判断时间应读取 `timestamps_s`，不要凭数组下标猜测秒数。

<a id="s03"></a>

## 3. 你的文件存放在哪里

本机有两个相关目录：**原始 SAM 3D 复现工程**和**海康适配工程**。运行海康整套流程时，工作目录应该是 `hik_g1`。

| 内容 | 本机实际路径 |
|---|---|
| 海康适配工程 | `C:\YXW\视觉\hik_g1` |
| SAM 3D 官方源码 | `C:\YXW\视觉\pose_world_sam3d_repro_20260918\pose_project\sam3d\repo` |
| SAM 3D 权重目录 | `C:\YXW\视觉\pose_world_sam3d_repro_20260918\pose_project\sam3d\checkpoints\user_dinov3_20260917` |
| 海康原视频 | `C:\Users\liuq8\MVS\Data\MV-CS020-10GC(DB2677346)` |
| 当前可用 Python | `C:\YXW\视觉\hik_g1\.venv\Scripts\python.exe` |
| 结果目录 | `C:\YXW\视觉\hik_g1\outputs` |
| 结果压缩包 | `C:\YXW\视觉\hik_g1\delivery\hikvision_g1_results.zip` |
| 注释前的两个源码备份 | `C:\YXW\视觉\hik_g1\source_annotation_backups\20260923` |

```text
C:\YXW\视觉\
├─ pose_world_sam3d_repro_20260918\
│  └─ pose_project\sam3d\
│     ├─ repo\                         SAM 3D 官方源码
│     ├─ dinov3_official\              本地 DINOv3 源码
│     └─ checkpoints\...\             model.ckpt、MHR 资产和配置
└─ hik_g1\
   ├─ .venv\                          已安装的运行环境
   ├─ run_pipeline.py                  整套流程入口
   ├─ infer_hik.py                     海康检测、抽帧和人体推理
   ├─ reconstruction\                 固定世界坐标、根平移降抖
   ├─ retarget\                       G1 逆运动学和质量检查
   ├─ visualization\                  对比视频渲染
   ├─ assets\                         YOLO 权重和 G1 模型资源
   ├─ outputs\                        每段视频的结果
   ├─ docs\                           本说明书和专题文档
   ├─ delivery\                       结果包与校验记录
   └─ source_annotation_backups\      加注释前的备份
```

`infer_hik.py` 按上述相邻目录关系定位原复现包。只把 `hik_g1` 单独搬走，权重路径可能失效。结果 ZIP 也没有包含整个 Python 环境、SAM 3D 权重和原始 AVI，详见第 21 章。

### 3.1 建议从哪些文件读起

| 顺序 | 文件 | 先看什么 |
|---|---|---|
| 1 | [run_pipeline.py](../run_pipeline.py) | `main()` 如何串起各个阶段 |
| 2 | [infer_hik.py](../infer_hik.py) | `prepare()` 和 `infer()` |
| 3 | [sam_3d_body_estimator.py](../../pose_world_sam3d_repro_20260918/pose_project/sam3d/repo/sam_3d_body/sam_3d_body_estimator.py) | 单张图怎样进入模型 |
| 4 | [sam3d_body.py](../../pose_world_sam3d_repro_20260918/pose_project/sam3d/repo/sam_3d_body/models/meta_arch/sam3d_body.py) | `run_inference()`、`forward_pose_branch()` |
| 5 | [fixed_camera_world.py](../reconstruction/fixed_camera_world.py) | `reconstruct()` 怎样变换坐标 |
| 6 | [world_to_g1.py](../retarget/world_to_g1.py) | 怎样构造 G1 目标并求关节角 |
| 7 | [render_comparison.py](../visualization/render_comparison.py) | 怎样把画面和动作同步成视频 |

<a id="s04"></a>

## 4. VS Code 和 PowerShell 怎样操作

### 4.1 打开工程与文件

在 VS Code 中选择“文件 → 打开文件夹”，选中 `C:\YXW\视觉\hik_g1`。左侧资源管理器是文件树，点击 `.py` 文件即可查看。要阅读官方模型，可以另开一个窗口，打开第 3 章中的 `repo` 目录。

常用快捷键：`Ctrl+P` 按文件名查找，`Ctrl+F` 在当前文件搜索，`Ctrl+Shift+F` 在工程里搜索，`Ctrl+S` 保存。查看 Markdown 手册时，可以使用 `Ctrl+Shift+V` 打开预览。

修改 Python 时保留原有缩进，不要把编辑器显示的行号复制进文件。只改注释不会改变结果；改过函数后，先保存，下一次新启动的 Python 进程才会读取新文件。

### 4.2 打开终端

选择“终端 → 新建终端”，使用 PowerShell。本手册中的 `powershell` 代码块在终端执行，`python` 代码块应保存为 `.py` 再由 Python 执行，二者不要混用。

```powershell
Set-Location -LiteralPath 'C:\YXW\视觉\hik_g1'
Get-Location
```

第一行切换目录，第二行显示当前目录。看到 `C:\YXW\视觉\hik_g1` 才说明位置正确。`PS C:\...>` 是终端提示符，不用手动输入。

相对路径 `outputs\...` 从“终端当前目录”开始寻找，和你正在看的编辑器标签页没有必然关系。

### 4.3 指定正确的 Python

本手册始终显式使用项目 Python，因此不用先激活虚拟环境：

```powershell
& 'C:\YXW\视觉\hik_g1\.venv\Scripts\python.exe' --version
```

PowerShell 的 `&` 用于执行后面的程序路径；引号保护中文、空格和括号。不要使用系统里另一个不确定的 `python` 来代替它。

如果使用 VS Code Python 扩展的运行按钮，应先选择解释器 `C:\YXW\视觉\hik_g1\.venv\Scripts\python.exe`。新手运行整个流程时优先复制本手册命令，因为很多脚本必须带参数，不能只点“运行当前文件”。

### 4.4 停止、恢复与日志

在运行终端按 `Ctrl+C` 可以中断当前前台任务。已经保存的结果通常仍在磁盘上；人体逐帧缓存可在相同输入和参数下继续使用。是否能恢复还取决于哪一阶段被中断，先查看输出目录和最后几行报错。

日志中的 `RUN ...` 表示开始一个阶段，`Detected ...` 表示正在检测抽样图片，`Completed ... camera poses` 表示人体序列导出完成。运行期间某个阶段只在若干帧后打印一次，不应把短暂无新日志直接当作卡死。

<a id="s05"></a>

## 5. 不运行模型，先看已有结果

### 5.1 主视频

在资源管理器中双击以下文件，也可以复制命令：

```powershell
Start-Process -FilePath 'C:\YXW\视觉\hik_g1\outputs\Video_20260922162126269\comparison.mp4'
```

![真实海康姿态与G1对比](../outputs/Video_20260922162126269/comparison.preview.png)

这张图片来自已生成的真实对比视频，不是新生成的示意结果。

画面左侧是源视频和人体二维投影骨架；右侧是对应的 G1 运动学姿态。青色和橙色分别表示人体与 G1 的骨盆轨迹，下方提供平面轨迹与端点距离。

右侧的世界相机保持固定，不跟着机器人移动。默认主展示按输入数据记录的相机 K 和相机到世界变换设置视角；由于这些参数仍含估计与先验，“和原相机视角一致”不等于“完成真实标定”。

### 5.2 原始版和平滑版

| 文件 | 内容 |
|---|---|
| `comparison.mp4` | 当前主展示：根平移降抖候选，原相机视角 |
| `comparison_raw.mp4` | 原始世界轨迹对应的 G1，全景视角 |
| `comparison.preview.png` | 单张预览图 |
| `comparison.contact_sheet.jpg` | 多时刻拼图，用于快速检查 |

两种视频同时改变了“运动版本”和“展示视角”，所以不能只凭观看它们来判断滤波优劣。要比较平滑本身，应使用相同相机设置分别渲染 raw 和 temporal 数据。

### 5.3 字幕怎样理解

| 字幕 | 含义 |
|---|---|
| `MODEL-SCALE / UNCALIBRATED` | 使用模型尺度，未验证真实物理米制 |
| `ROOT-SMOOTHED` | 对人体整体平移进行过时序降抖 |
| `NO SUPPORTED HUMAN POSE` | 当前时刻没有受支持的人体姿态 |
| `NO SUPPORTED ROBOT POSE` | 当前时刻没有受支持的机器人姿态 |
| `kinematic retarget / physics not validated` | 运动学重定向，尚未验证动力学 |
| `TRAJECTORY ENDPOINT DISTANCE` | 选定序列首尾位置的距离，不自动等于回环误差 |

约 60 FPS 的原视频每 4 帧推理一次，人体真实采样约 15 Hz。最终视频为 30 FPS：左侧重复显示对应推理样本的源帧，右侧对机器人姿态做显示插值。它并没有产生每秒 30 次新的模型观测。

<a id="s06"></a>

## 6. 第一次运行前，做四项检查

这些检查只读取环境，不重新运行整段模型。先进入工程目录：

```powershell
Set-Location -LiteralPath 'C:\YXW\视觉\hik_g1'
```

### 6.1 Python 与 GPU

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU')"
```

现有环境记录为 Python 3.11、PyTorch `2.8.0+cu128`、MuJoCo `3.3.7`。此前在本机 RTX 5060 Laptop 8 GB 上完成过推理。显存用量和耗时会受模式、图片人数和其他程序影响，不把历史速度当作保证。

当前官方调用链含硬编码 `.cuda()`，即使某个命令允许写 `--device cpu`，也不能认为整条流程已经支持 CPU。

### 6.2 视频与权重存在

```powershell
Test-Path -LiteralPath 'C:\Users\liuq8\MVS\Data\MV-CS020-10GC(DB2677346)\Video_20260922162126269.avi'
Test-Path -LiteralPath 'C:\YXW\视觉\pose_world_sam3d_repro_20260918\pose_project\sam3d\checkpoints\user_dinov3_20260917\model.ckpt'
Test-Path -LiteralPath 'C:\YXW\视觉\pose_world_sam3d_repro_20260918\pose_project\sam3d\checkpoints\user_dinov3_20260917\assets\mhr_model.pt'
Test-Path -LiteralPath 'C:\YXW\视觉\hik_g1\assets\unitree_g1\scene.xml'
```

四项应显示 `True`。`False` 表示对应路径不存在，先确认目录是否被移动，不要立即重新下载所有资源。

### 6.3 FFmpeg 工具

```powershell
Get-Command ffmpeg, ffprobe
```

`ffprobe` 读取视频时间戳；`ffmpeg` 编码结果视频。当前工具位于 `C:\YXW\视觉\tools\ffmpeg\ffmpeg-9.0.2-essentials_build\bin`。若新开的终端找不到它们，可以仅在当前终端加入路径：

```powershell
$env:Path = 'C:\YXW\视觉\tools\ffmpeg\ffmpeg-9.0.2-essentials_build\bin;' + $env:Path
```

### 6.4 查看程序接受哪些参数

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --help
.\.venv\Scripts\python.exe infer_hik.py --help
```

`--help` 只显示用法，不开始完整推理。今后修改入口后，以实际帮助信息和源码为准，不要凭记忆给脚本添加不存在的参数。

<a id="s07"></a>

## 7. 第一次试跑：用独立目录处理约 8 秒

### 7.1 复制这段完整命令

```powershell
Set-Location -LiteralPath 'C:\YXW\视觉\hik_g1'
$taskVideo = 'C:\Users\liuq8\MVS\Data\MV-CS020-10GC(DB2677346)\Video_20260922162126269.avi'
$taskOutput = Join-Path (Get-Location) ('outputs\tutorial_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
.\.venv\Scripts\python.exe run_pipeline.py --video $taskVideo --output $taskOutput --stride 4 --max-frames 120
Write-Output $taskOutput
```

它会新建带时间的输出目录，保留既有交付结果。`--max-frames 120` 指 **最多 120 个抽样时间点**，不是“原视频前 120 帧”，也不是“120 个有效人体”。在约 60 FPS、stride=4 时，它覆盖约 8 秒。

如果采样片段没有足够有效人体、持续躺卧、脚部被遮挡或地面估计不成立，世界重建阶段可能失败。不要为了让流程通过而凭空填写标定矩阵；先换成更完整的片段或处理完整主视频。

### 7.2 参数逐个解释

| 参数 | 作用 | 本例 |
|---|---|---|
| `--video` | 输入视频路径 | 主海康 AVI |
| `--output` | 本次输出目录 | 自动生成的 `tutorial_...` |
| `--stride` | 每隔多少个源帧选一个 | 4 |
| `--max-frames` | 限制抽样数，便于试跑 | 120 |
| `--raw-only` | 跳过根平移降抖，输出 raw 全景视频 | 本例未使用 |
| `--overview` | 用固定全景视角代替原相机视角 | 本例未使用 |

初次加载模型可能明显慢于后续帧。程序需要经过检测、人体推理、世界变换、G1 求解和渲染，不是加载一次权重后立刻弹出实时窗口。

### 7.3 完成后打开视频

仍在同一个终端中执行：

```powershell
Start-Process -FilePath (Join-Path $taskOutput 'comparison.mp4')
```

如果你已经关闭终端，`$taskOutput` 变量也随之消失。到 `outputs` 找到刚生成的 `tutorial_日期时间` 文件夹，双击 `comparison.mp4` 即可。

### 7.4 处理完整视频

重新运行下面这段，去掉 `--max-frames`，使用另一个新目录：

```powershell
Set-Location -LiteralPath 'C:\YXW\视觉\hik_g1'
$taskVideo = 'C:\Users\liuq8\MVS\Data\MV-CS020-10GC(DB2677346)\Video_20260922162126269.avi'
$taskOutput = Join-Path (Get-Location) ('outputs\full_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
.\.venv\Scripts\python.exe run_pipeline.py --video $taskVideo --output $taskOutput --stride 4
```

不写 `--output` 时，默认使用 `outputs\视频文件名去掉扩展名`。对于实验和修改后的重跑，建议一直明确指定新目录。

### 7.5 中断后继续

如果检测/抽帧已经成功，只是人体推理中途停止，可以在**同一终端、相同输入**下运行：

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --video $taskVideo --output $taskOutput --skip-prepare
```

这会复用已准备的图片和 tracks；人体推理会跳过已存在的逐帧 NPZ。若抽帧阶段本身没完成，应重新执行原命令，不能跳过准备。

**缓存不会自动检测所有源码与权重变化。** 修改 SAM 3D 计算逻辑或更换权重后，不要用旧逐帧缓存验证新算法；请改用新的输出目录。单纯增加注释不影响计算，可继续复用已有结果。

<a id="s08"></a>

## 8. 从输入到输出，程序实际上怎样执行

`run_pipeline.py` 是调度入口，用当前 Python 依次启动子脚本。某阶段报错时，后续阶段不会被当成成功继续完成。

| 阶段 | 执行位置 | 主要工作 | 主要输出 |
|---|---|---|---|
| 1. prepare | `infer_hik.py` | 时间戳、抽帧、YOLO 人体框、主要人物选择 | `camera/images`、`tracks.json`、`input.json` |
| 2. infer | `infer_hik.py` | 加载权重，逐样本运行 SAM 3D Body | `camera/frames`、`camera_sequence.npz` |
| 3. world | `fixed_camera_world.py` | 一个固定变换应用到全部帧 | `world/world_joints.npz`、`gauge.json` |
| 4. raw G1 | `world_to_g1.py` | 原始世界人体转换为 G1 | `g1_motion.npz` |
| 5. temporal | `temporal_root.py` | 评估根平移滤波候选 | `world_temporal`、`temporal_report.json` |
| 6. temporal G1 | `world_to_g1.py` | 为通过筛选的降抖人体再做重定向 | `g1_motion_temporal.npz` |
| 7. render | `render_comparison.py` | 同步时间、绘制骨架和机器人、编码视频 | `comparison.mp4` 和渲染报告 |

当没有合格的降抖候选时，流水线可保留原始人体/G1用于展示。`comparison.mp4` 这个名字本身不能证明一定做过平滑，应看字幕和 `temporal_report.json`。

默认运行一次并不同时生成 `comparison_raw.mp4`。已有交付目录中两种视频都在，是因为当时额外生成了原始对照版本。需要 raw 视频时可用 `--raw-only` 或单独调用渲染脚本。

质量审计、背景静止检查和逐帧视频解码验证也不全由入口自动执行。手册和交付报告里提到的检查，是当时另外执行的验证，不能把任意一次新运行自动称为已经通过全部审计。

### 8.1 主要人物怎样选择

当前 `select_box()` 初始倾向于面积与检测分数较大的框，之后依据框中心移动、大小变化和分数保持连续。它不是带人物身份识别能力的全场多人跟踪系统。

多人交叉、主体离场、另一个人走近镜头时，目标可能改变。因此准备场景的第一段视频只做了检测审查，没有作为可信运动序列交付。

### 8.2 body 模式是否只有身体没有手

现有海康入口明确使用 `inference_type='body'`。身体解码器仍输出完整的 MHR70 人体关键点，其中包含手部点，但没有额外裁剪左右手并进行专门细化。对走路的 G1 重定向，当前主要依赖髋、膝、踝、肩、肘、腕等核心关节点。

<a id="s09"></a>

## 9. 读懂 sam_3d_body_estimator.py

这个文件像模型的“接待与整理窗口”：把外部图像变成网络能接收的格式，再把复杂输出整理成每个人一份字典。

### 9.1 构造函数 __init__

它接收已加载的模型和配置，以及可选人体检测器、分割器、视场估计器。构造函数不负责从互联网下载权重。

`self.transform` 构成人体裁剪流程：从框得到中心和尺寸，按模型输入大小做仿射变换，再把图像转为 Tensor。`self.transform_hand` 为手部裁剪使用另一组 padding 设置。

`self.faces` 是人体表面的三角面索引。它只说明哪些顶点连接在一起，真正的顶点位置来自每次推理。

### 9.2 process_one_image 的参数

| 参数 | 应传什么 | 初学者最容易弄错的地方 |
|---|---|---|
| `img` | 路径字符串或 RGB 图像数组 | OpenCV 读到的通常是 BGR，数组不能直接当 RGB |
| `bboxes` | `[N,4]` 原图像素框，xyxy | 不是中心点加宽高，也不是 0～1 坐标 |
| `masks` | 与 N 个框对应的原图掩码 | 不是裁剪后的图；概率图需先合理处理 |
| `cam_int` | 当前实现需 Tensor `[1,3,3]` | 类型注解写了 NumPy，但代码实际调用 `.to()` |
| `bbox_thr` | 内置检测器的框分数门限 | 显式提供 bboxes 后不会再用它筛框 |
| `use_mask` | 是否请求可选分割器 | 没装分割组件不会自动提供真实分割 |
| `inference_type` | `body`、`full`、`hand` | 三种分支的返回结构和可用程度不同 |

现有海康入口在外部用 YOLO 检测，并把框显式传入；所以修改 Estimator 的 `bbox_thr` 不会改变当前海康人物检测。要调整海康检测，应看 `infer_hik.py` 中 `model.predict(... conf=.25 ...)`。

### 9.3 每张图片经过的步骤

1. 清理上一张图的交互缓存，关闭推理梯度记录。
2. 读取图片并确认 RGB/BGR 约定。
3. 按“显式框 → 检测器 → 整图作为一个框”的优先级取得人体框。
4. 若没有框，返回空列表，不强行生成一个人体。
5. 选择显式掩码、分割器掩码或无掩码。
6. 为每个人裁图，组装 `[1,N,3,Hin,Win]` 的批次，并迁移到 CUDA。
7. 按“显式 K → 视场估计器 → 默认先验 K”设置内参。
8. 调用核心模型的 `run_inference()`。
9. 将 Tensor 从 GPU 移到 CPU，转成 NumPy，逐人返回字段。

### 9.4 最重要的返回字段

| 字段 | 含义 |
|---|---|
| `pred_keypoints_3d` | 70 个三维关键点，尚未加相机平移 |
| `pred_cam_t` | 应加到上述点的整体平移，得到相机中的人体位置 |
| `pred_keypoints_2d` | 对三维结果进行的原图像素投影 |
| `pred_vertices` | 人体表面网格顶点，同样尚未加相机平移 |
| `global_rot` | MHR 的整体欧拉角表示，不是世界外参 |
| `body_pose_params` | MHR 身体参数，不是 G1 关节角 |
| `hand_pose_params` | MHR 左右手参数 |
| `pred_joint_coords` | MHR 内部骨架关节点，不等于 MHR70 的编号表 |
| `pred_global_rots` | MHR 内部骨架旋转，轴约定也需单独处理 |

相机点的关键计算是：

```python
camera_joints = prediction["pred_keypoints_3d"] + prediction["pred_cam_t"][None, :]
```

上面是一行概念示例，`prediction` 必须来自一次成功的人体推理。海康入口已经做了这次相加，因此读取 `camera_sequence.npz` 时不能再加一次。

### 9.5 阅读代码时要知道的现有接口限制

`hand` 核心分支返回 `mhr_hand`，而当前 Estimator 后处理统一读取 `mhr`，不能把它当作已经完整支持的独立手部接口。`full` 融合后会将 `pred_pose_raw` 清零使其失效，不能再把这个字段当成最终身体姿态。

这些地方在中文注释中标出了实际行为，本次仅补充说明，没有替你悄悄修复分支。

<a id="s10"></a>

## 10. 读懂 sam3d_body.py

文件较长，第一次不要从第一行逐字读完。先搜索 `run_inference`，理解整体分支，再顺着下面的调用链阅读。

```text
SAM3DBodyEstimator.process_one_image
└─ SAM3DBody.run_inference
   ├─ body → forward_step(body)
   │         └─ forward_pose_branch
   │            ├─ data_preprocess
   │            ├─ get_ray_condition
   │            ├─ backbone
   │            └─ forward_decoder
   │               ├─ Transformer 逐层更新 token
   │               ├─ MHR 头：姿态、形状、网格、关节点
   │               └─ 相机头：平移和二维投影
   └─ full → 先运行 body，再细化左右手并融合
```

### 10.1 初始化建立了哪些部件

| 部件 | 简单理解 |
|---|---|
| backbone | 从人体裁剪图片提取视觉特征，当前权重使用 DINOv3 |
| decoder | 从图像特征和提示逐步生成更合适的姿态表示 |
| token | 一个数值向量，承载姿态、关节点身份或提示等信息 |
| prompt encoder | 把“这个关节点应该在这里”或掩码变成模型可用的特征 |
| MHR head | 将网络向量解释为人体模型参数，再生成几何 |
| camera head | 将相机相关预测转换为人体相对相机的平移 |
| ray condition | 告诉模型一个裁剪像素在原相机中对应什么方向 |

不要随意改变这些模块的维度。权重与网络结构必须匹配，改了维度通常不是修改一个数字就能继续加载原模型。

### 10.2 一个裁剪批次怎样变成姿态

`forward_pose_branch()` 把图像数 B 和每图人数 N 展开成 P=B×N 个裁剪样本，做像素归一化，然后交给主干。主干返回特征图 `[P,C,h,w]`，不是直接返回人体坐标。

接着构造框位置与焦距有关的条件，以及必要的提示。第一次没有人工点击时仍会传一个 `label=-2` 的占位提示。这个标签不表示人体“检测失败”，而是提示输入为空。

解码器通常以一个姿态 token 开始，再带上提示和关节点 token。经过注意力层后，预测头输出人体和相机参数。程序可以在中间层将预测点投影回图片，在相应位置采样视觉特征，再提供给下一层。

### 10.3 几个关键函数做什么

| 函数 | 读代码时抓住的主线 |
|---|---|
| `_initialze_model` | 按配置创建模型；拼写沿用上游，不要为了改英文自行重命名 |
| `_get_decoder_condition` | 框中心相对主点的偏移、框大小，都除以焦距 |
| `forward_decoder` | 身体 token 与图像特征交互，回归身体结果 |
| `forward_decoder_hand` | 独立手部分支，保持统一输出结构 |
| `_full_to_crop` | 原图像素转为裁剪图的归一化位置 |
| `camera_project` | 加相机平移，再用内参投影 |
| `get_ray_condition` | 构造像素射线的横纵方向分量 |
| `forward_step` | 选择身体或手部批次索引 |
| `run_keypoint_prompt` | 同一图像上利用新提示细化结果，复用主干特征 |
| `_get_hand_box` | 将手部预测框转换回原图像素 |
| `keypoint_token_update_fn` | 根据当前二维位置采样视觉特征，更新 token |
| `keypoint3d_token_update_fn` | 使用骨盆相对三维结构更新位置编码 |

### 10.4 full 模式为什么更复杂

身体初估提供全身结果和手框。左手图片水平翻转后交给手部分支，右手直接裁剪；之后将左手参数和框恢复到相应坐标约定。

程序检查手腕旋转差、手框大小、投影是否在框内，以及身体与手部分支的手腕二维位置是否一致。通过筛选的手腕和手肘被组合成提示，再次调整身体，最后融合手指参数与手腕连接并重新计算网格。

此处的手腕 IK 是人体模型内部的连接调整，不是 G1 的重定向 IK。`full` 也不会自动获得固定世界坐标。

当前手部分支保留 70 点的输出结构，但将右手 21～41 之外的三维关键点置零。统一的数组长度不代表手部分支恢复了有效的全身结构。

### 10.5 为什么要区分三种编号

MHR70 关键点编号用于观测和显示；MHR 内部骨骼编号用于骨骼旋转；`body_pose` 参数槽位用于存放模型参数。同一个数字例如 41，出现在这三处时可能表示完全不同的东西。

修改时优先根据名称查找，不要因为看到“手腕 41”就把其他数组的第 41 项直接当成同一个量。

<a id="s11"></a>

## 11. 坐标系、单位和时间：最值得读懂的一章

### 11.1 四种位置不要混用

| 表示 | 原点/范围 | 单位 | 示例 |
|---|---|---|---|
| 原图二维像素 | 图像左上角，x 向右、y 向下 | 像素 | `(812,620)` |
| 裁剪归一化二维 | 人体或手部裁剪中的相对位置 | 无量纲 | `[-0.5,0.5]` 或提示的 `[0,1]` |
| 相机三维 | 光学相机原点，x 右、y 下、z 前 | 当前为模型名义米 | `joints_camera` |
| 固定世界三维 | 本片固定原点，右手系、z 向上 | 当前为模型名义米 | `world_joints` |

一个人在图像里向右移动，可能是世界横向运动，也可能包含透视、身体转身或深度变化。不能把图像像素位移直接当成地面上的米数。

### 11.2 内参 K 怎样把三维点投影到图片

```text
K = [ fx   0   cx ]
    [  0  fy   cy ]
    [  0   0    1 ]

u = fx × X/Z + cx
v = fy × Y/Z + cy
```

X、Y、Z 必须是已经加上 `pred_cam_t` 的相机点，Z 应在相机前方。这个公式没有包含镜头畸变，实际标定流程要同时考虑畸变和去畸变后的新 K。

当前未给实测内参时，默认 `fx=fy=sqrt(W²+H²)`，主点为图像中心。对于 1624×1240 图像，默认焦距约 2043.28 像素。这个数来自先验公式，不是镜头标称焦距，也不是棋盘格标定结果。

文件中的 `keypoints_2d_calibrated` 名称也不能单独证明真实标定。应同时读取 `metadata`、`K` 来源和尺度状态。

### 11.3 相机点怎样转成世界点

```text
p_world = R_world_from_camera × (s × p_camera) + t_world_from_camera
```

`R` 是 3×3 旋转矩阵，`t` 是 3 维平移，`s` 是整段共用的尺度。它们在同一段固定相机视频里保持不变。代码将 R 和 t 放进 4×4 的 `T_world_from_camera`。

尺度乘的是整个相机点，包括它相对于相机的距离，不能只把人体骨架变大而保持根轨迹不动。反向的“世界到相机”矩阵也不能直接填进这个字段。

### 11.4 模型米与真实米

模型输出遵循人体尺寸先验，可以用名义“米”组织几何；但如果没有独立的尺度验证，数字 `1.0` 不代表已经证明现实中正好移动一米。

用人体身高做尺度校正可以增加约束，但要考虑姿态、截断和形状估计误差。仅将 `--metric-scale` 设为一个数，只表明你提供了一个常量，程序不会自动证明它正确。

### 11.5 时间戳比帧序号更重要

`frame_indices` 保存原视频帧号，`timestamps_s` 保存原视频对应时间。主视频约 60 FPS、stride=4 时，数组相邻样本通常相隔约 0.0666 秒。

计算速度应使用真实时间差：位移除以 `t[i+1]-t[i]`。缺测或长间隔不能直接跨过去算作连续一步，更不能先把无效帧删除后假定剩下全是等间隔。

<a id="s12"></a>

## 12. 固定世界坐标与降抖怎样建立

### 12.1 没有标定文件时

`infer_ground_gauge()` 用肩部中点到髋部中点的方向估计身体上方向，再选取较低的足部候选点拟合一个固定参考平面。条件不足时会使用身体上方向与低足高度的退化方案。

世界 +Z 对应估计平面法线；世界 +X 通常由相机向右方向在平面上的投影确定，特殊退化时换用相机前向投影；+Y 按右手系得到。原点是最前面最多 15 个有效样本的骨盆中位位置投影到参考平面上的点。

整个序列只用一个变换。它不逐帧抬高或降低地面，不将每个支撑脚锁死，也不把最后一个点拉回第一个点。

**这是一种展示用的固定参考坐标。** 平面来自人体估计，不能把平面拟合残差称为相机标定误差或人体真实位置误差。相机是否固定也需要外部证据，本模块本身没有运行视觉里程计。

### 12.2 raw 与 temporal 的区别

`world/world_joints.npz` 保存固定变换后的原始世界点。`world_temporal/world_joints.npz` 保存整体平移降抖候选，同时保留原始点和修正量。

```text
某一帧的每个关节新位置 = 该关节原位置 + 同一个三维平移修正量
```

因此同一帧的骨长和关节相对形状基本保持，但骨盆和足部的世界轨迹会变化。它没有专门优化关节角，也没有接触锁定。

程序评估多个滤波候选，依据加速度改善、位移与地面相关门限选择，不按“首尾更接近”来挑选。主交付段选中的是 3 Hz 二阶零相位低通；零相位处理需要前后帧，是离线处理，不能直接声称具备零延迟实时能力。

### 12.3 原图骨架为何仍用原始投影

temporal 文件保留 `keypoints_2d` 作为原模型在真实图片上的投影证据。显示左侧骨架时使用这些原始投影，右侧则可能使用降抖后的机器人。它们时间同步，但不能声称左侧每个像素点是降抖后世界关节重新投影的结果。

### 12.4 缺测怎样处理

无效人体保留 `valid=False` 和 NaN，原始时间轴不被压缩。时序处理只作用于支持的连续段，渲染不会跨过超过门限的长缺口伪造连续运动。

不同模块有各自门限：当前渲染默认最大插值间隔为 0.1 秒，G1 时序连接默认门限为 0.2 秒。它们用途不同，不要把两个参数认为是同一项配置。

<a id="s13"></a>

## 13. 人体动作怎样变成 G1 动作

### 13.1 为什么不能直接复制人的关节角

人的关节结构、骨长和活动范围与 G1 不同。SAM 3D 的身体参数也不是机器人电机角度。重定向需要先构造适合机器人尺寸的目标，再求机器人关节角。

当前使用宇树官方 G1 29 自由度模型。模型局部轴约定为 +X 向前、+Y 向左、+Z 向上。

### 13.2 当前求解过程

1. 从名称中找到左右髋、膝、踝、肩、肘、腕等核心关节。
2. 用整段有效人体数据估计固定腿长比例和足踝高度差。
3. 用左右髋连线的水平分量确定根朝向，并采用重力直立先验。
4. 保留人体骨盆的世界 XY，按 G1 尺寸换算根高度 Z。
5. 保留人体肢段方向，按机器人自己的骨长构造手、脚和关节目标。
6. 用雅可比与阻尼最小二乘迭代调整 29 个关节，执行关节限位。
7. 加入较弱的中性姿态、连续帧和脚底穿地约束，输出误差与动作。

机器人根的 XY 与人体骨盆 XY 一致是这里的设计，不是算法得到的一份独立位置测量。机器人的 Z、身体比例和脚部形状不同，所以两者骨盆高度不应强制相同。

根朝向采用髋线和直立先验；骨盆俯仰、部分肢段扭转、手腕方向等并不能由少量位置点唯一恢复。当前根朝向没有另做平滑，相关跳变诊断保存在结果里。

### 13.3 qpos 和 qvel

| 数组 | 主段实际形状 | 每一行含义 |
|---|---|---|
| `qpos` | `[461,36]` | 根位置 3 + 根四元数 4 + 29 关节角 |
| `qvel` | `[461,35]` | 根线速度 3 + 根旋转速度 3 + 29 关节角速度 |
| `joint_names` | `[29]` | 29 个关节角对应的名字及顺序 |
| `xpos` | `[461,31,3]` | MuJoCo 各刚体的世界位置，包括 world body |
| `xquat` | `[461,31,4]` | 各刚体的世界旋转四元数 |

```text
qpos[0:3]   = x, y, z
qpos[3:7]   = qw, qx, qy, qz    （wxyz）
qpos[7:36]  = 29 个关节角       （弧度，顺序看 joint_names）
```

四元数是描述三维旋转的 4 个数，不是四个电机，也不是四个欧拉角。`qvel` 使用 MuJoCo 的速度空间，不应把 qpos 的四元数四项直接相减得到角速度；当前程序调用 `mj_differentiatePos` 来计算。

弧度转角度使用 `角度 = 弧度 × 180 / π`。机器人数据文件保留弧度，只有给人阅读时才按需转换。

### 13.4 怎样解释质量数字

IK 残差表示机器人部位离“为它构造的目标”有多远，不是人体位置与真实世界真值的距离。关节没有超限也不能推出动作一定能站稳。

`foot_near_floor` 只表示足部接近参考地面，不等于实际接触。接触诊断脚本还根据足部速度和连续性寻找候选支撑段；这些仍是运动学推断，不是脚底传感器测量。

<a id="s14"></a>

## 14. 输出目录里的文件怎样看

以下以 `outputs\Video_20260922162126269` 为例。

| 文件 | 用途 | 打开方式 |
|---|---|---|
| `camera/input.json` | 视频来源、尺寸、抽样参数、K 来源 | VS Code 文本查看 |
| `camera/tracks.json` | 每个抽样帧的人体框和有效性 | VS Code；检查是否选错人 |
| `camera/images/*.jpg` | 抽取的实际源图片 | 图片查看器 |
| `camera/frames/*.npz` | 单帧人体推理缓存 | Python/NumPy |
| `camera/camera_sequence.npz` | 相机坐标人体序列 | Python/NumPy |
| `camera/status.json` | 推理进度或完成状态 | VS Code |
| `camera/inference_provenance.json` | checkpoint、输入和运行来源 | VS Code |
| `camera/cache_contract.json` | 新版运行的缓存一致性约定 | 旧交付结果可能尚无此文件 |
| `world/world_joints.npz` | 原始固定世界人体 | Python/NumPy |
| `world/gauge.json` | 世界坐标定义、尺度和变换 | 首先查看此文件解释单位 |
| `world/closure_metrics.json` | 首尾窗口与轨迹统计 | 不应自动称为回环真值误差 |
| `world_temporal/temporal_report.json` | 滤波候选、门限与选择结果 | VS Code |
| `g1_motion.npz` | raw 人体对应的 G1 | Python/NumPy |
| `g1_motion_temporal.npz` | temporal 人体对应的 G1 | Python/NumPy |
| `g1_motion*.json` | 重定向来源与主要质量指标 | VS Code |
| `g1_motion*.contact_candidates.npz` | 单独审计生成的接触/筛查字段 | 新运行不会保证自动生成 |
| `comparison.render.json` | 视频使用了哪些输入、视角和参数 | VS Code |
| `comparison.validation.json` | 单独执行的视频验证记录 | 不要把旧记录套在新视频上 |
| `quality/` | 已交付的世界轨迹与端点审查材料 | 图片、Markdown、JSON |

### 14.1 人体 NPZ 最常用的字段

| 字段 | 主段形状 | 含义 |
|---|---|---|
| `joints_camera` | `[461,70,3]` | 已经加过相机平移的相机坐标 |
| `world_joints` / `joints_world` | `[461,70,3]` | 世界关节点，两个名称为别名 |
| `joint_names` | `[70]` | 关键点名称 |
| `pelvis_world` | `[461,3]` | 左右髋中点 |
| `keypoints_2d` | `[461,70,2]` | 源图片中的原始投影 |
| `valid` | `[461]` | 每个时间样本是否有效 |
| `frame_indices` | `[461]` | 对应原视频帧号 |
| `timestamps_s` | `[461]` | 视频时间，秒 |
| `T_world_from_camera` | `[4,4]` | 固定世界变换 |
| `constant_body_scale` | 标量 | 应用于相机点的固定尺度 |
| `metadata` | 标量字符串 | 内含 JSON，需要进一步解析 |

当前源码导出的 camera 文件还可包含检测框和检测分数，而既有交付文件可能没有这些新增字段。写读取程序时先用 `data.files` 看实际内容，不要仅凭文件名假设版本完全一致。

temporal 版本额外保存 `pelvis_world_raw`、`joints_world_raw`、`root_translation_correction_world_m` 和 `temporal_root_method` 等字段，用于追溯平滑前后变化。

### 14.2 如何判断文件来自哪里

文件里记录的 SHA256 是内容指纹，帮助判断文件有没有改变，不表示算法精度。`inference_provenance.json` 记录权重来源；`gauge.json` 记录世界定义；重定向 JSON 记录人体来源；`render.json` 记录渲染使用的输入。按这条链追溯，才能知道某个 MP4 对应的是哪版数据。

<a id="s15"></a>

## 15. 零基础读取动作数据

NPZ 不能像普通文本一样阅读。下面两个完整示例只读取文件、打印内容，不重新推理。

### 15.1 读取人体数据

在 `C:\YXW\视觉\hik_g1` 中新建 `read_human_example.py`，粘贴以下内容并保存：

```python
from pathlib import Path
import json
import numpy as np

path = Path(r"C:\YXW\视觉\hik_g1\outputs\Video_20260922162126269\world\world_joints.npz")
with np.load(path, allow_pickle=False) as data:
    print("字段：", data.files)
    joints = data["world_joints"]
    valid = data["valid"].astype(bool)
    times = data["timestamps_s"]
    names = [str(name).replace("-", "_").lower() for name in data["joint_names"]]
    metadata = json.loads(data["metadata"].item())

    ids = np.flatnonzero(valid)
    if len(ids) == 0:
        raise RuntimeError("文件中没有有效姿态")
    i = int(ids[0])
    left = names.index("left_hip")
    right = names.index("right_hip")
    pelvis = (joints[i, left] + joints[i, right]) / 2

    print("形状：", joints.shape)
    print("有效样本：", int(valid.sum()), "/", len(valid))
    print("第一个有效时间：", float(times[i]), "秒")
    print("对应源帧：", int(data["frame_indices"][i]))
    print("骨盆xyz：", pelvis)
    print("尺度状态：", metadata.get("metric_scale_status", "unknown"))
```

执行：

```powershell
Set-Location -LiteralPath 'C:\YXW\视觉\hik_g1'
.\.venv\Scripts\python.exe read_human_example.py
```

`r"..."` 表示原始字符串，适合 Windows 路径；`with np.load(...)` 在读取结束后关闭压缩文件；`names.index(...)` 通过关节名称查编号；`valid` 避免拿缺测帧做计算。当前人体文件实际使用 `left-hip` 这样的连字符名称，示例先统一成 `left_hip` 后再查询，和重定向入口的处理一致。

### 15.2 读取 G1 关节角

同样新建 `read_g1_example.py`：

```python
from pathlib import Path
import numpy as np

path = Path(r"C:\YXW\视觉\hik_g1\outputs\Video_20260922162126269\g1_motion_temporal.npz")
with np.load(path, allow_pickle=False) as data:
    ids = np.flatnonzero(data["valid"])
    if len(ids) == 0:
        raise RuntimeError("文件中没有有效机器人姿态")
    i = int(ids[0])
    q = data["qpos"][i]
    print("qpos / qvel：", data["qpos"].shape, data["qvel"].shape)
    print("根位置xyz：", q[:3])
    print("根四元数wxyz：", q[3:7])
    for name, angle in zip(data["joint_names"], q[7:]):
        print(f"{name}: {angle:.4f} rad = {np.degrees(angle):.2f} deg")
```

执行：

```powershell
Set-Location -LiteralPath 'C:\YXW\视觉\hik_g1'
.\.venv\Scripts\python.exe read_g1_example.py
```

打印出的关节角是动作参考，不是力矩、PID 参数或机器人通信指令。不要把这段读取代码理解为已经控制了真实 G1。

<a id="s16"></a>

## 16. 想修改程序，从哪里开始

### 16.1 先判断修改属于哪一层

| 你的目标 | 优先修改位置 | 需要重新计算到哪一步 |
|---|---|---|
| 改标题、颜色、镜头、视频分辨率 | `visualization/render_comparison.py` | 只重新渲染 |
| 改 G1 目标权重或 IK 次数 | `retarget/world_to_g1.py` | 重定向 + 渲染 |
| 改根平移滤波 | `reconstruction/temporal_root.py` | 降抖 + 对应 G1 + 渲染 |
| 改世界外参或统一尺度 | `fixed_camera_world.py` 或标定文件 | 世界 + 后续步骤 |
| 改相机 K、检测框、选人、采样间隔 | `infer_hik.py` / 参数 | 新目录准备与人体推理，再全部后续 |
| 改 SAM 3D 网络或推理算法 | 两个核心文件及相关模块 | 新缓存的人体推理，再全部后续 |
| 改 `.ckpt` 权重 | 模型加载路径与配置 | 新缓存的人体推理，再全部后续 |
| 只改 `#` 注释 | 对应源码 | 不需要重新推理 |

首次修改建议从标题或视角开始。网络维度、旋转表示、MHR 参数索引等需要理解训练与权重对应关系后再动。

### 16.2 第一个修改练习：只重渲染前 5 秒

下面复用真实主段的已有数据，创建新的独立视频，不改原 `comparison.mp4`。复制整个命令块，PowerShell 反引号必须是行尾最后一个字符，后面不能加空格。

```powershell
Set-Location -LiteralPath 'C:\YXW\视觉\hik_g1'
$taskClip = 'outputs\Video_20260922162126269'
$taskVideo = 'C:\Users\liuq8\MVS\Data\MV-CS020-10GC(DB2677346)\Video_20260922162126269.avi'
$taskPreview = 'outputs\manual_preview_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.mp4'
.\.venv\Scripts\python.exe visualization\render_comparison.py `
  --human "$taskClip\world_temporal\world_joints.npz" `
  --robot "$taskClip\g1_motion_temporal.npz" `
  --model assets\unitree_g1\scene.xml `
  --source-video $taskVideo `
  --match-source-camera `
  --quality-label 'ROOT-SMOOTHED / MODEL-SCALE / UNCALIBRATED' `
  --pose-label 'source 2D / raw SAM 3D projection' `
  --title 'My first SAM 3D and G1 preview' `
  --start-s 0 --duration-s 5 --fps 30 `
  --output $taskPreview
```

先用英文标题避免当前渲染器所选字体缺少中文字形。渲染参数中的尺寸要求是：宽度至少 1400、高度至少 900，且两者为偶数；默认 1600×1000。

要改成全景视角，删除 `--match-source-camera`，加入例如 `--azimuth 100 --elevation -18`。这只是改变观看方向，不改变数据的世界坐标。

### 16.3 第二个练习：只重新做 G1 重定向

```powershell
Set-Location -LiteralPath 'C:\YXW\视觉\hik_g1'
$taskRobot = 'outputs\g1_ik_trial_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.npz'
.\.venv\Scripts\python.exe retarget\world_to_g1.py outputs\Video_20260922162126269\world\world_joints.npz $taskRobot --iterations 25
```

这读取 raw 世界人体，生成新的机器人文件与 JSON。默认迭代次数是 20，增加次数只意味着求解预算增加，不保证真实姿态更准确、支撑更稳定或训练效果更好。

若渲染这个新机器人，人体输入应使用同一 raw 世界文件，避免把 raw 机器人与 temporal 人体误配。

### 16.4 如何只重跑后半段

`--skip-infer` 只跳过 SAM 3D 推理，**并不自动跳过 prepare**。当你确定准备数据和相机序列完全一致，要复用它们时应同时指定 `--skip-prepare --skip-infer`。

较稳妥的实验方式是新建输出目录，只复制必要的 `camera_sequence.npz` 后执行下游：

```powershell
Set-Location -LiteralPath 'C:\YXW\视觉\hik_g1'
$taskSource = 'outputs\Video_20260922162126269\camera\camera_sequence.npz'
$taskOutput = Join-Path (Get-Location) ('outputs\downstream_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Path (Join-Path $taskOutput 'camera') | Out-Null
Copy-Item -LiteralPath $taskSource -Destination (Join-Path $taskOutput 'camera\camera_sequence.npz')
.\.venv\Scripts\python.exe run_pipeline.py --video 'C:\Users\liuq8\MVS\Data\MV-CS020-10GC(DB2677346)\Video_20260922162126269.avi' --output $taskOutput --skip-prepare --skip-infer
```

这个目录用于下游实验，不是包含全部原始推理材料的完整交付包。保留源 camera 目录的来源记录，便于追溯。

### 16.5 修改后的最低验证

先检查语法；下面的命令不加载模型，但会生成或更新相应的 `__pycache__`：

```powershell
Set-Location -LiteralPath 'C:\YXW\视觉\hik_g1'
.\.venv\Scripts\python.exe -m py_compile infer_hik.py run_pipeline.py
```

再在新目录做小样本运行，确认图像和骨架同帧、人物身份正确、数据没有 NaN 被误当有效、机器人方向没有突跳，最后才扩展到完整视频。

`verify_delivery.py` 针对既有两段交付目录、特定质量状态和视频校验记录编写，不是一个自动适配任意新实验的万能测试入口。新算法应选择相关检查，不要只运行旧交付验证就宣布新方案通过。

<a id="s17"></a>

## 17. 想得到更可靠的真实世界坐标，需要什么

需要分别解决内参/畸变、相机到地面的外参、尺度以及独立误差验证。它们不能由一张“看起来很平”的渲染图代替。

### 17.1 内参和畸变

应在实际分辨率、镜头焦距和对焦设置下采集标定板图像，得到 K 和畸变参数。若视频经过裁剪或缩放，需要相应调整 K。

当前 `--intrinsics` 接收 JSON，其核心字段如下：

| 字段 | 类型和含义 |
|---|---|
| `K` | 3×3 数字数组，原图或去畸变后图像对应的内参 |
| `distortion` | 可选数字数组；非零值会被当前入口拒绝 |

当前入口不会替你完成去畸变。应先按实际标定将视频去畸变，再传入与处理后图像相匹配的新 K。不要把非零畸变系数简单改成零冒充已经处理。

### 17.2 外参和世界轴

`--world-calibration` 接收一个 JSON，支持下面两种形式之一：

| 形式 | 必需字段 |
|---|---|
| 4×4 形式 | `T_world_from_camera` |
| 旋转和平移分开 | `R_world_from_camera`（3×3）与 `t_world_from_camera_m`（3维） |

可选字段包括 `world_up`（必须为 `"z"`）、`world_frame_id`、`provenance`、`metric_scale` 和 `scale_provenance`。R 必须是合法旋转，4×4 矩阵最后一行为 `[0,0,0,1]`，平移使用世界单位。

这里刻意不填写虚构的标定数值。只有字段名正确而数值随意，并不能构成标定。OpenCV 标定方法经常输出“世界/标定板到相机”的姿态，传入本程序前要确认是否需要求逆。

### 17.3 尺度的优先级

没有提供尺度时使用 1.0，状态为 `model_prior_only`。通过 `--metric-scale` 或 world JSON 的 `metric_scale` 提供正数后，状态变为 `user_supplied_constant_scale`，不自动变成“测量已验证”。

命令行与 JSON 同时提供尺度时，两者必须一致，否则程序报错；这不是命令行悄悄覆盖 JSON 的规则。只想提供尺度、不提供外参时可用命令行 `--metric-scale`；单独只有尺度的 world JSON 不能代替外参文件。

### 17.4 准备好真实文件后怎样运行

以下命令以你已经创建了真实标定文件为前提，文件名不是当前已经存在的标定结果：

```powershell
Set-Location -LiteralPath 'C:\YXW\视觉\hik_g1'
.\.venv\Scripts\python.exe run_pipeline.py --video 'C:\你的数据\去畸变视频.avi' --output 'outputs\calibrated_trial' --intrinsics 'C:\你的标定\intrinsics.json' --world-calibration 'C:\你的标定\world.json'
```

改变 K 会改变人体推理条件，应从新输出目录重新推理。只改变世界外参或尺度时，可以复用对应输入不变的相机序列，重新计算世界坐标及后续结果。

`world_frame_id` 是坐标定义的标识。把两段视频的字符串改成一样，不会自动让它们共享真实房间原点；必须有实际一致的外参、轴、尺度与场景参照。

<a id="s18"></a>

## 18. 怎样判断是否实现了世界坐标回环

这里的“回环”指人在场景里绕行后回到同一个物理位置，重建轨迹也应返回相应位置。当前固定相机方案没有运行相机 SLAM 的闭环优化，两者不要混为一谈。

### 18.1 先建立可核对的起终点

在地面做明确标记，固定相机，完整拍摄脚部。开始和结束时站在同一标记、采用相近站姿与朝向，并各停留一段时间。更严谨时同时设置几个途中测量点，检查整条路径，而不只看起终点。

即便双脚回到同一位置，弯腰、跨步宽度或身体倾斜不同，骨盆也可能不在同一点。所以应明确你验收的是骨盆、足部还是地面标记之间的关系。

### 18.2 两种端点指标

视频里直接首末有效样本的距离，与 `closure_metrics.json` 中首尾窗口中位位置距离不是同一个统计量。后者默认使用首尾各约 0.25 秒的窗口，减少单帧抖动，但没有挑选“最靠近的两个时刻”。

两种统计都只能描述这段轨迹。如果现实里没有回到同一地点，它们不能被称为回环误差。若人为把最后位置减掉一个偏移强行闭合，则应另外记录修正量，也不能用修正后的零误差证明原始估计准确。

### 18.3 当前主视频的结论

主段 raw 首末样本距离约 1.676 个模型米；首尾窗口距离约 1.699 个模型米。原视频首尾实际站位不同，结束时更靠近白板，因此这段数据没有提供可直接验收的同起终点真值。

降抖后窗口距离约 1.694 个模型米，变化仅说明滤波改变了少量位置，不能据此宣称回环成功。所有交付结果都没有施加强制闭合。

### 18.4 建议保存的回环验收记录

记录地面起终点标记、实际采集时间、K 与畸变来源、外参与尺度来源、首尾选取规则、姿态差异，以及 raw/temporal 两套误差。误差阈值应根据机器人训练需求和测量精度预先确定，而不是看完结果再决定。

<a id="s19"></a>

## 19. 从动作参考到机器人走路，还缺哪些步骤

让 MuJoCo 按 `qpos` 摆出姿势并渲染，和让机器人在重力、接触和电机约束下走出这些动作，是不同任务。当前视频主要做前者。

```text
已有的人体与 G1 运动学参考
    ↓ 数据清洗、标定与接触检查
可用的动作片段和训练数据格式
    ↓ 设计跟踪任务、状态/动作、奖励与终止条件
仿真中的控制策略训练
    ↓ 平衡、力矩、接触、扰动、泛化等测试
通过验收的策略
    ↓ 另行完成硬件接口与部署验证
真实机器人执行
```

当前项目没有提供完成训练的策略网络、行走控制器或真实机器人部署流程。NPZ 可以作为后续参考数据的起点，但仍需适配你实际采用的训练框架。

### 19.1 先检查动作数据

检查缺测、人物切换、画面边缘截断、关节速度尖峰、朝向跳变、足部穿地、支撑期滑动与自碰撞。时序重采样时使用时间戳，四元数插值要考虑旋转空间，不能对欧拉角跨越 ±180° 的情形随意线性平均。

接触候选来自高度与速度规则，不等于真实接触标签；“通过参考质量筛查”只说明通过已有规则，也不表示可以直接训练或部署。

### 19.2 为什么降抖不一定让训练更好

当前主段平滑降低了根轨迹的加速度尖峰，但在固定的原始支撑候选样本上，G1 足部速度 P95 反而从约 0.228 增至 0.317 个模型米/秒。整体平移滤波可能扰动原先相对稳定的支撑足。

所以后续应同时比较 raw 与 temporal 的接触、速度、动态跟踪表现，而不是只选择看起来最顺滑的视频。

### 19.3 当前能直接复用哪些材料

可复用 `qpos`、`qvel`、`joint_names`、`timestamps_s`、`valid`、人体与机器人根轨迹、模型 XML 和来源记录。还需要根据训练框架确认关节顺序、根坐标约定、采样频率、初始状态和控制模式。

训练是否成功应通过实际仿真中的站立、行走、转身、跌倒率、跟踪误差和约束指标判断，不能通过运动学视频单独判断。

<a id="s20"></a>

## 20. 常见问题与排错

先读报错最后一行，再回看导致错误的文件和行号。下面给出的是当前工程中有意义的检查方向。

| 现象或报错 | 常见原因 | 先做什么 |
|---|---|---|
| 找不到 `.venv\Scripts\python.exe` | 当前目录不对，或环境已移动 | 使用绝对 Python 路径，检查第 3 章 |
| `ModuleNotFoundError` | 用了错误的 Python，或依赖不全 | 先确认解释器，避免对系统 Python 随意 pip install |
| `CUDA available: False` | 当前 PyTorch/驱动/GPU 状态不匹配 | 确认使用项目环境、检查 GPU 状态 |
| `CUDA out of memory` | 同时跑多个模型、图片人数多或模式更重 | 停止多余推理进程、先用当前单人 body 流程 |
| 加大 stride 后仍单帧显存不足 | stride 只减少帧数，不降低单帧模型需求 | 查单帧分辨率、人数、模型及模式 |
| `ffmpeg not found` / 无 ffprobe | 终端 PATH 中没有视频工具 | 按第 6 章给当前终端加路径 |
| `Cached poses belong to different...` | 输入、K 或检测记录与缓存不匹配 | 使用新输出目录，不绕过检查 |
| 改了模型代码但输出没变化 | 旧逐帧 NPZ 被复用 | 新建输出目录重新推理 |
| 没找到人体或有效帧很少 | 人太小、遮挡、截断或人物选择失败 | 查看抽帧图片和 tracks，再查检测配置 |
| 人体骨架明显错位 | RGB/BGR、框坐标、K、时间同步不一致 | 先核对单张图和同一个源帧号 |
| 没有世界坐标输出 | 前一步缺文件，或地面/上方向估计失败 | 看 camera 有效性、完整脚部和最后报错 |
| `hand` 模式后处理失败 | 当前 Estimator 未完整适配 mhr_hand | 先用已跑通的 body，不把模式名当兼容保证 |
| 机器人背朝反了或转身突跳 | 左右关节、髋线朝向或观测歧义 | 核对 joint_names、原图和 root_yaw 诊断 |
| 机器人穿地、滑脚 | 地面、尺度、人体误差或软约束权衡 | 看 raw/temporal、足部诊断，不先移动地面掩盖 |
| `G1 model/qpos mismatch` | 用错机器人自由度或模型版本 | 当前必须是 29 DOF、nq=36、nv=35 |
| 视频显示不出姿态 | 无效时间、输入时钟不匹配或缺少二维投影 | 查看 valid、时间戳、源帧号和 render.json |
| 改动后程序 SyntaxError | 括号、引号或缩进错误 | 保存后运行 py_compile，定位行号 |
| 原包 SHA256 校验报两个源码变化 | 已经主动添加中文注释 | 与 source_annotation_backups 对照，不覆盖旧清单冒充原件 |

### 20.1 为什么不能直接点击运行 sam3d_body.py

它主要定义模型类，没有独立的图像输入、权重加载和结果展示入口。只执行这个文件不等于完成推理。当前 Windows 海康任务应优先从 `run_pipeline.py` 或 `infer_hik.py` 进入。

官方 `demo.py` 是单图文件夹示例，另有检测器、权重、可选分割器和依赖要求。当前海康包装层还处理了本地 DINOv3 加载、中文路径等适配；直接照搬官方示例不一定等同于已跑通的本机流程。

### 20.2 为什么刚打开终端看不到人体

终端只显示文字。计算完成后打开 MP4，或另行实现交互查看器。当前项目未把所有运行步骤封装成 GUI，也没有自动启动海康相机采集。

### 20.3 为什么视频帧数和 NPZ 样本数不同

原视频约 60 FPS、人体抽样约 15 Hz、成片 30 FPS，它们本来就不同。左图根据实际推理帧重复显示，机器人在有效段内插值展示。应对齐 `timestamps_s` 和 `frame_indices`，不要直接用两个文件相同下标对应。

<a id="s21"></a>

## 21. 备份、版本和换电脑复现

### 21.1 当前注释修改怎样恢复

2026-09-23 已给两个核心文件添加中文注释。原始备份位于 `source_annotation_backups\20260923`，验证记录为 `annotation_verification.json`。检查确认非注释代码 token、语法树和原始代码行保持一致。

如要恢复，先另存自己的新修改，再从备份手动复制对应文件。不要在尚未确认修改内容时覆盖正在编辑的源码。

### 21.2 至少保留哪些内容

| 内容 | 为什么需要 |
|---|---|
| 原始视频与采集信息 | 保留真正的数据来源 |
| 实际运行的源码与参数 | 知道如何得到结果 |
| 权重与配置，及其来源/许可 | 神经网络不能只靠 `.py` 重建 |
| 相机内参、外参、尺度的测量记录 | 解释坐标和精度 |
| NPZ、metadata 与质量报告 | 便于重分析，不只剩视频 |
| G1 XML、mesh、来源记录 | 确保关节与模型一致 |
| 依赖版本与运行环境说明 | 方便在其他机器重建环境 |

### 21.3 结果 ZIP 包含什么

既有 `hikvision_g1_results.zip` 包含当时的适配代码、G1 资源、两段动作数据、视频与质量材料，不包含全部原始 AVI、SAM 3D 大权重、完整上游源码或 Python 环境。

它是此前生成的静态快照。后来添加的中文源码注释和本说明书，不会自动进入那个旧 ZIP；需要另行重新打包或单独传递本手册。

### 21.4 换电脑的大致顺序

准备 NVIDIA 驱动及匹配的 Python/PyTorch 环境，按实际项目依赖重建环境，放置已有授权的模型权重和 G1 资源，保持路径关系或修改配置，确认 ffmpeg/ffprobe，然后先跑小片段。

`requirements.lock.txt` 和 `environment_versions.json` 是已运行环境的记录。Windows、CUDA 版本、GPU 架构与某些原生库会影响安装，不能认为在另一台机器执行一次普通 `pip install` 就必然完全复现。不要把整份 `.venv` 复制过去当成可移植环境。

本手册描述本地实现，不替代上游 LICENSE 和模型访问条款。分享代码或权重前仍应查看相应资源附带的许可。

<a id="s22"></a>

## 22. 建议的学习练习与自检

### 22.1 按这五个练习入门

1. **只观看。** 打开主视频与原始版，指出骨盆轨迹、缺测提示和尺度标签各在哪里。
2. **只读取。** 运行第 15 章示例，打印第一个有效时间、人体骨盆与 G1 根位置。
3. **只渲染。** 用第 16 章命令重做 5 秒视频，改变标题或镜头，确认 NPZ 没有变化。
4. **小规模推理。** 在新目录执行 120 个抽样点的试跑，找到每一步的输出文件。
5. **有记录地修改。** 每次只改一个参数，保存原版、新版、运行命令与同设置的比较结果。

### 22.2 能回答这些问题，就读懂了主线

- 修改 `sam3d_body.py` 后为什么有时结果不变？——可能复用了旧人体缓存，模型逻辑变化要用新缓存验证。
- 为什么 raw 人体点还不能直接当世界点？——先确认是否已加相机平移，再应用固定世界变换。
- 为什么 G1 和人体 XY 一样不证明重建准确？——它们共享同一条输入根轨迹，重定向设计就保留了它。
- 为什么 30 FPS 视频不代表 30 Hz 的真实推理？——成片可能重复源帧并插值机器人。
- 为什么同一地面标记上的首尾骨盆仍可能不同？——身体姿态和站姿可能不同。
- 为什么 qpos 有 36 个数而 qvel 只有 35 个？——根朝向用四元数存 4 个数，但旋转速度只有 3 个分量。
- 为什么视频里机器人没倒，不代表能走？——当前是按运动学参考摆姿态和渲染，尚未完成动力学控制验收。

### 22.3 后续阅读材料

| 文档 | 内容 |
|---|---|
| [README_CN.md](../README_CN.md) | 工程简明入口与运行约定 |
| [RESULTS_CN.md](../RESULTS_CN.md) | 本次真实数据结果、质量数字与边界 |
| [世界重建审查](world_reconstruction_audit.md) | 固定世界定义与实际数据检查 |
| [单目焦距、深度和训练说明](monocular_focal_depth_and_training.md) | 尺度不确定性与后续训练问题 |
| [G1 重定向说明](../retarget/README.md) | 机器人输入输出与求解约定 |
| [渲染说明](../visualization/README.md) | 输入契约、相机视角、同步与成片 |
| [源码注释验证记录](../source_annotation_backups/20260923/annotation_verification.json) | 两个核心文件只增加注释的验证 |

本文依据这些本地材料和实际源码核对，命令参数与主段 NPZ 字段已检查。阅读手册时不需要下载外部网页；如果之后升级上游模型、改变目录或替换 G1 版本，应同步更新对应章节和验证记录。
