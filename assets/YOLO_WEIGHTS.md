# 人体框检测权重来源

本项目用 Ultralytics `yolo11n.pt` 检测人物框，人体三维姿态由 SAM 3D Body 推理。权重文件独立于源代码版本管理；此说明和 [YOLO_SOURCE.json](YOLO_SOURCE.json) 仅保存来源、默认路径、实际文件大小及 SHA256，不包含或复制权重。

- 官方资产仓库：<https://github.com/ultralytics/assets>
- 已记录的发布版本：<https://github.com/ultralytics/assets/releases/tag/v8.3.0>
- 对应下载地址：<https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt>
- 默认本地位置：`hik_g1/assets/yolo11n.pt`。

本次 SHA256 从当前本地权重重新计算，并与既有真实推理记录中的 `detector_sha256` 对照。此核查没有重新下载远程文件；线上资产是否变化不由本地记录保证。使用时应核对文件摘要，而非只核对文件名。

准备好权重后，在 `hik_g1` 根目录执行以下 PowerShell 校验：

```powershell
$taskYoloSource = Get-Content -LiteralPath '.\assets\YOLO_SOURCE.json' -Raw -Encoding UTF8 | ConvertFrom-Json
$taskYoloPath = Join-Path (Get-Location).Path $taskYoloSource.default_relative_path
$taskYoloHash = (Get-FileHash -LiteralPath $taskYoloPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($taskYoloHash -ne $taskYoloSource.sha256) { throw 'YOLO 权重 SHA256 与本项目已验证文件不同。' }
Write-Output 'YOLO 权重 SHA256 匹配。'
```

`infer_hik.py prepare --detector <文件路径>` 可以指定另一位置。`run_pipeline.py` 目前使用上述默认位置。此来源说明不赋予额外许可，也不把第三方权重改为本项目许可；获取和使用权重时保留并遵循其上游许可说明。
