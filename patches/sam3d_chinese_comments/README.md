# SAM 3D 中文阅读注释补丁

这两份补丁保存本项目给 SAM 3D Body 添加的中文阅读注释，不复制模型权重，也不改变推理逻辑。补丁由 `source_annotation_backups/20260923` 的原始备份与兄弟目录中当前源码逐字节对比生成。`manifest.json` 记录每个原文件、目标文件和补丁的 SHA256；`validation.json` 记录实际应用验证。

| 补丁 | 上游仓库中的目标路径 |
| --- | --- |
| `sam_3d_body_estimator.patch` | `sam_3d_body/sam_3d_body_estimator.py` |
| `sam3d_body.patch` | `sam_3d_body/models/meta_arch/sam3d_body.py` |

上游版本：

- SAM 3D Body：`facebookresearch/sam-3d-body`，commit `b5c765a0d89d789985e186d396315e7590887b94`。
- DINOv3：`facebookresearch/dinov3`，commit `6876159a11b4df116f30f667f8c9888617df0751`。这两份补丁不修改 DINOv3；此版本记录用于恢复推理依赖。

以上 commit 来源于原复现包的版本记录。补丁能否应用，以原文件 SHA256 校验为准；不要把相同仓库名称当作相同源码版本。源码随附的 SAM License 已原样复制为 [LICENSE.sam3d](LICENSE.sam3d)，原文件摘要见 manifest；补丁不改变第三方源码的许可归属。

## 当前目录关系

海康入口仍从现有兄弟目录加载 SAM，Git 中的补丁不改变该路径。新机器应恢复为：

```text
workspace/
├─ hik_g1/
│  └─ patches/sam3d_chinese_comments/
└─ pose_world_sam3d_repro_20260918/
   └─ pose_project/sam3d/
      ├─ repo/                       # SAM 3D Body 源码
      ├─ dinov3_official/             # 固定版本 DINOv3 源码
      └─ checkpoints/user_dinov3_20260917/
         ├─ model_config.yaml
         ├─ model.ckpt
         └─ assets/mhr_model.pt
```

恢复指定版本的 SAM 和 DINOv3 源码后，再应用补丁。已有完整交付包的当前源码通常已经包含这些注释，无须重复应用。模型文件独立放置；补丁既不下载模型，也不改变模型路径。

## 在 PowerShell 中验证并应用

先进入 `hik_g1` 根目录，再执行以下步骤。需要已安装 Git；这些命令不会创建仓库或提交。源码目录可以是 Git checkout，也可以是普通解压目录。

```powershell
$taskProject = (Get-Location).Path
$taskPatchDir = Join-Path $taskProject 'patches\sam3d_chinese_comments'
$taskSamRepo = Join-Path (Split-Path $taskProject -Parent) 'pose_world_sam3d_repro_20260918\pose_project\sam3d\repo'
$taskManifest = Get-Content -LiteralPath (Join-Path $taskPatchDir 'manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json

if (-not (Test-Path -LiteralPath $taskSamRepo -PathType Container)) {
    throw 'SAM 源码目录不存在；请先恢复上面的目录结构。'
}

# 先检查两份原文件与补丁自身；任一不符都停止，不强行应用。
foreach ($taskEntry in $taskManifest.files) {
    $taskTarget = Join-Path $taskSamRepo $taskEntry.target_path
    $taskPatch = Join-Path $taskPatchDir $taskEntry.patch
    $taskTargetHash = (Get-FileHash -LiteralPath $taskTarget -Algorithm SHA256).Hash.ToLowerInvariant()
    $taskPatchHash = (Get-FileHash -LiteralPath $taskPatch -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($taskPatchHash -ne $taskEntry.patch_sha256) { throw "补丁摘要不符：$taskPatch" }
    if ($taskTargetHash -eq $taskEntry.annotated_sha256) {
        throw "文件已经带有当前中文注释，不要重复应用：$taskTarget"
    }
    if ($taskTargetHash -ne $taskEntry.original_sha256) {
        throw "原文件版本或换行格式不同，请先核对来源：$taskTarget"
    }
}

$taskPatchFiles = @($taskManifest.files | ForEach-Object { Join-Path $taskPatchDir $_.patch })
& git -c core.autocrlf=false -c core.eol=lf -C $taskSamRepo apply --check @taskPatchFiles
if ($LASTEXITCODE -ne 0) { throw 'git apply --check 失败；尚未写入源码。' }

& git -c core.autocrlf=false -c core.eol=lf -C $taskSamRepo apply @taskPatchFiles
if ($LASTEXITCODE -ne 0) { throw 'git apply 失败，请检查 Git 输出。' }

foreach ($taskEntry in $taskManifest.files) {
    $taskTarget = Join-Path $taskSamRepo $taskEntry.target_path
    $taskTargetHash = (Get-FileHash -LiteralPath $taskTarget -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($taskTargetHash -ne $taskEntry.annotated_sha256) { throw "目标摘要不符：$taskTarget" }
}
Write-Output '两份补丁已应用，目标 SHA256 全部匹配。'
```

SHA256 对换行也敏感。本次原文件和目标文件使用 LF；上面的 `-c` 仅对本次 Git 命令关闭换行转换，不改变全局配置。如果 Git checkout 已将原源码转换成 CRLF，应先确认转换设置与来源，恢复 manifest 对应的原文件字节，再应用，不能略过校验。本目录 `.gitattributes` 保证补丁和许可证在检出时保持原字节。

## 已验证的内容

生成时在临时目录复制了两份原始备份，运行 `git apply --check` 后实际应用补丁，并逐字节对照当前注释版本；之后执行反向检查和反向应用，验证恢复出的字节与原备份完全一致。临时目录已清理，兄弟目录源码未被写入。

同时比较了忽略 COMMENT/NL 的 Python token、忽略位置的 AST，并进行了不导入依赖的语法编译。验证不启动 CUDA、不加载模型、不运行人体推理，也不代表修复了源码中已经存在的行为限制。
