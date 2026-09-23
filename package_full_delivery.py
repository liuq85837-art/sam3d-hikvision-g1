"""Create and verify a complete local source/model/Hikvision delivery ZIP64.

The portable folder layout keeps hik_g1 and the upstream reproduction package
next to each other. No installed environment or wheel cache is copied.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import time
import zipfile

ROOT = Path(__file__).resolve().parent
REPRO_NAME = 'pose_world_sam3d_repro_20260918'
SKIP_DIRS = {'.venv', 'venv', '__pycache__', '.git', '.pytest_cache', '.mypy_cache', 'node_modules'}
STORE_SUFFIXES = {'.npz', '.ckpt', '.pt', '.pth', '.avi', '.mp4', '.jpg', '.jpeg', '.png', '.gif', '.webp', '.zip', '.zst', '.gz'}
CHUNK = 8 * 1024 * 1024

VERIFY_SCRIPT = '''"""Check every delivered payload against MANIFEST.json; Python standard library only."""
from pathlib import Path, PurePosixPath
import hashlib, json, sys, time
root = Path(__file__).resolve().parent
manifest = json.loads((root / 'MANIFEST.json').read_text(encoding='utf-8'))
last = time.monotonic()
total = 0
for i, entry in enumerate(manifest['files'], 1):
    relative = PurePosixPath(entry['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise SystemExit('Unsafe manifest path: ' + entry['path'])
    path = root.joinpath(*relative.parts)
    if not path.is_file():
        raise SystemExit('Missing file: ' + entry['path'])
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
            size += len(chunk)
    if size != entry['bytes'] or digest.hexdigest() != entry['sha256']:
        raise SystemExit('File verification failed: ' + entry['path'])
    total += size
    if time.monotonic() - last > 15:
        print(f'Checked {i}/{len(manifest["files"])} files, {total / 1024**3:.2f} GiB', flush=True)
        last = time.monotonic()
print(f'PASS: {len(manifest["files"])} payload files; {total:,} bytes; all SHA256 match.')
print('This verifies file integrity, not calibration accuracy or robot dynamic feasibility.')
'''

LAUNCH_SCRIPT = r'''param(
    [Parameter(Mandatory=$true)][string]$PythonExe,
    [string]$Video,
    [int]$Stride = 4,
    [int]$MaxFrames = 120
)
$ErrorActionPreference = 'Stop'
$taskProject = Join-Path $PSScriptRoot 'hik_g1'
if (-not $Video) { $Video = Join-Path $PSScriptRoot 'raw_hikvision\Video_20260922162126269.avi' }
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) { throw 'Python executable not found. Supply a configured Python 3.11 environment.' }
if (-not (Test-Path -LiteralPath $Video -PathType Leaf)) { throw 'Source video not found.' }
if ($Stride -lt 1 -or $MaxFrames -lt 0) { throw 'Stride must be positive; MaxFrames must be zero (full video) or positive.' }
$taskOutput = Join-Path $taskProject ('outputs\package_run_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
$taskArgs = @((Join-Path $taskProject 'run_pipeline.py'), '--video', $Video, '--output', $taskOutput, '--stride', $Stride)
if ($MaxFrames -gt 0) { $taskArgs += @('--max-frames', $MaxFrames) }
Push-Location -LiteralPath $taskProject
try {
    & $PythonExe @taskArgs
    if ($LASTEXITCODE -ne 0) { throw "Pipeline failed with exit code $LASTEXITCODE" }
    Write-Output (Join-Path $taskOutput 'comparison.mp4')
} finally { Pop-Location }
'''


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b''):
            h.update(chunk)
    return h.hexdigest()


def included_file(path: Path, base: Path) -> bool:
    relative = path.relative_to(base)
    if any(part in SKIP_DIRS for part in relative.parts):
        return False
    if path.suffix.lower() in {'.pyc', '.pyo', '.tmp', '.partial'}:
        return False
    if path.name == '.env' or path.name.startswith('.env.') or path.suffix.lower() in {'.pem', '.key'}:
        return False
    return path.is_file() and not path.is_symlink()


def collect(videos: Path):
    result = {}

    def add_tree(base: Path, prefix: str, blocked_top=frozenset()):
        if not base.is_dir():
            raise FileNotFoundError(base)
        for path in base.rglob('*'):
            rel = path.relative_to(base)
            if rel.parts[0] in blocked_top:
                continue
            if included_file(path, base):
                result[f'{prefix}/{rel.as_posix()}'] = path

    add_tree(ROOT, 'hik_g1', {'delivery', 'wheels', '.venv', '__pycache__'})
    # Include source, official model assets, original ZED project code/results,
    # and every supplied checkpoint asset; omit the separate raw ZED SVO tree.
    repro = ROOT.parent / REPRO_NAME
    add_tree(repro / 'pose_project', REPRO_NAME + '/pose_project')
    if (repro / 'README_CN.md').is_file():
        result[REPRO_NAME + '/README_CN.md'] = repro / 'README_CN.md'
    # The old result verification is a historical record, not this ZIP's manifest.
    if (ROOT / 'delivery/verification.json').is_file():
        result['hik_g1/delivery/verification.json'] = ROOT / 'delivery/verification.json'
    for video in sorted(videos.glob('*.avi')):
        if video.is_symlink():
            raise ValueError(f'Unexpected video symlink: {video}')
        result['raw_hikvision/' + video.name] = video
    expected = ['Video_20260922155450711.avi', 'Video_20260922161535257.avi', 'Video_20260922162126269.avi']
    for name in expected:
        if 'raw_hikvision/' + name not in result:
            raise FileNotFoundError(videos / name)
    required = [
        'hik_g1/run_pipeline.py', 'hik_g1/infer_hik.py',
        'hik_g1/docs/SAM3D_零基础完整说明书.html',
        'hik_g1/docs/SAM3D_零基础完整说明书.md',
        f'{REPRO_NAME}/pose_project/sam3d/repo/sam_3d_body/models/meta_arch/sam3d_body.py',
        f'{REPRO_NAME}/pose_project/sam3d/repo/sam_3d_body/sam_3d_body_estimator.py',
        f'{REPRO_NAME}/pose_project/sam3d/dinov3_official/dinov3/hub/backbones.py',
        f'{REPRO_NAME}/pose_project/sam3d/checkpoints/user_dinov3_20260917/model.ckpt',
        f'{REPRO_NAME}/pose_project/sam3d/checkpoints/user_dinov3_20260917/assets/mhr_model.pt',
    ]
    for relative in required:
        if relative not in result:
            raise FileNotFoundError(relative)
    return result


def readme(bundle_name: str, files: dict, total: int) -> str:
    return f'''# SAM 3D / 海康人体运动 / 固定世界 / G1：完整源码与数据包

打包时间：{datetime.now().astimezone().isoformat(timespec='seconds')}
包名：`{bundle_name}`；打包前文件数：{len(files)}；源文件总大小：{total / 1024**3:.2f} GiB。

## 包含内容

- `hik_g1/`：最新海康适配源码、说明书（HTML/Markdown）、环境版本记录、G1 XML/网格和 YOLO 权重。
- `hik_g1/outputs/`：真实海康的抽帧图像、逐帧人体缓存、相机/世界序列、原始与降抖 G1 动作、视频和质量材料。
- `hik_g1/source_annotation_backups/`：添加中文注释前的两个核心源码，以及注释等价性校验记录。
- `{REPRO_NAME}/pose_project/`：原复现包的工程代码与结果，包括已加中文注释的 SAM 3D 源码、DINOv3 源码、ZED 相机/世界坐标相关代码与资料。
- `.../sam3d/checkpoints/user_dinov3_20260917/`：模型 checkpoint、配置、MHR TorchScript 及全部原附属资产，保留许可证/来源文件。
- `raw_hikvision/`：三段原始海康 AVI。准备场景的第一段只有检测审查；主要运动结果是后两段。
- `MANIFEST.json`、`verify_package.py`：逐文件 SHA256 与解压后完整性检查。
- `run_hikvision.ps1`：使用指定 Python 环境的小片段运行入口，默认最多 120 个抽样点。

## 不包含的内容

- 本机 `.venv` 和安装 wheel 缓存：它们约 15 GiB，不能代替跨机器重建环境。
- 原 ZED `raw_data/` 中的 SVO2/IMU 原始采集：原始 TAR.ZST 复现包仍单独保留，本包以海康数据复现为主。
- 旧交付 ZIP、Python 缓存、Git 数据和临时文件。

这不是原 TAR.ZST 的逐字节副本。原 README 中提到的 ZED raw_data 不在此包；当时保存的结果/JSON 可能包含原运行机器绝对路径和历史文件 hash。当前包的完整性以根目录 MANIFEST.json 为准，历史记录保留原样。

## 解压与阅读

本文件为 ZIP64，大于 4 GiB。请完整解压到一个有足够空间的目录，不要直接在 ZIP 内运行程序。保留根目录下 `hik_g1` 和 `{REPRO_NAME}` 的同级关系，海康入口据此寻找 SAM 3D 源码与权重。

双击 `hik_g1/docs/SAM3D_零基础完整说明书.html` 阅读完整 22 章手册。其文字和预览图片已内嵌，源码/报告链接及示例绝对路径仍面向原电脑；换电脑后按本包目录替换路径，或使用下方启动脚本。可编辑原稿是同名 Markdown。

VS Code 的主要修改入口：

1. `hik_g1/run_pipeline.py`：整套流程。
2. `hik_g1/infer_hik.py`：检测、抽帧和 SAM 3D 调用。
3. `{REPRO_NAME}/pose_project/sam3d/repo/sam_3d_body/sam_3d_body_estimator.py`：外层单图接口。
4. `{REPRO_NAME}/pose_project/sam3d/repo/sam_3d_body/models/meta_arch/sam3d_body.py`：核心模型，已含中文注释。

源码以打包时磁盘上已保存的版本为准，不包含编辑器尚未保存的缓冲区。

## 校验文件

在解压后的包根目录，用任意可用的 Python 3 执行（仅需标准库）：

```powershell
python .\\verify_package.py
```

成功会显示所有清单文件的 SHA256 一致。压缩包旁的 `.sha256` 可用 `Get-FileHash` 检查整个 ZIP。完整性通过不代表尺度标定、回环精度或机器人动力学通过。

## 运行海康小片段

先配置 Python 3.11、匹配本机 GPU 的 CUDA PyTorch、项目依赖和 ffmpeg/ffprobe。参考 `hik_g1/environment_versions.json`、`requirements.lock.txt` 及手册第 6/21 章；这些是环境记录，不是已打包的可移植环境。

在原电脑上，可使用已经跑通的环境；在包根目录执行：

```powershell
.\\run_hikvision.ps1 -PythonExe 'C:\\YXW\\视觉\\hik_g1\\.venv\\Scripts\\python.exe'
```

默认读取包内主海康视频，新建时间戳结果目录，stride=4、最多120个抽样点。处理完整视频可加 `-MaxFrames 0`。换电脑后把 `-PythonExe` 换成新配置环境中的真实解释器路径。如果 PowerShell 执行策略禁止脚本，可从终端直接运行 `hik_g1/run_pipeline.py`，手册已给出对应参数。

修改网络、权重或推理逻辑后，请新建输出目录，不能用旧逐帧缓存检验新算法。旧缓存不自动感知所有源码/权重变化。

## 结果边界与来源

海康结果使用固定参考世界坐标，当前尺度仍未实测标定；未强制闭合轨迹，主视频起终点本身不相同。G1 是运动学参考，没有完成动态平衡或机器人行走策略训练。完整解释见 `hik_g1/RESULTS_CN.md`。

SAM 3D、DINOv3、MHR、G1 和相关组件的许可证、来源和配置原样随包保留。权重来自用户提供的复现资料。本次仅本地打包，没有上传数据或模型。
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-videos', type=Path, default=(ROOT.parent / 'raw_hikvision' if (ROOT.parent / 'raw_hikvision').is_dir() else Path(r'C:\Users\liuq8\MVS\Data\MV-CS020-10GC(DB2677346)')))
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'delivery')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if output != (ROOT / 'delivery').resolve():
        try:
            output.relative_to(ROOT)
        except ValueError:
            pass
        else:
            raise ValueError('Alternate output must be outside hik_g1 to avoid recursively packaging itself')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    name = 'SAM3D_Hikvision_G1_full_' + stamp
    archive = output / (name + '.zip')
    partial = output / (name + '.partial.zip')
    if archive.exists() or partial.exists():
        raise FileExistsError(name)
    files = collect(args.input_videos.resolve())
    total = sum(p.stat().st_size for p in files.values())
    if shutil.disk_usage(output).free < total * 1.1 + 1024**3:
        raise RuntimeError('Insufficient free space for complete ZIP')
    generated = {
        'README_先读我.md': readme(name, files, total).encode('utf-8'),
        'verify_package.py': VERIFY_SCRIPT.encode('utf-8'),
        'run_hikvision.ps1': b'\xef\xbb\xbf' + LAUNCH_SCRIPT.encode('utf-8'),
    }
    compile(VERIFY_SCRIPT, 'verify_package.py', 'exec')
    manifest = {
        'schema_version': 1, 'created_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'bundle_root': name, 'scope': 'All source, annotated SAM3D, DINOv3, all supplied model weights/assets, original pose_project results, Hikvision original AVI and full derived results; no installed environments, wheel cache or raw ZED SVO',
        'manifest_does_not_hash_itself': True, 'files': [],
    }
    count, processed, last = 0, 0, time.monotonic()
    print(json.dumps({'stage': 'pack', 'files': len(files), 'input_bytes': total, 'archive': str(archive)}, ensure_ascii=False), flush=True)
    with zipfile.ZipFile(partial, 'x', allowZip64=True, compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for relative, source in sorted(files.items()):
            stat_before = source.stat()
            info = zipfile.ZipInfo.from_file(source, f'{name}/{relative}')
            info.compress_type = zipfile.ZIP_STORED if source.suffix.lower() in STORE_SUFFIXES else zipfile.ZIP_DEFLATED
            h = hashlib.sha256()
            copied = 0
            with source.open('rb') as src, z.open(info, 'w', force_zip64=True) as dst:
                for chunk in iter(lambda: src.read(CHUNK), b''):
                    h.update(chunk)
                    dst.write(chunk)
                    copied += len(chunk)
                    processed += len(chunk)
                    if time.monotonic() - last >= 15:
                        print(json.dumps({'stage': 'pack', 'files_done': count, 'bytes_done': processed, 'total_bytes': total, 'current': relative}, ensure_ascii=False), flush=True)
                        last = time.monotonic()
            stat_after = source.stat()
            if copied != stat_before.st_size or (stat_after.st_size, stat_after.st_mtime_ns) != (stat_before.st_size, stat_before.st_mtime_ns):
                raise RuntimeError(f'Source changed during packaging: {source}')
            manifest['files'].append({'path': relative, 'bytes': copied, 'sha256': h.hexdigest()})
            count += 1
        for relative, data in generated.items():
            z.writestr(f'{name}/{relative}', data)
            manifest['files'].append({'path': relative, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
        manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8')
        z.writestr(f'{name}/MANIFEST.json', manifest_bytes)
    print(json.dumps({'stage': 'verify_archive', 'payload_files': len(manifest['files']), 'zip_bytes': partial.stat().st_size}), flush=True)
    last = time.monotonic()
    verified = 0
    with zipfile.ZipFile(partial) as z:
        names = z.namelist()
        expected = {f'{name}/{e["path"]}' for e in manifest['files']} | {f'{name}/MANIFEST.json'}
        if len(names) != len(set(names)) or set(names) != expected:
            raise RuntimeError('Duplicate or unexpected archive members')
        if z.read(f'{name}/MANIFEST.json') != manifest_bytes:
            raise RuntimeError('Manifest readback differs')
        for i, entry in enumerate(manifest['files'], 1):
            h, size = hashlib.sha256(), 0
            with z.open(f'{name}/{entry["path"]}') as stream:
                for chunk in iter(lambda: stream.read(CHUNK), b''):
                    h.update(chunk)
                    size += len(chunk)
            if size != entry['bytes'] or h.hexdigest() != entry['sha256']:
                raise RuntimeError('Archive readback mismatch: ' + entry['path'])
            verified += size
            if time.monotonic() - last >= 15:
                print(json.dumps({'stage': 'verify_archive', 'files_done': i, 'bytes_verified': verified}), flush=True)
                last = time.monotonic()
    digest = sha256(partial)
    partial.rename(archive)
    manifest_path = output / (name + '.manifest.json')
    manifest_path.write_bytes(manifest_bytes)
    (output / (name + '.zip.sha256')).write_text(digest + '  ' + archive.name + '\n', encoding='ascii')
    report = {
        'status': 'passed', 'archive': str(archive), 'archive_format': 'ZIP64',
        'payload_files': len(manifest['files']), 'archive_entries': len(manifest['files']) + 1,
        'uncompressed_payload_bytes': verified, 'archive_bytes': archive.stat().st_size,
        'archive_sha256': digest, 'all_payloads_read_back_and_sha256_verified': True,
        'model_weights_included': True, 'original_hikvision_videos_included': True,
        'original_zed_svo_included': False, 'installed_environment_included': False,
    }
    (output / (name + '.verification.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
