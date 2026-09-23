#!/usr/bin/env python3
"""Build the Chinese SAM 3D manual as one offline HTML file.

Run with the bundled Python runtime; Python needs only its standard library.
Markdown is parsed by the already bundled Node/marked package, never a CDN.
Use --self-test to check parsing/layout contracts in memory without writing HTML.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from datetime import datetime
import html
from html.parser import HTMLParser
import json
import mimetypes
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit
from urllib.request import url2pathname


DOCS = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = DOCS / "SAM3D_零基础完整说明书.md"
DEFAULT_DEPS = (
    Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies"
)

# 字体全部使用系统字体；图片由 Python 内嵌，样式和交互随 HTML 一起保存。
CSS = r"""
:root{--ink:#203246;--muted:#66778b;--line:#dce5ee;--paper:#fff;--bg:#f2f5f9;
--accent:#075f6d;--blue:#2563eb;--sidebar:294px;scroll-behavior:smooth}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.88 "Microsoft YaHei UI","Microsoft YaHei","PingFang SC",system-ui,sans-serif}
a{color:var(--accent);text-underline-offset:3px;overflow-wrap:anywhere}
a:hover{color:var(--blue)}button{font:inherit;cursor:pointer}
.sidebar{position:fixed;inset:0 auto 0 0;width:var(--sidebar);background:#102b3d;
color:#dfedf3;padding:28px 20px;overflow-y:auto;z-index:5}
.brand{font-size:19px;font-weight:750;letter-spacing:.03em}.brand small{display:block;
font-weight:400;font-size:12px;letter-spacing:.08em;color:#9fbcc8;margin-top:5px}
.toc-title{margin:29px 10px 10px;font-size:12px;letter-spacing:.15em;color:#9fbcc8}
.toc a{display:block;color:#bed1dc;text-decoration:none;font-size:13px;line-height:1.7;
padding:7px 10px;border-radius:6px;border-left:3px solid transparent;margin:2px 0}
.toc a.level-3{padding-left:24px;font-size:12px}.toc a:hover,.toc a.active{
background:#24475a;color:white;border-left-color:#57c7bc}.sidebar-note{font-size:12px;
color:#9fbcc8;margin:26px 10px 0;border-top:1px solid #315062;padding-top:17px}
.main{margin-left:var(--sidebar);padding:32px clamp(20px,4vw,65px) 70px}
.topbar{max-width:1120px;margin:0 auto 22px;display:flex;justify-content:space-between;
align-items:center;gap:14px;flex-wrap:wrap}.offline{display:inline-flex;align-items:center;
gap:9px;font-size:13px;color:#175c5b}.offline:before{content:"";width:8px;height:8px;
border-radius:50%;background:#198277}.meta{font-size:12px;color:var(--muted)}
.tools{display:flex;gap:9px}.button{border:1px solid #c8d5df;background:white;color:#355065;
border-radius:7px;padding:5px 12px;font-size:13px}.menu-button{display:none}
.article{max-width:1120px;margin:auto;background:var(--paper);padding:clamp(25px,4vw,58px);
border:1px solid var(--line);border-radius:16px;box-shadow:0 9px 30px #20324608}
h1,h2,h3{line-height:1.4;color:#173a50;scroll-margin-top:24px;overflow-wrap:anywhere}
h1{font-size:clamp(28px,3vw,39px);margin:0 0 25px;letter-spacing:-.02em}
h2{font-size:25px;margin:56px 0 20px;border-top:1px solid var(--line);padding-top:28px}
h3{font-size:19px;margin:32px 0 14px}p{margin:15px 0}strong{color:#153a50}
li{margin:7px 0}ul,ol{padding-left:1.65em}li>p{margin:6px 0}
blockquote{margin:22px 0;padding:12px 20px;border-left:4px solid #4b9b99;
border-radius:0 8px 8px 0;background:#edf7f5;color:#34585e}blockquote p{margin:5px 0}
code,kbd{font:0.89em/1.7 Consolas,"Cascadia Code","Courier New",monospace}
:not(pre)>code{background:#edf2f6;border:1px solid #e1e8ef;padding:2px 5px;
border-radius:4px;overflow-wrap:anywhere;color:#295168}
pre{position:relative;background:#122b3d;color:#e1edf4;border-radius:10px;padding:19px 22px;
overflow-x:auto;white-space:pre;line-height:1.65;tab-size:4;margin:23px 0;
border:1px solid #243f51}pre code{font-size:13px;overflow-wrap:normal}
pre[data-language]:before{content:attr(data-language);display:block;font:11px/1.5 system-ui;
letter-spacing:.08em;text-transform:uppercase;color:#8fb7c8;margin-bottom:9px}
.table-wrap{overflow-x:auto;max-width:100%;margin:23px 0;border:1px solid var(--line);
border-radius:9px}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.7;
min-width:420px}th,td{padding:12px 15px;text-align:left;vertical-align:top;
border-right:1px solid #e5ebf1;border-bottom:1px solid #e5ebf1;overflow-wrap:anywhere}
th{background:#eaf1f6;color:#23465c;font-weight:650}tr:nth-child(even) td{background:#f8fafc}
tr:last-child td{border-bottom:0}th:last-child,td:last-child{border-right:0}
img{display:block;max-width:100%;height:auto;border-radius:10px;border:1px solid #dae3ec;
margin:24px auto}hr{border:0;border-top:1px solid var(--line);margin:35px 0}
a[id]{scroll-margin-top:24px}.footer{max-width:1120px;margin:22px auto 0;font-size:12px;
color:var(--muted)}.progress{position:fixed;top:0;left:var(--sidebar);right:0;height:3px;
background:transparent;z-index:10}.progress div{height:100%;width:0;background:#2ca99e}
@media(min-width:1600px){.main{padding-left:70px;padding-right:70px}}
@media(max-width:1020px){:root{--sidebar:245px}.sidebar{padding:22px 13px}.main{padding:22px 22px 50px}}
@media(max-width:780px){.sidebar{display:none;width:min(86vw,320px);box-shadow:10px 0 40px #0004}
body.menu-open .sidebar{display:block}.main{margin-left:0;padding:19px 12px 35px}
.article{padding:25px 21px;border-radius:10px}.menu-button{display:inline-block}
.progress{left:0}h2{font-size:23px}.meta{display:block}.topbar{padding:0 4px}}
@media(prefers-reduced-motion:reduce){:root{scroll-behavior:auto}}
@media print{@page{size:A4;margin:17mm 15mm}body{background:white;color:#111;font-size:10.5pt;
line-height:1.65}.sidebar,.tools,.progress{display:none!important}.main{margin:0;padding:0}
.topbar{margin-bottom:12px}.article{border:0;border-radius:0;box-shadow:none;padding:0;
max-width:none}.footer{font-size:8pt}h1{font-size:24pt}h2{font-size:17pt;margin-top:28px;
padding-top:15px;break-after:avoid}h3{font-size:12pt;break-after:avoid}p,li{orphans:3;widows:3}
pre{white-space:pre-wrap;overflow:visible;overflow-wrap:anywhere;word-break:break-word;
background:#f2f5f8!important;color:#142e40!important;border-color:#ccd5de;padding:12px;
break-inside:auto}pre code{font-size:8pt}.table-wrap{overflow:visible}table{min-width:0;
font-size:8.5pt}thead{display:table-header-group}tr{break-inside:avoid}th,td{padding:7px}
img{max-height:220mm;object-fit:contain;break-inside:avoid}a{color:inherit;text-decoration:none}
blockquote{break-inside:avoid}*{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
"""

JS = r"""
document.querySelectorAll('pre > code').forEach(code => {
  const language = [...code.classList].find(name => name.startsWith('language-'));
  if (language) code.parentElement.dataset.language = language.slice(9);
});
document.querySelector('[data-menu]').addEventListener('click', () => {
  const open = document.body.classList.toggle('menu-open');
  document.querySelector('[data-menu]').setAttribute('aria-expanded', String(open));
});
document.querySelector('[data-print]').addEventListener('click', () => window.print());
const links = [...document.querySelectorAll('.toc a')];
const sections = links.map(link => document.getElementById(decodeURIComponent(link.hash.slice(1))));
links.forEach(link => link.addEventListener('click', () => {
  document.body.classList.remove('menu-open');
  document.querySelector('[data-menu]').setAttribute('aria-expanded', 'false');
}));
let ticking = false;
function updateReadingPosition() {
  const max = document.documentElement.scrollHeight - innerHeight;
  document.querySelector('.progress div').style.width = (max > 0 ? scrollY / max * 100 : 100) + '%';
  let current = 0;
  sections.forEach((section, index) => { if (section && section.getBoundingClientRect().top <= 100) current = index; });
  links.forEach((link, index) => {
    link.classList.toggle('active', index === current);
    if(index === current) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  });
  ticking = false;
}
addEventListener('scroll', () => {if (!ticking) {requestAnimationFrame(updateReadingPosition); ticking = true;}}, {passive:true});
addEventListener('resize', updateReadingPosition);
updateReadingPosition();
"""


@dataclass
class Heading:
    level: int
    anchor: str
    label: str


def parse_markdown(source: str, node: Path, modules: Path) -> str:
    """Run marked through stdin; no shell interpolation and no network access."""
    package = modules / "marked"
    manifest = package / "package.json"
    if not node.is_file() or not manifest.is_file():
        raise RuntimeError("Bundled Node/marked missing; pass --node-runtime and --node-modules.")
    entry = package / json.loads(manifest.read_text(encoding="utf-8")).get(
        "module", "lib/marked.esm.js"
    )
    script = """
import {readFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
const {marked} = await import(pathToFileURL(process.argv[1]).href);
process.stdout.write(marked.parse(readFileSync(0, 'utf8'), {gfm: true, breaks: false}));
"""
    result = subprocess.run(
        [str(node), "--input-type=module", "-e", script, str(entry)],
        input=source, text=True, encoding="utf-8", capture_output=True,
        check=False, timeout=60,
    )
    if result.returncode:
        raise RuntimeError("Markdown parsing failed: " + result.stderr.strip())
    return result.stdout


def local_path(url: str, source_dir: Path) -> Path | None:
    """Resolve Markdown paths, including Windows drive paths and file: URIs."""
    if re.match(r"^[A-Za-z]:[\\/]", url):
        return Path(unquote(url))
    parts = urlsplit(url)
    if parts.scheme == "file":
        path = url2pathname(unquote(parts.path))
        if parts.netloc:
            path = "//" + parts.netloc + path
        return Path(path)
    if parts.scheme or parts.netloc or not parts.path:
        return None
    return (source_dir / unquote(parts.path)).resolve()


class OfflineDocument(HTMLParser):
    """Preserve Markdown HTML while adding navigation and inlining local images."""

    def __init__(self, source_dir: Path):
        super().__init__(convert_charrefs=False)
        self.source_dir = source_dir
        self.parts: list[str] = []
        self.headings: list[Heading] = []
        self.heading: Heading | None = None
        self.heading_text: list[str] = []
        self.image_count = 0
        self.image_bytes = 0
        self.used_ids: set[str] = set()

    def handle_starttag(self, tag, attrs):
        attrs = list(attrs)
        attr = dict(attrs)
        if tag in {"h1", "h2", "h3"}:
            anchor = attr.get("id") or "manual-section-" + str(len(self.headings) + 1)
            while anchor in self.used_ids:
                anchor += "-heading"
            attrs = [(key, value) for key, value in attrs if key != "id"] + [("id", anchor)]
            self.heading = Heading(int(tag[1]), anchor, "")
            self.heading_text = []
            self.headings.append(self.heading)
            self.used_ids.add(anchor)
        elif attr.get("id"):
            self.used_ids.add(attr["id"])
        if tag == "img":
            src = attr.get("src", "")
            if src.startswith("data:image/"):
                pass
            else:
                path = local_path(src, self.source_dir)
                if path is None:
                    raise ValueError("Offline manual requires local images, got: " + src)
                if not path.is_file():
                    raise FileNotFoundError("Markdown image not found: " + str(path))
                data = path.read_bytes()
                mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                if not mime.startswith("image/"):
                    raise ValueError("Not an image MIME type: " + str(path))
                src = "data:" + mime + ";base64," + base64.b64encode(data).decode("ascii")
                self.image_bytes += len(data)
            self.image_count += 1
            attrs = [(key, value) for key, value in attrs if key not in {"src", "srcset", "loading"}]
            attrs += [("src", src), ("loading", "eager")]
        if tag == "a" and attr.get("href"):
            href = attr["href"]
            path = local_path(href, self.source_dir)
            if path is not None:
                # 本地资源链接保留为绝对 file: URI，HTML 移动后仍能指向当前项目文件。
                fragment = urlsplit(href).fragment if not re.match(r"^[A-Za-z]:", href) else ""
                target = path.resolve().as_uri() + ("#" + fragment if fragment else "")
                attrs = [(key, target if key == "href" else value) for key, value in attrs]
                attrs += [("data-local-file", "true")]
        if tag == "table":
            self.parts.append('<div class="table-wrap" role="region" aria-label="可横向滚动的数据表格" tabindex="0">')
        self.parts.append("<" + tag + "".join(
            " " + key if value is None else " " + key + '="' + html.escape(value, quote=True) + '"'
            for key, value in attrs
        ) + ">")

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        self.parts.append("</" + tag + ">")
        if tag == "table":
            self.parts.append("</div>")
        if self.heading is not None and tag == "h" + str(self.heading.level):
            self.heading.label = " ".join("".join(self.heading_text).split())
            self.heading = None

    def handle_data(self, data):
        self.parts.append(data)
        if self.heading is not None:
            self.heading_text.append(data)

    def handle_entityref(self, name):
        self.parts.append("&" + name + ";")
        if self.heading is not None:
            self.heading_text.append(html.unescape("&" + name + ";"))

    def handle_charref(self, name):
        self.parts.append("&#" + name + ";")
        if self.heading is not None:
            self.heading_text.append(html.unescape("&#" + name + ";"))

    def handle_comment(self, data):
        self.parts.append("<!--" + data + "-->")


def build_html(markdown: str, source: Path, node: Path, modules: Path) -> tuple[str, dict]:
    parsed = OfflineDocument(source.parent)
    parsed.feed(parse_markdown(markdown, node, modules))
    parsed.close()
    title = next((h.label for h in parsed.headings if h.level == 1), source.stem)
    nav_headings = [h for h in parsed.headings if h.level in {2, 3}] or parsed.headings
    toc = "\n".join(
        '<a class="level-' + str(h.level) + '" href="#' + html.escape(h.anchor, quote=True)
        + '">' + html.escape(h.label) + '</a>' for h in nav_headings
    )
    built_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %z")
    result = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; font-src 'none'; base-uri 'none'; form-action 'none'">
<title>""" + html.escape(title) + "</title><style>" + CSS + """</style></head><body>
<aside class="sidebar" aria-label="说明书目录"><div class="brand">SAM 3D · 学习手册
<small>HUMAN MOTION → WORLD → G1</small></div><div class="toc-title">阅读目录</div>
<nav class="toc">""" + toc + """</nav><p class="sidebar-note">从视频中的人，到固定世界坐标，再到机器人的运动。按目录逐步阅读，也可搜索关键词。</p></aside>
<div class="progress" aria-hidden="true"><div></div></div><main class="main">
<div class="topbar"><div><span class="offline">离线可读 · 图片已内嵌 · 无需联网</span>
<span class="meta">&nbsp; 生成于 """ + built_at + """</span></div><div class="tools">
<button class="button menu-button" data-menu aria-expanded="false">目录</button>
<button class="button" data-print>打印 / 保存 PDF</button></div></div>
<article class="article">""" + "".join(parsed.parts) + """</article>
<footer class="footer">正文、样式和图片均保存在本 HTML 中；本地代码及视频链接仍需访问原项目文件。使用浏览器 Ctrl+F 可搜索全文。</footer>
</main><script>""" + JS + "</script></body></html>\n"
    report = {
        "title": title, "headings": len(parsed.headings), "toc_entries": len(nav_headings),
        "embedded_images": parsed.image_count, "embedded_image_bytes": parsed.image_bytes,
        "html_bytes": len(result.encode("utf-8")), "offline": True,
    }
    return result, report


def self_test(node: Path, modules: Path) -> dict:
    """Only construct strings in memory; do not create sample/output files."""
    sample = """# 零基础测试 & 参数

<a id="s01"></a>

## 第一节 `输入`

> 相机坐标和世界坐标需要区分。

- RGB 输入
- 骨盆轨迹

1. 识别人
2. 重定向

### 数据表

| 名称 | 形状 |
| --- | --- |
| joints | `T × 70 × 3` |

```python
print("中文 <tag>")
```

```powershell
& $python run_pipeline.py
```

```json
{"scale": "model_prior_only"}
```

```text
固定世界坐标
```

[返回](#s01) · [源文件](../run_pipeline.py)
"""
    preview = DOCS.parent / "outputs/Video_20260922162126269/comparison.preview.png"
    if preview.is_file():
        sample += "\n![真实海康姿态与G1对比](../outputs/Video_20260922162126269/comparison.preview.png)\n"
    else:
        sample += '\n![内嵌测试](data:image/png;base64,iVBORw0KGgo=)\n'
    page, report = build_html(sample, DEFAULT_SOURCE, node, modules)
    assertions = {
        "h1_h2_h3_and_toc": report["headings"] == 3 and report["toc_entries"] == 2,
        "explicit_internal_anchor": 'id="s01"' in page and 'href="#s01"' in page,
        "gfm_table": '<div class="table-wrap"' in page and '<table>' in page,
        "lists_and_quote": '<ul>' in page and '<ol>' in page and '<blockquote>' in page,
        "fenced_languages": all('language-' + lang in page for lang in ("python", "powershell", "json", "text")),
        "code_escaped": '&lt;tag&gt;' in page,
        "local_file_uri": 'href="file:///' in page and 'data-local-file="true"' in page,
        "embedded_image": report["embedded_images"] == 1 and 'src="data:image/png;base64,' in page,
        "print_css": '@media print' in page,
        "no_network_assets": 'src="http' not in page and 'href="https://cdn' not in page,
        "offline_notice": '离线可读 · 图片已内嵌 · 无需联网' in page,
    }
    if not all(assertions.values()):
        raise AssertionError(assertions)
    return {"self_test": "passed", "checks": assertions, **report}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--node-runtime", type=Path, default=Path(os.environ.get(
        "MANUAL_NODE_RUNTIME", str(DEFAULT_DEPS / "node/bin/node.exe"))))
    parser.add_argument("--node-modules", type=Path, default=Path(os.environ.get(
        "MANUAL_NODE_MODULES", str(DEFAULT_DEPS / "node/node_modules"))))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        if args.self_test:
            report = self_test(args.node_runtime, args.node_modules)
        else:
            source = args.source.resolve()
            output = (args.output or source.with_suffix(".html")).resolve()
            if output == source:
                raise ValueError("Output must differ from the Markdown source.")
            page, report = build_html(source.read_text(encoding="utf-8-sig"), source,
                                      args.node_runtime, args.node_modules)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(page, encoding="utf-8", newline="\n")
            report.update({"source": str(source), "output": str(output)})
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, AssertionError, subprocess.SubprocessError) as exc:
        print("Manual build failed: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
