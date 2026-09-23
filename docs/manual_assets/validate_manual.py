"""Validate local links and runnable, read-only examples in the beginner manual."""
from pathlib import Path
import ast
import json
import re
import subprocess
import sys

DOCS = Path(__file__).resolve().parent.parent
SOURCE = DOCS / 'SAM3D_零基础完整说明书.md'
PYTHON = DOCS.parent / '.venv/Scripts/python.exe'


def main():
    text = SOURCE.read_text(encoding='utf-8')
    blocks = re.findall(r'^```([^\n]*)\n(.*?)^```\s*$', text, re.M | re.S)
    anchors = set(re.findall(r'<a id="([^"]+)"></a>', text))
    links = re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text)
    for target in links:
        if target.startswith('#'):
            assert target[1:] in anchors, f'Broken anchor: {target}'
        elif not re.match(r'[a-z]+://', target):
            assert (SOURCE.parent / target).resolve().exists(), f'Broken local link: {target}'
    python_blocks = [body for lang, body in blocks if lang.strip() == 'python']
    for body in python_blocks:
        ast.parse(body)
        compile(body, str(SOURCE), 'exec')
    read_examples = [body for body in python_blocks if 'with np.load(path, allow_pickle=False)' in body]
    results = []
    for body in read_examples:
        completed = subprocess.run([str(PYTHON), '-X', 'utf8', '-c', body], cwd=DOCS.parent,
                                   capture_output=True, encoding='utf-8')
        if completed.returncode:
            raise RuntimeError(completed.stderr)
        results.append({'passed': True, 'output': completed.stdout})
    powershell = [body for lang, body in blocks if lang.strip() == 'powershell']
    ps_file = SOURCE.parent / 'manual_assets/manual_powershell_blocks.json'
    ps_file.write_text(json.dumps(powershell, ensure_ascii=False, indent=2), encoding='utf-8')
    report = {
        'source': str(SOURCE), 'chapters': len(anchors),
        'markdown_characters': len(text), 'local_and_anchor_links_checked': len(links),
        'python_blocks_compiled': len(python_blocks),
        'read_only_examples_executed': results,
        'powershell_blocks_for_syntax_check': len(powershell),
        'no_inference_or_retargeting_performed': True,
        'passed': True,
    }
    (DOCS / 'manual_assets/manual_validation.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='read_only_examples_executed'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
