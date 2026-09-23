"""Package code, official G1 assets, motion results and review evidence (no environments/weights)."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parent
delivery = ROOT/'delivery'
delivery.mkdir(exist_ok=True)
files = set()
for pattern in ['*.py','*.md','*.json','requirements.lock.txt']:
    files.update(ROOT.glob(pattern))
for folder in ['docs','reconstruction','retarget','visualization','assets/unitree_g1']:
    for p in (ROOT/folder).rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and 'qa_synthetic' not in p.parts:
            files.add(p)
for name in ['Video_20260922162126269','Video_20260922161535257']:
    for p in (ROOT/'outputs'/name).rglob('*'):
        if p.is_file() and 'images' not in p.parts and 'frames' not in p.parts and p.suffix != '.log':
            files.add(p)
for p in (ROOT/'input_audit').glob('*'):
    if p.is_file() and p.suffix.lower() in ('.json','.md','.jpg','.png'):
        files.add(p)
files.add(delivery/'verification.json')
manifest = {'scope':'Code + official G1 assets + true Hikvision motion results/videos + audits',
            'excluded':'Original Hikvision AVI, SAM3D source/checkpoint, Python environment and downloaded wheel cache; these remain locally available separately',
            'files':[]}
for p in sorted(files):
    manifest['files'].append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,
                              'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
manifest_path=delivery/'MANIFEST.json'
manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
archive=delivery/'hikvision_g1_results.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
    for p in sorted(files):
        z.write(p,'hik_g1/'+p.relative_to(ROOT).as_posix())
    z.write(manifest_path,'hik_g1/delivery/MANIFEST.json')
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
digest=hashlib.sha256(archive.read_bytes()).hexdigest()
(delivery/'hikvision_g1_results.zip.sha256').write_text(digest+'  '+archive.name+'\n',encoding='ascii')
print(json.dumps({'archive':str(archive),'files':len(files)+1,'bytes':archive.stat().st_size,'sha256':digest}))
