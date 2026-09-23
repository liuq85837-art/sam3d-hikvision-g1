"""Fetch the unmodified official Unitree G1 29-DOF MJCF and required meshes."""
from __future__ import annotations
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request
import time
import xml.etree.ElementTree as ET

DEST = Path(__file__).resolve().parents[1] / "assets" / "unitree_g1"
REPO = "unitreerobotics/unitree_mujoco"

def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "hik-g1-local-reconstruction"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=35) as response:
                return response.read()
        except OSError:
            if attempt == 4:
                raise
            time.sleep(1 + attempt)

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    commit = json.loads(download(f"https://api.github.com/repos/{REPO}/commits/main"))["sha"]
    previous_commit = (DEST / "FETCH_COMMIT.txt").read_text().strip() if (DEST / "FETCH_COMMIT.txt").exists() else commit
    if previous_commit != commit:
        raise RuntimeError("Existing partial model belongs to another commit; use a new assets directory")
    (DEST / "FETCH_COMMIT.txt").write_text(commit, encoding="utf-8")
    root = f"https://raw.githubusercontent.com/{REPO}/{commit}"
    xml = download(root + "/unitree_robots/g1/g1_29dof.xml")
    (DEST / "g1_29dof.xml").write_bytes(xml)
    meshes = sorted({m.attrib["file"] for m in ET.fromstring(xml).findall("./asset/mesh")})
    (DEST / "meshes").mkdir(exist_ok=True)
    def fetch_mesh(name):
        destination = DEST / "meshes" / name
        if destination.exists():
            data = destination.read_bytes()
        else:
            data = download(root + "/unitree_robots/g1/meshes/" + name)
            destination.write_bytes(data)
        return {"path": "meshes/" + name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(fetch_mesh, meshes))
    (DEST / "LICENSE.unitree").write_bytes(download(root + "/LICENSE"))
    provenance = {"repository": "https://github.com/" + REPO, "commit": commit,
                  "model_path": "unitree_robots/g1/g1_29dof.xml", "model_unmodified": True,
                  "model_sha256": hashlib.sha256(xml).hexdigest(), "meshes": records}
    (DEST / "SOURCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps({"commit": commit, "mesh_files": len(records), "bytes": sum(r["bytes"] for r in records)}))

if __name__ == "__main__":
    main()
