"""Download official PyPI pure-Python runtime wheels and verify PyPI SHA256."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "wheels"
NAMES = ["absl-py", "etils", "glfw", "pyopengl", "fsspec", "importlib-resources", "zipp", "typing-extensions"]

def curl(url, output):
    subprocess.run(["curl.exe", "-f", "-L", "--ssl-revoke-best-effort", "--proxy", "http://127.0.0.1:7897",
                    "--retry", "3", "--connect-timeout", "20", "--max-time", "120", "--silent", "--show-error",
                    "-o", str(output), url], check=True)

def fetch(name):
    info_path = DEST / (name + ".pypi.json")
    curl("https://pypi.org/pypi/" + name + "/json", info_path)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    wheels = [f for f in info["urls"] if f["filename"].endswith(".whl") and not f.get("yanked", False)]
    options = [f for f in wheels if "none-any" in f["filename"]]
    if not options:
        options = [f for f in wheels if "win_amd64" in f["filename"] and ("py3" in f["filename"] or "cp311" in f["filename"])]
    if not options:
        raise RuntimeError("No compatible wheel for " + name)
    file = options[0]
    output = DEST / file["filename"]
    if not output.exists() or hashlib.sha256(output.read_bytes()).hexdigest() != file["digests"]["sha256"]:
        curl(file["url"], output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    if digest != file["digests"]["sha256"]:
        raise RuntimeError("SHA256 mismatch: " + str(output))
    result = {"name": name, "version": info["info"]["version"], "filename": file["filename"], "url": file["url"], "sha256": digest}
    print(json.dumps(result), flush=True)
    return result

def main():
    DEST.mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, NAMES))
    (DEST / "small_dependencies_source.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
