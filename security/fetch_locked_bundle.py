"""Fetch only the hashed certifi wheel data from the lock; never install/import it."""
import hashlib
import io
from pathlib import Path
import sys
import tomllib
import subprocess
import zipfile

root = Path(__file__).resolve().parents[1]
lock = next(root.rglob("uv.lock"))
packages = tomllib.loads(lock.read_text())["package"]
certifi = next(package for package in packages if package["name"] == "certifi")
wheel = certifi["wheels"][0]
# Fixed HTTPS origin/path: do not turn lockfile data into an arbitrary URL fetch.
assert wheel["url"] == "https://files.pythonhosted.org/packages/c4/51/29046e48f693c54e5b7412216d447a1d8362c9653148d98c02cdf65fa733/certifi-2026.10.19-py3-none-any.whl"
data = subprocess.run([
    "curl", "--fail", "--silent", "--show-error", "--proto", "=https",
    "--tlsv1.2", "--max-time", "30", "https://files.pythonhosted.org/packages/c4/51/29046e48f693c54e5b7412216d447a1d8362c9653148d98c02cdf65fa733/certifi-2026.10.19-py3-none-any.whl",
], check=True, capture_output=True).stdout
assert "sha256:" + hashlib.sha256(data).hexdigest() == wheel["hash"], "Wheel hash mismatch"
with zipfile.ZipFile(io.BytesIO(data)) as archive:
    Path(sys.argv[1]).write_bytes(archive.read("certifi/cacert.pem"))
