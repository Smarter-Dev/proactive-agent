"""Security regression checks; run with python3 -S security/test_trust_policy.py."""
import hashlib
import json
import subprocess
import tempfile
import os
from pathlib import Path
import re
import ssl
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]
MIN_CERTIFI = (2026, 10, 19)


class TrustPolicy(unittest.TestCase):
    def test_python_trust_bundle_floor(self):
        locks = list(ROOT.rglob("uv.lock"))
        self.assertTrue(locks)
        for lock in locks:
            with self.subTest(lock=str(lock.relative_to(ROOT))):
                data = tomllib.loads(lock.read_text())
                certifi = next(p for p in data["package"] if p["name"] == "certifi")
                self.assertGreaterEqual(tuple(map(int, certifi["version"].split("."))), MIN_CERTIFI)
                manifest = tomllib.loads(lock.with_name("pyproject.toml").read_text())
                self.assertIn("certifi>=2026.10.19", manifest["project"]["dependencies"])
                self.assertEqual(manifest["tool"]["uv"]["required-version"], ">=0.13.4")

    def test_python_images_refresh_system_trust_and_uv(self):
        for dockerfile in ROOT.rglob("Dockerfile*"):
            text = dockerfile.read_text()
            if not re.search(r"^FROM python:", text, re.M):
                continue
            with self.subTest(image=str(dockerfile.relative_to(ROOT))):
                self.assertIn("--only-upgrade ca-certificates", text)
                self.assertIn("ghcr.io/astral-sh/uv:0.13.4", text)
                self.assertNotRegex(text, r"pip install[^\n]*\buv\b|astral.sh/uv/install.sh|uv:latest")

    @unittest.skipUnless(os.environ.get("TRUST_CA_BUNDLE"), "set TRUST_CA_BUNDLE to check the actual shipped PEM bundle")
    def test_actual_bundle_distrusts_emsign(self):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.load_verify_locations(cafile=os.environ["TRUST_CA_BUNDLE"])
        certificates = context.get_ca_certs(binary_form=True)
        self.assertGreater(len(certificates), 50)
        # emSign ECC Root CA - G3; other emSign roots are not distrusted.
        fingerprints = {hashlib.sha256(cert).hexdigest() for cert in certificates}
        self.assertNotIn("86a1ecba089c4a8d3bbe2734c612ba341d813e043cf9e8a862cd5c57a36bbe6b", fingerprints)


    @unittest.skipUnless(os.environ.get("TRUST_CA_BUNDLE"), "set TRUST_CA_BUNDLE to check actual certificate chains")
    def test_actual_bundle_rejects_compromised_chains(self):
        fixtures = json.loads((ROOT / "security/impacted-intermediates.json").read_text())
        self.assertEqual(len(fixtures), 16)
        with tempfile.TemporaryDirectory() as directory:
            for index, item in enumerate(fixtures):
                pem = Path(directory) / f"{index}.pem"
                pem.write_text(item["pem"])
                with self.subTest(certificate=item["name"], issuer=item["issued_by"]):
                    result = subprocess.run([
                        "openssl", "verify", "-no-CApath", "-no-CAstore",
                        "-CAfile", os.environ["TRUST_CA_BUNDLE"], str(pem),
                    ], capture_output=True, text=True)
                    # Failures to parse or launch are not evidence of distrust.
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertIn("verification failed", result.stderr)


if __name__ == "__main__":
    unittest.main()
