# Certificate trust regression checks

Run `python3 -S security/fetch_locked_bundle.py /tmp/certifi-trust.pem`, then
`TRUST_CA_BUNDLE=/tmp/certifi-trust.pem python3 -S security/test_trust_policy.py`.
This fetches and hashes the locked wheel as data without installing or importing
application dependencies. OpenSSL 3 verifies rejection of all 16 compromised
intermediate certificates, including cross-signed variants, with independent
CA-directory and system-store fallback disabled.

The public certificate fixtures come from the
[eMudhra incident disclosure](https://emudhra.we-have-a-problem.com/).
The Dockerfiles also check the real OS and installed certifi root stores when
images build. Rebuild and validate deployed images after review; passing these
source checks does not establish the trust stores of existing containers.
