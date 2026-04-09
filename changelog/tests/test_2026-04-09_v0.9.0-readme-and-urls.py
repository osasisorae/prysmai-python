"""
Tests for: v0.9.0 README URL fix + dev branch features.
2026-04-09-v0.9.0-readme-and-urls.md

Usage:
    cd /Users/Macintosh/LearningHub/prysmai-python
    python changelog/tests/test_2026-04-09_v0.9.0-readme-and-urls.py
"""
import sys
import re

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = []

def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    results.append(condition)


# ── 1. README URLs ────────────────────────────────────────────────────────────

print("\n── README URLs ──")
readme = open("README.md").read()
check("no localhost:3000 references", "localhost:3000" not in readme)
check("localhost:8000/v1 present for local dev", "localhost:8000/v1" in readme)
check("localhost:8000/mcp present for MCP", "localhost:8000/mcp" in readme)
check("production URL still correct", "https://prysmai.io/api/v1" in readme)


# ── 2. Version consistency ────────────────────────────────────────────────────

print("\n── Version consistency ──")
pyproject = open("pyproject.toml").read()
init = open("prysmai/__init__.py").read()

pyproject_version = re.search(r'^version = "(.+)"', pyproject, re.M)
init_version = re.search(r'^__version__ = "(.+)"', init, re.M)

pv = pyproject_version.group(1) if pyproject_version else None
iv = init_version.group(1) if init_version else None

check("pyproject.toml version is 0.9.0", pv == "0.9.0", f"got: {pv}")
check("__init__.py version is 0.9.0", iv == "0.9.0", f"got: {iv}")
check("versions match", pv == iv, f"pyproject={pv}, __init__={iv}")


# ── 3. Dev branch features importable ─────────────────────────────────────────

print("\n── Dev branch features importable ──")
try:
    from prysmai import PrysmClient, prysm_context
    from prysmai.governance import GovernanceSession, AsyncGovernanceSession
    from prysmai.detectors import FinancialAnomalyDetector, LoopDetector
    check("PrysmClient importable", True)
    check("AsyncGovernanceSession importable", True)
    check("prysm_context importable", True)
    check("detectors importable", True)
except ImportError as e:
    check("all imports succeed", False, str(e))

try:
    from prysmai.context import prysm_context
    check("prysm_context.from_headers exists", hasattr(prysm_context, "from_headers"))
except Exception as e:
    check("prysm_context.from_headers", False, str(e))

try:
    client_src = open("prysmai/client.py").read()
    check("last_trace_id property in client", "last_trace_id" in client_src)
    check("last_threat_level property in client", "last_threat_level" in client_src)
    check("last_threat_score property in client", "last_threat_score" in client_src)
except Exception as e:
    check("response header properties", False, str(e))


# ── Summary ───────────────────────────────────────────────────────────────────

passed = sum(results)
total = len(results)
print(f"\n{'─'*40}")
print(f"  {passed}/{total} passed")
if passed == total:
    print(f"  \033[92mAll tests passed.\033[0m")
else:
    print(f"  \033[91m{total - passed} test(s) failed.\033[0m")
print()

sys.exit(0 if passed == total else 1)
