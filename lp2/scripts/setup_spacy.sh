#!/usr/bin/env bash
set -e
python3 - <<'PY'
try:
    import spacy
    spacy.load("en_core_web_sm")
    print("[OK] spaCy model already present.")
except Exception:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    print("[OK] spaCy model installed.")
PY
