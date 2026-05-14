#!/usr/bin/env python
"""
verify_installation.py — ATS Resume Analyzer Environment Check

Run this after installing dependencies to verify everything is ready.
Usage: python verify_installation.py
"""

import sys
import importlib
from pathlib import Path

print("=" * 70)
print("🔍 ATS Resume Analyzer — Installation Verification")
print("=" * 70)
print()

# ── Python Version ────────────────────────────────────────────────────────────
py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
py_ok = sys.version_info >= (3, 9)
status = "✅" if py_ok else "❌"
print(f"{status} Python Version: {py_version} {'(OK)' if py_ok else '(Need 3.9+)'}")
print()

# ── Dependencies ──────────────────────────────────────────────────────────────
print("Checking Required Packages:")
print("-" * 70)

packages = [
    ("streamlit", "Streamlit (Frontend)"),
    ("pdfminer", "pdfminer.six (PDF Extraction)"),
    ("docx", "python-docx (DOCX Extraction)"),
    ("spacy", "SpaCy (NLP)"),
    ("sklearn", "scikit-learn (ML/Similarity)"),
    ("numpy", "NumPy (Numerical Computing)"),
    ("plotly", "Plotly (Visualizations)"),
    ("pandas", "Pandas (Data Manipulation)"),
]

all_ok = True
for import_name, display_name in packages:
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"✅ {display_name:40} v{version}")
    except ImportError:
        print(f"❌ {display_name:40} NOT INSTALLED")
        all_ok = False

print()

# ── SpaCy Model ───────────────────────────────────────────────────────────────
print("Checking SpaCy NER Model:")
print("-" * 70)

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    print(f"✅ SpaCy model 'en_core_web_sm' loaded successfully")
    print(f"   Components: {list(nlp.pipe_names)}")
except OSError:
    print(f"❌ SpaCy model 'en_core_web_sm' NOT FOUND")
    print(f"   Fix: python -m spacy download en_core_web_sm")
    all_ok = False

print()

# ── Project Files ─────────────────────────────────────────────────────────────
print("Checking Project Files:")
print("-" * 70)

project_root = Path(__file__).parent
required_files = [
    "app.py",
    "resume_parser.py",
    "analyzer.py",
    "security.py",
    "requirements.txt",
    "README.md",
]

for filename in required_files:
    filepath = project_root / filename
    exists = filepath.exists()
    status = "✅" if exists else "❌"
    print(f"{status} {filename}")
    if not exists:
        all_ok = False

print()

# ── Final Summary ─────────────────────────────────────────────────────────────
print("=" * 70)
if all_ok and py_ok:
    print("✅ ALL CHECKS PASSED — Ready to run!")
    print()
    print("Next steps:")
    print("  1. Activate your virtual environment")
    print("  2. Run: streamlit run app.py")
    print("  3. Open http://localhost:8501 in your browser")
    print()
    sys.exit(0)
else:
    print("❌ ISSUES FOUND — Please fix above before running")
    print()
    print("Troubleshooting:")
    print("  • Python version: python --version")
    print("  • Install dependencies: pip install -r requirements.txt")
    print("  • Download SpaCy model: python -m spacy download en_core_web_sm")
    print()
    sys.exit(1)
