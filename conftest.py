"""Put the repo root on sys.path so tests can do `from models.dna_lm import ...`
regardless of the working directory pytest is launched from.
"""

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
