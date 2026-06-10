"""Download the real-mode assets: DNABERT-2 weights/tokenizer and JASPAR PWMs.

Needs network. Run once; afterwards the pipeline works from the local cache.
    /home/aryan/anaconda3/envs/tinyinfer-gpu/bin/python scripts/download_assets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DNABERT2_ID  # noqa: E402


def main() -> None:
    print(f"Fetching DNABERT-2 tokenizer + model: {DNABERT2_ID}")
    try:
        from transformers import AutoModel, AutoTokenizer

        AutoTokenizer.from_pretrained(DNABERT2_ID, trust_remote_code=True)
        AutoModel.from_pretrained(DNABERT2_ID, trust_remote_code=True, attn_implementation="eager")
        print("  DNABERT-2 cached OK")
    except Exception as e:  # noqa: BLE001
        print(f"  DNABERT-2 download/load failed: {e}")
        print("  If this is a transformers-version incompatibility, see README caveats.")

    print("Fetching JASPAR 2024 CORE vertebrates PFMs")
    try:
        from data.jaspar import download_jaspar

        path = download_jaspar()
        print(f"  JASPAR saved to {path}")
    except Exception as e:  # noqa: BLE001
        print(f"  JASPAR download failed: {e}")

    print("To install BEND data (separate package): pip install bend")


if __name__ == "__main__":
    main()
