"""JASPAR position weight matrices and a from-scratch log-odds scanner.

A `Motif` holds base probabilities per column. Scoring is log-odds against a
background distribution, scanning both strands. Per the prompt this is
implemented from scratch (no MOODS, no BioPython) since the scanning logic is
simple and is where the biology lives.

Real mode downloads JASPAR 2024 CORE vertebrates. Offline, a small set of
consensus-derived PWMs for canonical TF motifs is provided so the motif-scan
analysis runs without network. The offline PWMs use the same consensuses the
synthetic generator plants, which is what makes the offline validation meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

BASE_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}
COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
UNIFORM_BG = np.array([0.25, 0.25, 0.25, 0.25])


@dataclass
class Motif:
    name: str
    probs: np.ndarray  # [4, width], columns sum to 1, rows are A,C,G,T

    @property
    def width(self) -> int:
        return self.probs.shape[1]

    def consensus(self) -> str:
        return "".join("ACGT"[i] for i in self.probs.argmax(axis=0))

    def reverse_complement(self) -> "Motif":
        # reverse columns and swap A<->T, C<->G (row order A,C,G,T -> T,G,C,A)
        rc = self.probs[::-1, ::-1].copy()
        return Motif(self.name + "_rc", rc)


def counts_to_probs(counts: np.ndarray, pseudocount: float = 0.8) -> np.ndarray:
    counts = counts.astype(np.float64) + pseudocount
    return counts / counts.sum(axis=0, keepdims=True)


def pwm_from_consensus(name: str, consensus: str, peak: float = 0.9) -> Motif:
    """Build a PWM that strongly prefers the consensus base in each column."""
    w = len(consensus)
    probs = np.full((4, w), (1.0 - peak) / 3.0)
    for j, base in enumerate(consensus.upper()):
        probs[BASE_INDEX[base], j] = peak
    return Motif(name, probs)


def builtin_motifs() -> list[Motif]:
    """Canonical TF motifs (consensus-derived) for offline use, plus a decoy.

    The decoy is a non-biological pattern used as a specificity control: a head
    that genuinely tracks biology should not correlate with it.
    """
    return [
        pwm_from_consensus("GATA", "GATAAG"),
        pwm_from_consensus("TATA", "TATAAA"),
        pwm_from_consensus("SP1", "GGGGCGGGG"),
        pwm_from_consensus("EBOX", "CACGTG"),
        pwm_from_consensus("DECOY", "ACACACAC"),
    ]


def log_odds_matrix(motif: Motif, background: np.ndarray = UNIFORM_BG) -> np.ndarray:
    """log2(p / background) per base per column. Shape [4, width]."""
    return np.log2(motif.probs / background[:, None])


def _encode_seq(seq: str) -> np.ndarray:
    return np.array([BASE_INDEX.get(b, -1) for b in seq.upper()], dtype=np.int64)


def window_scores(seq: str, motif: Motif, background: np.ndarray = UNIFORM_BG) -> np.ndarray:
    """Log-odds score of the motif at each start offset. Shape [len-width+1].

    Vectorized over windows via a sliding view; bases not in ACGT (encoded -1)
    contribute 0, matching a per-column gather. Equivalent to the per-window sum.
    """
    lom = log_odds_matrix(motif, background)
    enc = _encode_seq(seq)
    w = motif.width
    n = len(enc) - w + 1
    if n <= 0:
        return np.zeros(0)
    windows = np.lib.stride_tricks.sliding_window_view(enc, w)  # [n, w]
    cols = np.arange(w)
    valid = windows >= 0
    contrib = lom[np.where(valid, windows, 0), cols[None, :]]  # [n, w]
    return np.where(valid, contrib, 0.0).sum(axis=1)


def best_score(seq: str, motif: Motif, both_strands: bool = True, background: np.ndarray = UNIFORM_BG) -> float:
    """Best log-odds score over all offsets and (optionally) both strands."""
    fwd = window_scores(seq, motif, background)
    best = float(fwd.max()) if fwd.size else float("-inf")
    if both_strands:
        rev = window_scores(seq, motif.reverse_complement(), background)
        if rev.size:
            best = max(best, float(rev.max()))
    return best


def scan_sequence(
    seq: str, motif: Motif, threshold: float, both_strands: bool = True
) -> list[tuple[int, float, str]]:
    """Return (start, score, strand) for every window scoring above threshold."""
    hits: list[tuple[int, float, str]] = []
    fwd = window_scores(seq, motif)
    for i, s in enumerate(fwd):
        if s >= threshold:
            hits.append((i, float(s), "+"))
    if both_strands:
        rev = window_scores(seq, motif.reverse_complement())
        for i, s in enumerate(rev):
            if s >= threshold:
                hits.append((i, float(s), "-"))
    return hits


def parse_jaspar(path: str | Path) -> list[Motif]:
    """Parse a JASPAR-format PFM file into Motifs.

    Format per record:
        >MATRIX_ID NAME
        A [ c c c ... ]
        C [ ... ]
        G [ ... ]
        T [ ... ]
    """
    motifs: list[Motif] = []
    name = None
    rows: dict[str, list[float]] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None and len(rows) == 4:
                    motifs.append(_finish_record(name, rows))
                parts = line[1:].split()
                name = parts[1] if len(parts) > 1 else parts[0]
                rows = {}
            else:
                base = line[0].upper()
                nums = line[line.find("[") + 1 : line.find("]")] if "[" in line else line[1:]
                rows[base] = [float(x) for x in nums.split()]
        if name is not None and len(rows) == 4:
            motifs.append(_finish_record(name, rows))
    return motifs


def _finish_record(name: str, rows: dict[str, list[float]]) -> Motif:
    counts = np.array([rows["A"], rows["C"], rows["G"], rows["T"]], dtype=np.float64)
    return Motif(name, counts_to_probs(counts))


def download_jaspar(out_path: str | Path | None = None) -> Path:
    """Download JASPAR 2024 CORE vertebrates PFMs (real mode, needs network)."""
    import urllib.request

    from config import RAW

    url = (
        "https://jaspar.elixir.no/download/data/2024/CORE/"
        "JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt"
    )
    out_path = Path(out_path) if out_path else RAW / "JASPAR2024_CORE_vertebrates.txt"
    urllib.request.urlretrieve(url, out_path)
    return out_path


def load_motifs(jaspar_path: str | Path | None = None) -> list[Motif]:
    """Parse a JASPAR file if given/available, else return the built-in set."""
    if jaspar_path is not None and Path(jaspar_path).exists():
        return parse_jaspar(jaspar_path)
    return builtin_motifs()
