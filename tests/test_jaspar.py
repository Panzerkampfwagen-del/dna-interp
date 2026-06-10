import os
import tempfile

from data.jaspar import (
    best_score,
    builtin_motifs,
    parse_jaspar,
    pwm_from_consensus,
    scan_sequence,
)


def test_builtin_consensus():
    motifs = {m.name: m.consensus() for m in builtin_motifs()}
    assert motifs["GATA"] == "GATAAG"
    assert motifs["TATA"] == "TATAAA"


def test_scan_finds_planted_motif():
    gata = pwm_from_consensus("GATA", "GATAAG")
    seq = "A" * 20 + "GATAAG" + "T" * 20
    hits = scan_sequence(seq, gata, threshold=5.0, both_strands=False)
    starts = [h[0] for h in hits]
    assert 20 in starts


def test_motif_score_specificity():
    gata = pwm_from_consensus("GATA", "GATAAG")
    planted = best_score("A" * 10 + "GATAAG" + "A" * 10, gata)
    random = best_score("ACACACACACACACAC", gata)
    assert planted > random


def test_reverse_complement_detection():
    gata = pwm_from_consensus("GATA", "GATAAG")
    rc_seq = "A" * 10 + "CTTATC" + "A" * 10  # reverse complement of GATAAG
    assert best_score(rc_seq, gata, both_strands=True) > best_score(rc_seq, gata, both_strands=False)


def test_parse_jaspar_roundtrip():
    text = (
        ">MA0035.4\tGATA1\n"
        "A  [ 0 90 0 90 90 0 ]\n"
        "C  [ 3 3 3 3 3 3 ]\n"
        "G  [ 90 3 3 3 3 90 ]\n"
        "T  [ 3 3 90 3 3 3 ]\n"
    )
    path = tempfile.mktemp(suffix=".jaspar")
    with open(path, "w") as fh:
        fh.write(text)
    try:
        motifs = parse_jaspar(path)
        assert len(motifs) == 1
        assert motifs[0].name == "GATA1"
        assert motifs[0].consensus() == "GATAAG"
    finally:
        os.remove(path)
