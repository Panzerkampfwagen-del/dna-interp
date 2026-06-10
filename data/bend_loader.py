"""BEND dataset loading, tokenization, and an offline synthetic generator.

Real mode loads the BEND enhancer/TF datasets and tokenizes with the DNABERT-2
BPE tokenizer. Synthetic mode builds enhancer-like sequences with planted
biology so the interpretability tools can be validated without any download.

The DNABERT-2 tokenizer is not available offline, so `KmerTokenizer` provides a
HuggingFace-compatible stand-in (non-overlapping k-mers over the ACGT alphabet)
that the dataset, model wrapper, and interp code all accept unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

BASES = "ACGT"

# Canonical motif consensus strings used by the synthetic generator. These are
# real transcription-factor binding consensuses; the same strings show up in the
# built-in JASPAR PWMs (data/jaspar.py), which is what makes the offline
# motif-scan validation meaningful.
PLANT_MOTIFS = {
    "GATA": "GATAAG",
    "TATA": "TATAAA",
    "SP1": "GGGGCGGGG",
    "EBOX": "CACGTG",
}


class KmerTokenizer:
    """Minimal HuggingFace-style tokenizer: non-overlapping k-mers + CLS/SEP.

    Implements just enough of the `AutoTokenizer` call contract for this project
    (`__call__`, `decode`, `convert_ids_to_tokens`, the special-token ids and
    `vocab_size`). It is a stand-in for the DNABERT-2 BPE tokenizer when offline.
    """

    def __init__(self, k: int = 3, model_max_length: int = 512) -> None:
        self.k = k
        self.model_max_length = model_max_length
        specials = ["[PAD]", "[CLS]", "[SEP]", "[UNK]"]
        kmers = ["".join(p) for p in product(BASES, repeat=k)]
        self._vocab = {tok: i for i, tok in enumerate(specials + kmers)}
        self._ids = {i: tok for tok, i in self._vocab.items()}
        self.pad_token_id = self._vocab["[PAD]"]
        self.cls_token_id = self._vocab["[CLS]"]
        self.sep_token_id = self._vocab["[SEP]"]
        self.unk_token_id = self._vocab["[UNK]"]

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def _tokenize_one(self, seq: str) -> tuple[list[int], list[tuple[int, int]]]:
        seq = seq.upper()
        ids = [self.cls_token_id]
        offsets = [(0, 0)]
        for i in range(0, len(seq) - self.k + 1, self.k):
            ids.append(self._vocab.get(seq[i : i + self.k], self.unk_token_id))
            offsets.append((i, i + self.k))
        ids.append(self.sep_token_id)
        offsets.append((0, 0))
        return ids, offsets

    def __call__(
        self,
        sequences: str | Sequence[str],
        padding: bool | str = True,
        truncation: bool = True,
        max_length: int | None = None,
        return_tensors: str | None = "pt",
        return_offsets_mapping: bool = False,
    ) -> dict:
        if isinstance(sequences, str):
            sequences = [sequences]
        max_length = max_length or self.model_max_length
        rows, offs = zip(*[self._tokenize_one(s) for s in sequences])
        rows, offs = list(rows), list(offs)
        if truncation:
            for j, (r, o) in enumerate(zip(rows, offs)):
                if len(r) > max_length:
                    rows[j] = r[: max_length - 1] + [self.sep_token_id]
                    offs[j] = o[: max_length - 1] + [(0, 0)]
        width = max(len(r) for r in rows) if padding else None
        if padding == "max_length":
            width = max_length
        input_ids, attention_mask, offset_mapping = [], [], []
        for r, o in zip(rows, offs):
            if width is not None:
                pad = width - len(r)
                input_ids.append(r + [self.pad_token_id] * pad)
                attention_mask.append([1] * len(r) + [0] * pad)
                offset_mapping.append(o + [(0, 0)] * pad)
            else:
                input_ids.append(r)
                attention_mask.append([1] * len(r))
                offset_mapping.append(o)
        out = {"input_ids": input_ids, "attention_mask": attention_mask}
        if return_offsets_mapping:
            out["offset_mapping"] = offset_mapping
        if return_tensors == "pt":
            out["input_ids"] = torch.tensor(out["input_ids"], dtype=torch.long)
            out["attention_mask"] = torch.tensor(out["attention_mask"], dtype=torch.long)
            if return_offsets_mapping:
                out["offset_mapping"] = torch.tensor(offset_mapping, dtype=torch.long)
        return out

    def convert_ids_to_tokens(self, ids: Iterable[int]) -> list[str]:
        return [self._ids.get(int(i), "[UNK]") for i in ids]

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        toks = self.convert_ids_to_tokens(ids)
        if skip_special_tokens:
            toks = [t for t in toks if not (t.startswith("[") and t.endswith("]"))]
        return "".join(toks)


class BendDataset(Dataset):
    """PyTorch dataset over DNA sequences for enhancer or TF-binding tasks.

    Each item is the dict the prompt specifies: tokenized ids, attention mask,
    label, the raw sequence string (kept for interpretability), and position ids.
    """

    def __init__(
        self,
        sequences: Sequence[str],
        labels: Sequence,
        tokenizer,
        max_length: int = 128,
        multilabel: bool = False,
    ) -> None:
        if len(sequences) != len(labels):
            raise ValueError("sequences and labels must have equal length")
        self.sequences = list(sequences)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.multilabel = multilabel

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq = self.sequences[idx]
        enc = self.tokenizer(
            seq,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        if self.multilabel:
            label = torch.as_tensor(self.labels[idx], dtype=torch.float)
        else:
            label = torch.as_tensor(self.labels[idx], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label,
            "sequence": seq,
            "position_ids": torch.arange(input_ids.shape[0], dtype=torch.long),
        }


def collate(batch: list[dict]) -> dict:
    """Stack tensor fields, keep raw sequences as a list."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "position_ids": torch.stack([b["position_ids"] for b in batch]),
        "sequence": [b["sequence"] for b in batch],
    }


def load_dnabert2_tokenizer():
    """Load the real DNABERT-2 BPE tokenizer (real mode, needs the cached model)."""
    from transformers import AutoTokenizer

    from config import DNABERT2_ID

    return AutoTokenizer.from_pretrained(DNABERT2_ID, trust_remote_code=True)


def load_bend_enhancer(split: str = "train") -> tuple[list[str], list[int]]:
    """Load the BEND enhancer-annotation dataset (real mode).

    Tries the installed `bend` package, then a local CSV at
    `data/raw/enhancer_<split>.csv` with `sequence,label` columns. Raises with
    guidance if neither is available so the caller can fall back to synthetic.
    """
    try:
        from bend.utils import data_utils  # type: ignore

        ds = data_utils.load_dataset("enhancer_annotation", split=split)
        seqs = [r["sequence"] for r in ds]
        labels = [int(r["label"]) for r in ds]
        return seqs, labels
    except Exception:
        pass

    import csv

    from config import RAW

    path = RAW / f"enhancer_{split}.csv"
    if path.exists():
        seqs, labels = [], []
        with open(path) as fh:
            for row in csv.DictReader(fh):
                seqs.append(row["sequence"])
                labels.append(int(row["label"]))
        return seqs, labels

    raise FileNotFoundError(
        "BEND enhancer data unavailable. Install `bend` (needs network) or place "
        f"a CSV at {path} with columns sequence,label. For offline development "
        "use make_synthetic_enhancer_dataset()."
    )


def load_nt_enhancer(split: str = "train") -> tuple[list[str], list[int]]:
    """Load the Nucleotide Transformer downstream 'enhancers' task (real data).

    Used as the accessible stand-in for BEND enhancer annotation: real,
    experimentally-derived 200 bp sequences, binary enhancer label, balanced
    (14968 train / 400 test). Needs network on first call, then HF-cached.
    """
    from datasets import load_dataset

    ds = load_dataset(
        "InstaDeepAI/nucleotide_transformer_downstream_tasks",
        data_files={"train": "enhancers/train.parquet", "test": "enhancers/test.parquet"},
    )
    split = "test" if split in ("test", "valid", "validation") else "train"
    d = ds[split]
    return list(d["sequence"]), [int(x) for x in d["label"]]


@dataclass
class SyntheticDataset:
    """Synthetic enhancer dataset with planted biology and ground-truth metadata.

    `motif_positions[i]` maps a motif name to the base index where it was planted
    in sequence i (absent if not planted). This ground truth is what the patching
    and motif-scan validations check their discoveries against.
    """

    sequences: list[str]
    labels: list[int]
    motif_positions: list[dict[str, int]] = field(default_factory=list)
    length: int = 200


def make_synthetic_enhancer_dataset(
    n: int = 2000,
    length: int = 200,
    seed: int = 0,
    pos_gc: float = 0.62,
    neg_gc: float = 0.42,
    motifs: Sequence[str] = ("GATA", "TATA", "SP1"),
) -> SyntheticDataset:
    """Build a balanced enhancer dataset with controllable, known signal.

    Positives carry a GC bias and one or more planted TF motifs at known
    positions. Negatives are lower-GC background with no planted motifs. Because
    the ground truth is known, the interpretability tools can be scored: a good
    patcher localizes to planted-motif positions, a good probe recovers GC, and a
    good motif scan correlates head attention with the planted motif track.
    """
    rng = np.random.default_rng(seed)

    def background(gc: float) -> list[str]:
        # P(G)=P(C)=gc/2, P(A)=P(T)=(1-gc)/2
        probs = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]  # A C G T
        return list(rng.choice(list(BASES), size=length, p=probs))

    sequences: list[str] = []
    labels: list[int] = []
    motif_positions: list[dict[str, int]] = []

    for i in range(n):
        is_pos = i % 2 == 0
        seq = background(pos_gc if is_pos else neg_gc)
        planted: dict[str, int] = {}
        if is_pos:
            # plant motifs at non-overlapping positions in the central half so the
            # recorded positions stay valid ground truth and effects are localizable
            occupied: list[tuple[int, int]] = []
            lo = length // 4
            for name in motifs:
                consensus = PLANT_MOTIFS[name]
                hi = length - length // 4 - len(consensus)
                for _ in range(20):
                    start = int(rng.integers(lo, hi))
                    end = start + len(consensus)
                    if all(end <= a or start >= b for a, b in occupied):
                        seq[start:end] = list(consensus)
                        occupied.append((start, end))
                        planted[name] = start
                        break
        sequences.append("".join(seq))
        labels.append(1 if is_pos else 0)
        motif_positions.append(planted)

    order = rng.permutation(n)
    return SyntheticDataset(
        sequences=[sequences[i] for i in order],
        labels=[int(labels[i]) for i in order],
        motif_positions=[motif_positions[i] for i in order],
        length=length,
    )


@dataclass
class SyntheticTFDataset:
    """Synthetic TF-binding dataset: multi-label presence of planted TF motifs."""

    sequences: list[str]
    labels: np.ndarray  # [n, n_tf] float, 1 if that TF's motif is planted
    tf_names: list[str]
    motif_positions: list[dict[str, int]]
    length: int = 101


def make_synthetic_tf_binding_dataset(
    n: int = 2000,
    length: int = 101,
    seed: int = 0,
    tf_motifs: Sequence[str] = ("GATA", "TATA", "SP1", "EBOX"),
    gc: float = 0.5,
    plant_prob: float = 0.5,
) -> SyntheticTFDataset:
    """Each sequence independently plants each TF's motif with `plant_prob`.

    The label vector marks which TFs are present, giving a multi-label task whose
    ground truth (which motif sits where) lets the interp tools be scored exactly
    as in the enhancer case.
    """
    rng = np.random.default_rng(seed)
    tf_names = list(tf_motifs)
    probs = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]  # A C G T

    sequences: list[str] = []
    labels = np.zeros((n, len(tf_names)), dtype=np.float32)
    motif_positions: list[dict[str, int]] = []

    for i in range(n):
        seq = list(rng.choice(list(BASES), size=length, p=probs))
        occupied: list[tuple[int, int]] = []
        planted: dict[str, int] = {}
        for t, name in enumerate(tf_names):
            if rng.random() >= plant_prob:
                continue
            consensus = PLANT_MOTIFS[name]
            hi = length - len(consensus)
            for _ in range(20):
                start = int(rng.integers(0, hi))
                end = start + len(consensus)
                if all(end <= a or start >= b for a, b in occupied):
                    seq[start:end] = list(consensus)
                    occupied.append((start, end))
                    planted[name] = start
                    labels[i, t] = 1.0
                    break
        sequences.append("".join(seq))
        motif_positions.append(planted)

    return SyntheticTFDataset(sequences, labels, tf_names, motif_positions, length)


def load_bend_tf_binding(split: str = "train") -> tuple[list[str], np.ndarray]:
    """Load the BEND TF-binding dataset (real mode).

    Tries the installed `bend` package, then a local .npz at
    `data/raw/tf_binding_<split>.npz` with arrays `sequences` and `labels`.
    """
    try:
        from bend.utils import data_utils  # type: ignore

        ds = data_utils.load_dataset("chromatin_accessibility", split=split)
        seqs = [r["sequence"] for r in ds]
        labels = np.array([r["label"] for r in ds], dtype=np.float32)
        return seqs, labels
    except Exception:
        pass

    from config import RAW

    path = RAW / f"tf_binding_{split}.npz"
    if path.exists():
        blob = np.load(path, allow_pickle=True)
        return list(blob["sequences"]), blob["labels"].astype(np.float32)

    raise FileNotFoundError(
        "BEND TF-binding data unavailable. Install `bend` or place an .npz at "
        f"{path} with arrays sequences,labels. For offline development use "
        "make_synthetic_tf_binding_dataset()."
    )


def train_val_test_split(
    n: int, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproducible index split."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    return idx[n_val + n_test :], idx[:n_val], idx[n_val : n_val + n_test]
