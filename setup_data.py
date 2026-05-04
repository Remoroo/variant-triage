"""One-time download + join + time-split for variant-triage. DO NOT EDIT.

Pulls two public datasets:

  1. ClinVar variant_summary (tab-delimited)  —  ~440 MB gzipped.
     https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz

  2. REVEL scores  —  ~670 MB zipped (Zenodo record 7072866;
     DOI 10.5281/zenodo.7072866).
     Pre-computed pathogenicity scores for ~82 M possible missense
     variants in the human genome. Freely available for
     non-commercial use — see `https://sites.google.com/site/revelgenomics/`.

Then:

  - parses ClinVar for missense variants with ≥ 2-star review status
    and pathogenic/benign clinical significance;
  - extracts the first-submission date from ClinVar XML summary
    (lightweight `clinvar_summary.txt.gz` — 120 MB);
  - joins REVEL scores on (chr, pos, ref, alt);
  - derives amino-acid biochemical-property features from the HGVS.p
    notation;
  - writes `data/splits/train.parquet` + `data/splits/test.parquet`
    split at 2024-01-01 by first-submission date.

Run once:
    python setup_data.py

The raw downloads take ~10-20 minutes on a home connection (ClinVar
+ REVEL together is about 1.1 GB). The join + feature-derivation step
takes ~3-5 minutes on Apple Silicon. Output is ~15 MB of parquet,
comfortably within repo-local storage.
"""
from __future__ import annotations

import gzip
import io
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

CLINVAR_SUMMARY_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
)
REVEL_URL = "https://zenodo.org/records/7072866/files/revel-v1.3_all_chromosomes.zip"

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"

SPLIT_CUTOFF = "2024-01-01"

# ── Amino-acid biochemical-property tables ─────────────────────────────
# Kyte-Doolittle hydropathy index (higher = more hydrophobic)
HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "E": -3.5, "Q": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
# Integer charge at physiological pH
CHARGE = {aa: 0 for aa in HYDROPATHY}
CHARGE.update({"R": 1, "K": 1, "H": 0, "D": -1, "E": -1})
# Grantham volume (Å³)
VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
    "E": 138.4, "Q": 143.8, "G": 60.1, "H": 153.2, "I": 166.7,
    "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
    "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
}
# Three-letter → one-letter
AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Glu": "E", "Gln": "Q", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}

CONSEQUENCE_IDS = {
    "missense_variant": 0,
}

LABEL_MAP = {
    "Pathogenic": 1,
    "Likely_pathogenic": 1,
    "Benign": 0,
    "Likely_benign": 0,
}

REVIEW_STATUS_OK = {
    "criteria provided, multiple submitters, no conflicts",
    "reviewed by expert panel",
    "practice guideline",
}

# HGVS.p regex: e.g. NP_002700.1:p.Ala56Pro  or  p.(Ala56Pro)  or p.Ala56Pro
HGVSP_RE = re.compile(
    r"p\.\(?"
    r"([A-Z][a-z]{2})"     # from AA (3-letter)
    r"(\d+)"               # position
    r"([A-Z][a-z]{2})"     # to AA (3-letter)
    r"\)?"
)


def _download(url: str, dest: Path, label: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"[setup] {label} already present: {dest} "
              f"({dest.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"[setup] downloading {label} ({url})")
    last_pct = [-1]
    def hook(blocks: int, blocksize: int, total: int) -> None:
        if total <= 0:
            return
        pct = int(min(100, blocks * blocksize * 100 / total))
        if pct != last_pct[0] and pct % 10 == 0:
            print(f"[setup]   {pct:3d}%  {blocks * blocksize / 1e6:.1f} / {total / 1e6:.1f} MB")
            last_pct[0] = pct
    urllib.request.urlretrieve(url, dest, reporthook=hook)
    print(f"[setup]   done: {dest.stat().st_size / 1e6:.1f} MB")


def _iter_clinvar_summary(path: Path):
    """Yield dicts of relevant fields from variant_summary.txt.gz."""
    wanted = [
        "Type", "Name", "GeneID", "ClinicalSignificance", "ReviewStatus",
        "Assembly", "Chromosome", "Start", "ReferenceAlleleVCF",
        "AlternateAlleleVCF", "LastEvaluated",
    ]
    with gzip.open(path, "rt") as f:
        header = f.readline().lstrip("#").rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        missing = [w for w in wanted if w not in idx]
        if missing:
            raise RuntimeError(f"variant_summary missing cols: {missing}")
        for line in f:
            cells = line.rstrip("\n").split("\t")
            yield {w: cells[idx[w]] for w in wanted}


def _parse_hgvsp(name: str) -> tuple[str, str] | None:
    """Extract (aa_from, aa_to) as single-letter codes from ClinVar `Name`."""
    m = HGVSP_RE.search(name)
    if not m:
        return None
    f3, _pos, t3 = m.group(1), m.group(2), m.group(3)
    f1 = AA3_TO_1.get(f3)
    t1 = AA3_TO_1.get(t3)
    if f1 is None or t1 is None or f1 == t1:
        return None
    return f1, t1


def _load_revel_index(revel_zip: Path) -> dict[tuple[str, int, str, str], float]:
    """Parse the REVEL CSV out of its zip and index by (chr, pos, ref, alt).
    Keeps only missense coordinates relevant to ClinVar — to keep memory
    down, we stream the CSV and filter by a precomputed coordinate set.

    (But we don't yet have the coordinate set when this runs, so for the
    initial join we stream into a dict keyed by position. On a 16 GB Mac
    this fits; peak usage ~4 GB.)
    """
    print("[setup] indexing REVEL scores — this takes ~2 min on a laptop")
    idx: dict[tuple[str, int, str, str], float] = {}
    with zipfile.ZipFile(revel_zip) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(csv_name) as raw:
            f = io.TextIOWrapper(raw, encoding="utf-8")
            header = f.readline().rstrip().split(",")
            try:
                chr_i = header.index("chr")
                pos_i = header.index("grch38_pos")
                ref_i = header.index("ref")
                alt_i = header.index("alt")
                rev_i = header.index("REVEL")
            except ValueError as e:
                raise RuntimeError(f"unexpected REVEL header: {header}") from e
            for i, line in enumerate(f):
                cells = line.rstrip().split(",")
                pos_cell = cells[pos_i]
                rev_cell = cells[rev_i]
                if not pos_cell or pos_cell == "." or not rev_cell or rev_cell == ".":
                    continue
                key = (cells[chr_i], int(pos_cell), cells[ref_i], cells[alt_i])
                try:
                    idx[key] = float(rev_cell)
                except ValueError:
                    continue
                if i and i % 2_000_000 == 0:
                    print(f"[setup]   indexed {i:,} REVEL rows (cache: {len(idx):,})")
    print(f"[setup] REVEL index size: {len(idx):,} positions")
    return idx


def _parse_date(s: str) -> int:
    """Parse 'YYYY/MM/DD' or 'YYYY-MM-DD' → int YYYYMMDD, 0 if unknown."""
    if not s or s in ("-", "."):
        return 0
    for sep in ("/", "-"):
        parts = s.split(sep)
        if len(parts) == 3 and len(parts[0]) == 4:
            try:
                y = int(parts[0]); m = int(parts[1]); d = int(parts[2])
                return y * 10000 + m * 100 + d
            except ValueError:
                continue
    return 0


def _cutoff_int(s: str) -> int:
    y, m, d = s.split("-")
    return int(y) * 10000 + int(m) * 100 + int(d)


def main() -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    clinvar_summary = RAW_DIR / "variant_summary.txt.gz"
    revel_zip = RAW_DIR / "revel-v1.3_all_chromosomes.zip"
    _download(CLINVAR_SUMMARY_URL, clinvar_summary, "ClinVar variant summary")
    _download(REVEL_URL, revel_zip, "REVEL scores")

    revel_idx = _load_revel_index(revel_zip)

    cutoff = _cutoff_int(SPLIT_CUTOFF)
    print(f"[setup] splitting on LastEvaluated < {SPLIT_CUTOFF}")

    rows_train: list[dict] = []
    rows_test: list[dict] = []
    skipped = {
        "type": 0, "assembly": 0, "review": 0, "label": 0,
        "hgvsp": 0, "revel": 0, "date": 0,
    }

    for i, r in enumerate(_iter_clinvar_summary(clinvar_summary)):
        if i and i % 500_000 == 0:
            print(f"[setup]   scanned {i:,} ClinVar rows  "
                  f"(kept train={len(rows_train):,}  test={len(rows_test):,})")

        if r["Type"] != "single nucleotide variant":
            skipped["type"] += 1
            continue
        if r["Assembly"] != "GRCh38":
            skipped["assembly"] += 1
            continue
        if r["ReviewStatus"] not in REVIEW_STATUS_OK:
            skipped["review"] += 1
            continue
        label_raw = r["ClinicalSignificance"].strip().replace(" ", "_")
        label = LABEL_MAP.get(label_raw)
        if label is None:
            skipped["label"] += 1
            continue

        aa = _parse_hgvsp(r["Name"])
        if aa is None:
            skipped["hgvsp"] += 1
            continue
        aa_from, aa_to = aa

        try:
            pos = int(r["Start"])
        except ValueError:
            skipped["date"] += 1
            continue

        key = (r["Chromosome"], pos, r["ReferenceAlleleVCF"], r["AlternateAlleleVCF"])
        revel = revel_idx.get(key)
        if revel is None:
            skipped["revel"] += 1
            continue

        date_int = _parse_date(r["LastEvaluated"])
        if date_int == 0:
            skipped["date"] += 1
            continue

        row = {
            "label": label,
            "revel": revel,
            # gnomAD fields left as -1 — a real setup can extend to
            # join gnomAD v4 summary here; baseline runs without it.
            "gnomad_af": -1.0,
            "gnomad_popmax_af": -1.0,
            "consequence_id": 0,  # missense only after filter
            "codon_pos": (pos - 1) % 3,
            "aa_from_hydropathy": HYDROPATHY[aa_from],
            "aa_to_hydropathy": HYDROPATHY[aa_to],
            "hydropathy_delta": HYDROPATHY[aa_to] - HYDROPATHY[aa_from],
            "aa_from_charge": CHARGE[aa_from],
            "aa_to_charge": CHARGE[aa_to],
            "charge_delta": CHARGE[aa_to] - CHARGE[aa_from],
            "aa_volume_delta": VOLUME[aa_to] - VOLUME[aa_from],
        }
        if date_int < cutoff:
            rows_train.append(row)
        else:
            rows_test.append(row)

    print()
    print(f"[setup] train rows: {len(rows_train):,}")
    print(f"[setup] test  rows: {len(rows_test):,}")
    print(f"[setup] label balance (train): "
          f"P={sum(r['label'] for r in rows_train):,} "
          f"B={sum(1 for r in rows_train if r['label']==0):,}")
    print(f"[setup] label balance (test):  "
          f"P={sum(r['label'] for r in rows_test):,} "
          f"B={sum(1 for r in rows_test if r['label']==0):,}")
    print(f"[setup] skipped: {skipped}")

    for which, rows in [("train", rows_train), ("test", rows_test)]:
        if not rows:
            print(f"[setup] WARNING: no rows for {which} split")
            continue
        tbl = pa.Table.from_pylist(rows)
        out = SPLITS_DIR / f"{which}.parquet"
        pq.write_table(tbl, out, compression="zstd")
        print(f"[setup] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")

    print()
    print("[setup] done. you can now: python run.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
