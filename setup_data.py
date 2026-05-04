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
ALPHAMISSENSE_URL = (
    "https://storage.googleapis.com/dm_alphamissense/AlphaMissense_hg38.tsv.gz"
)

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

# Grantham (1974) amino-acid distance matrix — composition + polarity +
# molecular volume. Only the non-zero upper triangle; we symmetrize in
# `grantham_distance()`. Higher distance = more disruptive substitution.
# Source: Grantham R, *Science* 185:862–864 (1974).
GRANTHAM_DATA = {
    ("S", "R"): 110, ("L", "R"): 102, ("P", "R"): 103, ("T", "R"): 71,
    ("A", "R"): 112, ("V", "R"): 96, ("G", "R"): 125, ("I", "R"): 97,
    ("F", "R"): 97, ("Y", "R"): 77, ("C", "R"): 180, ("H", "R"): 29,
    ("Q", "R"): 43, ("N", "R"): 86, ("K", "R"): 26, ("D", "R"): 96,
    ("E", "R"): 54, ("M", "R"): 91, ("W", "R"): 101,
    ("L", "S"): 145, ("P", "S"): 74, ("T", "S"): 58, ("A", "S"): 99,
    ("V", "S"): 124, ("G", "S"): 56, ("I", "S"): 142, ("F", "S"): 155,
    ("Y", "S"): 144, ("C", "S"): 112, ("H", "S"): 89, ("Q", "S"): 68,
    ("N", "S"): 46, ("K", "S"): 121, ("D", "S"): 65, ("E", "S"): 80,
    ("M", "S"): 135, ("W", "S"): 177,
    ("P", "L"): 98, ("T", "L"): 92, ("A", "L"): 96, ("V", "L"): 32,
    ("G", "L"): 138, ("I", "L"): 5, ("F", "L"): 22, ("Y", "L"): 36,
    ("C", "L"): 198, ("H", "L"): 99, ("Q", "L"): 113, ("N", "L"): 153,
    ("K", "L"): 107, ("D", "L"): 172, ("E", "L"): 138, ("M", "L"): 15,
    ("W", "L"): 61,
    ("T", "P"): 38, ("A", "P"): 27, ("V", "P"): 68, ("G", "P"): 42,
    ("I", "P"): 95, ("F", "P"): 114, ("Y", "P"): 110, ("C", "P"): 169,
    ("H", "P"): 77, ("Q", "P"): 76, ("N", "P"): 91, ("K", "P"): 103,
    ("D", "P"): 108, ("E", "P"): 93, ("M", "P"): 87, ("W", "P"): 147,
    ("A", "T"): 58, ("V", "T"): 69, ("G", "T"): 59, ("I", "T"): 89,
    ("F", "T"): 103, ("Y", "T"): 92, ("C", "T"): 149, ("H", "T"): 47,
    ("Q", "T"): 42, ("N", "T"): 65, ("K", "T"): 78, ("D", "T"): 85,
    ("E", "T"): 65, ("M", "T"): 81, ("W", "T"): 128,
    ("V", "A"): 64, ("G", "A"): 60, ("I", "A"): 94, ("F", "A"): 113,
    ("Y", "A"): 112, ("C", "A"): 195, ("H", "A"): 86, ("Q", "A"): 91,
    ("N", "A"): 111, ("K", "A"): 106, ("D", "A"): 126, ("E", "A"): 107,
    ("M", "A"): 84, ("W", "A"): 148,
    ("G", "V"): 109, ("I", "V"): 29, ("F", "V"): 50, ("Y", "V"): 55,
    ("C", "V"): 192, ("H", "V"): 84, ("Q", "V"): 96, ("N", "V"): 133,
    ("K", "V"): 97, ("D", "V"): 152, ("E", "V"): 121, ("M", "V"): 21,
    ("W", "V"): 88,
    ("I", "G"): 135, ("F", "G"): 153, ("Y", "G"): 147, ("C", "G"): 159,
    ("H", "G"): 98, ("Q", "G"): 87, ("N", "G"): 80, ("K", "G"): 127,
    ("D", "G"): 94, ("E", "G"): 98, ("M", "G"): 127, ("W", "G"): 184,
    ("F", "I"): 21, ("Y", "I"): 33, ("C", "I"): 198, ("H", "I"): 94,
    ("Q", "I"): 109, ("N", "I"): 149, ("K", "I"): 102, ("D", "I"): 168,
    ("E", "I"): 134, ("M", "I"): 10, ("W", "I"): 61,
    ("Y", "F"): 22, ("C", "F"): 205, ("H", "F"): 100, ("Q", "F"): 116,
    ("N", "F"): 158, ("K", "F"): 102, ("D", "F"): 177, ("E", "F"): 140,
    ("M", "F"): 28, ("W", "F"): 40,
    ("C", "Y"): 194, ("H", "Y"): 83, ("Q", "Y"): 99, ("N", "Y"): 143,
    ("K", "Y"): 85, ("D", "Y"): 160, ("E", "Y"): 122, ("M", "Y"): 36,
    ("W", "Y"): 37,
    ("H", "C"): 174, ("Q", "C"): 154, ("N", "C"): 139, ("K", "C"): 202,
    ("D", "C"): 154, ("E", "C"): 170, ("M", "C"): 196, ("W", "C"): 215,
    ("Q", "H"): 24, ("N", "H"): 68, ("K", "H"): 32, ("D", "H"): 81,
    ("E", "H"): 40, ("M", "H"): 87, ("W", "H"): 115,
    ("N", "Q"): 46, ("K", "Q"): 53, ("D", "Q"): 61, ("E", "Q"): 29,
    ("M", "Q"): 101, ("W", "Q"): 130,
    ("K", "N"): 94, ("D", "N"): 23, ("E", "N"): 42, ("M", "N"): 142,
    ("W", "N"): 174,
    ("D", "K"): 101, ("E", "K"): 56, ("M", "K"): 95, ("W", "K"): 110,
    ("E", "D"): 45, ("M", "D"): 160, ("W", "D"): 181,
    ("M", "E"): 126, ("W", "E"): 152,
    ("M", "W"): 67,
}
# Build a symmetric lookup; self-distance = 0.
GRANTHAM: dict[tuple[str, str], int] = {}
for (a, b), v in GRANTHAM_DATA.items():
    GRANTHAM[(a, b)] = v
    GRANTHAM[(b, a)] = v
for a in HYDROPATHY:
    GRANTHAM[(a, a)] = 0

# BLOSUM62 substitution matrix — log-odds of AA substitutions in
# conserved protein alignments. Higher = more commonly observed.
# Encoded as the standard 20×20 matrix (Henikoff & Henikoff 1992).
_BLOSUM62_ROWS = """
#  A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""

def _load_blosum62() -> dict[tuple[str, str], int]:
    """Parse the embedded BLOSUM62 table into a {(from, to): score} dict.
    The first non-empty line is a `#`-prefixed column-letter header."""
    non_empty = [ln for ln in _BLOSUM62_ROWS.strip().splitlines() if ln.strip()]
    header_tokens = non_empty[0].lstrip("#").split()
    m: dict[tuple[str, str], int] = {}
    for row in non_empty[1:]:
        cells = row.split()
        a = cells[0]
        for b, v in zip(header_tokens, cells[1:]):
            m[(a, b)] = int(v)
    return m

BLOSUM62 = _load_blosum62()

# Chromosome → integer id (X/Y/MT are mapped to 23/24/25 to stay
# numeric for tree models).
CHROM_ID = {str(i): i for i in range(1, 23)}
CHROM_ID.update({"X": 23, "Y": 24, "MT": 25, "M": 25})
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
    # Unambiguous single-direction classifications. Combined records
    # (e.g. "Pathogenic/Likely pathogenic") share the same direction so
    # we keep them; mixed-direction records ("Conflicting ...") are
    # correctly dropped.
    "Pathogenic": 1,
    "Likely_pathogenic": 1,
    "Pathogenic/Likely_pathogenic": 1,
    "Pathogenic,_low_penetrance": 1,
    "Likely_pathogenic,_low_penetrance": 1,
    "Benign": 0,
    "Likely_benign": 0,
    "Benign/Likely_benign": 0,
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
        "Type", "Name", "GeneID", "GeneSymbol", "ClinicalSignificance", "ReviewStatus",
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


def _annotate_with_revel(
    candidates: dict[tuple[str, int, str, str], float],
    revel_zip: Path,
) -> int:
    """Stream REVEL once, fill in REVEL scores for any key already
    present in `candidates`. Mutates `candidates` in place.

    The REVEL zip contains one extension-less file
    (`revel_with_transcript_ids`, 6.5 GB CSV). The same (chr, pos, ref,
    alt) key can appear multiple times — once per Ensembl transcript —
    but the REVEL score is identical across rows so the first hit wins.

    Peak memory stays bounded by the ClinVar candidate set (a few
    hundred thousand rows, tens of MB) rather than the full 82 M-row
    REVEL corpus.
    """
    print("[setup] streaming REVEL — this takes ~5-8 min on a laptop")
    matched = 0
    with zipfile.ZipFile(revel_zip) as zf:
        # REVEL v1.3 ships a single extension-less data file inside
        # the zip; pick whichever file is largest so we're robust to
        # future repackagings.
        names = [
            n for n in zf.namelist()
            if not n.endswith("/") and not zf.getinfo(n).is_dir()
        ]
        if not names:
            raise RuntimeError(f"no data files inside {revel_zip}")
        names.sort(key=lambda n: zf.getinfo(n).file_size, reverse=True)
        data_name = names[0]
        print(f"[setup]   reading `{data_name}` "
              f"({zf.getinfo(data_name).file_size / 1e9:.2f} GB uncompressed)")

        with zf.open(data_name) as raw:
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
                try:
                    key = (cells[chr_i], int(pos_cell), cells[ref_i], cells[alt_i])
                except ValueError:
                    continue
                existing = candidates.get(key)
                if existing is None or existing != -999.0:
                    continue  # already annotated (dupe) or not a candidate
                try:
                    candidates[key] = float(rev_cell)
                    matched += 1
                except ValueError:
                    continue
                if i and i % 5_000_000 == 0:
                    print(f"[setup]   scanned {i:,} REVEL rows  "
                          f"({matched:,} candidates annotated so far)")

    print(f"[setup] REVEL pass done: {matched:,} / {len(candidates):,} "
          f"ClinVar candidates annotated")
    return matched


def _load_gnomad_constraint(path: Path) -> dict[str, tuple[float, float, float, float, float]]:
    """Read gnomAD v4.1 per-gene constraint TSV. Returns {gene_symbol:
    (lof_pLI, lof_oe, mis_oe, mis_z, lof_z)} from the canonical/MANE row."""
    if not path.exists():
        return {}
    out: dict[str, tuple[float, float, float, float, float]] = {}
    with open(path, "r") as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        gi = idx["gene"]; mane = idx.get("mane_select"); canon = idx.get("canonical")
        cols = ["lof.pLI", "lof.oe", "mis.oe", "mis.z_score", "lof.z_score"]
        ci = [idx[c] for c in cols]
        def _f(s: str) -> float:
            try:
                return float(s)
            except (ValueError, TypeError):
                return float("nan")
        for line in f:
            cells = line.rstrip("\n").split("\t")
            if len(cells) <= max(ci):
                continue
            gene = cells[gi]
            is_mane = mane is not None and cells[mane] == "true"
            is_canon = canon is not None and cells[canon] == "true"
            existing = out.get(gene)
            # Prefer MANE > canonical > first-seen
            tup = tuple(_f(cells[i]) for i in ci)
            if existing is None or is_mane or (is_canon and gene not in out):
                out[gene] = tup
    return out


def _annotate_with_alphamissense(
    candidates: dict[tuple[str, int, str, str], float],
    am_tsv: Path,
) -> int:
    """Stream AlphaMissense hg38 TSV (gz), fill candidate keys.

    AlphaMissense uses 'chrN' prefix; ClinVar uses bare 'N'. Strip 'chr'.
    Same (chr,pos,ref,alt) can repeat across transcripts; first hit wins.
    """
    print("[setup] streaming AlphaMissense…")
    matched = 0
    with gzip.open(am_tsv, "rt") as f:
        header = None
        for line in f:
            if line.startswith("#CHROM"):
                header = line.lstrip("#").rstrip("\n").split("\t")
                break
            if line.startswith("#") or not line.strip():
                continue
        if header is None:
            raise RuntimeError("AlphaMissense TSV missing #CHROM header")
        chr_i = header.index("CHROM")
        pos_i = header.index("POS")
        ref_i = header.index("REF")
        alt_i = header.index("ALT")
        am_i = header.index("am_pathogenicity")
        for i, line in enumerate(f):
            cells = line.rstrip("\n").split("\t")
            if len(cells) <= am_i:
                continue
            chrom = cells[chr_i]
            if chrom.startswith("chr"):
                chrom = chrom[3:]
            try:
                pos = int(cells[pos_i])
            except ValueError:
                continue
            key = (chrom, pos, cells[ref_i], cells[alt_i])
            existing = candidates.get(key)
            if existing is None or existing != -999.0:
                continue
            try:
                candidates[key] = float(cells[am_i])
                matched += 1
            except ValueError:
                continue
            if i and i % 10_000_000 == 0:
                print(f"[setup]   AM scanned {i:,} rows ({matched:,} matched)")
    return matched


def _parse_date(s: str) -> int:
    """Parse ClinVar LastEvaluated → int YYYYMMDD, 0 if unknown.

    Handles all three formats we've seen in ClinVar dumps:
    - `YYYY-MM-DD` and `YYYY/MM/DD` (old tab-delimited style)
    - `"Jun 11, 2025"` (current variant_summary.txt.gz style)
    """
    if not s or s in ("-", "."):
        return 0
    s = s.strip()
    # numeric formats first (cheap)
    for sep in ("/", "-"):
        parts = s.split(sep)
        if len(parts) == 3 and len(parts[0]) == 4:
            try:
                y = int(parts[0]); m = int(parts[1]); d = int(parts[2])
                return y * 10000 + m * 100 + d
            except ValueError:
                continue
    # month-name format: "Jun 11, 2025" / "June 11, 2025"
    from datetime import datetime
    for fmt in ("%b %d, %Y", "%B %d, %Y"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.year * 10000 + dt.month * 100 + dt.day
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
    am_tsv = RAW_DIR / "AlphaMissense_hg38.tsv.gz"
    _download(CLINVAR_SUMMARY_URL, clinvar_summary, "ClinVar variant summary")
    _download(REVEL_URL, revel_zip, "REVEL scores")
    _download(ALPHAMISSENSE_URL, am_tsv, "AlphaMissense hg38 scores")

    cutoff = _cutoff_int(SPLIT_CUTOFF)

    # ── Pass 1: build ClinVar candidate dict ──────────────────────
    # Sentinel -999.0 means "candidate awaiting REVEL annotation";
    # _annotate_with_revel swaps in the real score.
    print(f"[setup] scanning ClinVar (splitting on LastEvaluated < {SPLIT_CUTOFF})")
    candidate_revel: dict[tuple[str, int, str, str], float] = {}
    candidate_meta: dict[tuple[str, int, str, str], dict] = {}
    skipped = {
        "type": 0, "assembly": 0, "review": 0, "label": 0,
        "hgvsp": 0, "date": 0, "pos": 0,
    }

    for i, r in enumerate(_iter_clinvar_summary(clinvar_summary)):
        if i and i % 500_000 == 0:
            print(f"[setup]   scanned {i:,} ClinVar rows  "
                  f"(candidates so far: {len(candidate_meta):,})")

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
            skipped["pos"] += 1
            continue
        date_int = _parse_date(r["LastEvaluated"])
        if date_int == 0:
            skipped["date"] += 1
            continue

        key = (r["Chromosome"], pos, r["ReferenceAlleleVCF"], r["AlternateAlleleVCF"])
        # If a variant re-appears with different review/date, keep the
        # most-recently-reviewed record so the test split has the
        # freshest label decisions.
        existing = candidate_meta.get(key)
        if existing is not None and existing["date_int"] >= date_int:
            continue

        candidate_revel[key] = -999.0  # sentinel
        try:
            gene_id = int(r["GeneID"]) if r["GeneID"] not in ("", "-", ".") else 0
        except ValueError:
            gene_id = 0
        candidate_meta[key] = {
            "label": label,
            "date_int": date_int,
            "aa_from": aa_from,
            "aa_to": aa_to,
            "codon_pos": (pos - 1) % 3,
            "chrom": r["Chromosome"],
            "pos": pos,
            "gene_id": gene_id,
            "gene_symbol": r["GeneSymbol"],
        }

    print(f"[setup] ClinVar candidates awaiting REVEL: {len(candidate_meta):,}")
    print(f"[setup] skipped (ClinVar pass): {skipped}")

    # ── Pass 2: stream REVEL once, annotate candidates ───────────-
    matched = _annotate_with_revel(candidate_revel, revel_zip)
    if matched == 0:
        print("[setup] ERROR: no REVEL annotations matched — REVEL file format may have changed")
        return 1

    # ── Pass 2b: stream AlphaMissense, annotate candidates ───────-
    candidate_am: dict[tuple[str, int, str, str], float] = {k: -999.0 for k in candidate_meta}
    am_matched = _annotate_with_alphamissense(candidate_am, am_tsv)
    print(f"[setup] AlphaMissense matched {am_matched:,} / {len(candidate_meta):,}")

    # ── Pass 2c: gnomAD per-gene constraint metrics ───────────────-
    constraint = _load_gnomad_constraint(RAW_DIR / "gnomad_constraint.tsv")
    print(f"[setup] gnomAD constraint loaded for {len(constraint):,} genes")

    # ── Pass 3: emit train/test parquet splits ──────────────────-
    rows_train: list[dict] = []
    rows_test: list[dict] = []
    skipped_revel = 0

    for key, meta in candidate_meta.items():
        revel = candidate_revel[key]
        if revel == -999.0:
            # REVEL score is required during JOIN only (keeps the split
            # aligned with the published REVEL reference benchmark),
            # but is NOT emitted as a feature.
            skipped_revel += 1
            continue
        am_score = candidate_am.get(key, -999.0)
        if am_score == -999.0:
            am_score = float("nan")
        cmetrics = constraint.get(meta["gene_symbol"], (float("nan"),) * 5)
        aa_from = meta["aa_from"]
        aa_to = meta["aa_to"]
        row = {
            "label": meta["label"],
            # biochemistry of the substitution
            "aa_from_hydropathy": HYDROPATHY[aa_from],
            "aa_to_hydropathy": HYDROPATHY[aa_to],
            "hydropathy_delta": HYDROPATHY[aa_to] - HYDROPATHY[aa_from],
            "aa_from_charge": CHARGE[aa_from],
            "aa_to_charge": CHARGE[aa_to],
            "charge_delta": CHARGE[aa_to] - CHARGE[aa_from],
            "aa_volume_delta": VOLUME[aa_to] - VOLUME[aa_from],
            "grantham_distance": GRANTHAM.get((aa_from, aa_to), 0),
            "blosum62_score": BLOSUM62.get((aa_from, aa_to), 0),
            # positional / genomic context
            "codon_pos": meta["codon_pos"],
            "chrom_id": CHROM_ID.get(meta["chrom"], 0),
            "pos_mod1000": meta["pos"] % 1000,
            # reserved: molecular-consequence id (0 = missense; filter
            # currently keeps only missense, but leaving the column in
            # means future extensions can add non-missense variants
            # without a schema migration).
            "consequence_id": 0,
            # Allowed extension features (program.md §"Allowed extension features"):
            "revel_score": float(revel),
            "alphamissense_score": float(am_score),
            "gene_id": meta["gene_id"],
            "gene_lof_pLI": float(cmetrics[0]),
            "gene_lof_oe": float(cmetrics[1]),
            "gene_mis_oe": float(cmetrics[2]),
            "gene_mis_z": float(cmetrics[3]),
            "gene_lof_z": float(cmetrics[4]),
        }
        if meta["date_int"] < cutoff:
            rows_train.append(row)
        else:
            rows_test.append(row)

    print()
    print(f"[setup] train rows: {len(rows_train):,}")
    print(f"[setup] test  rows: {len(rows_test):,}")
    if rows_train:
        p_tr = sum(r["label"] for r in rows_train)
        print(f"[setup] label balance (train): P={p_tr:,} "
              f"B={len(rows_train) - p_tr:,}")
    if rows_test:
        p_te = sum(r["label"] for r in rows_test)
        print(f"[setup] label balance (test):  P={p_te:,} "
              f"B={len(rows_test) - p_te:,}")
    print(f"[setup] ClinVar candidates without REVEL match: {skipped_revel:,}")

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
