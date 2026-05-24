"""Preprocess protein-DNA complex mmCIF files into joint NPZ feature/label caches.

Usage
-----
Single PDB ID (downloads from RCSB):

    python scripts/preprocess_protein_dna.py \\
        --pdb-ids 1aoi 1d66 1tf6 \\
        --protein-chain A --dna-chains B C \\
        --features-dir data/pdna_features \\
        --labels-dir   data/pdna_labels

Local mmCIF file:

    python scripts/preprocess_protein_dna.py \\
        --mmcif /path/to/1aoi.cif \\
        --pdb-id 1aoi --protein-chain A --dna-chains B C \\
        --features-dir data/pdna_features \\
        --labels-dir   data/pdna_labels

Output
------
For each complex "<pdb_id>_<protein_chain>_<dna_chains>" two files are written:

* ``features-dir/<complex_id>.npz`` — model inputs:
  - ``aatype``           (N_total,) int64 — protein tokens 0-20, DNA tokens 21-25
  - ``chain_type``       (N_total,) int64 — 0=protein, 1=DNA
  - ``residue_index``    (N_total,) int64 — protein 0..N_p-1, DNA N_p+200..N_p+200+N_d-1
  - ``between_segment_residues`` (N_total,) int64 — 1 at first DNA residue
  - ``msa``              (1, N_total) int64 — single-sequence MSA, zeros for DNA
  - ``deletions``        (1, N_total) int64 — zeros
  - ``template_aatype``  (0, N_total) — empty templates
  - ``template_atom14_positions`` (0, N_total, 14, 3) — empty
  - ``template_atom14_mask``      (0, N_total, 14) — empty

* ``labels-dir/<complex_id>.npz`` — supervision:
  - ``atom14_positions`` (N_total, 14, 3) float32 — ground-truth heavy atoms (Å)
  - ``atom14_mask``      (N_total, 14)    float32 — 1 where atom is resolved
  - ``resolution``       float32

Design notes
------------
* DNA positions receive zero MSA (no evolutionary co-variation available for
  DNA strands). The protein chain uses a single-sequence "MSA" (just the query
  sequence); for real training you would run a homology search and append those
  rows here.
* The 200-residue index gap between protein and DNA chains causes RelPos to
  treat protein-DNA and DNA-DNA pairs as "far apart", which is correct — the
  learned relative positional encoding only has meaningful signal within a chain.
* Template features are left empty. Adding protein-structure templates for the
  protein chain is straightforward (run the existing preprocess_openproteinset
  pipeline for the protein-only part) but is omitted here to keep the script
  self-contained.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
_RESTYPE_3TO1 = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
    'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
    'UNK': 20,
}
_DNA_RESTYPE = {'DA': 21, 'DC': 22, 'DG': 23, 'DT': 24}
_DNA_TOKEN_OFFSET = 21
_CHAIN_TYPE_PROTEIN = 0
_CHAIN_TYPE_DNA = 1


def _download_mmcif(pdb_id: str, out_dir: Path) -> Path:
    url = _RCSB_URL.format(pdb_id=pdb_id.upper())
    dest = out_dir / f"{pdb_id.lower()}.cif"
    if dest.exists():
        print(f"  [cache] {dest}")
        return dest
    print(f"  Downloading {url} …", end="", flush=True)
    urllib.request.urlretrieve(url, dest)
    print(" done")
    return dest


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_complex(
    mmcif_path: Path,
    pdb_id: str,
    protein_chain: str,
    dna_chains: List[str],
    features_dir: Path,
    labels_dir: Path,
    chain_id_sep: str = "_",
) -> str:
    """Extract joint features and labels for one protein-DNA complex."""
    from minalphafold.mmcif import extract_chain_atoms, extract_dna_chain_atoms

    # ------------------------------------------------------------------ #
    # 1. Extract protein chain
    # ------------------------------------------------------------------ #
    prot = extract_chain_atoms(mmcif_path, pdb_id, protein_chain)
    N_prot = len(prot.sequence)

    # Map sequence to integer tokens (0-20)
    prot_aatype = np.array(
        [_RESTYPE_3TO1.get(res.upper(), 20) for res in prot.sequence],
        dtype=np.int64,
    )

    # ------------------------------------------------------------------ #
    # 2. Extract DNA chain(s) and concatenate
    # ------------------------------------------------------------------ #
    dna_aatype_list: list[np.ndarray] = []
    dna_pos_list:    list[np.ndarray] = []
    dna_mask_list:   list[np.ndarray] = []

    for dc in dna_chains:
        dna = extract_dna_chain_atoms(mmcif_path, pdb_id, dc)
        # Map nuctype (0-3 for a/c/g/t, 4=DUNK) to joint-vocab tokens 21-25
        dna_tok = (dna.nuctype + _DNA_TOKEN_OFFSET).astype(np.int64)
        dna_aatype_list.append(dna_tok)
        dna_pos_list.append(dna.atom14_positions)
        dna_mask_list.append(dna.atom14_mask)

    dna_aatype   = np.concatenate(dna_aatype_list,  axis=0)
    dna_positions = np.concatenate(dna_pos_list,    axis=0)
    dna_atom_mask = np.concatenate(dna_mask_list,   axis=0)
    N_dna = len(dna_aatype)

    # ------------------------------------------------------------------ #
    # 3. Build joint arrays
    # ------------------------------------------------------------------ #
    N_total = N_prot + N_dna

    # Residue index: protein 0..N_p-1, DNA N_p+200..N_p+200+N_d-1
    prot_ri = np.arange(N_prot, dtype=np.int64)
    dna_ri  = np.arange(N_prot + 200, N_prot + 200 + N_dna, dtype=np.int64)
    residue_index = np.concatenate([prot_ri, dna_ri])

    # chain_type
    chain_type = np.concatenate([
        np.zeros(N_prot, dtype=np.int64),
        np.ones(N_dna,  dtype=np.int64),
    ])

    # between_segment_residues: mark the first DNA residue as a chain break
    between_seg = np.zeros(N_total, dtype=np.int64)
    if N_dna > 0:
        between_seg[N_prot] = 1

    # Joint aatype
    aatype = np.concatenate([prot_aatype, dna_aatype])

    # atom14 positions and mask (protein first, then DNA)
    prot_pos  = prot.atom14_positions   # (N_prot, 14, 3)
    prot_mask = prot.atom14_mask         # (N_prot, 14)
    atom14_positions = np.concatenate([prot_pos,  dna_positions], axis=0)
    atom14_mask      = np.concatenate([prot_mask, dna_atom_mask], axis=0)

    # Single-sequence "MSA": protein residues encode aatype-as-hhblits-token,
    # DNA positions are padded with 21 (gap token in the HHblits alphabet).
    # Protein HHblits tokens ≈ aatype values (0-20 for AAs, 20 for UNK);
    # we use aatype directly here as a minimal stand-in.
    msa_row = np.concatenate([
        prot_aatype.astype(np.int64),
        np.full(N_dna, 21, dtype=np.int64),  # gap token for DNA
    ])
    msa       = msa_row[np.newaxis, :]          # (1, N_total)
    deletions = np.zeros_like(msa)              # (1, N_total)

    # Empty templates
    template_aatype          = np.zeros((0, N_total), dtype=np.int64)
    template_atom14_positions = np.zeros((0, N_total, 14, 3), dtype=np.float32)
    template_atom14_mask      = np.zeros((0, N_total, 14),    dtype=np.float32)

    # ------------------------------------------------------------------ #
    # 4. Write NPZs
    # ------------------------------------------------------------------ #
    complex_id = pdb_id.lower() + chain_id_sep + protein_chain + chain_id_sep + "".join(dna_chains)
    features_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        features_dir / f"{complex_id}.npz",
        aatype=aatype,
        chain_type=chain_type,
        between_segment_residues=between_seg,
        residue_index=residue_index,
        msa=msa,
        deletions=deletions,
        template_aatype=template_aatype,
        template_atom14_positions=template_atom14_positions,
        template_atom14_mask=template_atom14_mask,
    )
    np.savez(
        labels_dir / f"{complex_id}.npz",
        atom14_positions=atom14_positions.astype(np.float32),
        atom14_mask=atom14_mask.astype(np.float32),
        resolution=np.float32(getattr(prot, "resolution", 0.0)),
    )

    print(f"  Wrote {complex_id}: {N_prot} protein + {N_dna} DNA residues")
    return complex_id


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdb-ids",       nargs="*", default=[],      help="RCSB PDB IDs to download and process")
    parser.add_argument("--mmcif",                    default=None,     help="Path to a local mmCIF file (single complex)")
    parser.add_argument("--pdb-id",                   default=None,     help="PDB ID for a local mmCIF file")
    parser.add_argument("--protein-chain", required=True,               help="Chain ID of the protein chain, e.g. A")
    parser.add_argument("--dna-chains",    nargs="+",  required=True,   help="Chain ID(s) of the DNA chain(s), e.g. B C")
    parser.add_argument("--features-dir",  required=True,               help="Output directory for feature NPZs")
    parser.add_argument("--labels-dir",    required=True,               help="Output directory for label NPZs")
    parser.add_argument("--download-dir",  default="/tmp/pdna_mmcif",   help="Cache dir for downloaded mmCIF files")
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    labels_dir   = Path(args.labels_dir)
    download_dir = Path(args.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    pdb_sources: list[tuple[Path, str]] = []

    # Local file
    if args.mmcif:
        if not args.pdb_id:
            parser.error("--pdb-id is required when using --mmcif")
        pdb_sources.append((Path(args.mmcif), args.pdb_id))

    # Remote downloads
    for pdb_id in (args.pdb_ids or []):
        mmcif_path = _download_mmcif(pdb_id, download_dir)
        pdb_sources.append((mmcif_path, pdb_id))

    if not pdb_sources:
        parser.error("Provide at least one PDB ID via --pdb-ids or a local file via --mmcif")

    ok = 0
    for mmcif_path, pdb_id in pdb_sources:
        try:
            process_complex(
                mmcif_path=mmcif_path,
                pdb_id=pdb_id,
                protein_chain=args.protein_chain,
                dna_chains=args.dna_chains,
                features_dir=features_dir,
                labels_dir=labels_dir,
            )
            ok += 1
        except Exception as exc:
            print(f"  ERROR processing {pdb_id}: {exc}", file=sys.stderr)

    print(f"\nDone: {ok}/{len(pdb_sources)} complexes processed successfully.")


if __name__ == "__main__":
    main()
