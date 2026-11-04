"""
Packages the camera-ready NeurIPS/ICLR preprint submission archive with SHA-256 manifest.
"""

from __future__ import annotations
import os
import hashlib
import zipfile
from pathlib import Path


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def package_submission(output_zip: str = "dist/icia_neurips_submission.zip"):
    dist_dir = Path("dist")
    dist_dir.mkdir(parents=True, exist_ok=True)
    paper_dir = Path("paper")

    files_to_pack = [
        paper_dir / "main.tex",
        paper_dir / "references.bib",
    ]

    # Include figures
    fig_dir = paper_dir / "figures"
    if fig_dir.exists():
        for fig in sorted(fig_dir.glob("*.*")):
            if fig.is_file():
                files_to_pack.append(fig)

    # Include embedded C99 core header
    c99_header = Path("include/icia_core.h")
    if c99_header.exists():
        files_to_pack.append(c99_header)

    # Generate Manifest
    manifest_lines = [
        "# ICIA PREPRINT SUBMISSION MANIFEST",
        "## Intention Is All You Need: A Post-Transformer Architecture for Latent Teleological Computation",
        "",
        "### Included Files & SHA-256 Checksums",
        "| File Path | SHA-256 Checksum | Size (Bytes) |",
        "| :--- | :--- | :--- |"
    ]

    for p in files_to_pack:
        sha = compute_sha256(p)
        size = p.stat().st_size
        manifest_lines.append(f"| `{p.as_posix()}` | `{sha}` | {size:,} |")

    manifest_content = "\n".join(manifest_lines) + "\n"
    manifest_path = paper_dir / "MANIFEST.md"
    manifest_path.write_text(manifest_content, encoding="utf-8")
    files_to_pack.append(manifest_path)

    # Write Zip
    zip_path = Path(output_zip)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files_to_pack:
            arcname = p.as_posix()
            zf.write(p, arcname)

    print(f"Successfully packaged {len(files_to_pack)} submission files into {zip_path} ({zip_path.stat().st_size:,} bytes)")
    print("\nPackage Contents:")
    for p in files_to_pack:
        print(f"  + {p.as_posix()} ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    package_submission()
