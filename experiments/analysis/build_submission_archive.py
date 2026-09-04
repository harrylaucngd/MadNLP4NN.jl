#!/usr/bin/env python3
"""Build or verify the anonymous, deterministic manuscript source archive."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path


STATIC_MEMBERS = (
    "iclr_workshop.tex",
    "references_iclr.bib",
    "iclr2026/fancyhdr.sty",
    "iclr2026/iclr2026_conference.bst",
    "iclr2026/iclr2026_conference.sty",
    "iclr2026/math_commands.tex",
    "figures/controlled_regime.pdf",
    "figures/darcy_iclr.pdf",
    "figures/pfr_iclr.pdf",
)
GLOBS = ("sections/*.tex", "generated/*.tex")
FORBIDDEN_MEMBERS = {"author_metadata.tex", "iclr_workshop_author.tex"}


def expected_members(paper_dir: Path) -> list[str]:
    members = set(STATIC_MEMBERS)
    for pattern in GLOBS:
        members.update(
            path.relative_to(paper_dir).as_posix()
            for path in paper_dir.glob(pattern)
            if path.is_file()
        )
    missing = sorted(name for name in members if not (paper_dir / name).is_file())
    if missing:
        raise FileNotFoundError(f"missing archive inputs: {', '.join(missing)}")
    return sorted(members)


def check_archive(paper_dir: Path, output: Path) -> list[str]:
    expected = expected_members(paper_dir)
    errors: list[str] = []
    if not output.is_file():
        return [f"archive does not exist: {output}"]
    with zipfile.ZipFile(output) as archive:
        actual = sorted(name for name in archive.namelist() if not name.endswith("/"))
        if actual != expected:
            errors.append("archive member list differs from the anonymous allowlist")
        leaked = FORBIDDEN_MEMBERS.intersection(actual)
        if leaked:
            errors.append(f"archive contains author-only files: {sorted(leaked)}")
        for name in sorted(set(actual).intersection(expected)):
            if archive.read(name) != (paper_dir / name).read_bytes():
                errors.append(f"stale archive member: {name}")
    return errors


def build_archive(paper_dir: Path, output: Path) -> None:
    members = expected_members(paper_dir)
    temporary = output.with_suffix(output.suffix + ".tmp")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in members:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (paper_dir / name).read_bytes(), compresslevel=9)
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-dir", type=Path, default=Path("paper"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    paper_dir = args.paper_dir.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else paper_dir / "iclr_workshop_source.zip"
    )
    if args.check:
        errors = check_archive(paper_dir, output)
        if errors:
            print("\n".join(errors), file=sys.stderr)
            raise SystemExit(1)
        print(f"Anonymous source archive is current: {output}")
        return
    build_archive(paper_dir, output)
    print(f"Wrote anonymous source archive: {output}")


if __name__ == "__main__":
    main()
