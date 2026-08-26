#!/usr/bin/env python3
"""
Inspect and extract a large research-data zip that `unzip` refuses.

    python scripts/unpack_archive.py FACED_nm000112_v1.1.3.zip
    python scripts/unpack_archive.py FACED_nm000112_v1.1.3.zip --extract-to raw/

Info-ZIP's zip-bomb guard rejects archives whose entry data ranges overlap. Two
very different things produce that message and they need opposite responses:

  * A FALSE POSITIVE on a legitimate ZIP64 archive. Large multi-gigabyte zips
    written by some tools carry local-header offsets the heuristic reads as
    overlapping. The data is fine and simply needs an extractor without that
    heuristic.
  * A TRUNCATED OR CORRUPT DOWNLOAD. A zip cut short has a central directory
    pointing past the end of the file, which also looks like overlap. Disabling
    the check here does not recover the data; it extracts a prefix and leaves
    you with a corpus that is quietly missing files.

So this measures first. The compression ratio is what separates a bomb from a
big archive: a zip bomb runs to thousands-to-one, and EEG recordings run about
one-and-a-half to three.
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:,.1f} {unit}"
        n /= 1024.0
    return f"{n:,.1f} PB"


def inspect(path: str, ratio_limit: float) -> tuple[int, zipfile.ZipFile | None]:
    size = os.path.getsize(path)
    print(f"{path}")
    print(f"  file on disk   {human(size)}")

    try:
        z = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        sys.stdout.flush()
        print(f"\n  BAD ZIP: {exc}", file=sys.stderr)
        print("\n  The central directory could not be read at all. That is a "
              "truncated or corrupt download, not the zip-bomb heuristic.\n"
              "  Re-fetch the archive; do not try to force it open.",
              file=sys.stderr)
        return 1, None

    infos = z.infolist()
    comp = sum(i.compress_size for i in infos)
    uncomp = sum(i.file_size for i in infos)
    ratio = uncomp / max(1, comp)
    print(f"  entries        {len(infos):,}")
    print(f"  compressed     {human(comp)}")
    print(f"  uncompressed   {human(uncomp)}")
    print(f"  ratio          {ratio:.2f}x")
    print(f"  ZIP64          {'yes' if size >= 2**32 or len(infos) >= 65535 else 'no'}")

    # Does the archive actually claim what it occupies? A central directory
    # describing more compressed bytes than the file holds is the truncation
    # case, and it is the one that must not be forced.
    sys.stdout.flush()
    if comp > size:
        print(f"\n  TRUNCATED: the directory describes {human(comp)} of "
              f"compressed data but the file is only {human(size)}.",
              file=sys.stderr)
        print("  Re-download it. Forcing this extracts a prefix and leaves a "
              "corpus silently missing files.", file=sys.stderr)
        return 1, None

    sys.stdout.flush()
    if ratio > ratio_limit:
        print(f"\n  REFUSING: {ratio:.0f}x expansion is far beyond what "
              f"recorded signal compresses to (~1.5-3x).", file=sys.stderr)
        print(f"  --ratio-limit {ratio_limit} is the threshold; raise it only "
              f"if you know why this archive expands that much.", file=sys.stderr)
        return 1, None

    print(f"\n  Consistent with an archive of recorded signal, not a bomb: "
          f"{ratio:.2f}x expansion,")
    print(f"  and the directory's {human(comp)} fits inside the "
          f"{human(size)} on disk.")
    return 0, z


def extract(z: zipfile.ZipFile, dest: str) -> int:
    infos = z.infolist()
    total = sum(i.file_size for i in infos)
    os.makedirs(dest, exist_ok=True)
    print(f"\nextracting {len(infos):,} entries -> {dest}")
    done = 0
    for n, info in enumerate(infos, 1):
        # Path traversal is the other thing a hostile archive does, and the
        # heuristic we are stepping around is the one that would have caught it.
        target = os.path.realpath(os.path.join(dest, info.filename))
        if not target.startswith(os.path.realpath(dest) + os.sep) and \
                target != os.path.realpath(dest):
            print(f"\n  REFUSING {info.filename!r}: escapes {dest}",
                  file=sys.stderr)
            return 1
        z.extract(info, dest)
        done += info.file_size
        if n % 200 == 0 or n == len(infos):
            pct = done / max(1, total) * 100
            print(f"  {n:,}/{len(infos):,}  {human(done)}  {pct:5.1f}%",
                  flush=True)
    print(f"\ndone: {human(done)} in {len(infos):,} entries under {dest}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("archive")
    p.add_argument("--extract-to", default=None, metavar="DIR",
                   help="extract here. Omitted: inspect and report only.")
    p.add_argument("--ratio-limit", type=float, default=50.0,
                   help="refuse above this expansion factor (default 50)")
    args = p.parse_args(argv)

    if not os.path.isfile(args.archive):
        print(f"ERROR: no such file: {args.archive}", file=sys.stderr)
        return 1

    rc, z = inspect(args.archive, args.ratio_limit)
    if rc or z is None:
        return rc
    if args.extract_to is None:
        print(f"\nTo extract:\n  python {sys.argv[0]} {args.archive} "
              f"--extract-to <dir>")
        return 0
    with z:
        return extract(z, args.extract_to)


if __name__ == "__main__":
    raise SystemExit(main())
