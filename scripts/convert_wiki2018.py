#!/usr/bin/env python3
"""Convert the Wiki2018 CDN tar.gz trace into a 10M-line gzip prefix.

Input:
    data/raw/wiki2018/wiki2018.tr.tar.gz

Output:
    data/raw/wiki2018/wiki2018.gz
"""

from __future__ import annotations

import gzip
import tarfile
from pathlib import Path


INPUT_PATH = Path("data/raw/wiki2018/wiki2018.tr.tar.gz")
OUTPUT_PATH = Path("data/raw/wiki2018/wiki2018.gz")
MAX_LINES = 10_000_000


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input trace not found: {INPUT_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with tarfile.open(INPUT_PATH, mode="r:gz") as tar:
        member = next((m for m in tar.getmembers() if m.name.endswith(".tr")), None)
        if member is None:
            raise RuntimeError(f"No .tr member found inside {INPUT_PATH}")

        src = tar.extractfile(member)
        if src is None:
            raise RuntimeError(f"Could not extract {member.name} from {INPUT_PATH}")

        with gzip.open(OUTPUT_PATH, "wb") as dst:
            for line in src:
                if written >= MAX_LINES:
                    break
                dst.write(line)
                written += 1

    print(f"Wrote {written:,} lines to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
