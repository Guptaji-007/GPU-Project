
#!/usr/bin/env python3

import csv
import re
import sys
from pathlib import Path

HEADER_RE = re.compile(r"^=====+\s*(\S+)")
TIME_RE = re.compile(r"^Average GPU (.+?) Time\D*\*?(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
GFLOPS_RE = re.compile(r"^GPU (.+?) GFLOP/s:\s*\**(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
CHECK_RE = re.compile(r"^Checking (.+?) Correctness:\s*\**(PASS|FAIL)\**", re.IGNORECASE)
CPU_RE = re.compile(r"^CPU (SpMV|SpMM) Time\D*\*?(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")

NAIVE_SPMM = "CSR naive SpMM"


def normalize(name: str) -> str:
    # Naive SpMM is printed as SpMM (time/check) and naive SpMM (GFLOP/s).
    name = name.strip()

    if name == "SpMM" or name == "naive SpMM":
        return NAIVE_SPMM

    return name


def to_ms(value: float, line: str) -> float:
    # Convert to ms based on the unit found in the line (default: ms).
    low = line.lower()

    if re.search(r"\bms\b", low):
        return value

    if re.search(r"\b(us|µs|usec)\b", low):
        return value / 1000.0

    if re.search(r"\b(s|sec|secs|seconds)\b", low):
        return value * 1000.0

    return value


def parse(path: Path):
    rows = {}
    matrix = None

    # Get an existing row or create a new one.
    def row(kernel):
        key = (matrix, kernel)

        if key not in rows:
            rows[key] = {
                "matrix": matrix,
                "kernel": kernel,
                "time_ms": "",
                "gflops": "",
                "pass": ""
            }

        return rows[key]

    with path.open() as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()

            if not line:
                continue

            m = HEADER_RE.match(line)

            if m:
                matrix = Path(m.group(1)).stem
                continue

            m = TIME_RE.match(line)
            kind = None

            if m:
                kind = "time"
            else:
                m = GFLOPS_RE.match(line)

                if m:
                    kind = "gflops"
                else:
                    m = CHECK_RE.match(line)

                    if m:
                        kind = "pass"
                    else:
                        m = CPU_RE.match(line)

                        if m:
                            kind = "cpu"

            if kind is None:
                continue

            if matrix is None:
                print(
                    f"warning: line {lineno} appears before any ===== header: {line}",
                    file=sys.stderr
                )
                continue

            if kind == "time":
                kernel = normalize(m.group(1))
                value = float(m.group(2))
                row(kernel)["time_ms"] = to_ms(value, line)

            elif kind == "gflops":
                kernel = normalize(m.group(1))
                row(kernel)["gflops"] = float(m.group(2))

            elif kind == "pass":
                kernel = normalize(m.group(1))
                row(kernel)["pass"] = m.group(2).upper()

            else:
                # CPU baseline
                kernel = "CPU " + m.group(1)
                value = float(m.group(2))
                row(kernel)["time_ms"] = to_ms(value, line)

    return list(rows.values())


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)

    src = Path(sys.argv[1])

    default_out = (
        Path(__file__).resolve().parent.parent
        / "results"
        / "results.csv"
    )

    if len(sys.argv) > 2:
        out = Path(sys.argv[2])
    else:
        out = default_out

    data = parse(src)

    if not data:
        sys.exit(f"error: no results parsed from {src}")

    for r in data:
        is_cpu = r["kernel"].startswith("CPU")
        has_time = r["time_ms"] != ""
        missing_gflops = r["gflops"] == ""

        if not is_cpu and has_time and missing_gflops:
            print(
                f"warning: {r['matrix']} / {r['kernel']} has a time but no GFLOP/s",
                file=sys.stderr
            )

    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="") as f:
        fieldnames = ["matrix", "kernel", "time_ms", "gflops", "pass"]

        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(data)

    matrices = len({r["matrix"] for r in data})

    print(f"wrote {len(data)} rows for {matrices} matrices to {out}")


if __name__ == "__main__":
    main()
