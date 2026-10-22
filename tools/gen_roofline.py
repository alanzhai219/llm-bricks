"""Minimal CPU Roofline script.

Default: show the plot window.
If OUTPUT is set, save it to that path instead.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PEAK_GFLOPS = 153.6
DRAM_GBPS = 22.39
L3_GBPS = None
L2_GBPS = None
L1_GBPS = None
POINTS = [("GEMV 7680x30720", 0.5, 10.23),
          ("GEMV REF 7680x30720", 0.52, 2.96)]
OUTPUT = None

def main():
    parser = argparse.ArgumentParser(description="Minimal CPU Roofline")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="If set, save the plot to this path instead of showing it")
    args = parser.parse_args()

    out = args.output if args.output is not None else OUTPUT

    print("CPU Roofline")
    print(f"Peak: {PEAK_GFLOPS:g} GFLOP/s")
    print(f"DRAM: {DRAM_GBPS:g} GB/s")
    for name, value in (("L3", L3_GBPS), ("L2", L2_GBPS), ("L1", L1_GBPS)):
        if value is not None:
            print(f"{name}: {value:g} GB/s")
    for name, ai, perf in POINTS:
        print(f"{name}: AI={ai:g}, perf={perf:g} GFLOP/s")

    ai = np.logspace(-2, 2, 500)
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, value in (("DRAM", DRAM_GBPS), ("L3", L3_GBPS), ("L2", L2_GBPS), ("L1", L1_GBPS)):
        if value is not None:
            ax.loglog(ai, np.minimum(PEAK_GFLOPS, ai * value), label=f"{name} {value:g} GB/s")
            knee_ai = PEAK_GFLOPS / value
            ax.annotate(f"({knee_ai:.2g}, {PEAK_GFLOPS:.2g})",
                        xy=(knee_ai, PEAK_GFLOPS),
                        xytext=(8, 8),
                        textcoords="offset points",
                        fontsize=9,
                        color="black")

    for name, ai_value, perf in POINTS:
        ax.plot(ai_value, perf, "o", label=name)
        ax.annotate(f"({ai_value:.2g}, {perf:.2g})",
                    xy=(ai_value, perf),
                    xytext=(8, 8),
                    textcoords="offset points",
                    fontsize=9,
                    color="black")

    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("Performance (GFLOP/s)")
    ax.set_title("CPU Roofline")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()

    if out is None:
        plt.show()
    else:
        out_path = out if out.is_absolute() else Path(__file__).resolve().parent.parent / out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        print(f"Saved {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
