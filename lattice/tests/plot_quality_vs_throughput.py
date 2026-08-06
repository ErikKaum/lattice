"""Plot quality vs throughput for the deployment variant matrix.

X-axis: throughput (M tokens/sec) on wiki-5k, log-scaled.
Y-axis: NDCG@10 on decontaminated BEIR (12-mean, LightOn-comparable).
Colors: bit-width (32/8/4/2). Markers: axis (dim = circle, row = square).
Annotations: weight footprint in MB.
Dashed line: Pareto frontier (no other point has both higher throughput AND higher NDCG).

Reads:
- data/throughput.csv               (one row per variant, tokens/sec)
- data/<slug>/eval/decontam_beir.json  (12-mean NDCG@10 at model's native dim)

Writes:
- data/quality_vs_throughput.png
- data/quality_vs_throughput.pdf
- data/quality_vs_throughput.csv (the underlying point table)
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from adjustText import adjust_text

DATA = Path(__file__).resolve().parents[2] / "data"

BITS_COLOR = {32: "#404040", 8: "#1f77b4", 4: "#d97a1f", 2: "#c43030"}
AXIS_MARKER = {"dim": "o", "row": "s"}


def parse_slug(slug: str) -> tuple[int, str, int]:
    """Returns (bits, axis, dim) from a slug like 'int4-dim-512' or 'fp32-dim-1024'."""
    m = re.match(r"(fp32|int(\d+))-(dim|row)-(\d+)", slug)
    if not m:
        raise ValueError(f"unrecognized slug: {slug}")
    if m.group(1) == "fp32":
        bits = 32
    else:
        bits = int(m.group(2))
    axis = m.group(3)
    dim = int(m.group(4))
    return bits, axis, dim


def load_throughput() -> dict[str, dict]:
    out = {}
    with open(DATA / "throughput.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["variant"]] = {
                "size_mb": float(row["size_mb"]),
                "tok_s": float(row["tokens_per_s"]),
                "embed_us_per_doc": float(row["embed_us_per_doc"]),
            }
    return out


def load_quality(slug: str, dim: int) -> float | None:
    p = DATA / slug / "eval" / "decontam_beir.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    return d["aggregates"]["12_mean_ndcg@10"].get(str(dim))


def collect_points() -> list[dict]:
    tput = load_throughput()
    points = []
    for slug, t in tput.items():
        try:
            bits, axis, dim = parse_slug(slug)
        except ValueError:
            continue
        ndcg = load_quality(slug, dim)
        if ndcg is None:
            print(f"skipping {slug} — no decontam JSON yet")
            continue
        points.append({
            "slug": slug,
            "bits": bits,
            "axis": axis,
            "dim": dim,
            "size_mb": t["size_mb"],
            "tok_s": t["tok_s"],
            "ndcg12": ndcg,
        })
    return points


def pareto_set(points: list[dict]) -> list[dict]:
    """Non-dominated set: no other point has both higher tok_s AND higher ndcg12."""
    out = []
    for p in points:
        dominated = any(
            q["tok_s"] > p["tok_s"] and q["ndcg12"] > p["ndcg12"]
            for q in points if q is not p
        )
        if not dominated:
            out.append(p)
    return sorted(out, key=lambda p: p["tok_s"])


def main() -> None:
    points = collect_points()
    if not points:
        print("No points to plot.")
        return

    fig, ax = plt.subplots(figsize=(13, 8.5))

    # Plot one scatter per (bits, axis) for a clean legend. No connecting
    # lines this time — only the Pareto frontier is drawn.
    by_group: dict[tuple[int, str], list[dict]] = {}
    for p in points:
        by_group.setdefault((p["bits"], p["axis"]), []).append(p)

    for (bits, axis), pts in sorted(by_group.items()):
        label = "fp32" if bits == 32 else f"int{bits}-{axis}"
        ax.scatter(
            [p["tok_s"] / 1e6 for p in pts],
            [p["ndcg12"] for p in pts],
            s=130,
            c=BITS_COLOR[bits],
            marker=AXIS_MARKER[axis],
            edgecolors="black",
            linewidth=0.7,
            label=label,
            zorder=3,
        )

    # Per-point annotations: `d=<dim> · <size>MB`. No connector arrows
    # (they look like graph lines and confuse the chart). Color-match each
    # label to its point's marker so dense clusters remain unambiguous.
    #
    # Pre-offset each label by ~12 pixels up-and-right of its point so the
    # label starts CLEAR of its own marker; `adjustText` then only has to
    # resolve label-vs-label and label-vs-other-marker collisions.
    import numpy as np
    fig.canvas.draw()  # ensure transform is realized before we use it

    def offset_in_pixels(x_data, y_data, dx_px, dy_px):
        """Convert (data_x, data_y) shifted by (dx, dy) pixels back to data coords."""
        display = ax.transData.transform([x_data, y_data])
        shifted = display + np.array([dx_px, dy_px])
        return ax.transData.inverted().transform(shifted)

    texts = []
    for p in points:
        label = f"d={p['dim']} · {p['size_mb']:.1f}MB"
        # ~18 px right, ~10 px up; circle/square markers at s=130 have
        # visual radius ~9 px so this leaves a ~9 px clear gap before
        # the label glyphs start.
        lx, ly = offset_in_pixels(p["tok_s"] / 1e6, p["ndcg12"], 18, 10)
        texts.append(ax.text(
            lx, ly, label,
            fontsize=8, color=BITS_COLOR[p["bits"]], zorder=4,
            ha="left", va="bottom",
            weight="medium",
        ))
    # Now that labels start clear of their own markers, adjustText only
    # resolves the remaining collisions. Modest force_static so labels
    # don't drift into other markers; gentle force_text so they don't
    # spread apart from each other unnecessarily.
    adjust_text(
        texts, ax=ax,
        expand=(1.12, 1.22),
        force_text=(0.18, 0.28),
        force_static=(0.55, 0.7),
        only_move={"text": "xy", "static": "xy"},
        max_move=25,
    )

    # Post-check: warn if any text label's bbox still intersects a marker.
    # (At this point adjustText has finished; we just report what we see.)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    marker_radius_px = 9.0  # for s=130 scatter
    overlaps = []
    for p, t in zip(points, texts):
        tx, ty = ax.transData.transform([p["tok_s"] / 1e6, p["ndcg12"]])
        bb = t.get_window_extent(renderer=renderer)
        # Closest point of text bbox to marker center (axis-aligned).
        cx = max(bb.xmin, min(tx, bb.xmax))
        cy = max(bb.ymin, min(ty, bb.ymax))
        dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
        if dist < marker_radius_px + 1:
            overlaps.append((p["slug"], dist))
    if overlaps:
        print(f"[warn] {len(overlaps)} label(s) still touch their own marker:")
        for slug, d in overlaps:
            print(f"   {slug:<22} gap = {d:.1f} px (want > {marker_radius_px:.0f})")
    else:
        print(f"All {len(points)} labels clear their own markers (>= {marker_radius_px:.0f} px gap).")

    # Pareto frontier
    pareto = pareto_set(points)
    if pareto:
        ax.plot(
            [p["tok_s"] / 1e6 for p in pareto],
            [p["ndcg12"] for p in pareto],
            linestyle="--",
            color="gray",
            alpha=0.7,
            linewidth=1.3,
            label="Pareto frontier",
            zorder=2,
        )

    ax.set_xlabel("Throughput (M tokens/sec, 8-core Apple M2, 12 worker threads, wiki-5k corpus)")
    ax.set_ylabel("Quality (NDCG@10, decontaminated BEIR 12-mean)")
    ax.set_title("Quality vs throughput across the lattice-retrieval variant matrix\n"
                 "(circle = axis=dim,  square = axis=row,  color = bit-width)")
    ax.set_xscale("linear")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="lower left", framealpha=0.9, fontsize=10)
    plt.tight_layout()

    png_path = DATA / "quality_vs_throughput.png"
    pdf_path = DATA / "quality_vs_throughput.pdf"
    csv_path = DATA / "quality_vs_throughput.csv"
    plt.savefig(png_path, dpi=150)
    plt.savefig(pdf_path)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")

    # Also dump the point table as CSV
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["slug", "bits", "axis", "dim", "size_mb", "tok_s", "ndcg12"],
            lineterminator="\n",
        )
        w.writeheader()
        for p in sorted(points, key=lambda p: (p["bits"], p["dim"])):
            w.writerow(p)
    print(f"Saved {csv_path}")

    # Also print a summary table
    print()
    print(f"{'variant':<22} {'size_mb':>8} {'tok/s (M)':>10} {'NDCG@10 (12)':>14}")
    print("-" * 56)
    for p in sorted(points, key=lambda x: -x["ndcg12"]):
        print(f"{p['slug']:<22} {p['size_mb']:>8.2f} {p['tok_s']/1e6:>10.2f} {p['ndcg12']:>14.4f}")


if __name__ == "__main__":
    main()
