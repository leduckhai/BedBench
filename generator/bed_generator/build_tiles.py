"""Render separated color tiles from T5 targets with the original per-object shuffle seeds. See README.md for usage."""
import argparse
import json
import math
import random
from pathlib import Path

from datasets import load_from_disk
from matplotlib.patches import Rectangle


from .config import GeneratorConfig
from .render import save_panel

GAP = 0.6  # tile size 1; gap keeps every tile separate


def draw_tiles(ax, colors: list[str], cfg: GeneratorConfig) -> None:
    cols = math.ceil(math.sqrt(len(colors)))
    rows = math.ceil(len(colors) / cols)
    ax.set_facecolor((1.0, 1.0, 1.0))
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    step = 1 + GAP
    for i, color in enumerate(colors):
        r, c = divmod(i, cols)
        ax.add_patch(Rectangle((c * step, (rows - 1 - r) * step), 1, 1, linewidth=cfg.linewidth,
                               edgecolor=(0, 0, 0), facecolor=color, alpha=1.0, fill=True))
    size = max(cols, rows) * step - GAP
    ax.set_xlim(-0.8, size + 0.8)
    ax.set_ylim(-0.8, size + 0.8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True, help="local T5_color dataset directory")
    ap.add_argument("--out", type=Path, default=Path("tile_inputs"))
    args = ap.parse_args()
    cfg = GeneratorConfig()
    ds = load_from_disk(str(args.dataset))
    out = []
    for row in ds:
        dist = json.loads(row["color_distribution_json"])
        colors = [c for c, n in dist.items() for _ in range(n)]
        random.Random(int(row["sample_id"])).shuffle(colors)
        path = args.out / "tiles" / f"{row['sample_id']}.png"
        save_panel(str(path), lambda ax: draw_tiles(ax, colors, cfg), cfg)
        out.append({"sample_id": row["sample_id"], "image": f"tiles/{path.name}",
                    "color_distribution_json": row["color_distribution_json"]})
    (args.out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    print(len(out), "tiles written")


if __name__ == "__main__":
    main()
