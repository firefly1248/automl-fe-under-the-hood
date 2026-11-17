"""Picture of the transformation map (article figure): `python -m afe.analysis.transform_map` in the core env.

Reads results/tidy/transform_map.csv (the `<tool>_short` columns) and transform_map_evidence.csv (does_it,
status) and writes results/figures/article/transform_map.png. Size rule: the PNG is 1800 px wide and the font is
14 * 1800 / 700 = 36 px (1 pt = 1 px at 72 dpi), so the text is 14 px tall after downscaling to a 700 px column.
Cell text must be short: a word wider than a column is an error. transform_map.csv and
transform_map_evidence.csv are hand-curated tables (written once by a script that is not in the repository,
backed by exp01, exp03, exp05 outputs, the probes exp03_dates_text.csv and exp06_h2o_probe.csv and the cited
source lines); `make article` regenerates only this picture from them.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from afe.paths import RESULTS

WIDTH, COLUMN_PX, MIN_PT = 1800, 700, 14
FONT = MIN_PT * WIDTH / COLUMN_PX
PAD, LINE = 10, 1.22 * FONT
FAMILY_W = 300
TOOLS = ["ag_default", "ag_tabprep", "lama", "h2o", "openfe", "autofeat", "mljar_golden", "manual_baseline"]
HEAD = {"ag_default": "ag_\ndefault", "ag_tabprep": "ag_\ntabprep", "mljar_golden": "mljar_\ngolden",
        "manual_baseline": "manual_\nbaseline"}
FAMILY = {"categorical_encoding": "categorical\nencoding"}
SYMBOL = {"confirmed_run": "●", "confirmed_source": "◐", "not_confirmed": "○"}
FILL = {"yes": "#cfe8d5", "partial": "#f6e3b4", "no": "#e4e4e4"}
LEGEND = ["● run output   ○ not confirmed",
          "◐ source code (a 'not done' cell: no such code at the cited place)",
          "green: done   amber: partial   grey: not done",
          "h2o: default and TE in one column; lama: some cells from the lama_adv arm"]


def wrap(text: str, width: float, measure) -> list[str]:
    lines, cur = [], ""
    words = text.split()
    glued = [" ".join(words[:2])] + words[2:]  # the status symbol stays with the first word when that fits
    for word in glued if measure(glued[0]) <= width else words:
        if measure(word) > width:
            raise ValueError(f"word {word!r} does not fit a {width:.0f} px column; shorten the cell text")
        trial = f"{cur} {word}".strip()
        if cur and measure(trial) > width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return lines + [cur]


def main() -> None:
    tidy = RESULTS / "tidy"
    short = pd.read_csv(tidy / "transform_map.csv").set_index("family")
    ev = pd.read_csv(tidy / "transform_map_evidence.csv").set_index(["family", "tool"])
    tool_w = (WIDTH - FAMILY_W) / len(TOOLS)

    fig = plt.figure(figsize=(WIDTH / 72, 10), dpi=72)
    probe = fig.text(0, 0, "", fontsize=FONT)
    renderer = fig.canvas.get_renderer()

    def measure(s: str) -> float:
        probe.set_text(s)
        return probe.get_window_extent(renderer).width

    for fam in short.index:
        for part in FAMILY.get(fam, fam).split("\n"):
            if measure(part) > FAMILY_W - 2 * PAD:
                raise ValueError(f"family label {part!r} does not fit the first column")
    cells = {}
    for fam in short.index:
        for tool in TOOLS:
            e = ev.loc[(fam, tool)]
            text = f"{SYMBOL[e.status]} {short.loc[fam, tool + '_short']}"
            cells[fam, tool] = (wrap(text, tool_w - 2 * PAD, measure), e.does_it)
    probe.remove()
    row_h = {fam: max(len(cells[fam, t][0]) for t in TOOLS) * LINE + 2 * PAD for fam in short.index}
    head_h = 2 * LINE + 2 * PAD
    foot_h = len(LEGEND) * LINE + 2 * PAD
    height = head_h + sum(row_h.values()) + foot_h

    fig.set_size_inches(WIDTH / 72, height / 72)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(height, 0)
    ax.axis("off")
    kw = dict(fontsize=FONT, family="DejaVu Sans", va="center", color="#1a1a1a")

    def box(x, y, w, h, fc):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec="white", lw=3))

    for i, tool in enumerate(TOOLS):
        x = FAMILY_W + i * tool_w
        box(x, 0, tool_w, head_h, "#2b3a4a")
        ax.text(x + tool_w / 2, head_h / 2, HEAD.get(tool, tool), ha="center", linespacing=1.1, **{**kw, "color": "white", "weight": "bold"})
    y = head_h
    for fam in short.index:
        h = row_h[fam]
        box(0, y, FAMILY_W, h, "#eef1f4")
        ax.text(PAD, y + h / 2, FAMILY.get(fam, fam), ha="left", linespacing=1.1, **{**kw, "weight": "bold"})
        for i, tool in enumerate(TOOLS):
            lines, does = cells[fam, tool]
            x = FAMILY_W + i * tool_w
            box(x, y, tool_w, h, FILL[does])
            ax.text(x + PAD, y + PAD, "\n".join(lines), ha="left", va="top", linespacing=LINE / FONT, **{k: v for k, v in kw.items() if k != "va"})
        y += h
    for j, s in enumerate(LEGEND):
        ax.text(PAD, y + PAD + (j + 0.5) * LINE, s, ha="left", **kw)

    out = RESULTS / "figures" / "article"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "transform_map.png", dpi=72)
    plt.close(fig)
    print(f"article/transform_map.png {WIDTH} x {round(height)} px")


if __name__ == "__main__":
    main()
