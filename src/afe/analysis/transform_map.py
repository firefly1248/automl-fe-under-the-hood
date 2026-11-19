"""Pictures of the transformation map (article figures): `python -m afe.analysis.transform_map` in the core env.

Reads results/tidy/transform_map.csv (the `<tool>_short` columns) and transform_map_evidence.csv (does_it) and writes
two PNGs in results/figures/article: `transform_map_automl.png` (ag_default, ag_tabprep, lama, h2o) and
`transform_map_generators.png` (openfe, autofeat, mljar_golden, manual_baseline), each with a CSV of the rendered
cells. Only the cell colour carries a status (done, partial, not done); how a cell was confirmed (run output or
source code) and the references are in transform_map_evidence.csv, not in the picture. Size rule: the PNG is 1800 px
wide and the font is 14 * 1800 / 700 = 36 px (1 pt = 1 px at 72 dpi), so the text is 14 px tall after downscaling to
a 700 px column. Cell text must be short: a word wider than a column is an error. The two tables come from
`afe.analysis.transform_map_build` (`make article` runs it first).
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
GROUPS = {
    "transform_map_automl": (["ag_default", "ag_tabprep", "lama", "h2o"],
                             ["green: done   amber: partial   grey: not done",
                              "h2o: default and target encoding in one column",
                              "lama: some cells come from the lama_adv arm"]),
    "transform_map_generators": (["openfe", "autofeat", "mljar_golden", "manual_baseline", "skrub_tv"],
                                 ["green: done   amber: partial   grey: not done"]),
}
HEAD = {"ag_default": "ag_\ndefault", "ag_tabprep": "ag_\ntabprep", "mljar_golden": "mljar_\ngolden",
        "manual_baseline": "manual_\nbaseline", "skrub_tv": "skrub_\ntv"}
FAMILY = {"categorical_encoding": "categorical\nencoding"}
FILL = {"yes": "#cfe8d5", "partial": "#f6e3b4", "no": "#e4e4e4"}


def wrap(text: str, width: float, measure) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        if measure(word) > width:
            raise ValueError(f"word {word!r} does not fit a {width:.0f} px column; shorten the cell text")
        trial = f"{cur} {word}".strip()
        if cur and measure(trial) > width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return lines + [cur]


def render(name: str, tools: list[str], legend: list[str]) -> None:
    tidy = RESULTS / "tidy"
    short = pd.read_csv(tidy / "transform_map.csv").set_index("family")
    ev = pd.read_csv(tidy / "transform_map_evidence.csv").set_index(["family", "tool"])
    tool_w = (WIDTH - FAMILY_W) / len(tools)

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
    cells = {(fam, tool): (wrap(short.loc[fam, tool + "_short"], tool_w - 2 * PAD, measure), ev.loc[(fam, tool)].does_it)
             for fam in short.index for tool in tools}
    probe.remove()
    row_h = {fam: max(len(FAMILY.get(fam, fam).split("\n")), *(len(cells[fam, t][0]) for t in tools)) * LINE + 2 * PAD
             for fam in short.index}
    head_h = 2 * LINE + 2 * PAD
    foot_h = len(legend) * LINE + 2 * PAD
    height = head_h + sum(row_h.values()) + foot_h

    fig.set_size_inches(WIDTH / 72, height / 72)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(height, 0)
    ax.axis("off")
    kw = dict(fontsize=FONT, family="DejaVu Sans", va="center", color="#1a1a1a")

    def box(x, y, w, h, fc):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec="white", lw=3))

    for i, tool in enumerate(tools):
        x = FAMILY_W + i * tool_w
        box(x, 0, tool_w, head_h, "#2b3a4a")
        ax.text(x + tool_w / 2, head_h / 2, HEAD.get(tool, tool), ha="center", linespacing=1.1, **{**kw, "color": "white", "weight": "bold"})
    y = head_h
    for fam in short.index:
        h = row_h[fam]
        box(0, y, FAMILY_W, h, "#eef1f4")
        ax.text(PAD, y + h / 2, FAMILY.get(fam, fam), ha="left", linespacing=1.1, **{**kw, "weight": "bold"})
        for i, tool in enumerate(tools):
            lines, does = cells[fam, tool]
            x = FAMILY_W + i * tool_w
            box(x, y, tool_w, h, FILL[does])
            ax.text(x + PAD, y + PAD, "\n".join(lines), ha="left", va="top", linespacing=LINE / FONT, **{k: v for k, v in kw.items() if k != "va"})
        y += h
    for j, s in enumerate(legend):
        ax.text(PAD, y + PAD + (j + 0.5) * LINE, s, ha="left", **kw)

    out = RESULTS / "figures" / "article"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=72)
    plt.close(fig)
    rows = [{"family": fam, **{t: " ".join(cells[fam, t][0]) for t in tools},
             **{f"{t}_does_it": cells[fam, t][1] for t in tools}} for fam in short.index]
    pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
    print(f"article/{name}.png {WIDTH} x {round(height)} px")


def main() -> None:
    for name, (tools, legend) in GROUPS.items():
        render(name, tools, legend)


if __name__ == "__main__":
    main()
