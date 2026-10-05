"""Recreate the two report figures from checked evidence; never simulate."""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import CASES, read, rows, write_json


def build(data, output):
    global OUT
    OUT = Path(output)
    OUT.mkdir(parents=True, exist_ok=True)
    data = Path(data)
    weights = read(data / "weights.json")["policies"]
    curves = rows(OUT.parent / "tables/native_learning.csv")
    curve_rows = rows(OUT.parent / "tables/classic_learning.csv")
    b7 = read(data / "baselines/native_b7.json")
    b11 = read(data / "baselines/native_b11.json")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.4,
            "axes.unicode_minus": False,
            "pdf.use14corefonts": True,
            "ps.useafm": True,
        }
    )
    C = {
        "A": "#2166ac",
        "B": "#d95f02",
        "C": "#16816a",
        "base": "#444444",
        "sel": "#956900",
    }

    def panel(ax, label):
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.text(0, 1.03, label, transform=ax.transAxes, va="bottom", fontsize=9)

    def save(fig, name):
        fig.savefig(OUT / (name + ".pdf"), bbox_inches="tight", pad_inches=0.03)
        fig.savefig(
            OUT / (name + ".png"), dpi=180, bbox_inches="tight", pad_inches=0.03
        )
        plt.close(fig)

    base = {
        f'b{b}_{r["regime"]}': r["summary"]["return"]["mean"]
        for b, p in [(7, b7), (11, b11)]
        for r in p["runs"]
        if r["policy"] == "direct"
    }
    sel = {case: weights["native_" + case]["selected_transitions"] for case in CASES}
    fig, axs = plt.subplots(2, 2, figsize=(6.75, 3.8))
    fig.subplots_adjust(
        left=0.09, right=0.985, bottom=0.11, top=0.94, hspace=0.55, wspace=0.27
    )
    for row, b in enumerate((7, 11)):
        for col, reg in enumerate(("full", "partial")):
            ax = axs[row, col]
            case = f"b{b}_{reg}"
            if b == 7:
                selected = sorted(
                    [r for r in curves if r["case"] == case and r["variant"] == "A"],
                    key=lambda r: int(r["transitions"]),
                )
                ax.plot(
                    [int(r["transitions"]) / 1e6 for r in selected],
                    [float(r["return_mean"]) for r in selected],
                    color=C["A"],
                    lw=1.1,
                    marker="o",
                    ms=1.7,
                    label="DQN",
                )
            else:
                control = sorted(
                    [r for r in curves if r["case"] == case and r["variant"] == "A"],
                    key=lambda r: int(r["transitions"]),
                )
                selected = sorted(
                    [
                        r
                        for r in curves
                        if r["case"] == case
                        and (
                            (r["variant"] == "A" and int(r["transitions"]) <= 1024000)
                            or (
                                r["variant"] == "B"
                                and 1024000 < int(r["transitions"]) <= 2048000
                            )
                            or (r["variant"] == "C" and int(r["transitions"]) > 2048000)
                        )
                    ],
                    key=lambda r: int(r["transitions"]),
                )
                ax.plot(
                    [int(r["transitions"]) / 1e6 for r in control],
                    [float(r["return_mean"]) for r in control],
                    color="#aaaaaa",
                    lw=0.9,
                    label="Constant rate",
                )
                ax.plot(
                    [int(r["transitions"]) / 1e6 for r in selected],
                    [float(r["return_mean"]) for r in selected],
                    color=C["A"],
                    lw=1.1,
                    marker="o",
                    ms=1.7,
                    label="Selected schedule",
                )
            ax.axhline(base[case], ls="--", color=C["base"], lw=0.9, label="Direct")
            r = next(r for r in selected if int(r["transitions"]) == sel[case])
            ax.scatter(
                sel[case] / 1e6,
                float(r["return_mean"]),
                marker="*",
                s=55,
                color=C["sel"],
                zorder=5,
            )
            ax.set_xlabel("Training transitions (millions)")
            ax.set_ylabel("Validation return")
            ax.set_xlim(0, 2.1 if b == 7 else 4.15)
            ax.set_ylim(
                (-110, 145) if b == 7 else ((-70, 85) if reg == "full" else (-110, 45))
            )
            panel(ax, f"({chr(97+row*2+col)}) {b} x {b}, {reg}")
            if b == 7:
                inset = ax.inset_axes([0.44, 0.19, 0.53, 0.36])
                v = [r for r in selected if int(r["transitions"]) >= 256000]
                inset.plot(
                    [int(r["transitions"]) / 1e6 for r in v],
                    [float(r["return_mean"]) for r in v],
                    color=C["A"],
                    lw=0.8,
                )
                inset.axhline(base[case], ls="--", color=C["base"], lw=0.6)
                inset.set_ylim((115, 136) if reg == "full" else (108, 124))
                inset.set_xticks([0.512, 2.048])
                inset.tick_params(labelsize=6)
                inset.grid(axis="y")
    axs[0, 0].legend(loc="lower left", fontsize=7)
    axs[1, 0].legend(loc="lower right", fontsize=7, handlelength=1.5)
    save(fig, "native_learning")
    classic_series = {}
    fig, axs = plt.subplots(2, 2, figsize=(6.75, 3.5))
    fig.subplots_adjust(
        left=0.09, right=0.985, bottom=0.12, top=0.93, hspace=0.58, wspace=0.28
    )
    for col, reg in enumerate(("full", "partial")):
        baseline = {
            p: read(data / f"baselines/classic_{reg}_{p}.json")["summary"]
            for p in ("bfs", "greedy")
        }
        by = {int(r["counter"]): r for r in curve_rows if r["run"] == "scratch_" + reg}
        by.update(
            {int(r["counter"]): r for r in curve_rows if r["run"] == "extended_" + reg}
        )
        if reg == "full":
            by.update(
                {int(r["counter"]): r for r in curve_rows if r["run"] == "full_control"}
            )
        ordered = [by[k] for k in sorted(by)]
        step = weights["classic_" + reg]["selected_transitions"]
        classic_series[reg] = {
            "counters": [int(r["counter"]) for r in ordered],
            "wins": [float(r["wins"]) for r in ordered],
            "mean_reward": [float(r["mean_reward"]) for r in ordered],
            "selected_counter": step,
        }
        for row, (metric, ylabel) in enumerate(
            [("wins", "Wins per 100 games"), ("mean_reward", "Mean validation return")]
        ):
            ax = axs[row, col]
            ax.plot(
                [int(r["counter"]) / 1e6 for r in ordered],
                [float(r[metric]) for r in ordered],
                color=C["A"],
                lw=1.1,
                marker="o",
                ms=1.7,
                label="DQN",
            )
            if reg == "full":
                ax.axvspan(4.096, 5.12, color="#f0f0f0", zorder=0)
                ax.axvline(4.096, color="#888888", lw=0.6, ls=":")
            ax.scatter(
                step / 1e6,
                float(by[step][metric]),
                marker="*",
                s=55,
                color=C["sel"],
                zorder=5,
            )
            if row == 0:
                ax.axhline(0, color="#555555", ls="--", lw=0.8, label="BFS / Greedy")
                ax.set_ylim((-2, 62) if reg == "full" else (-0.15, 3.6))
                if reg == "partial":
                    ax.set_yticks([0, 1, 2, 3])
            else:
                for p, ls in [("bfs", "--"), ("greedy", ":")]:
                    ax.axhline(
                        baseline[p][metric],
                        color="#444444" if p == "bfs" else "#999999",
                        ls=ls,
                        lw=0.9,
                        label=p.upper() if p == "bfs" else "Greedy",
                    )
                ax.set_ylim((-5, 82) if reg == "full" else (-2, 15))
            ax.set_xlabel("Training transitions (millions)")
            ax.set_ylabel(ylabel)
            ax.set_xlim(0, 5.25)
            panel(
                ax,
                f"({chr(97+row*2+col)}) {reg.capitalize()}: "
                + ("completion" if row == 0 else "return"),
            )
    axs[0, 0].legend(loc="upper left", fontsize=7)
    axs[1, 0].legend(loc="upper left", fontsize=7, handlelength=1.5)
    save(fig, "classic_learning")
    write_json(
        OUT / "provenance.json",
        dict(native_selected=sel, classic_learning=classic_series, seed=0),
    )
