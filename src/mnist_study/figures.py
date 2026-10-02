"""Static PNG figures for the README, drawn only from results/ (plus raw images for the gallery)."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mnist_study.analyze import RESULTS
from mnist_study.data import RAW_PATH, load_npz
from mnist_study.models import MODEL_NAMES

FIGURES = Path("docs/figures")
INK, MUTED, GRID = "#0f172a", "#64748b", "#e2e8f0"
COLORS = {"mlp": "#d97706", "cnn": "#0d9488"}
LABELS = {"mlp": "MLP (109k params)", "cnn": "CNN (225k params)"}
NAMES = {"mlp": "MLP", "cnn": "CNN", "mlp_wide": "Wide MLP"}
WIDE_LABEL = "Wide MLP (235k params)"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def _rows(name: str) -> list[dict]:
    with (RESULTS / name).open() as fh:
        return list(csv.DictReader(fh))


def _summary() -> dict:
    return json.loads((RESULTS / "summary.json").read_text())


def _save(fig: plt.Figure, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path)
    plt.close(fig)
    return path


def hero() -> Path:
    s = _summary()
    runs = _rows("runs.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.5]})

    # Bottom to top: CNN, the wide MLP (capacity control, same parameter count), MLP.
    bars = [("cnn", LABELS["cnn"], COLORS["cnn"]), ("mlp_wide", WIDE_LABEL, MUTED)]
    bars.append(("mlp", LABELS["mlp"], COLORS["mlp"]))
    for i, (name, _, color) in enumerate(bars):
        errs = [int(r["test_errors"]) for r in runs if r["model"] == name]
        mean = np.mean(errs)
        ax1.barh(i, mean, color=color, height=0.55, alpha=0.9)
        ax1.scatter(errs, [i] * len(errs), color=INK, s=14, zorder=3)
        ax1.text(max(errs) + 8, i, f"{mean:.0f} errors", va="center", color=INK, fontsize=10)
    ax1.set_yticks(range(len(bars)), [label for _, label, _ in bars])
    ax1.set_xlabel("Misclassified test digits (of 10,000); dots = seeds")
    ax1.set_xlim(0, max(int(r["test_errors"]) for r in runs) * 1.35)
    ax1.grid(axis="y", visible=False)
    ax1.set_title("Clean test set\n(doubling the MLP's width does not help)")

    px = np.arange(len(s["models"]["mlp"]["shift_accuracy_mean"]))
    for name in ("mlp", "cnn"):
        m = s["models"][name]
        mean = np.array(m["shift_accuracy_mean"]) * 100
        ax2.plot(px, mean, color=COLORS[name], lw=2, marker="o", ms=5)
        if m["shift_accuracy_std"] is not None:  # no band for a single seed
            std = np.array(m["shift_accuracy_std"]) * 100
            ax2.fill_between(px, mean - std, mean + std, color=COLORS[name], alpha=0.15, lw=0)
        ax2.annotate(
            f"{mean[-1]:.1f}%",
            (px[-1], mean[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=INK,
        )
        ax2.text(
            px[1] + 0.05,
            mean[1] + (2 if name == "cnn" else -5),
            name.upper(),
            color=COLORS[name],
            fontweight="bold",
        )
    ax2.set_xticks(px)
    ax2.set_xlabel("Translation of the test digit (pixels, averaged over 8 directions)")
    ax2.set_ylabel("Test accuracy (%)")
    ax2.set_xlim(-0.2, px[-1] + 0.6)
    ax2.set_title("Same digits, shifted")

    d = s["models"]
    e_mlp, e_cnn = d["mlp"]["test_errors_mean"], d["cnn"]["test_errors_mean"]
    fig.suptitle(
        f"The CNN cuts test errors by {1 - e_cnn / e_mlp:.0%} ({e_mlp:.0f} → {e_cnn:.0f} per 10k) "
        f"and tolerates small shifts better, but neither is shift-invariant: at {px[-1]} px "
        f"accuracy drops to {d['mlp']['shift_accuracy_mean'][-1]:.0%} and "
        f"{d['cnn']['shift_accuracy_mean'][-1]:.0%}",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
        wrap=True,
    )
    fig.tight_layout()
    return _save(fig, "hero.png")


def budget_failure_note(s: dict, names=tuple(NAMES)) -> str:
    """One sentence naming the runs where no validation threshold met the budget, or ''."""
    failed = [
        f"{NAMES[name]} seed{'s' * (len(seeds) > 1)} {', '.join(map(str, seeds))}"
        for name in names
        if (seeds := s["models"][name]["error_budget_failed_seeds"])
    ]
    if not failed:
        return ""
    return (
        f"No validation threshold met the error budget for {'; '.join(failed)}: such runs "
        "would automate nothing and are left out of the coverage and error means."
    )


def risk_coverage() -> Path:
    s = _summary()
    curves = defaultdict(lambda: ([], []))
    for r in _rows("risk_coverage.csv"):
        if r["model"] not in MODEL_NAMES:
            continue
        curves[r["model"]][0].append(float(r["coverage"]) * 100)
        curves[r["model"]][1].append(float(r["selective_error"]) * 100)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for name, (cov, err) in curves.items():
        ax.plot(cov, err, color=COLORS[name], lw=2, label=LABELS[name])
        m = s["models"][name]
        if m["error_budget_coverage_mean"] is None:  # no seed met the budget on validation
            continue
        c, e = m["error_budget_coverage_mean"] * 100, m["error_budget_error_mean"] * 100
        ax.scatter([c], [e], color=COLORS[name], s=60, zorder=3, edgecolor="white", lw=1.5)
    # A text key instead of arrows: any arrow to the CNN point would cross the MLP curve.
    cov = {m: s["models"][m]["error_budget_coverage_mean"] for m in ("mlp", "cnn")}
    shown = {m: "none" if c is None else f"{c:.1%}" for m, c in cov.items()}
    ax.text(
        51,
        0.56,
        "Dots: threshold chosen on validation, applied to test\n"
        f"MLP {shown['mlp']} automated · CNN {shown['cnn']}",
        color=INK,
        fontsize=9.5,
        va="top",
    )
    budget = s["error_budget"] * 100
    ax.axhline(budget, color=MUTED, lw=1, ls="--")
    ax.text(
        50.5,
        budget,
        f"error budget {budget:.1f}%",
        color=MUTED,
        fontsize=9,
        ha="left",
        va="bottom",
    )
    ax.set_xlabel("Coverage: share of digits accepted automatically (%)")
    ax.set_ylabel("Error rate among accepted digits (%)")
    ax.set_xlim(50, 100)
    ax.set_ylim(0, 0.8)
    ax.legend(frameon=False, loc="upper left")
    if None in cov.values():
        ax.set_title(f"Coverage when errors are held to ≤{budget:.1f}%")
    else:
        ax.set_title(
            f"Holding errors to ≤{budget:.1f}%, the CNN reads {cov['cnn']:.0%} of digits "
            f"automatically, the MLP {cov['mlp']:.0%}"
        )
    if note := budget_failure_note(s, ("mlp", "cnn")):
        fig.text(0.01, -0.02, note, color=MUTED, fontsize=9, va="top", wrap=True)
    return _save(fig, "risk_coverage.png")


def reliability() -> Path:
    s = _summary()
    by = defaultdict(list)
    for r in _rows("reliability.csv"):
        by[r["model"]].append(r)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, name in zip(axes, ("mlp", "cnn"), strict=True):
        rows = by[name]
        conf = np.array([float(r["mean_confidence"]) for r in rows])
        acc = np.array([float(r["accuracy"]) for r in rows])
        cnt = np.array([int(r["count"]) for r in rows])
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
        ax.scatter(conf, acc, s=10 + 120 * np.sqrt(cnt / cnt.max()), color=COLORS[name], alpha=0.85)
        ax.plot(conf, acc, color=COLORS[name], lw=1.2)
        ax.set_title(f"{name.upper()}: ECE {s['models'][name]['ece_mean'] * 100:.2f}%")
        ax.set_xlabel("Mean predicted confidence")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Observed accuracy")
    fig.suptitle(
        "Both models are well calibrated: expected calibration error below "
        f"{max(s['models'][m]['ece_mean'] for m in ('mlp', 'cnn')) * 100:.1f}%\n"
        f"ECE is the mean over seeds; the diagram pools the {len(s['models']['mlp']['seeds'])} "
        "seeds (marker area = number of digits)",
        x=0.01,
        ha="left",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, "reliability.png")


def learning_curves() -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    by = defaultdict(lambda: ([], []))
    for r in _rows("learning_curves.csv"):
        if r["model"] not in MODEL_NAMES:
            continue
        key = (r["model"], r["seed"])
        by[key][0].append(int(r["epoch"]))
        by[key][1].append(float(r["val_loss"]))
    for (name, _), (ep, vl) in by.items():
        ax.plot(ep, vl, color=COLORS[name], lw=1.6, alpha=0.85)
        best = int(np.argmin(vl))
        ax.scatter([ep[best]], [vl[best]], color=COLORS[name], s=28, zorder=3)
    for name in ("mlp", "cnn"):
        ep, vl = next(v for k, v in by.items() if k[0] == name)
        ax.text(
            ep[-1] + 0.15,
            vl[-1],
            name.upper(),
            color=COLORS[name],
            va="center",
            fontweight="bold",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation cross-entropy")
    best = {
        m: sorted({int(r["best_epoch"]) for r in _rows("runs.csv") if r["model"] == m})
        for m in COLORS
    }
    ax.set_title(
        f"CNN: lower validation loss in fewer epochs (restored epoch "
        f"{'/'.join(map(str, best['cnn']))} vs {'/'.join(map(str, best['mlp']))})"
    )
    return _save(fig, "learning_curves.png")


def _count_phrase(k: int, n: int) -> str:
    return f"all {n}" if k == n else f"{k} of them"


def gallery() -> Path:
    s = _summary()
    _, _, x_test, _ = load_npz(RAW_PATH)
    items = s["gallery"]["mistakes"]
    cols = 6
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.45, rows * 1.75))
    for ax in axes.flat:
        ax.axis("off")
    for ax, it in zip(axes.flat, items, strict=False):
        ax.imshow(x_test[it["index"]], cmap="gray_r")
        mlp_ok = it["mlp_pred"] == it["label"]
        ax.set_title(
            f"true {it['label']} · CNN {it['cnn_pred']}\nMLP {it['mlp_pred']}"
            + (" ✓" if mlp_ok else ""),
            fontsize=8.5,
            loc="center",
            fontweight="normal",
            color=COLORS["cnn"] if mlp_ok else INK,
        )
    fig.suptitle(
        f"The CNN's {len(items)} most confident mistakes (seed {s['gallery']['seed']}): "
        "the MLP makes the same "
        f"wrong call on {_count_phrase(sum(it['mlp_pred'] == it['cnn_pred'] for it in items), len(items))}",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, "cnn_confident_errors.png")


def render_all() -> list[Path]:
    paths = [hero(), risk_coverage(), reliability(), learning_curves(), gallery()]
    for p in paths:
        print(p)
    return paths
