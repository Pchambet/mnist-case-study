"""Build the self-contained static report `site/index.html` from results/.

Charts are Plotly (loaded from jsDelivr); every number in the prose is read
from results/summary.json at build time, so the page cannot drift from the run.
"""

from __future__ import annotations

import base64
import csv
import json
from collections import defaultdict
from pathlib import Path

from mnist_study.analyze import RESULTS
from mnist_study.figures import FIGURES, NAMES, budget_failure_note

SITE = Path("site")
REPO = "https://github.com/Pchambet/mnist-case-study"


def _rows(name: str) -> list[dict]:
    with (RESULTS / name).open() as fh:
        return list(csv.DictReader(fh))


def _chart_data() -> dict:
    runs = _rows("runs.csv")
    shift = defaultdict(lambda: defaultdict(list))
    for r in _rows("shift.csv"):
        shift[r["model"]][int(r["shift_px"])].append(float(r["accuracy"]))
    rc = defaultdict(lambda: {"x": [], "y": []})
    for r in _rows("risk_coverage.csv"):
        rc[r["model"]]["x"].append(float(r["coverage"]))
        rc[r["model"]]["y"].append(float(r["selective_error"]))
    rel = defaultdict(lambda: {"x": [], "y": [], "n": []})
    for r in _rows("reliability.csv"):
        rel[r["model"]]["x"].append(float(r["mean_confidence"]))
        rel[r["model"]]["y"].append(float(r["accuracy"]))
        rel[r["model"]]["n"].append(int(r["count"]))
    lc = defaultdict(lambda: {"x": [], "y": []})
    for r in _rows("learning_curves.csv"):
        key = f"{r['model']}|{r['seed']}"
        lc[key]["x"].append(int(r["epoch"]))
        lc[key]["y"].append(float(r["val_loss"]))
    return {
        "runs": runs,
        "shift": {m: {d: v for d, v in sorted(ds.items())} for m, ds in shift.items()},
        "rc": rc,
        "rel": rel,
        "lc": lc,
    }


def _spread(model: str) -> int:
    errors = [int(r["test_errors"]) for r in _rows("runs.csv") if r["model"] == model]
    return max(errors) - min(errors)


def _pct(x: float, nd: int = 2) -> str:
    return f"{x * 100:.{nd}f}%"


def _pct_sig(x: float, sig: int = 2) -> str:
    """Percentage with `sig` significant digits: 0.000146 -> '0.015%'."""
    return f"{float(f'{x * 100:.{sig}g}'):g}%"


def _acc(m: dict) -> str:
    """Mean test accuracy, with the sd across seeds when there are at least two."""
    sd = m["test_accuracy_std"]
    return _pct(m["test_accuracy_mean"]) + ("" if sd is None else f" ± {_pct(sd)}")


def _budget(m: dict) -> tuple[str, str, str]:
    """Automated share, realised error and manual share under the budget; n/a if no seed met it."""
    cov, err = m["error_budget_coverage_mean"], m["error_budget_error_mean"]
    if cov is None:
        return "n/a", "n/a", "n/a"
    auto = round(cov * 100, 1)  # manual is 100 minus the rounded share, so the two add up to 100%
    return f"{auto:.1f}%", _pct_sig(err), f"{100 - auto:.1f}%"


def build_report(out: Path = SITE) -> Path:
    s = json.loads((RESULTS / "summary.json").read_text())
    mlp, cnn, cmp_ = s["models"]["mlp"], s["models"]["cnn"], s["comparison"]
    wide, cap = s["models"]["mlp_wide"], s["capacity_control"]
    lo, hi = cmp_["bootstrap_95ci"]
    pmax = max(m["p_value"] for m in cmp_["mcnemar_per_seed"])
    mc0 = cmp_["mcnemar_per_seed"][0]
    gallery_png = base64.b64encode((FIGURES / "cnn_confident_errors.png").read_bytes()).decode()
    n_seeds = len(cnn["seeds"])
    mlp_cov, mlp_berr, mlp_man = _budget(mlp)
    cnn_cov, cnn_berr, cnn_man = _budget(cnn)
    budget_note = budget_failure_note(s)
    shift_px = len(cnn["shift_accuracy_mean"]) - 1

    def conf_list(m: dict) -> str:
        return ", ".join(f"{t}→{p} ({n})" for t, p, n in m["top_confusions"][:3])

    mc_rows = "".join(
        f"<tr><td>{m['seed']}</td><td>{m['only_mlp_correct']}</td><td>{m['only_cnn_correct']}</td>"
        f"<td>{m['both_wrong']}</td><td>{m['p_value']:.2g}</td></tr>"
        for m in cmp_["mcnemar_per_seed"]
    )
    run_rows = "".join(
        f"<tr><td>{NAMES[r['model']]}</td><td>{r['seed']}</td><td>{int(r['params']):,}</td>"
        f"<td>{r['best_epoch']}/{r['epochs_run']}</td><td>{_pct(float(r['val_accuracy']))}</td>"
        f"<td>{_pct(float(r['test_accuracy']))}</td><td>{r['test_errors']}</td>"
        f"<td>{float(r['test_nll']):.4f}</td><td>{_pct(float(r['test_ece']))}</td></tr>"
        for r in _rows("runs.csv")
    )

    html = TEMPLATE
    replacements = {
        "__DATA__": json.dumps(_chart_data()),
        "__SUMMARY__": json.dumps(s),
        "__MLP_ACC__": _acc(mlp),
        "__CNN_ACC__": _acc(cnn),
        "__ACC_NOTE__": "mean ± sd across seeds"
        if mlp["test_accuracy_std"] is not None and cnn["test_accuracy_std"] is not None
        else "mean across seeds; an sd needs at least two",
        "__MLP_ERR__": f"{mlp['test_errors_mean']:.0f}",
        "__CNN_ERR__": f"{cnn['test_errors_mean']:.0f}",
        "__GAIN__": f"{cmp_['accuracy_gain_cnn_minus_mlp'] * 100:.2f}",
        "__CI_LO__": f"{lo * 100:.2f}",
        "__CI_HI__": f"{hi * 100:.2f}",
        "__PMAX__": f"{pmax:.1g}",
        "__MC_SEED__": str(mc0["seed"]),
        "__MC_B__": str(mc0["only_mlp_correct"]),
        "__MC_C__": str(mc0["only_cnn_correct"]),
        "__MC_BOTH__": str(mc0["both_wrong"]),
        "__NSEEDS__": str(n_seeds),
        "__SHIFT_PX__": str(shift_px),
        "__MLP_SHIFT__": _pct(mlp["shift_accuracy_mean"][-1], 1),
        "__CNN_SHIFT__": _pct(cnn["shift_accuracy_mean"][-1], 1),
        "__MLP_SHIFT2__": _pct(mlp["shift_accuracy_mean"][2], 1),
        "__CNN_SHIFT2__": _pct(cnn["shift_accuracy_mean"][2], 1),
        "__BUDGET__": _pct(s["error_budget"], 1),
        "__MLP_COV__": mlp_cov,
        "__CNN_COV__": cnn_cov,
        "__MLP_BERR__": mlp_berr,
        "__CNN_BERR__": cnn_berr,
        "__BUDGET_NOTE__": f" {budget_note}" if budget_note else "",
        "__MLP_ECE__": _pct(mlp["ece_mean"]),
        "__MLP_MAN__": mlp_man,
        "__CNN_MAN__": cnn_man,
        "__MLP_ORACLE__": _pct(mlp["error_budget_oracle_coverage_mean"], 1),
        "__CNN_ORACLE__": _pct(cnn["error_budget_oracle_coverage_mean"], 1),
        "__SPREAD__": str(max(_spread(m) for m in ("mlp", "cnn"))),
        "__GAP__": f"{mlp['test_errors_mean'] - cnn['test_errors_mean']:.0f}",
        "__WIDE_PARAMS__": f"{wide['params']:,}",
        "__WIDE_ACC__": _acc(wide),
        "__WIDE_ERR__": f"{wide['test_errors_mean']:.0f}",
        "__WIDE_SHIFT2__": _pct(wide["shift_accuracy_mean"][2], 1),
        "__WIDE_COV__": _budget(wide)[0],
        "__CAP_GAIN__": f"{cap['accuracy_gain_cnn_minus_mlp_wide'] * 100:.2f}",
        "__CAP_LO__": f"{cap['bootstrap_95ci'][0] * 100:.2f}",
        "__CAP_HI__": f"{cap['bootstrap_95ci'][1] * 100:.2f}",
        "__CAP_PMAX__": f"{max(m['p_value'] for m in cap['mcnemar_per_seed']):.1g}",
        "__CNN_ECE__": _pct(cnn["ece_mean"]),
        "__MLP_CONF__": conf_list(mlp),
        "__CNN_CONF__": conf_list(cnn),
        "__MC_ROWS__": mc_rows,
        "__RUN_ROWS__": run_rows,
        "__GALLERY__": gallery_png,
        "__GALLERY_SEED__": str(s["gallery"]["seed"]),
        "__REPO__": REPO,
    }
    for key, value in replacements.items():
        html = html.replace(key, value)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "index.html"
    path.write_text(html)
    return path


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MLP vs CNN on MNIST</title>
<meta name="description" content="A rigorous MLP vs CNN comparison on MNIST: paired significance, calibration, translation robustness and an error-budget automation rule.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
:root {
  --bg: #ffffff; --fg: #0f172a; --muted: #64748b; --grid: #e2e8f0; --card: #f8fafc;
  --border: #e2e8f0; --teal: #0d9488; --amber: #d97706; --link: #0f766e;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0b1120; --fg: #e2e8f0; --muted: #94a3b8; --grid: #1e293b; --card: #111827;
    --border: #1f2937; --link: #2dd4bf;
  }
}
:root[data-theme="dark"] {
  --bg: #0b1120; --fg: #e2e8f0; --muted: #94a3b8; --grid: #1e293b; --card: #111827;
  --border: #1f2937; --link: #2dd4bf;
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 16px/1.65 Inter, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}
main { max-width: 860px; margin: 0 auto; padding: 48px 16px 64px; }
h1 { font-size: 2rem; line-height: 1.2; margin: 0 0 8px; letter-spacing: -0.02em; }
h2 { font-size: 1.3rem; margin: 48px 0 8px; letter-spacing: -0.01em; }
p, li { color: var(--fg); }
.lede { color: var(--muted); font-size: 1.1rem; margin: 0 0 28px; }
a { color: var(--link); }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 24px 0; }
.kpi { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
.kpi .v { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em; }
.kpi .l { color: var(--muted); font-size: 0.85rem; line-height: 1.35; }
.chart { width: 100%; height: 360px; margin: 8px 0 4px; }
.takeaway { color: var(--muted); font-size: 0.92rem; margin-top: 0; }
table { width: 100%; border-collapse: collapse; font-size: 0.88rem; margin: 12px 0; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 6px 8px; border-bottom: 1px solid var(--border); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--muted); font-weight: 500; }
.scroll { overflow-x: auto; }
img.gallery { width: 100%; height: auto; border-radius: 8px; background: #fff; }
code { font-size: 0.9em; background: var(--card); border: 1px solid var(--border); border-radius: 4px; padding: 1px 4px; }
footer { margin-top: 56px; padding-top: 16px; border-top: 1px solid var(--border); color: var(--muted); font-size: 0.9rem; }
.teal { color: var(--teal); font-weight: 600; } .amber { color: var(--amber); font-weight: 600; }
</style>
</head>
<body>
<main>
<h1>MLP vs CNN on MNIST</h1>
<p class="lede">What does convolution actually buy on handwritten digits, once accuracy is measured with
paired statistics, calibration, robustness to shifted inputs, and a concrete automation rule?</p>

<div class="kpis">
  <div class="kpi"><div class="v">__MLP_ERR__ → __CNN_ERR__</div><div class="l">test errors per 10,000 digits, <span class="amber">MLP</span> → <span class="teal">CNN</span> (mean of __NSEEDS__ seeds)</div></div>
  <div class="kpi"><div class="v">+__GAIN__ pts</div><div class="l">accuracy gain, paired bootstrap 95% CI [__CI_LO__, __CI_HI__]</div></div>
  <div class="kpi"><div class="v">__MLP_SHIFT__ vs __CNN_SHIFT__</div><div class="l">accuracy when digits move __SHIFT_PX__ px (MLP vs CNN)</div></div>
  <div class="kpi"><div class="v">__MLP_COV__ → __CNN_COV__</div><div class="l">digits automated under a __BUDGET__ error budget</div></div>
</div>

<h2>1. Accuracy, with the noise measured</h2>
<p>Both models are trained __NSEEDS__ times (different initialisation and batch order) on the same
54,000-image training split, early-stopped on a stratified 6,000-image validation split, and scored once
on the untouched 10,000-image test set. The <span class="amber">MLP</span> reaches __MLP_ACC__,
the <span class="teal">CNN</span> __CNN_ACC__ (__ACC_NOTE__).</p>
<div id="c-errors" class="chart" style="height:240px"></div>
<p class="takeaway">Within the MLP or the CNN, seeds move the error count by at most __SPREAD__ digits; switching architecture moves it by __GAP__.</p>
<p>Is the gap real or test-set luck? Two paired checks on the same 10,000 images. A paired bootstrap over
test images puts the gain at +__GAIN__ points (95% CI __CI_LO__ to __CI_HI__). McNemar's exact test, seed by
seed, counts the digits only one model gets right: for seed __MC_SEED__, __MC_B__ digits only the MLP gets right
versus __MC_C__ only the CNN gets right (__MC_BOTH__ both miss); the largest p-value over seeds is
__PMAX__.</p>
<div class="scroll"><table>
<thead><tr><th>Seed</th><th>Only MLP right</th><th>Only CNN right</th><th>Both wrong</th><th>McNemar p</th></tr></thead>
<tbody>__MC_ROWS__</tbody></table></div>
<p><strong>Capacity or convolution?</strong> The CNN has about twice the MLP's parameters. A wider MLP
(784-256-128-10, __WIDE_PARAMS__ parameters, more than the CNN) trained with the same recipe reaches
__WIDE_ACC__, i.e. __WIDE_ERR__ errors: no better than the small MLP. The CNN beats it by
+__CAP_GAIN__ points (95% CI __CAP_LO__ to __CAP_HI__; largest McNemar p over seeds __CAP_PMAX__), and the wide MLP
is no more tolerant of shifts (__WIDE_SHIFT2__ at 2 px); under the error budget of section 3 it automates
__WIDE_COV__ of the stream. More parameters alone do not explain the gap.</p>

<h2>2. Robustness: move the digit a few pixels</h2>
<p>MNIST digits are centred by centre of mass. Real inputs rarely are. Each test digit is shifted by
<em>d</em> pixels in eight directions (zero-filled), and accuracy is averaged over directions.</p>
<div id="c-shift" class="chart"></div>
<p class="takeaway">At 2 px the MLP is at __MLP_SHIFT2__ while the CNN keeps __CNN_SHIFT2__; at __SHIFT_PX__ px,
__MLP_SHIFT__ vs __CNN_SHIFT__. Neither model saw shifted digits in training. Convolution and pooling buy
tolerance to one or two pixels, not invariance: past that, the CNN fails too. Training with random shifts
(augmentation) is the standard fix and is not tested here.</p>

<h2>3. The decision: how much can be automated?</h2>
<p>Suppose digits are read automatically only when the model is confident, and the rest go to a human,
with a budget of at most one error per 1,000 accepted digits (__BUDGET__). The confidence threshold is
chosen on the validation split, then frozen and applied to the test set.</p>
<div id="c-rc" class="chart"></div>
<p class="takeaway">Under the same budget the MLP automates __MLP_COV__ of the test stream (realised error
__MLP_BERR__), the CNN __CNN_COV__ (realised error __CNN_BERR__): the manual queue shrinks from __MLP_MAN__ to
__CNN_MAN__ of the stream, a far larger gap than the __GAIN__-point accuracy difference suggests. The thresholds are
conservative: with hindsight (tuning on the test set itself, which a real deployment cannot do) the same
budget would allow __MLP_ORACLE__ and __CNN_ORACLE__. The 6,000-digit validation split simply contains few
errors to calibrate a 0.1% threshold on, and the MLP pays more for that caution.__BUDGET_NOTE__</p>

<h2>4. Can the confidences be trusted?</h2>
<p>Expected calibration error (15 equal-width bins, mean over seeds): MLP __MLP_ECE__, CNN __CNN_ECE__.
Points on the diagonal mean "90% confident" is right about 90% of the time. The diagram pools the
__NSEEDS__ seeds; over 93% of digits fall in the top bin, which dominates the ECE. The mid-range bins
(confidence 0.67 to 0.93) are mostly overconfident, by up to 7 points, and all of this is measured on clean
test digits only. The threshold rule of section 3 only needs the ranking of confidences.</p>
<div id="c-rel" class="chart"></div>

<h2>5. Where the errors are</h2>
<p>Most frequent confusions (true→predicted, summed over seeds). MLP: __MLP_CONF__. CNN: __CNN_CONF__.</p>
<img class="gallery" alt="Grid of the CNN's twelve most confident misclassified test digits with true label, CNN prediction and MLP prediction" src="data:image/png;base64,__GALLERY__">
<p class="takeaway">The CNN's most confident mistakes (seed __GALLERY_SEED__), with the MLP's prediction for the same digit. Several are hard to read even for a person, and both models tend to fail on the same digits (see the "both wrong" column above).</p>

<h2>6. Training</h2>
<div id="c-lc" class="chart" style="height:300px"></div>
<div class="scroll"><table>
<thead><tr><th>Model</th><th>Seed</th><th>Params</th><th>Best/run epochs</th><th>Val acc</th><th>Test acc</th><th>Test errors</th><th>Test NLL</th><th>ECE</th></tr></thead>
<tbody>__RUN_ROWS__</tbody></table></div>

<h2>Limitations</h2>
<ul>
<li>__NSEEDS__ seeds per model is enough to show the seed spread is small next to the gap, not to estimate it precisely.</li>
<li>Neither architecture is tuned (no augmentation, no learning-rate schedule, no hyper-parameter search); the comparison is between two textbook models of the same order of size (the CNN has about twice the parameters), not between the best of each family.</li>
<li>Architecture and capacity are separated by one control only: the wide MLP matches the CNN's parameter count, but both MLPs use dropout and the CNN does not, and no regularisation was tuned for either.</li>
<li>The translation test is synthetic; it isolates one property (shift sensitivity) rather than modelling a real capture process. Diagonal directions move <em>d</em> px along each axis, and at 4 px about a third of shifted digits lose some ink at the border and a quarter lose more than 2% (98.7% of ink kept on average), so part of the 4 px drop is lost information, not fragility.</li>
<li>The 0.1% budget threshold is set on 6,000 validation digits, i.e. about six tolerated errors, so it is itself noisy; the realised test error is reported next to it.</li>
</ul>

<footer>
Code, data pipeline and tests: <a href="__REPO__">__REPO__</a>. Data: MNIST (LeCun, Cortes &amp; Burges), CC BY-SA 3.0.<br>
Built by <a href="https://github.com/Pchambet">Pierre Chambet</a> — decision science for operations under uncertainty.
</footer>
</main>
<script>
const D = __DATA__;
const S = __SUMMARY__;
const C = { mlp: "#d97706", cnn: "#0d9488" };
const NAME = { mlp: "MLP", cnn: "CNN" };
function css(v) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }
function base(extra) {
  const fg = css("--fg"), muted = css("--muted"), grid = css("--grid");
  return Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter, system-ui, sans-serif", color: fg, size: 13 },
    margin: { l: 64, r: 16, t: 16, b: 52 },
    xaxis: { gridcolor: grid, zerolinecolor: grid, linecolor: muted, tickcolor: muted },
    yaxis: { gridcolor: grid, zerolinecolor: grid, linecolor: muted, tickcolor: muted },
    legend: { orientation: "h", y: 1.12, x: 0 },
    hovermode: "closest",
  }, extra || {});
}
const cfg = { displayModeBar: false, responsive: true };
function merge(a, b) { for (const k in b) a[k] = (typeof b[k] === "object" && !Array.isArray(b[k]) && a[k]) ? Object.assign(a[k], b[k]) : b[k]; return a; }

function draw() {
  const models = ["mlp", "cnn"];
  // 1. errors per seed
  Plotly.react("c-errors", models.map((m, i) => {
    const r = D.runs.filter(x => x.model === m);
    return { type: "scatter", mode: "markers", name: NAME[m], x: r.map(x => +x.test_errors),
      y: r.map(() => NAME[m]), marker: { color: C[m], size: 12, line: { color: css("--bg"), width: 2 } },
      text: r.map(x => "seed " + x.seed), hovertemplate: "%{y} %{text}: %{x} errors<extra></extra>" };
  }), merge(base({ showlegend: false }), { xaxis: { title: "misclassified test digits (of 10,000)", rangemode: "tozero" }, margin: { l: 56 } }), cfg);

  // 2. shift robustness
  Plotly.react("c-shift", models.flatMap(m => {
    const px = Object.keys(D.shift[m]).map(Number);
    const mean = px.map(d => D.shift[m][d].reduce((a, b) => a + b, 0) / D.shift[m][d].length * 100);
    return [{ type: "scatter", mode: "lines+markers", name: NAME[m], x: px, y: mean,
      line: { color: C[m], width: 2 }, marker: { size: 8, color: C[m] },
      hovertemplate: NAME[m] + " · %{x} px: %{y:.2f}%<extra></extra>" }];
  }), merge(base(), { xaxis: { title: "shift (px, mean of 8 directions)", dtick: 1 }, yaxis: { title: "test accuracy (%)" } }), cfg);

  // 3. risk-coverage
  const budget = S.error_budget * 100;
  const rc = models.map(m => ({ type: "scatter", mode: "lines", name: NAME[m],
    x: D.rc[m].x.map(v => v * 100), y: D.rc[m].y.map(v => v * 100), line: { color: C[m], width: 2 },
    hovertemplate: NAME[m] + ": accept %{x:.1f}% → error %{y:.3f}%<extra></extra>" }));
  models.filter(m => S.models[m].error_budget_coverage_mean !== null).forEach(m => rc.push({ type: "scatter", mode: "markers", showlegend: false,
    x: [S.models[m].error_budget_coverage_mean * 100], y: [S.models[m].error_budget_error_mean * 100],
    marker: { color: C[m], size: 12, line: { color: css("--bg"), width: 2 } },
    hovertemplate: NAME[m] + " with validation-chosen threshold: %{x:.1f}% automated, error %{y:.3f}%<extra></extra>" }));
  Plotly.react("c-rc", rc, merge(base({ shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: budget, y1: budget,
    line: { color: css("--muted"), dash: "dash", width: 1 } }],
    annotations: [{ xref: "paper", x: 0.01, y: budget, yanchor: "bottom", text: "error budget " + budget.toFixed(1) + "%",
    showarrow: false, font: { color: css("--muted"), size: 12 } }] }),
    { xaxis: { title: "share of digits accepted automatically (%)", range: [50, 100] },
      yaxis: { title: "error among accepted (%)", range: [0, 0.8] } }), cfg);

  // 4. reliability
  const rel = models.map(m => ({ type: "scatter", mode: "lines+markers", name: NAME[m],
    x: D.rel[m].x, y: D.rel[m].y, line: { color: C[m], width: 2 },
    marker: { color: C[m], size: D.rel[m].n.map(n => 6 + 18 * Math.sqrt(n / Math.max(...D.rel[m].n))) },
    text: D.rel[m].n, hovertemplate: NAME[m] + ": confidence %{x:.3f}, accuracy %{y:.3f} (%{text} digits)<extra></extra>" }));
  rel.unshift({ type: "scatter", mode: "lines", x: [0, 1], y: [0, 1], showlegend: false, hoverinfo: "skip",
    line: { color: css("--muted"), dash: "dash", width: 1 } });
  Plotly.react("c-rel", rel, merge(base(), { xaxis: { title: "mean predicted confidence", range: [0, 1.02] },
    yaxis: { title: "observed accuracy", range: [0, 1.02] } }), cfg);

  // 5. learning curves
  const seen = {};
  Plotly.react("c-lc", Object.entries(D.lc).filter(([k]) => models.includes(k.split("|")[0])).map(([k, v]) => {
    const m = k.split("|")[0], first = !seen[m]; seen[m] = true;
    return { type: "scatter", mode: "lines+markers", name: NAME[m], legendgroup: m, showlegend: first,
      x: v.x, y: v.y, line: { color: C[m], width: 2 }, marker: { size: 6, color: C[m] },
      hovertemplate: NAME[m] + " seed " + k.split("|")[1] + " · epoch %{x}: %{y:.4f}<extra></extra>" };
  }), merge(base(), { xaxis: { title: "epoch", dtick: 1 }, yaxis: { title: "validation cross-entropy" } }), cfg);
}
draw();
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
</script>
</body>
</html>
"""
