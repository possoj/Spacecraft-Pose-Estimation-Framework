"""
Copyright (c) 2026 Julien Posso

Publish fixed, manually confirmed filters using the existing cache and scoring.

The fixed evaluation is descriptive calibration on the sweep videos.
"""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.spe.utils import quat2euler
from src.temporal.benchmark_filters import DT, pmf_filter, qdistance, unit

METHODS = ("raw", "moving_average", "ema", "adaptive")
LABELS = {"raw": "Raw", "moving_average": "Moving average", "ema": "EMA", "adaptive": "Adaptive"}
METRICS = ("mean", "median", "p95", "std", "rmse", "jitter", "rpe25", "lag_frames")
COLORS = {"raw": "0.68", "moving_average": "#e68a00", "ema": "#0072b2", "adaptive": "#a43b90"}


def read_parameters(path):
    """
    Read and validate the confirmed fixed-filter settings.

    Args:
        path (str or Path): JSON file with attitude and position filter settings.

    Returns:
        tuple[dict, dict]: Original settings and parameters formatted for pmf_filter.

    Raises:
        ValueError: If gains, sensitivities or window lengths are invalid.
    """
    settings = json.loads(Path(path).read_text())
    parameters = {}
    for component in ("ori", "pos"):
        selected = settings[component]
        alpha, n = (float(selected["adaptive"][key]) for key in ("alpha", "n"))
        beta = float(selected["ema"]["beta"])
        window = selected["moving_average"]["window"]
        if not np.all(np.isfinite([alpha, n, beta])) or alpha <= 0 or not 0 < n <= 1 or not 0 < beta <= 1:
            raise ValueError("Expected finite alpha > 0, and n, beta in (0, 1]")
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError("Moving-average windows must be positive integers")
        parameters[component] = {"adaptive": (alpha, n), "ema": beta, "moving_average": window}
    return settings, parameters


def write_csv(path, rows):
    """
    Write nonempty evaluation records to a CSV file.

    Args:
        path (str or Path): Destination file.
        rows (list[dict]): Nonempty records sharing the same columns.
    """
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_legacy(args, rows, parameters, output):
    """
    Compare matching fixed results with available thesis workbooks.

    Checks apply to complete mursop_fp32_dspeed runs. Adaptive comparisons are
    limited to settings matching the historical values; workbooks are read only.

    Args:
        args (argparse.Namespace): Command-line settings for paths and execution options.
        rows (list[dict]): Per-sequence/component/method evaluation metrics.
        parameters (dict): Fixed filter parameters indexed by component and method.
        output (Path): Directory receiving legacy_check.csv when comparisons exist.

    Raises:
        AssertionError: If an attitude difference exceeds 0.1 deg or a position difference exceeds 0.01 m.
    """
    legacy = Path("experiments/temporal_phd_thesis/mursop_fp32_dspeed/n and alpha tuned")
    if (Path(getattr(args, "model", "")).name != "mursop_fp32_dspeed"
            or getattr(args, "max_frames", None)):
        return
    references = {"raw": "still_metrics.xlsx", "adaptive": "video_metrics.xlsx"}
    expected = {"ori": (16.49, 0.8), "pos": (48.64, 0.5)}
    checks = []
    for method, filename in references.items():
        if not (legacy / filename).exists():
            continue
        import openpyxl
        workbook = openpyxl.load_workbook(legacy / filename, read_only=True, data_only=True)
        try:
            for row in rows:
                component = row["component"]
                if row["method"] != method or row["sequence"] not in workbook.sheetnames:
                    continue
                if method == "adaptive" and parameters[component][method] != expected[component]:
                    continue
                table = iter(workbook[row["sequence"]].values)
                headers = list(next(table))
                mean_row = next(line for line in table if line[0] == "mean")
                column = component + "_err" + ("_video" if method == "adaptive" else "")
                reference = float(mean_row[headers.index(column)])
                difference = abs(reference - row["mean"])
                checks.append({"sequence": row["sequence"], "method": method, "component": component,
                               "legacy": reference, "current": row["mean"],
                               "absolute_difference": difference,
                               "within_tolerance": difference <= (0.1 if component == "ori" else 0.01)})
        finally:
            workbook.close()
    if checks:
        write_csv(output / "legacy_check.csv", checks)
        failures = sum(not row["within_tolerance"] for row in checks)
        if failures:
            raise AssertionError(f"Fixed results differ from {failures} thesis metrics; see legacy_check.csv")


def grouped_average(rows, group_key, identifiers):
    """
    Average finite scalar sequence metrics within each group and method.

    Args:
        rows (list[dict]): Per-sequence evaluation records.
        group_key (str): Field identifying the group, typically family.
        identifiers (tuple[str]): Identifier fields retained in each aggregate record.

    Returns:
        list[dict]: Group/component/method averages; entirely undefined metrics remain NaN.
    """
    result = []
    for identifier in sorted({row[group_key] for row in rows}):
        for component in ("ori", "pos"):
            for method in METHODS:
                members = [r for r in rows if r[group_key] == identifier and
                           r["component"] == component and r["method"] == method]
                values = {}
                for metric in METRICS:
                    finite = [r[metric] for r in members if np.isfinite(r[metric])]
                    values[metric] = float(np.mean(finite)) if finite else np.nan
                result.append({**{key: members[0][key] for key in identifiers}, **values})
    return result


def summarize(rows, family_rows):
    """
    Summarize metrics with equal family weights and equal sequence weights.

    Scalar sequence statistics are averaged; frame errors are never pooled.

    Args:
        rows (list[dict]): Per-sequence metric records.
        family_rows (list[dict]): Within-family metric averages.

    Returns:
        list[dict]: Family-weighted and sequence-weighted summaries for each method/component.
    """
    result = []
    for weighting, source in (("family", family_rows), ("sequence", rows)):
        for component in ("ori", "pos"):
            for method in METHODS:
                members = [r for r in source if r["component"] == component and r["method"] == method]
                means = {}
                for metric in METRICS:
                    finite = [r[metric] for r in members if np.isfinite(r[metric])]
                    means[metric] = float(np.mean(finite)) if finite else np.nan
                result.append({"weighting": weighting, "component": component, "method": method, **means})
    return result


def bootstrap_families(family_rows):
    """
    Compute descriptive percentile intervals with fixed filter settings.

    The same 2,000 family resamples, seeded with zero, are used for all methods
    and metrics. Lag is excluded and parameters are never retuned.

    Args:
        family_rows (list[dict]): Complete within-family evaluation records.

    Returns:
        list[dict]: Family means and descriptive 95% intervals for each metric except lag.
    """
    families = sorted({r["family"] for r in family_rows})
    samples = np.random.default_rng(0).integers(0, len(families), size=(2000, len(families)))
    lookup = {(r["family"], r["component"], r["method"]): r for r in family_rows}
    rows = []
    for component in ("ori", "pos"):
        for method in METHODS:
            for metric in METRICS[:-1]:
                values = np.array([lookup[f, component, method][metric] for f in families])
                means = np.mean(values[samples], axis=1)
                low, high = np.percentile(means, [2.5, 97.5])
                rows.append({"component": component, "method": method, "metric": metric,
                             "estimate": float(np.mean(values)), "ci95_low": low, "ci95_high": high})
    return rows


def plot_sequence(path, name, tracks):
    """
    Save the eight-panel pose and error comparison as PNG and SVG files.

    Euler branch alignment is used only for display; errors use quaternions.

    Args:
        path (Path): Output filename stem; PNG and SVG suffixes are added.
        name (str): Sequence name for the figure title.
        tracks (dict): Frame identifiers, ground truth and decoded poses for all methods.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = np.arange(len(tracks["frame"])) * DT
    true_angles = np.array([quat2euler(q) for q in tracks["true_ori"]])
    unwrapped = np.rad2deg(np.unwrap(np.deg2rad(true_angles), axis=0))
    fig, axes = plt.subplots(4, 2, figsize=(16, 12), sharex=True)
    for component, column in (("ori", 0), ("pos", 1)):
        truth = tracks[f"true_{component}"]
        true_display = unwrapped if component == "ori" else truth
        for axis in range(3):
            axes[axis, column].plot(time, true_display[:, axis], color="#168038",
                                   linewidth=1.6, label="Ground truth", zorder=5, linestyle="--")
        for method in METHODS:
            values = tracks[f"{method}_{component}"]
            if component == "ori":
                angles = np.array([quat2euler(q) for q in values])
                display = unwrapped + (angles - true_angles + 180) % 360 - 180
                error = np.rad2deg(qdistance(truth, values))
            else:
                display = values
                error = np.linalg.norm(values - truth, axis=1)
            style = dict(color=COLORS[method], linewidth=0.65 if method == "raw" else 1.05,
                         alpha=0.65 if method == "raw" else 0.95, label=LABELS[method])
            for axis in range(3):
                axes[axis, column].plot(time, display[:, axis], **style)
            axes[3, column].plot(time, error, **style)
    for axis, label in enumerate(("Yaw (deg)", "Pitch (deg)", "Roll (deg)", "Attitude error (deg)")):
        axes[axis, 0].set_ylabel(label)
    for axis, label in enumerate(("x (m)", "y (m)", "z (m)", "Position error (m)")):
        axes[axis, 1].set_ylabel(label)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
        ax.set_xlim(time[0], time[-1])
    for ax in axes[3]:
        ax.set_xlabel("Time (s)")
        ax.set_ylim(bottom=0)
    axes[0, 0].set_title("Attitude — Euler angles (ZYX)")
    axes[0, 1].set_title("Position")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 0.957), frameon=False)
    fig.suptitle(f"D-SPEED / {name} — fixed temporal-filter settings", fontsize=15, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def result_table(rows):
    """
    Format the main accuracy and residual-jitter table in Markdown.

    Args:
        rows (list[dict]): One record per pose component and method.

    Returns:
        str: Markdown table containing means, within-sequence SDs and jitter.
    """
    lookup = {(r["component"], r["method"]): r for r in rows}
    lines = ["| Method | Mean $e_q$ (deg) | SD $e_q$ (deg) | $J_q$ (deg/s) | Mean $e_t$ (m) | SD $e_t$ (m) | $J_t$ (m/s) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        ori, pos = lookup["ori", method], lookup["pos", method]
        numbers = [ori[m] for m in ("mean", "std", "jitter")] + [pos[m] for m in ("mean", "std", "jitter")]
        lines.append("| " + LABELS[method] + " | " + " | ".join(f"{x:.4f}" for x in numbers) + " |")
    return "\n".join(lines)


def write_document(output, settings, rows, family_rows, summary, regressions, protocol):
    """
    Fill the evaluation Markdown template with settings, metrics and figure links.

    Args:
        output (Path): Evaluation output directory.
        settings (dict): Confirmed settings in the original JSON representation.
        rows (list[dict]): Per-sequence evaluation metrics.
        family_rows (list[dict]): Within-family metric averages.
        summary (list[dict]): Family-weighted and sequence-weighted summaries.
        regressions (list[dict]): Signed metric changes relative to raw predictions.
        protocol (dict): Sequence names, frame counts and cache/model provenance.
    """
    family_summary = [r for r in summary if r["weighting"] == "family"]
    sequence_summary = [r for r in summary if r["weighting"] == "sequence"]
    text = Path(__file__).with_name("evaluation_template.md").read_text()
    parameters = ["| Method | Attitude | Position |", "|---|---|---|"]
    parameters.extend([
        rf"| Adaptive | $\alpha = {settings['ori']['adaptive']['alpha']:g},\ n = {settings['ori']['adaptive']['n']:g}$ | $\alpha = {settings['pos']['adaptive']['alpha']:g},\ n = {settings['pos']['adaptive']['n']:g}$ |",
        rf"| EMA | $\beta = {settings['ori']['ema']['beta']:g}$ | $\beta = {settings['pos']['ema']['beta']:g}$ |",
        rf"| Moving average | $W = {settings['ori']['moving_average']['window']}$ frames | $W = {settings['pos']['moving_average']['window']}$ frames |",
    ])
    extras = ["| Component | Method | Median | 95th percentile | RMSE | RPE25 RMS | Lag proxy (frames) |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in family_summary:
        extras.append("| " + ("Attitude (deg)" if row["component"] == "ori" else "Position (m)") +
                      f" | {LABELS[row['method']]} | " +
                      " | ".join(f"{row[m]:.4f}" for m in ("median", "p95", "rmse", "rpe25", "lag_frames")) + " |")
    losses = ["| Method | Sequences with higher attitude mean error | Sequences with higher position mean error |",
              "|---|---|---|"]
    for method in METHODS[1:]:
        cells = []
        for component in ("ori", "pos"):
            names = [r["sequence"] for r in regressions if r["method"] == method and
                     r["component"] == component and r["delta_mean"] > 0]
            cells.append(", ".join(names) if names else "None")
        losses.append(f"| {LABELS[method]} | " + " | ".join(cells) + " |")
    sections = []
    for name in protocol["sequences"]:
        members = [r for r in rows if r["sequence"] == name]
        sections.extend([f"### {name}", "", result_table(members), "",
                         f"![{name}: Euler angles, positions and errors for all four methods](figures/{name}.png)", "",
                         f"[Vector figure (SVG)](figures/{name}.svg) · [Pose arrays (NPZ)](tracks/{name}.npz)", ""])
    family_tables = []
    for family in sorted({r["family"] for r in family_rows}):
        family_tables.extend([f"### {family}", "", result_table([r for r in family_rows if r["family"] == family]), ""])
    replacements = {"PARAMETERS": "\n".join(parameters), "FAMILY_RESULTS": result_table(family_summary),
                    "SEQUENCE_RESULTS": result_table(sequence_summary), "EXTRA_RESULTS": "\n".join(extras),
                    "REGRESSIONS": "\n".join(losses), "SEQUENCES": "\n".join(sections),
                    "FAMILIES": "\n".join(family_tables), "N_SEQUENCES": str(len(protocol["sequences"])),
                    "N_FAMILIES": str(len({r["family"] for r in rows})),
                    "N_FRAMES": str(sum(protocol["frame_counts"].values())),
                    "MODEL_HASH": protocol["cache_manifest"]["model_sha256"],
                    "CONFIG_HASH": protocol["cache_manifest"]["config_sha256"]}
    for key, value in replacements.items():
        text = text.replace("{{" + key + "}}", value)
    (output / "README.md").write_text(text)


def run_evaluation(args, cached_sequences, bins, component_metrics, family_of):
    """
    Evaluate fixed filters and export metrics, trajectories, figures and documentation.

    Settings are shared across sequences. This descriptive evaluation reuses the
    calibration videos and does not retune parameters or rerun inference.

    Args:
        args (argparse.Namespace): Command-line settings for paths and execution options.
        cached_sequences (iterable): Pairs of sequence names and cached prediction arrays.
        bins (dict): Attitude quaternion bins and position bin centers.
        component_metrics (Callable): Compute scalar metrics from truth and predicted poses.
        family_of (Callable): Map a sequence name to its trajectory family.

    Raises:
        FileExistsError: If the output directory is nonempty.
        ValueError: If settings are invalid or no sequences are supplied.
        AssertionError: If available matching thesis metrics exceed comparison tolerances.
    """
    settings, parameters = read_parameters(args.selected_parameters)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Evaluation already exists: {output}; choose a new --output-dir")
    manifest = json.loads((Path(args.cache_dir) / "manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    for directory in ("figures", "tracks"):
        (output / directory).mkdir()
    (output / "selected_parameters.json").write_text(json.dumps(settings, indent=2) + "\n")
    protocol = {"status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
                "selection": "Manually inspected and confirmed by the author; fixed across all sequences",
                "evaluation_scope": "Descriptive, using the calibration videos; no held-out claim",
                "fps": 1 / DT, "cache_manifest": manifest, "parameters": settings,
                "bootstrap": {"replicates": 2000, "seed": 0, "unit": "trajectory family", "retuning": False},
                "sequences": [], "frame_counts": {}}
    rows = []
    for name, arrays in cached_sequences:
        print(f"Evaluation {name}: {len(arrays['frame'])} frames", flush=True)
        protocol["sequences"].append(name)
        protocol["frame_counts"][name] = len(arrays["frame"])
        tracks = {"frame": arrays["frame"]}
        for component in ("ori", "pos"):
            truth = arrays[f"true_{component}"].astype(np.float64)
            raw = arrays[f"pred_{component}"].astype(np.float64)
            if component == "ori":
                truth, raw = unit(truth), unit(raw)
            tracks[f"true_{component}"] = truth
            for method in METHODS:
                prediction = raw if method == "raw" else pmf_filter(
                    arrays[f"{component}_pdf"], bins[component], component,
                    method, parameters[component][method], return_pdfs=False)[0]
                tracks[f"{method}_{component}"] = prediction
                rows.append({"sequence": name, "family": family_of(name), "component": component,
                             "method": method, "frames": len(truth),
                             **component_metrics(truth, prediction, component)})
        np.savez_compressed(output / "tracks" / f"{name}.npz", **tracks)
        plot_sequence(output / "figures" / name, name, tracks)
    if not rows:
        raise ValueError("No sequences supplied")
    check_legacy(args, rows, parameters, output)
    family_rows = grouped_average(rows, "family", ("family", "component", "method"))
    summary = summarize(rows, family_rows)
    baseline = {(r["sequence"], r["component"]): r for r in rows if r["method"] == "raw"}
    regressions = [{"sequence": r["sequence"], "component": r["component"], "method": r["method"],
                    **{f"delta_{m}": r[m] - baseline[r["sequence"], r["component"]][m]
                       for m in ("mean", "std", "jitter", "rmse")}}
                   for r in rows if r["method"] != "raw"]
    for filename, values in (("per_sequence.csv", rows), ("per_family.csv", family_rows),
                             ("summary.csv", summary), ("bootstrap_ci.csv", bootstrap_families(family_rows)),
                             ("changes_vs_raw.csv", regressions)):
        write_csv(output / filename, values)
    write_document(output, settings, rows, family_rows, summary, regressions, protocol)
    protocol["status"] = "complete"
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    files = sorted(p for p in output.rglob("*") if p.is_file())
    (output / "SHA256SUMS").write_text("".join(
        hashlib.sha256(p.read_bytes()).hexdigest() + "  " + str(p.relative_to(output)) + "\n" for p in files))
    print(f"Evaluation complete: {output / 'README.md'}", flush=True)
