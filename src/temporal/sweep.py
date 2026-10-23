"""
Copyright (c) 2026 Julien Posso

Offline parameter sweep for manual inspection of successive smoothing levels.
"""

import csv
import html
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.spe.utils import quat2euler
from src.temporal.benchmark_filters import DT, pmf_filter, qdistance, unit


SWEEP_METHODS = ("adaptive", "ema", "moving_average")


def sweep_parameters(method, component, alphas, maximum_window):
    """
    List candidates from weaker to stronger smoothing, excluding zero gain.

    Args:
        component (str): Pose component, 'ori' for attitude or 'pos' for position.
        method (str): Adaptive, EMA or moving-average filter name.
        alphas (dict): Fixed adaptive sensitivities for both pose components.
        maximum_window (int): Largest moving-average window in frames.

    Returns:
        list: Ordered windows, EMA gains or adaptive (alpha, n) pairs.

    Raises:
        ValueError: If the filter method is unsupported.
    """
    if method == "moving_average":
        return list(range(1, maximum_window + 1))
    coefficients = [i / 10 for i in range(10, 0, -1)]
    if method == "ema":
        return coefficients
    if method == "adaptive":
        return [(alphas[component], n) for n in coefficients]
    raise ValueError(method)


def _label(method, parameter):
    """
    Format a filter parameter for plot legends and interactive controls.

    Args:
        method (str): Filter name.
        parameter (int, float or tuple): Window, EMA gain or adaptive (alpha, n).

    Returns:
        str: Human-readable parameter label.
    """
    if method == "adaptive":
        return f"n={parameter[1]:g}"
    if method == "ema":
        return f"beta={parameter:g}"
    return f"W={parameter}"


def _finite_metrics(metrics):
    """
    Replace nonfinite metric values with JSON-compatible null values.

    Args:
        metrics (dict): Scalar metric values, possibly containing NaN.

    Returns:
        dict: Floating-point values or None.
    """
    return {key: float(value) if np.isfinite(value) else None for key, value in metrics.items()}


def _display_track(values, truth, component, true_angles=None, unwrapped=None):
    """
    Prepare rounded trajectories and errors for the interactive viewer.

    Euler branch alignment affects only display, never the pose-error metrics.

    Args:
        component (str): Pose component, 'ori' for attitude or 'pos' for position.
        values (np.ndarray): Predicted poses or ground truth.
        truth (np.ndarray): Reference poses.
        true_angles (np.ndarray or None): Ground-truth Euler angles in degrees for attitude.
        unwrapped (np.ndarray or None): Unwrapped ground-truth Euler angles for display.

    Returns:
        dict: Component-major display values and framewise errors, rounded to six decimals.
    """
    if component == "ori":
        angles = np.asarray([quat2euler(q) for q in values])
        # Euler branch alignment is display-only, as in temporal_benchmark.py.
        display = unwrapped + (angles - true_angles + 180) % 360 - 180
        error = np.rad2deg(qdistance(truth, values))
    else:
        display = values
        error = np.linalg.norm(values - truth, axis=1)
    return {"values": np.round(display.T, 6).tolist(), "error": np.round(error, 6).tolist()}


def _overview(path, name, method, component, candidates, raw_error, alpha):
    """
    Save a PNG overview of candidate errors and aggregate sweep metrics.

    Args:
        component (str): Pose component, 'ori' for attitude or 'pos' for position.
        path (Path): Output PNG path.
        name (str): Sequence name.
        method (str): Filter name.
        candidates (list[dict]): Candidate labels, error curves and metrics.
        raw_error (array-like): Unfiltered framewise pose errors.
        alpha (float): Fixed adaptive sensitivity used in the figure title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    unit_label = "deg" if component == "ori" else "m"
    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0, vmax=max(1, len(candidates) - 1))
    time = np.arange(len(raw_error)) * DT
    axes[0].plot(time, raw_error, color="0.7", linewidth=0.7, label="Raw")
    for index, candidate in enumerate(candidates):
        axes[0].plot(time, candidate["error"], color=cmap(norm(index)), linewidth=0.8)
    axes[0].set(xlabel="Time (s)", ylabel=f"Pose error ({unit_label})")
    axes[0].legend(loc="upper right")
    ticks = np.unique(np.linspace(0, len(candidates) - 1, min(6, len(candidates)), dtype=int))
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes[0], ticks=ticks)
    bar.ax.set_yticklabels([candidates[i]["label"] for i in ticks])
    for metric in ("mean", "p95"):
        axes[1].plot(range(len(candidates)), [c["metrics"][metric] for c in candidates],
                     marker=".", label=metric)
    axes[1].set_xticks(ticks)
    axes[1].set_xticklabels([candidates[i]["label"] for i in ticks])
    axes[1].set(xlabel="Increasing smoothing", ylabel=f"Pose error ({unit_label})")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    suffix = f", alpha={alpha:g} fixed" if method == "adaptive" else ""
    fig.suptitle(f"{name} — {component} — {method}{suffix}\nManual sweep: no parameter selected")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_csv(path, rows):
    """
    Write nonempty metric rows with a header derived from the first row.

    Args:
        path (Path): Destination CSV path.
        rows (list[dict]): Nonempty records sharing the same columns.
    """
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_index(output, names, maximum_window, alphas):
    """
    Write the English HTML and Markdown navigation indexes for a sweep.

    Args:
        output (Path): Sweep output directory.
        names (list[str]): Sequence names in display order.
        maximum_window (int): Largest explored moving-average window.
        alphas (dict): Fixed attitude and position adaptive sensitivities.
    """
    intro = ("Choose a sequence, then a filter and pose component. The slider compares the current setting "
             "with the immediately preceding setting, ground truth and raw predictions. "
             "The first setting is compared with the raw prediction. Zoom in on rapid changes; "
             "zoom is retained when changing settings. No parameter is selected automatically.")
    header = (f"n and beta: 1.0 to 0.1 in steps of 0.1. Windows: 1 to {maximum_window} frames. "
              f"Adaptive alpha fixed at {alphas['ori']:g} (attitude) and {alphas['pos']:g} (position). "
              "Data are reused from the mursop_fp32_dspeed cache without rerunning inference.")
    links = []
    for name in names:
        pngs = " · ".join(f'<a href="{html.escape(name)}/{method}_{component}.png">{method}/{component}</a>'
                           for method in SWEEP_METHODS for component in ("ori", "pos"))
        links.append(f'<tr><td><a href="{html.escape(name)}/index.html">{html.escape(name)}</a></td>'
                     f'<td>{pngs}</td></tr>')
    rapid = " · ".join(f'<a href="{name}/index.html">{name}</a>' for name in
                       ("CATR", "TITR_eb", "TITR_st", "S_st") if name in names)
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>D-SPEED parameter sweep</title>'
            '<style>body{font:16px system-ui;max-width:1250px;margin:35px auto;padding:0 20px;line-height:1.5}'
            'td,th{padding:12px;text-align:left;border-bottom:1px solid #ddd}a{color:#155ca2}'
            '</style><h1>Visual sweep of temporal filters</h1>'
            f'<p>{intro}</p><p>{header}</p><p>Start with: {rapid}</p>'
            '<p><a href="metrics.csv">Metrics and differences between successive settings</a> · '
            '<a href="manual_selection.json">Parameter selection file</a></p>'
            '<table><tr><th>Interactive sequence</th><th>PNG overviews</th></tr>'
            + "".join(links) + '</table></html>')
    (output / "index.html").write_text(page)
    lines = ["# D-SPEED visual parameter sweep", "", intro, "", header, "",
             "[Open the interactive viewer](index.html)", "",
             "| Sequence | Interactive comparison |", "|---|---|"]
    lines.extend(f"| {name} | [open]({name}/index.html) |" for name in names)
    (output / "index.md").write_text("\n".join(lines) + "\n")


def run_sweep(args, cached_sequences, bins, component_metrics, alphas):
    """
    Export candidate trajectories, metrics, figures and interactive pages.

    Existing causal filters and scoring functions are reused. Parameter selection
    is manual; a new sweep exports an empty selection sheet.

    Args:
        args (argparse.Namespace): Command-line settings for paths and execution options.
        cached_sequences (iterable): Pairs of sequence names and cached prediction arrays.
        bins (dict): Attitude quaternion bins and position bin centers.
        component_metrics (Callable): Compute scalar metrics from truth and predicted poses.
        alphas (dict): Fixed adaptive sensitivities for attitude and position.

    Raises:
        FileExistsError: If the output directory is nonempty.
        RuntimeError: If neither an explicit Plotly asset nor installed Plotly is available.
    """
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Sweep output already exists: {output}; choose another --output-dir")
    if args.plotly_js is not None:
        asset = Path(args.plotly_js).read_text()
    else:
        try:
            from plotly.offline import get_plotlyjs
        except ImportError as error:
            raise RuntimeError("Provide --plotly-js /path/to/plotly.min.js for offline interactive plots") from error
        asset = get_plotlyjs()
    template = Path(__file__).with_name("sweep_viewer.html").read_text()
    output.mkdir(parents=True, exist_ok=True)
    (output / "plotly.min.js").write_text(asset)
    maximum_window = args.sweep_window_max
    protocol = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "model": args.model, "cache_dir": args.cache_dir, "fps": 1 / DT,
                "selection": "manual; no automatic choice", "fixed_alpha": alphas,
                "parameters": {method: {component: sweep_parameters(method, component, alphas, maximum_window)
                                         for component in ("ori", "pos")} for method in SWEEP_METHODS},
                "display": "Euler branches aligned to ground truth for display only; metrics use quaternions"}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    selection = {component: {"adaptive": {"alpha": alphas[component], "n": None},
                             "ema": {"beta": None}, "moving_average": {"window": None}}
                 for component in ("ori", "pos")}
    (output / "manual_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    rows, names = [], []
    for name, arrays in cached_sequences:
        print(f"Sweep {name}: {len(arrays['frame'])} frames", flush=True)
        sequence_dir = output / name
        sequence_dir.mkdir()
        payload = {"name": name, "time": np.round(np.arange(len(arrays["frame"])) * DT, 6).tolist(),
                   "frame": arrays["frame"].tolist(), "alpha": alphas}
        tracks = {key: arrays[key] for key in ("frame", "true_ori", "true_pos", "pred_ori", "pred_pos")}
        for component in ("ori", "pos"):
            truth = arrays[f"true_{component}"].astype(np.float64)
            raw = arrays[f"pred_{component}"].astype(np.float64)
            if component == "ori":
                truth, raw = unit(truth), unit(raw)
                true_angles = np.asarray([quat2euler(q) for q in truth])
                unwrapped = np.rad2deg(np.unwrap(np.deg2rad(true_angles), axis=0))
            else:
                true_angles = unwrapped = None
            display_args = (truth, component, true_angles, unwrapped)
            raw_metrics = component_metrics(truth, raw, component)
            payload[component] = {"truth": _display_track(truth, *display_args),
                                  "raw": dict(_display_track(raw, *display_args), metrics=_finite_metrics(raw_metrics)),
                                  "candidates": {}}
            for method in SWEEP_METHODS:
                previous_metrics = raw_metrics
                previous_label = "raw"
                candidates = []
                for step, parameter in enumerate(sweep_parameters(method, component, alphas, maximum_window)):
                    values, _ = pmf_filter(arrays[f"{component}_pdf"], bins[component],
                                           component, method, parameter, return_pdfs=False)
                    metrics = component_metrics(truth, values, component)
                    label = _label(method, parameter)
                    candidates.append(dict(_display_track(values, *display_args),
                                           label=label, parameter=parameter, metrics=_finite_metrics(metrics)))
                    tracks[f"{method}_{component}_{step:02d}"] = values
                    row = {"sequence": name, "component": component, "method": method,
                           "step": step, "parameter": json.dumps(parameter), "previous": previous_label,
                           **metrics, **{f"delta_{key}": metrics[key] - previous_metrics[key]
                                          for key in ("mean", "p95", "rmse", "jitter")}}
                    rows.append(row)
                    previous_metrics, previous_label = metrics, label
                payload[component]["candidates"][method] = candidates
                _overview(sequence_dir / f"{method}_{component}.png", name, method, component,
                          candidates, payload[component]["raw"]["error"], alphas[component])
        np.savez_compressed(sequence_dir / "tracks.npz", **tracks)
        data = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
        page = template.replace("%%TITLE%%", html.escape(name)).replace("%%DATA%%", data)
        (sequence_dir / "index.html").write_text(page)
        names.append(name)
        _write_csv(output / "metrics.csv", rows)
        _write_index(output, names, maximum_window, alphas)
        print(f"  {name}: interactive comparison + 6 PNGs saved", flush=True)
    protocol.update(status="complete", sequences=names,
                    finished_utc=datetime.now(timezone.utc).isoformat())
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(f"Visual sweep ready: {output / 'index.html'}", flush=True)
