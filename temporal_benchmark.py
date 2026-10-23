"""
Copyright (c) 2026 Julien Posso

Reproducible, causal D-SPEED temporal-filter comparison.

Example (from the repository root, in pose_brevitas:latest):
    python temporal_benchmark.py  # infer/reuse the cache, then evaluate confirmed settings
    python temporal_benchmark.py --stage sweep  # inspect candidates before manual selection
    python temporal_benchmark.py --stage evaluate  # fixed settings, existing cache
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from src.temporal.benchmark_filters import (
    DT, qconj, qdistance, qlog, qmul,
)


SWEEP_ALPHAS = {"ori": 16.49, "pos": 48.64}
DEFAULT_MODEL = "models/mursop_fp32_dspeed"
DEFAULT_DATASET = "../datasets/dspeed/video"
DEFAULT_CACHE = "experiments/temporal_benchmark_ieee_taes/cache"
DEFAULT_OUTPUT = "experiments/temporal_benchmark_ieee_taes/evaluation"


def sha256(path):
    """
    Compute a file checksum without loading the whole file into memory.

    Args:
        path (str or Path): File to hash.

    Returns:
        str: Hexadecimal SHA256 digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequences(dataset, selected=None):
    """
    Find video sequences and validate the requested subset.

    Args:
        dataset (str or Path): Directory containing video sequence folders.
        selected (list[str] or None): Sequence names to keep; None selects all 21 videos.

    Returns:
        list[Path]: Sequence directories sorted by name.

    Raises:
        ValueError: If no sequences exist, a requested name is unknown, or the full set is incomplete.
    """
    paths = sorted(p for p in Path(dataset).iterdir() if p.is_dir() and (p / "pose.json").exists())
    if selected:
        names = set(selected)
        paths = [p for p in paths if p.name in names]
        if {p.name for p in paths} != names:
            raise ValueError("Unknown sequence name in --sequences")
    if not paths:
        raise ValueError("No D-SPEED video sequences found")
    if not selected and len(paths) != 21:
        raise ValueError(f"Expected 21 D-SPEED v1 video sequences, found {len(paths)}")
    return paths


def family_of(name):
    """
    Group Earth and star variants under the same motion-family name.

    Args:
        name (str): Sequence name, optionally ending in _eb or _st.

    Returns:
        str: Motion-family name; singleton names are unchanged.
    """
    stem, _, suffix = name.rpartition("_")
    return stem if suffix in ("eb", "st") else name


def load_bins(config_path):
    """
    Build the pose-decoding bins from the model configuration.

    Args:
        config_path (str or Path): Model YAML configuration.

    Returns:
        tuple[dict, dict]: Attitude/position bins and the loaded configuration.

    Raises:
        ValueError: If either model head does not predict classification PMFs.
    """
    import yaml
    from src.spe.classification_utils import OrientationSoftClassification, PositionSoftClassification
    with open(config_path) as stream:
        cfg = yaml.safe_load(stream)
    head, data = cfg["MODEL"]["HEAD"], cfg["DATA"]
    if head["ORI"] != "classification" or head["POS"] != "classification":
        raise ValueError("The benchmark requires PMFs for both attitude and position")
    ori = OrientationSoftClassification(head["N_ORI_BINS_PER_DIM"], data["ORI_SMOOTH_FACTOR"],
                                        head["ORI_DELETE_UNUSED_BINS"])
    pos = PositionSoftClassification(head["N_POS_BINS_PER_DIM"], data["POS_SMOOTH_FACTOR"],
                                     np.array([-16, -12, -2]), np.array([16, 12, 40]))
    return {"ori": ori.histogram, "pos": pos.histogram}, cfg


def manifest_for(model, dataset, selected):
    """
    Record model and label identities used to validate prediction caches.

    Args:
        selected (list[Path]): Selected video sequence directories.
        model (str or Path): Model directory containing weights and configuration.
        dataset (str or Path): Video dataset directory.

    Returns:
        dict: Cache schema, file hashes, source paths and sequence-label hashes.
    """
    model = Path(model)
    return {
        "schema": 1,
        "model": str(model.resolve()),
        "model_sha256": sha256(model / "model" / "parameters.pt"),
        "config_sha256": sha256(model / "config.yaml"),
        "dataset": str(Path(dataset).resolve()),
        "sequence_labels": {p.name: sha256(p / "pose.json") for p in selected},
        "image_count_limit": None,
    }


def ensure_manifest(cache_dir, expected, max_frames):
    """
    Create cache metadata or check that an existing cache is compatible.

    Absolute model and dataset paths may differ between host and container mounts.

    Args:
        cache_dir (Path): Prediction cache directory.
        expected (dict): Expected model, configuration and label identities.
        max_frames (int or None): Per-sequence frame limit for a smoke run.

    Raises:
        ValueError: If cache identities differ or a nonempty cache has no manifest.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    expected = dict(expected, image_count_limit=max_frames)
    path = cache_dir / "manifest.json"
    if path.exists():
        with open(path) as stream:
            found = json.load(stream)
        # The same mounted files have different absolute paths inside Docker.
        identity = lambda manifest: {key: value for key, value in manifest.items()
                                     if key not in ("model", "dataset")}
        if identity(found) != identity(expected):
            raise ValueError(f"Cache metadata differ in {path}; choose a new --cache-dir")
    else:
        if any(cache_dir.iterdir()):
            raise ValueError(f"Nonempty cache without manifest: {cache_dir}")
        path.write_text(json.dumps(expected, indent=2) + "\n")


def infer(args, selected, bins, cfg):
    """
    Cache frame predictions and PMFs, reusing completed sequence files.

    A complete cache is reused before importing the neural-network runtime.

    Args:
        args (argparse.Namespace): Command-line settings for paths and execution options.
        selected (list[Path]): Selected video sequence directories.
        bins (dict): Attitude quaternion bins and position bin centers.
        cfg (dict): Loaded model and preprocessing configuration.

    Raises:
        RuntimeError: If CUDA is explicitly requested but unavailable.
        ValueError: If a full sequence does not contain 1,500 frames.
    """
    cache_dir = Path(args.cache_dir)
    if all((cache_dir / f"{path.name}.npz").exists() for path in selected):
        print(f"Reuse complete prediction cache: {cache_dir}", flush=True)
        return
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    from src.data.import_dataset import load_camera
    from src.modeling.model import import_model
    from src.spe.spe_torch import SPETorch
    from src.spe.spe_utils import SPEUtils

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this Docker container")
    print(f"Inference device: {device}", flush=True)
    head, data = cfg["MODEL"]["HEAD"], cfg["DATA"]
    model_cfg = cfg["MODEL"]
    camera = load_camera(data["PATH"])
    spe_utils = SPEUtils(camera, head["ORI"], head["N_ORI_BINS_PER_DIM"], data["ORI_SMOOTH_FACTOR"],
                         head["ORI_DELETE_UNUSED_BINS"], head["POS"], head["N_POS_BINS_PER_DIM"],
                         data["POS_SMOOTH_FACTOR"], head["KEYPOINTS_PATH"])
    height, width = data["IMG_SIZE"]
    dummy = {"sample": [({"torch": torch.zeros((1, 3, height, width))}, {})]}
    weights = Path(args.model) / "model" / "parameters.pt"
    model, _ = import_model(dummy, model_cfg["BACKBONE"]["NAME"], head["NAME"], str(weights), None,
                            manual_copy=False, residual=model_cfg["BACKBONE"]["RESIDUAL"],
                            quantization=model_cfg["QUANTIZATION"], ori_mode=head["ORI"],
                            n_ori_bins=len(bins["ori"]), pos_mode=head["POS"],
                            n_pos_bins=len(bins["pos"]))
    engine = SPETorch(model, torch.device(device), spe_utils)
    transform = transforms.Compose([transforms.Resize((height, width)), transforms.ToTensor()])
    for path in selected:
        destination = cache_dir / f"{path.name}.npz"
        if destination.exists():
            print(f"Reuse {destination}", flush=True)
            continue
        with open(path / "pose.json") as stream:
            labels = json.load(stream)
        labels.sort(key=lambda row: int(Path(row["filename"]).stem))
        if args.max_frames is None and len(labels) != 1500:
            raise ValueError(f"Expected 1,500 frames in {path.name}, found {len(labels)}")
        if args.max_frames:
            labels = labels[:args.max_frames]
        names, truth_ori, truth_pos = [], [], []
        pred_ori, pred_pos, ori_pdf, pos_pdf = [], [], [], []
        print(f"Infer {path.name}: {len(labels)} frames", flush=True)
        for start in range(0, len(labels), args.batch_size):
            batch = labels[start:start + args.batch_size]
            tensors = []
            for row in batch:
                with Image.open(path / "images" / row["filename"]) as image:
                    tensors.append(transform(image.convert("RGB")))
            prediction, _ = engine.predict(torch.stack(tensors))
            for j, row in enumerate(batch):
                names.append(row["filename"])
                truth_ori.append(row["q"])
                truth_pos.append(row["t"])
                pred_ori.append(prediction["ori"][j])
                pred_pos.append(prediction["pos"][j])
                ori_pdf.append(prediction["ori_soft"][j])
                pos_pdf.append(prediction["pos_soft"][j])
            if start % (args.batch_size * 20) == 0:
                print(f"  {min(start + args.batch_size, len(labels))}/{len(labels)}", flush=True)
        partial = destination.with_suffix(".npz.partial")
        with open(partial, "wb") as stream:
            np.savez_compressed(stream, frame=np.asarray(names), true_ori=np.asarray(truth_ori, np.float32),
                                true_pos=np.asarray(truth_pos, np.float32), pred_ori=np.asarray(pred_ori, np.float32),
                                pred_pos=np.asarray(pred_pos, np.float32), ori_pdf=np.asarray(ori_pdf, np.float32),
                                pos_pdf=np.asarray(pos_pdf, np.float32))
        os.replace(partial, destination)


def _errors(true_values, pred_values, component):
    """
    Compute framewise geodesic attitude or Euclidean position errors.

    Args:
        component (str): Pose component, 'ori' for attitude or 'pos' for position.
        true_values (np.ndarray): Ground-truth poses, shaped (N, 4) or (N, 3).
        pred_values (np.ndarray): Corresponding predicted poses.

    Returns:
        np.ndarray: One error per frame, in degrees or meters.
    """
    if component == "ori":
        return np.rad2deg(qdistance(true_values, pred_values))
    return np.linalg.norm(pred_values - true_values, axis=1)


def _component_metrics(truth, prediction, component):
    """
    Measure pose accuracy, residual jitter, relative motion error and lag.

    All frames are scored without time alignment. RPE uses 25-frame intervals;
    lag is a diagnostic search over 0-10 frames and may be undefined.

    Args:
        component (str): Pose component, 'ori' for attitude or 'pos' for position.
        truth (np.ndarray): Ground-truth poses in chronological order.
        prediction (np.ndarray): Predicted poses in the same order.

    Returns:
        dict: mean, median, p95, std, rmse, jitter, rpe25 and lag_frames.
            Pose errors use degrees or meters, jitter uses degrees/s or m/s,
            and lag uses frames; unavailable diagnostics are NaN.
    """
    errors = _errors(truth, prediction, component)
    if component == "ori":
        residual = qlog(qmul(qconj(truth), prediction))
        jitter = np.rad2deg(np.linalg.norm(np.diff(residual, axis=0), axis=1)) / DT
        if len(truth) > 25:
            relative_true = qmul(qconj(truth[:-25]), truth[25:])
            relative_pred = qmul(qconj(prediction[:-25]), prediction[25:])
            rpe = np.rad2deg(qdistance(relative_true, relative_pred))
        else:
            rpe = np.array([np.nan])
    else:
        residual = prediction - truth
        jitter = np.linalg.norm(np.diff(residual, axis=0), axis=1) / DT
        rpe = (np.linalg.norm((prediction[25:]-prediction[:-25])-(truth[25:]-truth[:-25]), axis=1)
               if len(truth) > 25 else np.array([np.nan]))
    # Delay is evaluated, never used to shift or improve the prediction.
    movement = (np.max(np.linalg.norm(truth - truth[0], axis=1)) if component == "pos" else
                np.max(np.rad2deg(qdistance(truth[0], truth))))
    threshold = 0.3 if component == "pos" else 3.0
    if movement > threshold and len(truth) > 35:
        lag_cost = [np.mean(_errors(truth[:len(truth)-10], prediction[lag:len(truth)-10+lag], component)**2)
                    for lag in range(11)]
        # Positive lag means the prediction follows an earlier ground-truth frame.
        lag_frames = int(np.argmin(lag_cost))
    else:
        lag_frames = np.nan
    return {
        "mean": float(np.mean(errors)), "median": float(np.median(errors)),
        "p95": float(np.percentile(errors, 95)), "std": float(np.std(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "jitter": float(np.sqrt(np.mean(jitter**2))) if len(jitter) else np.nan,
        "rpe25": float(np.sqrt(np.mean(rpe**2))), "lag_frames": lag_frames,
    }


def _load_cache(cache_dir, selected, bins):
    """
    Load and validate cached sequence predictions one sequence at a time.

    Args:
        selected (list[Path]): Selected video sequence directories.
        bins (dict): Attitude quaternion bins and position bin centers.
        cache_dir (Path): Directory containing per-sequence NPZ files.

    Yields:
        tuple[str, dict]: Sequence name and cached arrays.

    Raises:
        FileNotFoundError: If a selected sequence has no cache file.
        ValueError: If array lengths, PMF dimensions or PMF normalization are invalid.
    """
    for path in selected:
        source = cache_dir / f"{path.name}.npz"
        if not source.exists():
            raise FileNotFoundError(f"Missing frame cache: {source}; run --stage infer first")
        with np.load(source, allow_pickle=False) as contents:
            arrays = {name: contents[name] for name in contents.files}
        count = len(arrays["frame"])
        for key in ("true_ori", "true_pos", "pred_ori", "pred_pos", "ori_pdf", "pos_pdf"):
            if len(arrays[key]) != count:
                raise ValueError(f"Bad cache length for {source}: {key}")
        if arrays["ori_pdf"].shape[1] != len(bins["ori"]) or arrays["pos_pdf"].shape[1] != len(bins["pos"]):
            raise ValueError(f"PMF shape does not match model config in {source}")
        if not np.allclose(np.sum(arrays["ori_pdf"], axis=1), 1, atol=1e-5):
            raise ValueError(f"Invalid orientation PMF in {source}")
        if not np.allclose(np.sum(arrays["pos_pdf"], axis=1), 1, atol=1e-5):
            raise ValueError(f"Invalid position PMF in {source}")
        yield path.name, arrays


def evaluate(args, selected, bins):
    """
    Run the fixed-setting evaluation using the existing prediction cache.

    Args:
        args (argparse.Namespace): Command-line settings for paths and execution options.
        selected (list[Path]): Selected video sequence directories.
        bins (dict): Attitude quaternion bins and position bin centers.
    """
    from src.temporal.evaluation import run_evaluation
    run_evaluation(args, _load_cache(Path(args.cache_dir), selected, bins), bins,
                   _component_metrics, family_of)


def main():
    """
    Parse command-line options and run inference, a sweep or fixed evaluation.

    The default stage reuses or completes the cache, then evaluates the confirmed
    settings. Nonempty output directories are never overwritten.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("infer", "evaluate", "sweep", "all"), default="all",
                        help=("all (default): infer/reuse cache, then fixed evaluation; "
                              "evaluate: fixed evaluation from cache; sweep: parameter exploration"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", help="Defaults to evaluation, or sweep for --stage sweep")
    parser.add_argument("--selected-parameters", type=Path,
                        default=Path("experiments/temporal_benchmark_ieee_taes/sweep/manual_selection.json"),
                        help="Confirmed manual settings for evaluate and all")
    parser.add_argument("--sweep-window-max", type=int, default=25,
                        help="Largest moving-average window in the visual sweep (default: 25 frames)")
    parser.add_argument("--plotly-js", type=Path,
                        help="Offline plotly.min.js asset if Plotly is not installed (for --stage sweep)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequences", nargs="+")
    parser.add_argument("--max-frames", type=int, help="Short smoke run; use a separate cache and output directory")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = ("experiments/temporal_benchmark_ieee_taes/sweep"
                           if args.stage == "sweep" else DEFAULT_OUTPUT)
    if args.sweep_window_max < 1:
        parser.error("--sweep-window-max must be positive")
    if args.batch_size < 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error("Batch size and frame limit must be positive")
    if args.stage in ("evaluate", "all"):
        from src.temporal.evaluation import read_parameters
        read_parameters(args.selected_parameters)
    if args.stage != "infer":
        output = Path(args.output_dir)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Output already exists: {output}; choose a new --output-dir")
    selected = sequences(args.dataset, args.sequences)
    bins, cfg = load_bins(Path(args.model) / "config.yaml")
    ensure_manifest(Path(args.cache_dir), manifest_for(args.model, args.dataset, selected), args.max_frames)
    if args.stage in ("infer", "all"):
        infer(args, selected, bins, cfg)
    if args.stage in ("evaluate", "all"):
        evaluate(args, selected, bins)
    if args.stage == "sweep":
        from src.temporal.sweep import run_sweep
        run_sweep(args, _load_cache(Path(args.cache_dir), selected, bins), bins,
                  _component_metrics, SWEEP_ALPHAS)


if __name__ == "__main__":
    main()
