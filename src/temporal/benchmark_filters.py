"""
Copyright (c) 2026 Julien Posso

Causal pose filters used by the D-SPEED temporal benchmark.

All quaternions are scalar-first. A filter receives only the current and past
network predictions; ground truth is never used in a filter update.
"""

import numpy as np

from src.spe.utils import conjugate_quaternion, multiply_quaternions
from src.temporal.moving_average import MovingAverage, ExponentialMovingAverage
from src.temporal.pdf_compare import TemporalPDF


DT = 1.0 / 25.0


def unit(q):
    """
    Normalize scalar-first quaternions along the last dimension.

    Args:
        q (array-like): Quaternion or quaternion array, with last dimension 4.

    Returns:
        np.ndarray: Unit quaternions with the same shape.

    Raises:
        ValueError: If any quaternion norm is smaller than 1e-12.
    """
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("Zero quaternion")
    return q / norm


def qconj(q):
    """
    Compute the conjugate of scalar-first quaternions.

    Args:
        q (array-like): Quaternion or quaternion array, with last dimension 4.

    Returns:
        np.ndarray: Conjugated quaternions.
    """
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 1:
        return conjugate_quaternion(q)
    result = q.copy()
    result[..., 1:] *= -1
    return result


def qmul(a, b):
    """
    Compute Hamilton products with NumPy broadcasting.

    Args:
        a (array-like): Left scalar-first quaternion operand.
        b (array-like): Right scalar-first quaternion operand.

    Returns:
        np.ndarray: Quaternion products with broadcast-compatible leading dimensions.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.ndim == b.ndim == 1:
        return multiply_quaternions(a, b)
    # Reuse the existing scalar helper. This vectorized
    # branch handles the full trajectories used by the benchmark metrics.
    a, b = np.broadcast_arrays(a, b)
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack((aw*bw-ax*bx-ay*by-az*bz,
                     aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx,
                     aw*bz+ax*by-ay*bx+az*bw), axis=-1)


def qexp(rotvec):
    """
    Convert rotation vectors to scalar-first unit quaternions.

    Args:
        rotvec (array-like): Rotation vectors in radians, with last dimension 3.

    Returns:
        np.ndarray: Unit quaternions with last dimension 4.
    """
    v = np.asarray(rotvec, dtype=np.float64)
    angle = np.linalg.norm(v, axis=-1, keepdims=True)
    half = angle / 2
    scale = np.where(angle > 1e-10, np.sin(half) / np.maximum(angle, 1e-12), 0.5-angle**2/48)
    return unit(np.concatenate((np.cos(half), v * scale), axis=-1))


def qlog(q):
    """
    Convert quaternions to shortest-angle rotation vectors.

    Quaternion signs are aligned to a nonnegative scalar component.

    Args:
        q (array-like): Scalar-first quaternions, with last dimension 4.

    Returns:
        np.ndarray: Rotation vectors in radians, with last dimension 3.
    """
    q = unit(q)
    q = np.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2 * np.arctan2(norm, np.clip(q[..., :1], 0, 1))
    return v * np.where(norm > 1e-10, angle / np.maximum(norm, 1e-12), 2.0)


def qdistance(a, b):
    """
    Compute the shortest angular distance between quaternion pairs.

    Args:
        a (array-like): First scalar-first quaternion operand.
        b (array-like): Second scalar-first quaternion operand.

    Returns:
        np.ndarray or np.float64: Angular distances in radians.
    """
    return np.linalg.norm(qlog(qmul(qconj(a), b)), axis=-1)


def mean_quaternion(quats, weights):
    """
    Decode a weighted quaternion mean using its dominant eigenvector.

    The result is invariant to the signs of the input quaternions.

    Args:
        quats (np.ndarray): Unit quaternion bins, shaped (B, 4).
        weights (array-like): Corresponding bin weights, shaped (B,).

    Returns:
        np.ndarray: Scalar-first unit quaternion, shaped (4,).
    """
    weights = np.asarray(weights, dtype=np.float64)
    matrix = np.einsum("n,ni,nj->ij", weights, quats, quats)
    _, vectors = np.linalg.eigh(matrix)
    return unit(vectors[:, -1])


def normalize_pdf(pdf):
    """
    Validate and normalize a probability mass function.

    Small negative values within the numerical tolerance are clipped to zero.

    Args:
        pdf (array-like): One-dimensional bin probabilities.

    Returns:
        np.ndarray: Nonnegative float64 probabilities summing to one.

    Raises:
        ValueError: If values are nonfinite, below -1e-7, or have no positive mass.
    """
    pdf = np.asarray(pdf, dtype=np.float64)
    if not np.all(np.isfinite(pdf)) or np.any(pdf < -1e-7):
        raise ValueError("Invalid probability distribution")
    pdf = np.maximum(pdf, 0)
    total = np.sum(pdf)
    if total <= 0:
        raise ValueError("Empty probability distribution")
    return pdf / total


def decode_pdf(pdf, bins, component):
    """
    Decode an attitude or position PMF into a pose component.

    Args:
        component (str): Pose component, 'ori' for attitude or 'pos' for position.
        pdf (np.ndarray): Normalized bin probabilities.
        bins (np.ndarray): Quaternion bins (B, 4) or position centers (B, 3).

    Returns:
        np.ndarray: Unit quaternion (4,) or position vector (3,).

    Raises:
        ValueError: If the pose component is unsupported.
    """
    if component == "pos":
        return np.asarray(pdf @ bins, dtype=np.float64)
    if component == "ori":
        return mean_quaternion(bins, pdf)
    raise ValueError(component)


def pmf_filter(pdfs, bins, component, method, parameter, return_pdfs=True):
    """
    Apply a causal PMF filter and decode the resulting trajectory.

    Filter state is initialized separately on each call. Ground truth is never
    used in a filter update.

    Args:
        component (str): Pose component, 'ori' for attitude or 'pos' for position.
        pdfs (np.ndarray): Chronological PMFs, shaped (N, B).
        bins (np.ndarray): Quaternion bins (B, 4) or position centers (B, 3).
        method (str): 'raw', 'moving_average', 'ema' or 'adaptive'.
        parameter (int, float, tuple or None): Window W, EMA gain, adaptive (alpha, n), or None for raw.
        return_pdfs (bool): Whether to retain the filtered PMFs in the output.

    Returns:
        tuple[np.ndarray, np.ndarray or None]: Decoded poses and filtered PMFs.
            Poses are float64; stored PMFs are float32 or None when omitted.

    Raises:
        ValueError: If a PMF, method or component is invalid.
    """
    output = np.empty((len(pdfs), bins.shape[1]), dtype=np.float64)
    filtered = np.empty(pdfs.shape, dtype=np.float32) if return_pdfs else None
    moving = MovingAverage(int(parameter)) if method == "moving_average" else None
    exponential = ExponentialMovingAverage(float(parameter)) if method == "ema" else None
    adaptive = None
    if method == "adaptive":
        alpha, n = parameter
        adaptive = TemporalPDF(n=n, alpha=alpha, distance_metric="l2")
    for i, pdf in enumerate(pdfs):
        pdf = normalize_pdf(pdf)
        if method == "raw":
            current = pdf
        elif method == "moving_average":
            moving.add_data(pdf)
            current = normalize_pdf(moving.predict())
        elif method == "ema":
            exponential.add_data(pdf)
            current = normalize_pdf(exponential.predict())
        elif method == "adaptive":
            current, _ = adaptive.update_pdf(pdf)
            current = normalize_pdf(current)
        else:
            raise ValueError(method)
        if filtered is not None:
            filtered[i] = current
        output[i] = decode_pdf(current, bins, component)
    return output, filtered
