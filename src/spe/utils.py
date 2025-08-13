"""
Copyright (c) 2024 Julien Posso
"""

import numpy
import numpy as np
import warnings


def quat2dcm(q: np.ndarray) -> np.ndarray:
    """
    Converts scalar-first unit quaternion to a Direction Cosine Matrix (DCM).

    Conventions:
        Right-handed coordinate system.
        Hamilton's multiplication.
        Scalar-first quaternion.
        Active rotation.

    References:
        Moti Ben-Ari section 4.6 "A Tutorial on Euler Angles and Quaternions"
        https://raw.githubusercontent.com/motib/mathematics/master/quaternions/quaternion-tutorial.pdf
        Hannes Sommer et al., "Why and How to Avoid the Flipped Quaternion Multiplication"
        https://arxiv.org/abs/1801.07478

    Args:
        q (numpy.ndarray): A scalar-first unit quaternion representing the rotation.

    Returns:
        numpy.ndarray: A 3x3 Direction Cosine Matrix representing the rotation.
    """

    # Verify that input quaternion is unit
    assert np.isclose(np.linalg.norm(q), 1.0) == 1.0, "q must be unit quaternion"

    q0, q1, q2, q3 = q

    dcm = np.zeros((3, 3))

    dcm[0, 0] = 2 * q0 ** 2 - 1 + 2 * q1 ** 2
    dcm[1, 1] = 2 * q0 ** 2 - 1 + 2 * q2 ** 2
    dcm[2, 2] = 2 * q0 ** 2 - 1 + 2 * q3 ** 2

    dcm[0, 1] = 2 * q1 * q2 - 2 * q0 * q3
    dcm[0, 2] = 2 * q1 * q3 + 2 * q0 * q2

    dcm[1, 0] = 2 * q1 * q2 + 2 * q0 * q3
    dcm[1, 2] = 2 * q2 * q3 - 2 * q0 * q1

    dcm[2, 0] = 2 * q1 * q3 - 2 * q0 * q2
    dcm[2, 1] = 2 * q2 * q3 + 2 * q0 * q1

    return dcm


def dcm2quat(dcm: np.ndarray, enforce_north: bool=False) -> np.ndarray:
    """
    Converts a Direction Cosine Matrix (DCM) to a scalar-first unit quaternion.

    Conventions:
        Right-handed coordinate system.
        Hamilton's multiplication.
        Scalar-first quaternion.
        Active rotation.

    References:
        Richard A. Spurrier "Comment on "Singularity-Free Extraction of a Quaternion from a Direction-Cosine Matrix""
        https://arc.aiaa.org/doi/10.2514/3.57311

    Args:
        dcm (numpy.ndarray): A Direction Cosine Matrix (DCM) representing the rotation.
        enforce_north (bool): Whether to enforce quaternion in the North Pole (positive scalar part).

    Returns:
        numpy.ndarray: q (numpy.ndarray): A scalar-first unit quaternion representing the same rotation as the DCM.
    """
    # Extract the elements for readability
    m11, m12, m13 = dcm[0, 0], dcm[0, 1], dcm[0, 2]
    m21, m22, m23 = dcm[1, 0], dcm[1, 1], dcm[1, 2]
    m31, m32, m33 = dcm[2, 0], dcm[2, 1], dcm[2, 2]
    trace = np.trace(dcm)

    # If trace is the largest, compute equations 2 and 3, else equations 4, 5 and 6 of the referenced paper
    if trace > max(m11, m22, m33):
        # Cyclic order (i, j, k) = (1, 2, 3)
        q0 = np.sqrt(1 + trace) / 2
        q1 = (m32 - m23) / (4 * q0)
        q2 = (m13 - m31) / (4 * q0)
        q3 = (m21 - m12) / (4 * q0)
    elif m11 > max(trace, m22, m33):
        # Cyclic order (i, j, k) = (1, 2, 3)
        q1 = np.sqrt((m11 / 2) + (1 - trace) / 4)
        q0 = (m32 - m23) / (4 * q1)
        q2 = (m21 + m12) / (4 * q1)
        q3 = (m31 + m13) / (4 * q1)
    elif m22 > max(trace, m11, m33):
        # Cyclic order (i, j, k) = (2, 3, 1)
        q2 = np.sqrt((m22 / 2) + (1 - trace) / 4)
        q0 = (m13 - m31) / (4 * q2)
        q3 = (m32 + m23) / (4 * q2)
        q1 = (m12 + m21) / (4 * q2)
    else:
        # Cyclic order (i, j, k) = (3, 1, 2)
        q3 = np.sqrt((m33 / 2) + (1 - trace) / 4)
        q0 = (m21 - m12) / (4 * q3)
        q1 = (m13 + m31) / (4 * q3)
        q2 = (m23 + m32) / (4 * q3)

    q = np.array([q0, q1, q2, q3])

    if enforce_north:
        if q[0] < 0:
            q = np.sign(q[0]) * q

    # Ensure unit quaternion
    q = q / np.linalg.norm(q)

    return q


def quat2euler(q: np.ndarray) -> tuple:
    """
    Converts scalar-first unit quaternion to euler angles.

    Conventions:
        Axis rotation sequence: 3, 2, 1 (Yaw (Z), then Pitch(Y), then Roll (X)).
        Right-handed coordinate system.
        Hamilton's multiplication.
        Scalar-first quaternion.
        Active rotation.

    References:
        Moti Ben-Ari section 4.6 + 3.7 "A Tutorial on Euler Angles and Quaternions"
        https://raw.githubusercontent.com/motib/mathematics/master/quaternions/quaternion-tutorial.pdf

    Args:
        q (numpy.ndarray): A scalar-first unit quaternion representing the rotation.

    Returns:
        tuple: Euler angles (yaw, pitch, roll) in degrees.
    """

    # Verify that input quaternion is unit
    assert np.isclose(np.linalg.norm(q), 1.0), "q must be unit quaternion"

    q0, q1, q2, q3 = q

    y_rad = np.arctan2(2 * (q0 * q3 + q1 * q2), 2 * (q0 ** 2 + q1 ** 2) - 1)
    # Add clipping for numerical precision issues
    p_rad_clip_arg = np.clip(1 - (2 * (q1 * q3 - q0 * q2)) ** 2, 0, 1)
    p_rad = np.arctan2(-2 * (q1 * q3 - q0 * q2), np.sqrt(p_rad_clip_arg))
    r_rad = np.arctan2(2 * (q0 * q1 + q2 * q3), 2 * (q0 ** 2 + q3 ** 2) - 1)

    yaw = np.rad2deg(y_rad)
    pitch = np.rad2deg(p_rad)
    roll = np.rad2deg(r_rad)

    # Check for gimbal lock condition
    if p_rad_clip_arg <= np.finfo(float).eps:
        warnings.simplefilter('always', UserWarning)
        warnings.warn("quat2euler: gimbal lock detected. Precision may be lost in yaw and roll calculations.",
                      UserWarning)

    return yaw, pitch, roll


def euler2quat(
        yaw: float,
        pitch: float,
        roll: float,
        enforce_north: bool = False,
        gymbal_check: bool = True
) -> np.ndarray:
    """
    Convert Euler angles in degrees to a scalar-first unit quaternion.

    Conventions:
        Axis rotation sequence: 3, 2, 1 (Yaw (Z), then Pitch(Y), then Roll (X)).
        Right-handed coordinate system.
        Hamilton's multiplication.
        Scalar-first quaternion.
        Active rotation.

    References:
        NASA Appendix A-(10) "Euler angles, quaternions, and transformation matrices for space shuttle analysis"
        https://ntrs.nasa.gov/citations/19770019231
        Moti Ben-Ari section 4.8 "A Tutorial on Euler Angles and Quaternions"
        https://raw.githubusercontent.com/motib/mathematics/master/quaternions/quaternion-tutorial.pdf
        Hannes Sommer et al., "Why and How to Avoid the Flipped Quaternion Multiplication"
        https://arxiv.org/abs/1801.07478

    Args:
        yaw (float): Yaw angle in degrees.
        pitch (float): Pitch angle in degrees.
        roll (float): Roll angle in degrees.
        enforce_north (bool): Whether to enforce quaternion in the North Pole (positive scalar part). Defaults to False.
        gymbal_check (bool): Whether to check for gymbal lock. Defaults to True.
    Returns:
        numpy.ndarray: Scalar-first unit quaternion [q1, q2, q3, q4].
    """

    assert -180 <= yaw <= 180, f"yaw is outside the range [-180° ; +180°] range: {yaw}°"
    assert -90 <= pitch <= 90, f"pitch is outside the range [-90° ; +90°] range: {pitch}°"
    assert -180 <= roll <= 180, f"roll is outside the range [-180° ; +180°] range: {roll}°"

    if gymbal_check:
        if np.isclose(pitch, 90) or np.isclose(pitch, -90):
            warnings.simplefilter('always', UserWarning)
            warnings.warn(f"euler2quat: gymbal lock detected, input pitch = {pitch:.3f}°", UserWarning)

    cy = np.cos(np.deg2rad(yaw) / 2)
    sy = np.sin(np.deg2rad(yaw) / 2)
    cp = np.cos(np.deg2rad(pitch) / 2)
    sp = np.sin(np.deg2rad(pitch) / 2)
    cr = np.cos(np.deg2rad(roll) / 2)
    sr = np.sin(np.deg2rad(roll) / 2)

    q = np.array([
        cy * cp * cr + sy * sp * sr,  # scalar part (q1)
        cy * cp * sr - sy * sp * cr,  # vector part (q2)
        cy * sp * cr + sy * cp * sr,  # vector part (q3)
        sy * cp * cr - cy * sp * sr   # vector part (q4)
    ])

    if enforce_north:
        if q[0] < 0:
            q = np.sign(q[0]) * q

    # Ensure unit quaternion
    q = q / np.linalg.norm(q)

    return q


def euler2dcm(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """
    Converts Euler angles in degrees to a Direction Cosine Matrix (DCM).

    Conventions:
        Axis rotation sequence: 3, 2, 1 (Yaw (Z), then Pitch(Y), then Roll (X)).
        Right-handed coordinate system.
        Active rotation.

    References:
        NASA Appendix A-(10) "Euler angles, quaternions, and transformation matrices for space shuttle analysis"
        https://ntrs.nasa.gov/citations/19770019231
        Moti Ben-Ari section 3.7 "A Tutorial on Euler Angles and Quaternions"
        https://raw.githubusercontent.com/motib/mathematics/master/quaternions/quaternion-tutorial.pdf
        Hannes Sommer et al., "Why and How to Avoid the Flipped Quaternion Multiplication"
        https://arxiv.org/abs/1801.07478

    Args:
        yaw (float): Yaw angle in degrees.
        pitch (float): Pitch angle in degrees.
        roll (float): Roll angle in degrees.

    Returns:
        numpy.ndarray: A 3x3 Direction Cosine Matrix representing the rotation.
    """
    assert -180 <= yaw <= 180, f"yaw is outside the range [-180° ; +180°] range: {yaw}°"
    assert -90 <= pitch <= 90, f"pitch is outside the range [-90° ; +90°] range: {pitch}°"
    assert -180 <= roll <= 180, f"roll is outside the range [-180° ; +180°] range: {roll}°"

    if np.isclose(pitch, 90) or np.isclose(pitch, -90):
        warnings.simplefilter('always', UserWarning)
        warnings.warn(f"euler2dcm: gymbal lock detected, input pitch = {pitch:.3f}°", UserWarning)

    cy = np.cos(np.deg2rad(yaw))
    sy = np.sin(np.deg2rad(yaw))
    cp = np.cos(np.deg2rad(pitch))
    sp = np.sin(np.deg2rad(pitch))
    cr = np.cos(np.deg2rad(roll))
    sr = np.sin(np.deg2rad(roll))

    dcm = np.zeros((3, 3))

    dcm[0, 0] = cy * cp
    dcm[1, 1] = sy * sp * sr + cy * cr
    dcm[2, 2] = cp * cr

    dcm[0, 1] = cy * sp * sr - sy * cr
    dcm[0, 2] = cy * sp * cr + sy * sr

    dcm[1, 0] = sy * cp
    dcm[1, 2] = sy * sp * cr - cy * sr

    dcm[2, 0] = -sp
    dcm[2, 1] = cp * sr

    return dcm


def dcm2euler(dcm: np.ndarray) -> tuple:
    """
    Converts a Direction Cosine Matrix (DCM) to Euler angles in degrees.
    The code does not explicitly handle Gymbal lock. Gymbal lock occurs when the second angle is +/-90°. In ZYX order
    it corresponds to the Pitch angle. However, the division by zero is avoided with the atan2 function.

    Conventions:
        Axis rotation sequence: 3, 2, 1 (Yaw (Z), then Pitch(Y), then Roll (X)).
        Right-handed coordinate system.
        Active rotation.

    References:
        NASA Appendix A-(10) "Euler angles, quaternions, and transformation matrices for space shuttle analysis"
        https://ntrs.nasa.gov/citations/19770019231

    Args:
        dcm (numpy.ndarray): Direction Cosine Matrix (DCM) representing the rotation.

    Returns:
        tuple: Euler angles (yaw, pitch, roll) in degrees.
    """
    # Extract the elements for readability
    m11, m12, m13 = dcm[0, 0], dcm[0, 1], dcm[0, 2]
    m21, m22, m23 = dcm[1, 0], dcm[1, 1], dcm[1, 2]
    m31, m32, m33 = dcm[2, 0], dcm[2, 1], dcm[2, 2]

    # Detect gimbal lock condition
    if np.isclose(abs(m31), 1.0):
        warnings.simplefilter('always', UserWarning)
        warnings.warn("dcm2euler: gimbal lock detected. Precision may be lost in yaw and roll calculations.", UserWarning)

    yaw = np.rad2deg(np.arctan2(m21, m11))
    pitch = np.rad2deg(np.arctan2(-m31, np.sqrt(1 - (m31 ** 2))))
    roll = np.rad2deg(np.arctan2(m32, m33))

    return yaw, pitch, roll


def multiply_quaternions(qa, qb):
    """
    Multiplies two scalar-first quaternions and returns the product.

    Conventions:
        Hamilton's multiplication.
        Scalar-first quaternion.

    References:
        Moti Ben-Ari section 4.2 "A Tutorial on Euler Angles and Quaternions"
        https://raw.githubusercontent.com/motib/mathematics/master/quaternions/quaternion-tutorial.pdf

    Args:
        qa (np.ndarray): First quaternion as an array [w1, x1, y1, z1].
        qb (np.ndarray): Second quaternion as an array [w2, x2, y2, z2].

    Returns:
        np.ndarray: The product of the two quaternions as an array [w, x, y, z].
    """
    q0, q1, q2, q3 = qa
    p0, p1, p2, p3 = qb

    w = q0 * p0 - q1 * p1 - q2 * p2 - q3 * p3
    x = q0 * p1 + q1 * p0 + q2 * p3 - q3 * p2
    y = q0 * p2 + q2 * p0 - q1 * p3 + q3 * p1
    z = q0 * p3 + q3 * p0 + q1 * p2 - q2 * p1

    q = np.array([w, x, y, z])

    # Ensure unit quaternion
    q = q / np.linalg.norm(q)

    return q


def conjugate_quaternion(q: np.ndarray) -> np.ndarray:
    """
    Convert an active rotation scalar-first quaternion to its corresponding passive rotation quaternion or vice-versa.

    Conventions:
        Hamilton's multiplication.
        Scalar-first quaternion.

    References:
        Moti Ben-Ari section 4.4 "A Tutorial on Euler Angles and Quaternions"
        https://raw.githubusercontent.com/motib/mathematics/master/quaternions/quaternion-tutorial.pdf

    Parameters:
    - q: A quaternion in the format [w, x, y, z], where w is the scalar part and x, y, z are the vector part.

    Returns:
    - The conjugate of the quaternion [w, -x, -y, -z].
    """
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def euler_angle_difference(angle1: float, angle2: float) -> float:
    """
    Compute the difference between two Euler angles in degrees, accounting for the circular nature of Yaw and Roll angles.

    Note:
        The Pitch angle is constrained between [-90, +90] degrees, so conditions such as `diff > 180` or `diff < -180`
        do not apply to it. However, the program remains valid for computing angle differences in general.

    Args:
        angle1 (float): The first angle in degrees.
        angle2 (float): The second angle in degrees.

    Returns:
        float: The angle difference, normalized to the range [-180, 180] degrees.
    """
    # Calculate the raw difference between the two angles
    diff = angle2 - angle1

    # Adjust the difference to account for the circular nature of angles
    if diff > 180:
        diff = diff - 360  # Wrap around the positive overflow
    elif diff < -180:
        diff = diff + 360  # Wrap around the negative overflow

    return diff


def generate_orientation(n_samples: int) -> np.ndarray:
    """
    Generate n_samples uniform random unit quaternions.
    Generated quaternions can be interpreted either as scalar-first or vector-first.
    Sources:
        Sharma et al., "POSE ESTIMATION FOR NON-COOPERATIVE SPACECRAFT RENDEZVOUS USING NEURAL NETWORKS",
        algorithm 1 https://arxiv.org/pdf/1906.09868.pdf
        K. Shoemake, "III.6 - UNIFORM RANDOM ROTATIONS"
        https://www.sciencedirect.com/science/article/abs/pii/B9780080507552500361?via%3Dihub
    """
    # Step 1-3: Generate m uniformly distributed random values in [0, 1]
    x0 = np.random.rand(n_samples)
    x1 = np.random.rand(n_samples)
    x2 = np.random.rand(n_samples)

    # Step 4-5: Calculate theta values
    theta1 = 2 * np.pi * x1
    theta2 = 2 * np.pi * x2

    # Step 6-9: Calculate sine and cosine of theta values
    s1 = np.sin(theta1)
    s2 = np.sin(theta2)
    c1 = np.cos(theta1)
    c2 = np.cos(theta2)

    # Step 10-11: Calculate the square roots
    r1 = np.sqrt(1 - x0)
    r2 = np.sqrt(x0)

    # Step 12: Assemble the quaternions
    q = np.array([s1 * r1, c1 * r1, s2 * r2, c2 * r2]).T

    return q
