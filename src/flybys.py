"""
Reusable patched-conics flyby utilities.
"""

import math
import numpy as np
from typing import Optional


def hyperbola_eccentricity(vinf_mag: float, rp: float, mu: float) -> float:
    return 1.0 + (rp * vinf_mag**2) / mu


def turn_angle(vinf_mag: float, rp: float, mu: float) -> float:
    """
    Maximum flyby turn angle, radians.

    delta = 2 asin(1/e)
    """
    e = hyperbola_eccentricity(vinf_mag, rp, mu)
    return 2.0 * math.asin(1.0 / e)


def periapsis_radius_for_turn(vinf_mag: float, delta: float, mu: float) -> float:
    """
    Required periapsis radius for a desired turn angle.
    """
    e = 1.0 / math.sin(delta / 2.0)
    return mu * (e - 1.0) / vinf_mag**2


def rotate_vector(vec: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate vec about axis by angle using Rodrigues' rotation formula.
    """
    axis = np.asarray(axis, dtype=float)
    vec = np.asarray(vec, dtype=float)

    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0:
        raise ValueError("Rotation axis cannot be zero.")

    k = axis / axis_norm

    return (
        vec * math.cos(angle)
        + np.cross(k, vec) * math.sin(angle)
        + k * np.dot(k, vec) * (1.0 - math.cos(angle))
    )


def flyby_outgoing_vinf(
    vinf_in: np.ndarray,
    rp: float,
    mu: float,
    axis: np.ndarray,
    turn_sign: int = 1,
) -> tuple[np.ndarray, float, float]:
    """
    Rotate incoming v_inf through the maximum turn angle.

    Returns:
        vinf_out, delta, e_hyp
    """
    vinf_in = np.asarray(vinf_in, dtype=float)

    vinf_mag = np.linalg.norm(vinf_in)
    if vinf_mag == 0:
        raise ValueError("Incoming v_inf cannot be zero.")

    e_hyp = hyperbola_eccentricity(vinf_mag, rp, mu)
    delta = turn_angle(vinf_mag, rp, mu)

    vinf_out = rotate_vector(vinf_in, axis, turn_sign * delta)

    return vinf_out, delta, e_hyp


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """
    Angle between two vectors, radians.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na == 0 or nb == 0:
        raise ValueError("Cannot compute angle with zero vector.")

    c = np.dot(a, b) / (na * nb)
    c = np.clip(c, -1.0, 1.0)

    return math.acos(c)


def flyby_summary(
    vinf_in: np.ndarray,
    vinf_out_required: Optional[np.ndarray],
    rp: float,
    mu: float,
    body_radius: float = 0.0,
) -> dict:
    """
    Compute flyby summary.
    """
    vinf_in_mag = np.linalg.norm(vinf_in) 
    e_hyp = hyperbola_eccentricity(vinf_in_mag, rp, mu)
    delta_max = turn_angle(vinf_in_mag, rp, mu)

    out = {
        "vinf_in_mag_km_s": vinf_in_mag,
        "rp_km": rp,
        "e_hyp": e_hyp,
        "max_turn_rad": delta_max,
        "max_turn_deg": math.degrees(delta_max),
    }

    if vinf_out_required is not None:
        required_turn = angle_between(vinf_in, vinf_out_required)
        vinf_out_mag = np.linalg.norm(vinf_out_required)

        rp_required = periapsis_radius_for_turn(vinf_in_mag, required_turn, mu)
        out.update(
            {
                "vinf_out_required_mag_km_s": vinf_out_mag,
                "required_turn_rad": required_turn,
                "required_turn_deg": math.degrees(required_turn),
                "turn_feasible": required_turn <= delta_max,
                "vinf_mag_mismatch_km_s": vinf_out_mag - vinf_in_mag,
                "required_altitude_km": rp_required - body_radius,
            }
        )

    return out