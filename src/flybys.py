"""
Functions for patched conic flyby calculations.
"""

import math
import numpy as np

def hyperbola_eccentricity(vinf_mag: float, rp: float, mu: float) -> float:
    """
    Flyby hyperbola eccentricity 

    """
    a_hyp = -mu / vinf_mag**2  # semi-major axis 
    return rp / abs(a_hyp) + 1.0 


def turn_angle(vinf_mag: float, rp: float, mu: float) -> float:
    """
    Maximum flyby turn angle, radians.

    """
    e = hyperbola_eccentricity(vinf_mag, rp, mu)
    return 2.0 * math.asin(1.0 / e)


def periapsis_radius_for_turn(vinf_mag: float, delta: float, mu: float) -> float:
    """
    Required periapsis radius for a desired turn angle.
    """
    e = 1.0 / math.sin(delta / 2.0)
    return mu * (e - 1.0) / vinf_mag**2

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


def compute_flyby(
    vinf_in: np.ndarray,
    vinf_out_required: np.ndarray,
    mu: float,
    body_radius: float = 0.0,
) -> dict:
    """
    Compute flyby summary. Turn angle is derived from vinf_in and vinf_out_required, then periapsis radius is solved from that angle.
    """
    vinf_in_mag = np.linalg.norm(vinf_in)
    vinf_out_mag = np.linalg.norm(vinf_out_required)
    required_turn = angle_between(vinf_in, vinf_out_required)
    rp = periapsis_radius_for_turn(vinf_in_mag, required_turn, mu)
    e_hyp = hyperbola_eccentricity(vinf_in_mag, rp, mu)

    return {
        "vinf_in_mag_km_s": vinf_in_mag,
        "vinf_out_required_mag_km_s": vinf_out_mag,
        "rp_km": rp,
        "required_altitude_km": rp - body_radius,
        "e_hyp": e_hyp,
        "required_turn_rad": required_turn,
        "required_turn_deg": math.degrees(required_turn),
        "vinf_mag_mismatch_km_s": vinf_out_mag - vinf_in_mag,
    }