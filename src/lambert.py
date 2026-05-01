"""
Lambert solver and two-body propagator.

This implementation follows the MECH 578 Lecture 18/19 alpha-beta
Lambert formulation for elliptic transfers.

Supported elliptic transfer classes:

    1A: transfer angle < 180 deg, short-period ellipse
    1B: transfer angle < 180 deg, long-period ellipse
    2A: transfer angle > 180 deg, short-period ellipse
    2B: transfer angle > 180 deg, long-period ellipse

Lecture mapping:
    Type 1 = transfer angle < 180 deg
    Type 2 = transfer angle > 180 deg
    A      = smaller time of flight branch
    B      = larger time of flight branch
"""

import math
import numpy as np

from .constants import MU_SUN


def solve_lambert(
    r1_vec: np.ndarray,
    r2_vec: np.ndarray,
    tof: float,
    mu: float = MU_SUN,
    transfer_type: str = "1A",
    tol: float = 1e-6,
    max_iter: int = 200,
) -> tuple:
    """
    Solve Lambert's problem for an elliptic transfer.

    Parameters
    ----------
    r1_vec : ndarray
        Initial position vector, km.

    r2_vec : ndarray
        Final position vector, km.

    tof : float
        Time of flight, seconds.

    mu : float
        Gravitational parameter of central body, km^3/s^2.

    transfer_type : str
        One of: "1A", "1B", "2A", "2B".

        1 = transfer angle < 180 deg
        2 = transfer angle > 180 deg
        A = short-period branch
        B = long-period branch

    tol : float
        Time-of-flight convergence tolerance, seconds.

    max_iter : int
        Maximum number of bisection iterations.

    Returns
    -------
    v1_vec : ndarray
        Velocity at r1, km/s.

    v2_vec : ndarray
        Velocity at r2, km/s.

    a : float
        Semi-major axis, km.

    p : float
        Semi-latus rectum, km.

    e : float
        Eccentricity.

    Notes
    -----
    This is an elliptic Lambert solver. It does not currently support
    hyperbolic or multi-revolution Lambert solutions.
    """

    transfer_type = transfer_type.upper().strip()

    if transfer_type not in {"1A", "1B", "2A", "2B"}:
        raise ValueError(
            f"transfer_type must be one of 1A, 1B, 2A, 2B. Got {transfer_type}"
        )

    # ------------------------------------------------------------------
    # Lecture 19 Step 1:
    # Determine the space triangle: r1, r2, chord c, semi-perimeter s.
    # ------------------------------------------------------------------

    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)

    cos_phi = float(np.dot(r1_vec, r2_vec) / (r1 * r2))
    cos_phi = np.clip(cos_phi, -1.0, 1.0)

    # The chord connects P1 to P2.
    c = math.sqrt(r1**2 + r2**2 - 2.0 * r1 * r2 * cos_phi)

    # Semi-perimeter of the space triangle.
    s = 0.5 * (r1 + r2 + c)

    # Minimum possible semi-major axis for an elliptic transfer.
    # Lecture notation: a_min = s / 2.
    a_min = 0.5 * s

    # ------------------------------------------------------------------
    # Type 1 vs Type 2 controls the sign of sin(phi).
    #
    # Type 1: transfer angle < 180 deg
    # Type 2: transfer angle > 180 deg
    #
    # The f/g velocity equations need signed sin(phi).
    # ------------------------------------------------------------------

    sin_phi_mag = np.linalg.norm(np.cross(r1_vec, r2_vec)) / (r1 * r2)

    if transfer_type.startswith("1"):
        sin_phi = +sin_phi_mag
    else:
        sin_phi = -sin_phi_mag

    # ------------------------------------------------------------------
    # Lecture 19 Step 2:
    # Calculate parabolic TOF.
    #
    # For Type 1:
    #   TOF_p = (1/3) * sqrt(2/mu) * [s^(3/2) - (s-c)^(3/2)]
    #
    # For Type 2:
    #   TOF_p = (1/3) * sqrt(2/mu) * [s^(3/2) + (s-c)^(3/2)]
    #
    # If TOF < TOF_p, the transfer would be hyperbolic.
    # This solver only handles elliptic transfers.
    # ------------------------------------------------------------------

    tof_parabolic = _tof_parabolic(s, c, mu, transfer_type)

    if tof <= tof_parabolic:
        raise ValueError(
            f"TOF = {tof/86400:.3f} days is below the parabolic TOF "
            f"= {tof_parabolic/86400:.3f} days for {transfer_type}. "
            "This requires a hyperbolic Lambert solver."
        )

    # ------------------------------------------------------------------
    # Lecture 19:
    # A/B branch decision.
    #
    # At a = a_min, alpha_0 = pi.
    # This gives the minimum-energy time of flight for that Type/branch.
    #
    # Type A exists for:
    #   TOF_p < TOF <= TOF_min_energy
    #
    # Type B exists for:
    #   TOF >= TOF_min_energy
    #
    # This is exactly why Earth->Mars is 1A while Mars->Earth is 1B.
    # ------------------------------------------------------------------

    tof_min_energy = _tof_elliptic(a_min * (1.0 + 1e-12), s, c, mu, transfer_type)

    branch = transfer_type[1]

    if branch == "A" and tof > tof_min_energy:
        raise ValueError(
            f"{transfer_type} requested, but TOF = {tof/86400:.3f} days "
            f"is larger than the minimum-energy TOF "
            f"= {tof_min_energy/86400:.3f} days. "
            f"Use {transfer_type[0]}B instead."
        )

    if branch == "B" and tof < tof_min_energy:
        raise ValueError(
            f"{transfer_type} requested, but TOF = {tof/86400:.3f} days "
            f"is smaller than the minimum-energy TOF "
            f"= {tof_min_energy/86400:.3f} days. "
            f"Use {transfer_type[0]}A instead."
        )

    # ------------------------------------------------------------------
    # Lecture 19 Step 6:
    # Iterate on a in Lambert's equation.
    #
    # For A branches:
    #   TOF decreases from TOF_min_energy toward TOF_parabolic as a grows.
    #
    # For B branches:
    #   TOF increases from TOF_min_energy as a grows.
    #
    # Each branch is monotonic, so bisection works.
    # ------------------------------------------------------------------

    if branch == "A":
        a = _bisect_a_type_A(
            tof=tof,
            a_min=a_min,
            s=s,
            c=c,
            mu=mu,
            transfer_type=transfer_type,
            tol=tol,
            max_iter=max_iter,
        )
    else:
        a = _bisect_a_type_B(
            tof=tof,
            a_min=a_min,
            s=s,
            c=c,
            mu=mu,
            transfer_type=transfer_type,
            tol=tol,
            max_iter=max_iter,
        )

    # ------------------------------------------------------------------
    # Lecture 19 Step 5:
    # Calculate alpha and beta with quadrant rules.
    #
    # First compute alpha_0 and beta_0.
    #
    # Then:
    #   1A: alpha = alpha_0,      beta =  beta_0
    #   1B: alpha = 2pi-alpha_0,  beta =  beta_0
    #   2A: alpha = alpha_0,      beta = -beta_0
    #   2B: alpha = 2pi-alpha_0,  beta = -beta_0
    # ------------------------------------------------------------------

    alpha, beta = _alpha_beta(a, s, c, transfer_type)

    # ------------------------------------------------------------------
    # Lecture 19 Step 7:
    # Find p and e.
    #
    # p = [4a(s-r1)(s-r2)/c^2] * sin^2((alpha ± beta)/2)
    #
    # Sign convention from Lecture 19:
    #   1B and 2A -> plus sign
    #   1A and 2B -> minus sign
    # ------------------------------------------------------------------

    if transfer_type in {"1A", "2B"}:
        p_angle = 0.5 * (alpha + beta)
    else:
        p_angle = 0.5 * (alpha - beta)

    p = (4.0 * a * (s - r1) * (s - r2) / c**2) * math.sin(p_angle) ** 2

    if p <= 0:
        raise RuntimeError(f"Computed invalid p = {p}. Check transfer type/geometry.")

    e = math.sqrt(max(0.0, 1.0 - p / a))

    # ------------------------------------------------------------------
    # Compute velocity vectors using Lagrange f and g coefficients.
    #
    # This is equivalent to the velocity-vector step in the lecture, but
    # easier to implement robustly in vector form.
    # ------------------------------------------------------------------

    f = 1.0 - (r2 / p) * (1.0 - cos_phi)
    g = r1 * r2 * sin_phi / math.sqrt(mu * p)
    g_dot = 1.0 - (r1 / p) * (1.0 - cos_phi)

    if abs(g) < 1e-12:
        raise RuntimeError("Computed g is nearly zero; transfer geometry is singular.")

    v1_vec = (r2_vec - f * r1_vec) / g
    v2_vec = (g_dot * r2_vec - r1_vec) / g

    return v1_vec, v2_vec, a, p, e


# ----------------------------------------------------------------------
# Internal helper functions
# ----------------------------------------------------------------------

def _alpha_beta(a: float, s: float, c: float, transfer_type: str) -> tuple:
    """
    Compute alpha and beta using Lecture 19 quadrant rules.
    """

    alpha0 = 2.0 * math.asin(math.sqrt(np.clip(s / (2.0 * a), 0.0, 1.0)))
    beta0 = 2.0 * math.asin(math.sqrt(np.clip((s - c) / (2.0 * a), 0.0, 1.0)))

    if transfer_type == "1A":
        alpha = alpha0
        beta = beta0

    elif transfer_type == "1B":
        alpha = 2.0 * math.pi - alpha0
        beta = beta0

    elif transfer_type == "2A":
        alpha = alpha0
        beta = -beta0

    elif transfer_type == "2B":
        alpha = 2.0 * math.pi - alpha0
        beta = -beta0

    else:
        raise ValueError(f"Invalid transfer_type: {transfer_type}")

    return alpha, beta


def _tof_elliptic(
    a: float,
    s: float,
    c: float,
    mu: float,
    transfer_type: str,
) -> float:
    """
    Elliptic Lambert time of flight.

    Lecture form:

        TOF = sqrt(a^3 / mu) *
              [(alpha - beta) - (sin(alpha) - sin(beta))]
    """

    alpha, beta = _alpha_beta(a, s, c, transfer_type)

    return math.sqrt(a**3 / mu) * (
        (alpha - beta) - (math.sin(alpha) - math.sin(beta))
    )


def _tof_parabolic(s: float, c: float, mu: float, transfer_type: str) -> float:
    """
    Parabolic time of flight from Lecture 19.

    Type 1:
        TOF_p = (1/3) sqrt(2/mu) [s^(3/2) - (s-c)^(3/2)]

    Type 2:
        TOF_p = (1/3) sqrt(2/mu) [s^(3/2) + (s-c)^(3/2)]
    """

    base = (1.0 / 3.0) * math.sqrt(2.0 / mu)

    if transfer_type.startswith("1"):
        return base * (s**1.5 - (s - c) ** 1.5)

    return base * (s**1.5 + (s - c) ** 1.5)


def _bisect_a_type_A(
    tof: float,
    a_min: float,
    s: float,
    c: float,
    mu: float,
    transfer_type: str,
    tol: float,
    max_iter: int,
) -> float:
    """
    Bisection for Type A branches.

    Type A TOF decreases as a increases:
        at a_min: TOF = TOF_min_energy
        as a -> infinity: TOF -> TOF_parabolic
    """

    a_lo = a_min * (1.0 + 1e-12)
    a_hi = a_min * 2.0

    # Expand upper bound until the Type A TOF is below target.
    while _tof_elliptic(a_hi, s, c, mu, transfer_type) > tof:
        a_hi *= 2.0

        if a_hi > 1e16:
            raise RuntimeError("Could not bracket Type A Lambert solution.")

    for _ in range(max_iter):
        a_mid = 0.5 * (a_lo + a_hi)
        tof_mid = _tof_elliptic(a_mid, s, c, mu, transfer_type)

        if abs(tof_mid - tof) < tol:
            return a_mid

        # Type A is decreasing in a.
        if tof_mid > tof:
            a_lo = a_mid
        else:
            a_hi = a_mid

    return a_mid


def _bisect_a_type_B(
    tof: float,
    a_min: float,
    s: float,
    c: float,
    mu: float,
    transfer_type: str,
    tol: float,
    max_iter: int,
) -> float:
    """
    Bisection for Type B branches.

    Type B TOF increases as a increases:
        at a_min: TOF = TOF_min_energy
        as a grows: TOF grows
    """

    a_lo = a_min * (1.0 + 1e-12)
    a_hi = a_min * 2.0

    # Expand upper bound until the Type B TOF exceeds target.
    while _tof_elliptic(a_hi, s, c, mu, transfer_type) < tof:
        a_hi *= 2.0

        if a_hi > 1e16:
            raise RuntimeError("Could not bracket Type B Lambert solution.")

    for _ in range(max_iter):
        a_mid = 0.5 * (a_lo + a_hi)
        tof_mid = _tof_elliptic(a_mid, s, c, mu, transfer_type)

        if abs(tof_mid - tof) < tol:
            return a_mid

        # Type B is increasing in a.
        if tof_mid < tof:
            a_lo = a_mid
        else:
            a_hi = a_mid

    return a_mid