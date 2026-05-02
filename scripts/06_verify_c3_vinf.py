"""
Verify C3 and v_inf at SOI crossings for Europa Clipper using JPL Horizons.

Three-phase refinement per crossing (day -> hour -> minute) brackets each
boundary without interpolation. Uses body-centered coordinates
(CENTER=500@399 / 500@499), so state vectors are already planet-relative.

  C3 = v_soi^2 - 2*mu/r,  v_inf = sqrt(C3) when C3 > 0

Run from project root:
    python scripts/06_verify_c3_vinf.py

Outputs:
    data/verification/exact_earth_soi_crossing.csv
    data/verification/exact_mars_soi_crossing.csv
"""

from __future__ import annotations

import csv
import io
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.constants import (
    EARTH_CENTER,
    EARTH_SOI_KM,
    MARS_CENTER,
    MARS_SOI_KM,
    MU_EARTH,
    MU_MARS,
    SPACECRAFT_ID,
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "verification"

Direction = Literal["outward", "inward"]


@dataclass(frozen=True)
class Epoch:
    jd: float
    datestr: str
    dt: datetime
    r: np.ndarray
    v: np.ndarray

    @property
    def r_mag(self) -> float:
        return float(np.linalg.norm(self.r))

    @property
    def v_mag(self) -> float:
        return float(np.linalg.norm(self.v))


# ---------------------------------------------------------------------------
# Horizons helpers
# ---------------------------------------------------------------------------

def _horizons_params(body_id: str, center: str, start: str, stop: str, step: str) -> dict[str, str]:
    """Build a params dict instead of hand-concatenating a URL.

    This matters because Horizons is picky about quoting/escaping values such as
    COMMAND=-168, CENTER=500@399, and STEP_SIZE='1 min'. Let requests encode it.
    """
    return {
        "format": "text",
        "COMMAND": f"'{body_id}'",
        "CENTER": f"'{center}'",
        "EPHEM_TYPE": "VECTORS",
        "REF_PLANE": "ECLIPTIC",
        "REF_SYSTEM": "ICRF",
        "OUT_UNITS": "KM-S",
        "VEC_TABLE": "2",
        "VEC_LABELS": "YES",
        "CSV_FORMAT": "YES",
        "OBJ_DATA": "NO",
        "START_TIME": f"'{start}'",
        "STOP_TIME": f"'{stop}'",
        "STEP_SIZE": f"'{step}'",
    }


def _parse_horizons_datetime(raw: str) -> datetime:
    """Parse common Horizons calendar strings.

    Examples observed from Horizons CSV vector tables include:
      2024-Oct-14 12:00:00.0000
      A.D. 2024-Oct-14 12:00:00.0000
      2024-Oct-14 12:00
    """
    s = raw.strip().strip('"')
    s = re.sub(r"^(A\.D\.|B\.C\.)\s+", "", s)
    s = s.split(".")[0]

    for fmt in ("%Y-%b-%d %H:%M:%S", "%Y-%b-%d %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(f"Could not parse Horizons datetime: {raw!r}")


def _parse_all_epochs(text: str) -> list[Epoch]:
    match = re.search(r"\$\$SOE\s*(.*?)\s*\$\$EOE", text, re.DOTALL)
    if not match:
        raise ValueError("No $$SOE...$$EOE block in Horizons response.\n" + text[:1200])

    block = match.group(1).strip()
    rows = csv.reader(io.StringIO(block), skipinitialspace=True)

    epochs: list[Epoch] = []
    skipped: list[str] = []

    for row in rows:
        if not row:
            continue

        # With CSV_FORMAT=YES and VEC_TABLE=2, the useful rows are normally:
        #   JD, Calendar Date, X, Y, Z, VX, VY, VZ, ...
        # but Horizons sometimes appends extra columns. We only need first 8.
        if len(row) < 8:
            skipped.append(",".join(row))
            continue

        try:
            jd = float(row[0])
            datestr = row[1].strip().strip('"')
            dt = _parse_horizons_datetime(datestr)
            r = np.array([float(row[2]), float(row[3]), float(row[4])], dtype=float)
            v = np.array([float(row[5]), float(row[6]), float(row[7])], dtype=float)
        except Exception:
            skipped.append(",".join(row))
            continue

        epochs.append(Epoch(jd=jd, datestr=datestr, dt=dt, r=r, v=v))

    if not epochs:
        preview = "\n".join(skipped[:10]) or block[:1200]
        raise ValueError("Found $$SOE...$$EOE but parsed zero vector rows. Preview:\n" + preview)

    return epochs


def query_range(body_id: str, center: str, start: str, stop: str, step: str) -> list[Epoch]:
    resp = requests.get(
        "https://ssd.jpl.nasa.gov/api/horizons.api",
        params=_horizons_params(body_id, center, start, stop, step),
        timeout=90,
    )
    if not resp.ok:
        raise RuntimeError(f"Horizons HTTP {resp.status_code}:\n{resp.text[:1200]}")

    # Horizons can return status 200 with an error message in the body.
    if "$$SOE" not in resp.text:
        raise RuntimeError("Horizons returned no vector table. Response preview:\n" + resp.text[:1200])

    return _parse_all_epochs(resp.text)


# ---------------------------------------------------------------------------
# Bracketing/refinement
# ---------------------------------------------------------------------------

def _signed_distance(ep: Epoch, soi_km: float, direction: Direction) -> float:
    """Boundary event function whose sign changes in the desired crossing direction."""
    if direction == "outward":
        return ep.r_mag - soi_km
    if direction == "inward":
        return soi_km - ep.r_mag
    raise ValueError("direction must be 'outward' or 'inward'")


def _find_bracket(epochs: Iterable[Epoch], soi_km: float, direction: Direction) -> tuple[Epoch, Epoch] | None:
    """Return the two consecutive samples that straddle the requested crossing."""
    prev: Epoch | None = None
    prev_f: float | None = None

    for ep in epochs:
        f = _signed_distance(ep, soi_km, direction)
        if prev is not None and prev_f is not None:
            # prev_f <= 0 and f >= 0 means the requested boundary crossing happened
            # between prev and ep. This works for outward and inward because the
            # signed distance function is direction-aware.
            if prev_f <= 0.0 <= f:
                return prev, ep
        prev = ep
        prev_f = f

    return None


def _iso(dt: datetime) -> str:
    # Include seconds so Horizons receives unambiguous timestamps.
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _sweep_and_bracket(
    *,
    center: str,
    soi_km: float,
    direction: Direction,
    start: datetime | str,
    stop: datetime | str,
    step: str,
    phase_name: str,
) -> tuple[Epoch, Epoch]:
    start_s = _iso(start) if isinstance(start, datetime) else start
    stop_s = _iso(stop) if isinstance(stop, datetime) else stop

    print(f"\n  {phase_name} ({start_s} -> {stop_s}, step={step})")
    epochs = query_range(SPACECRAFT_ID, center, start_s, stop_s, step)

    if len(epochs) < 2:
        raise RuntimeError(f"{phase_name} returned fewer than two epochs.")

    bracket = _find_bracket(epochs, soi_km, direction)
    if bracket is None:
        first = epochs[0]
        last = epochs[-1]
        raise RuntimeError(
            f"{phase_name} did not bracket the {direction} SOI crossing.\n"
            f"  first: {first.datestr}  r={first.r_mag:,.3f} km\n"
            f"  last:  {last.datestr}  r={last.r_mag:,.3f} km\n"
            f"  SOI:   {soi_km:,.3f} km"
        )

    before, after = bracket
    print(
        f"  bracket: {before.datestr}  r={before.r_mag:,.3f} km"
        f"  ->  {after.datestr}  r={after.r_mag:,.3f} km"
    )
    return before, after


def find_soi_crossing(
    *,
    center: str,
    soi_km: float,
    mu: float,
    window_start: str,
    window_stop: str,
    direction: Direction,
    label: str,
) -> list[dict[str, float | str]]:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  CENTER = {center}   SOI = {soi_km:,.0f} km   direction = {direction}")

    # Phase 1: broad daily sweep.
    before, after = _sweep_and_bracket(
        center=center,
        soi_km=soi_km,
        direction=direction,
        start=window_start,
        stop=window_stop,
        step="1 d",
        phase_name="Phase 1 - daily",
    )

    # Phase 2: sweep the whole prior bracket, with a small pad. This avoids
    # shifting the search to the wrong place if the prior samples are not exactly
    # midnight or if Horizons treats STOP_TIME as exclusive-ish at a step boundary.
    before, after = _sweep_and_bracket(
        center=center,
        soi_km=soi_km,
        direction=direction,
        start=before.dt - timedelta(hours=1),
        stop=after.dt + timedelta(hours=1),
        step="1 h",
        phase_name="Phase 2 - hourly",
    )

    # Phase 3: sweep the whole prior bracket, with a small pad.
    before, after = _sweep_and_bracket(
        center=center,
        soi_km=soi_km,
        direction=direction,
        start=before.dt - timedelta(minutes=2),
        stop=after.dt + timedelta(minutes=2),
        step="1 min",
        phase_name="Phase 3 - minute",
    )

    print(f"\n  *** Crossing is between {before.datestr} and {after.datestr} UTC ***")

    def stats(ep: Epoch) -> dict[str, float]:
        r_mag = ep.r_mag
        v_soi = ep.v_mag
        c3 = v_soi**2 - 2.0 * mu / r_mag
        v_inf = float(np.sqrt(c3)) if c3 >= 0.0 else float("nan")
        return {"r_km": r_mag, "v_soi_km_s": v_soi, "c3_km2_s2": c3, "v_inf_km_s": v_inf}

    rows = []
    for name, ep in (("before", before), ("after", after)):
        s = stats(ep)
        print(
            f"  {name:>6} {ep.datestr}: "
            f"r={s['r_km']:,.3f} km  "
            f"v_soi={s['v_soi_km_s']:.6f} km/s  "
            f"C3={s['c3_km2_s2']:.6f} km^2/s^2  "
            f"v_inf={s['v_inf_km_s']:.6f} km/s"
        )
        rows.append(
            {
                "epoch": name,
                "time_utc": ep.datestr,
                "r_km": round(s["r_km"], 3),
                "v_soi_km_s": round(s["v_soi_km_s"], 6),
                "c3_km2_s2": round(s["c3_km2_s2"], 6),
                "v_inf_km_s": round(s["v_inf_km_s"], 6),
                "rx_km": round(float(ep.r[0]), 3),
                "ry_km": round(float(ep.r[1]), 3),
                "rz_km": round(float(ep.r[2]), 3),
                "vx_km_s": round(float(ep.v[0]), 6),
                "vy_km_s": round(float(ep.v[1]), 6),
                "vz_km_s": round(float(ep.v[2]), 6),
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    earth_rows = find_soi_crossing(
        center=EARTH_CENTER,
        soi_km=EARTH_SOI_KM,
        mu=MU_EARTH,
        window_start="2024-10-14 16:30:00",
        window_stop="2024-10-22 00:00:00",
        direction="outward",
        label="Earth SOI departure (Europa Clipper)",
    )
    out_e = OUTPUT_DIR / "exact_earth_soi_crossing.csv"
    pd.DataFrame(earth_rows).to_csv(out_e, index=False)
    print(f"\n  Saved -> {out_e}")

    mars_rows = find_soi_crossing(
        center=MARS_CENTER,
        soi_km=MARS_SOI_KM,
        mu=MU_MARS,
        window_start="2025-02-15 00:00:00",
        window_stop="2025-03-05 00:00:00",
        direction="inward",
        label="Mars SOI arrival (Europa Clipper)",
    )
    out_m = OUTPUT_DIR / "exact_mars_soi_crossing.csv"
    pd.DataFrame(mars_rows).to_csv(out_m, index=False)
    print(f"\n  Saved -> {out_m}")

    print("\nDone.")


if __name__ == "__main__":
    main()
