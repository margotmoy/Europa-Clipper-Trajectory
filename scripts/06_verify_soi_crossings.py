"""
Verification: C3 and hyperbolic excess velocity (v_inf) at SOI crossings.

  C3    = v_rel^2 - 2*mu/r   (vis-viva energy at SOI boundary)
  v_inf = |v_rel|             (patched-conics hand-off speed at SOI)

Checks:
  - Earth departure : spacecraft exits Earth SOI (~Oct 2024)
  - Mars arrival    : spacecraft enters Mars SOI (~Mar 2025)

Run from project root:

    python scripts/06_verify_soi_crossings.py

Outputs:
    data/verification/earth_soi_crossing.csv
    data/verification/mars_soi_crossing.csv
"""

from contextlib import redirect_stdout
from pathlib import Path
from datetime import datetime, timedelta
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.request_jpl import query_horizons

OUTPUT_DIR = PROJECT_ROOT / "data" / "verification"

MU_EARTH = 398600.4418   # km^3/s^2
EARTH_SOI_KM = 925000.0  # km

MU_MARS = 42828.375      # km^3/s^2
MARS_SOI_KM = 577000.0   # km

SPACECRAFT_ID = "-168"   # Europa Clipper (NAIF ID)
EARTH_ID = "399"
MARS_ID = "499"


def vector_columns(prefix, vec):
    return {
        f"{prefix}_x": vec[0],
        f"{prefix}_y": vec[1],
        f"{prefix}_z": vec[2],
    }


def datetime_range(start, stop, step_minutes):
    times = []
    current = datetime.fromisoformat(start)
    end = datetime.fromisoformat(stop)
    while current <= end:
        times.append(current.strftime("%Y-%m-%dT%H:%M"))
        current += timedelta(minutes=step_minutes)
    return times


def query_state(body_id, time):
    stop = (datetime.fromisoformat(time) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M")
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
        return query_horizons(body_id=body_id, start=time, stop=stop)


def relative_state(spacecraft_id, body_id, time):
    sc = query_state(spacecraft_id, time)
    body = query_state(body_id, time)
    return sc["r"] - body["r"], sc["v"] - body["v"]


def find_soi_crossing(spacecraft_id, body_id, mu, soi_km, sweep_start, sweep_stop,
                      step_minutes=60, direction="outward"):
    """Sweep in hourly steps, bracket the SOI crossing, and interpolate to the boundary."""
    times = datetime_range(sweep_start, sweep_stop, step_minutes)
    previous = None

    for t in times:
        r_rel, v_rel = relative_state(spacecraft_id, body_id, t)
        r_mag = np.linalg.norm(r_rel)

        if previous is not None:
            crossed = (
                (direction == "outward" and previous["r_mag"] < soi_km <= r_mag) or
                (direction == "inward"  and previous["r_mag"] > soi_km >= r_mag)
            )

            if crossed:
                frac = abs(soi_km - previous["r_mag"]) / abs(r_mag - previous["r_mag"])

                r_cross = previous["r_rel"] + frac * (r_rel - previous["r_rel"])
                v_cross = previous["v_rel"] + frac * (v_rel - previous["v_rel"])

                v_inf = np.linalg.norm(v_cross)
                c3 = v_inf**2 - 2.0 * mu / soi_km

                return {
                    "time_before": previous["time"],
                    "time_after": t,
                    "frac": frac,
                    "r_soi_km": soi_km,
                    "v_inf_km_s": v_inf,
                    "c3_km2_s2": c3,
                    **vector_columns("r_rel_km", r_cross),
                    **vector_columns("v_rel_km_s", v_cross),
                }

        previous = {"time": t, "r_mag": r_mag, "r_rel": r_rel, "v_rel": v_rel}

    return None


def report(result, outpath):
    if result is None:
        print("  No crossing found in search window.")
        return

    print(f"\n  Crossed between {result['time_before']} and {result['time_after']}")
    print(f"  v_inf = {result['v_inf_km_s']:.6f} km/s  (= |v_rel| at SOI)")
    print(f"  C3    = {result['c3_km2_s2']:.6f} km^2/s^2")
    pd.DataFrame([result]).to_csv(outpath, index=False)
    print(f"  Saved: {outpath}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Earth departure — SOI exit ===")
    print(f"Earth SOI = {EARTH_SOI_KM:,.0f} km\n")
    earth = find_soi_crossing(
        spacecraft_id=SPACECRAFT_ID,
        body_id=EARTH_ID,
        mu=MU_EARTH,
        soi_km=EARTH_SOI_KM,
        sweep_start="2024-10-14T16:16",
        sweep_stop="2024-10-20T00:00",
        direction="outward",
    )
    report(earth, OUTPUT_DIR / "earth_soi_crossing.csv")

    print("\n=== Mars arrival — SOI entry ===")
    print(f"Mars SOI = {MARS_SOI_KM:,.0f} km\n")
    mars = find_soi_crossing(
        spacecraft_id=SPACECRAFT_ID,
        body_id=MARS_ID,
        mu=MU_MARS,
        soi_km=MARS_SOI_KM,
        sweep_start="2025-02-15T00:00",
        sweep_stop="2025-03-03T00:00",
        direction="inward",
    )
    report(mars, OUTPUT_DIR / "mars_soi_crossing.csv")


if __name__ == "__main__":
    main()
