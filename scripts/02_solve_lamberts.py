"""
Solve Lambert transfers from mission_events.csv.

Example: test only Earth -> Mars

    python scripts/02_solve_transfers.py --leg E-M

Outputs:
    data/processed_states/transfer_results.csv
"""

from pathlib import Path
import sys
import argparse

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.lambert import solve_lambert
from src.constants import MU_SUN


def get_state(states: pd.DataFrame, body: str, date: str):
    match = states[(states["body"] == body) & (states["date"] == date)]

    if match.empty:
        raise ValueError(f"No ephemeris found for {body} on {date}")

    row = match.iloc[0]

    r = np.array([row["x_km"], row["y_km"], row["z_km"]], dtype=float)
    v = np.array([row["vx_km_s"], row["vy_km_s"], row["vz_km_s"]], dtype=float)

    return r, v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--leg",
        default=None,
        help="Optional leg name to solve, e.g. E-M",
    )
    args = parser.parse_args()

    events_path = PROJECT_ROOT / "config" / "mission_events.csv"
    states_path = PROJECT_ROOT / "data" / "processed_states" / "ephemerides.csv"
    output_path = PROJECT_ROOT / "data" / "processed_states" / "transfer_results.csv"

    events = pd.read_csv(events_path)
    states = pd.read_csv(states_path)

    if args.leg:
        events = events[events["leg"] == args.leg]

        if events.empty:
            raise ValueError(f"No leg named {args.leg} found in mission_events.csv")

    rows = []

    # Iterate through each leg 
    for _, leg in events.iterrows():
        leg_name = leg["leg"]
        start_body = leg["start_body"]
        end_body = leg["end_body"]
        start_date = leg["start_date"]
        end_date = leg["end_date"]

        print(f"\nSolving {leg_name}: {start_body} -> {end_body}")
        print(f"  {start_date} to {end_date}")

        r1, v_body_start = get_state(states, start_body, start_date)
        r2, v_body_end = get_state(states, end_body, end_date)

        tof_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        tof_seconds = tof_days * 86400.0

        transfer_type = leg.get("transfer_type", "1A")

        v1_sc, v2_sc, a, p, e = solve_lambert(
            r1_vec=r1,
            r2_vec=r2,
            tof=tof_seconds,
            mu=MU_SUN,
            transfer_type=transfer_type,
        )

        # spacecraft velocity relative to body
        vinf_depart = v1_sc - v_body_start 
        vinf_arrive = v2_sc - v_body_end

        c3 = np.dot(vinf_depart, vinf_depart)

        print(f"  TOF: {tof_days:.2f} days")
        print(f"  a:   {a:.3e} km")
        print(f"  e:   {e:.6f}")
        print(f"  |v_inf depart|: {np.linalg.norm(vinf_depart):.3f} km/s")
        print(f"  C3:             {c3:.3f} km^2/s^2")
        print(f"  |v_inf arrive|: {np.linalg.norm(vinf_arrive):.3f} km/s")

        rows.append(
            {
                "leg": leg_name,
                "start_body": start_body,
                "end_body": end_body,
                "start_date": start_date,
                "end_date": end_date,
                "tof_days": tof_days,
                "a_km": a,
                "p_km": p,
                "e": e,
                "v1x_km_s": v1_sc[0],
                "v1y_km_s": v1_sc[1],
                "v1z_km_s": v1_sc[2],
                "v2x_km_s": v2_sc[0],
                "v2y_km_s": v2_sc[1],
                "v2z_km_s": v2_sc[2],
                "vinf_depart_x_km_s": vinf_depart[0],
                "vinf_depart_y_km_s": vinf_depart[1],
                "vinf_depart_z_km_s": vinf_depart[2],
                "vinf_depart_mag_km_s": np.linalg.norm(vinf_depart),
                "c3_km2_s2": c3,
                "vinf_arrive_x_km_s": vinf_arrive[0],
                "vinf_arrive_y_km_s": vinf_arrive[1],
                "vinf_arrive_z_km_s": vinf_arrive[2],
                "vinf_arrive_mag_km_s": np.linalg.norm(vinf_arrive),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)

    print(f"\nSaved transfer results to: {output_path}")


if __name__ == "__main__":
    main()