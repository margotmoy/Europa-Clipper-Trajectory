"""
Compute flyby patched-conics quantities from transfer_results.csv.

Run:
    python scripts/03_compute_flybys.py

Run only Mars:
    python scripts/03_compute_flybys.py --flyby Mars
"""

from pathlib import Path
import sys
import argparse

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.flybys import flyby_summary


BODY_DATA = {
    "Earth": {
        "mu": 398600.4418,
        "radius": 6378.1363,
        "default_altitude": 3100.0,
    },
    "Mars": {
        "mu": 42828.3,
        "radius": 3396.19,
        "default_altitude": 1041.0,
    },
    "Jupiter": {
        "mu": 126686534.0,
        "radius": 71492.0,
        "default_altitude": 100000.0,
    },
}


def vec_from_row(row: pd.Series, prefix: str) -> np.ndarray:
    return np.array(
        [
            row[f"{prefix}_x_km_s"],
            row[f"{prefix}_y_km_s"],
            row[f"{prefix}_z_km_s"],
        ],
        dtype=float,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flyby", default=None, help="Example: Mars")
    args = parser.parse_args()

    transfers_path = PROJECT_ROOT / "data" / "processed_states" / "transfer_results.csv"
    output_path = PROJECT_ROOT / "data" / "processed_states" / "flyby_results.csv"

    transfers = pd.read_csv(transfers_path)
    rows = []

    for _, incoming in transfers.iterrows(): 
        
        # Determine if we need to calculate a flyby
        flyby_body = incoming["end_body"]

        if args.flyby and flyby_body != args.flyby:
            continue

        if flyby_body not in BODY_DATA:
            continue

        body = BODY_DATA[flyby_body]
        rp = body["radius"] + body["default_altitude"]

        vinf_in = vec_from_row(incoming, "vinf_arrive")

        outgoing_matches = transfers[
            (transfers["start_body"] == flyby_body) &
            (transfers["start_date"] == incoming["end_date"])
        ]

        if outgoing_matches.empty:
            continue

        outgoing = outgoing_matches.iloc[0]
        outgoing_leg = outgoing["leg"]
        vinf_out_required = vec_from_row(outgoing, "vinf_depart")

        summary = flyby_summary(
            vinf_in=vinf_in,
            vinf_out_required=vinf_out_required,
            rp=rp,
            mu=body["mu"],
            body_radius=body["radius"],
        )

        rows.append({
            "flyby_body": flyby_body,
            "flyby_date": incoming["end_date"],
            "incoming_leg": incoming["leg"],
            "outgoing_leg": outgoing_leg,
            "altitude_km": body["default_altitude"],
            "vinf_in_x_km_s": vinf_in[0],
            "vinf_in_y_km_s": vinf_in[1],
            "vinf_in_z_km_s": vinf_in[2],
            **summary,
        })

        print(f"\nFlyby: {flyby_body}")
        print(f"  incoming leg:   {incoming['leg']}")
        print(f"  date:           {incoming['end_date']}")
        print(f"  altitude:       {body['default_altitude']:.1f} km")
        print(f"  |v_inf in|:     {summary['vinf_in_mag_km_s']:.3f} km/s")
        print(f"  e_hyp:          {summary['e_hyp']:.6f}")
        print(f"  max turn:       {summary['max_turn_deg']:.3f} deg")
        print(f"  outgoing leg:   {outgoing_leg}")
        print(f"  required turn:  {summary['required_turn_deg']:.3f} deg")
        print(f"  required alt:   {summary['required_altitude_km']:.1f} km")

        print(f"\n  [debug] v_inf in  (from {incoming['leg']} arrival):  "
              f"[{vinf_in[0]:+.4f}, {vinf_in[1]:+.4f}, {vinf_in[2]:+.4f}] km/s  "
              f"|v| = {np.linalg.norm(vinf_in):.4f} km/s")
        print(f"  [debug] v_inf out (for {outgoing_leg} departure): "
              f"[{vinf_out_required[0]:+.4f}, {vinf_out_required[1]:+.4f}, {vinf_out_required[2]:+.4f}] km/s  "
              f"|v| = {np.linalg.norm(vinf_out_required):.4f} km/s")
        print(f"  [debug] angle between them: {summary['required_turn_deg']:.4f} deg")
        print(f"  feasible:       {summary['turn_feasible']}")
        print(f"  v_inf mismatch: {summary['vinf_mag_mismatch_km_s']:.6f} km/s")

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"\nSaved flyby results to: {output_path}")


if __name__ == "__main__":
    main()