"""
Request heliocentric ephemerides for all bodies/dates in mission_events.csv.

Run from project root:

    python scripts/01_request_ephemerides.py

Outputs:
    data/processed_states/ephemerides.csv
"""

from pathlib import Path
import sys

import pandas as pd


# Allow imports from project root / src
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.request_jpl import query_horizons


BODY_IDS = {
    "Sun": "10",
    "Mercury": "199",
    "Venus": "299",
    "Earth": "399",
    "Mars": "499",
    "Jupiter": "599",
    "Saturn": "699",
    "Uranus": "799",
    "Neptune": "899",
}

def collect_required_queries(events: pd.DataFrame) -> list[tuple[str, str]]:
    """Return unique body/date pairs needed for Lambert endpoints."""
    queries = set()

    for _, row in events.iterrows():
        queries.add((row["start_body"], row["start_date"]))
        queries.add((row["end_body"], row["end_date"]))

    return sorted(queries)


def main() -> None:
    config_path = PROJECT_ROOT / "config" / "mission_events.csv"
    output_dir = PROJECT_ROOT / "data" / "processed_states"
    output_path = output_dir / "ephemerides.csv"

    output_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(config_path)
    queries = collect_required_queries(events) 

    rows = []

    for body, date in queries:
        if body not in BODY_IDS:
            raise ValueError(f"No Horizons ID defined for body: {body}")

        body_id = BODY_IDS[body]
        start = str(date)
        stop = (pd.to_datetime(start) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        print(f"Requesting {body} at {start}...")

        state = query_horizons(body_id=body_id, start=start, stop=stop)

        rows.append(
            {
                "body": body,
                "date": start,
                "jd": state["jd"], 
                "horizons_date": state["date"], # include, because JPL has it's own step grid
                "x_km": state["r"][0],
                "y_km": state["r"][1],
                "z_km": state["r"][2],
                "vx_km_s": state["v"][0],
                "vy_km_s": state["v"][1],
                "vz_km_s": state["v"][2],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print(f"\nSaved ephemerides to: {output_path}")


if __name__ == "__main__":
    main()