# Europa Clipper Patched-Conics Design

Trajectory design tool for the Europa Clipper mission using patched conics. Models the Earth → Mars → Earth → Jupiter transfer sequence, pulling ephemerides from JPL Horizons.

## Pipeline

Run scripts in order from the project root:

| Script | Description |
|--------|-------------|
| `scripts/01_request_ephemerides.py` | Fetch heliocentric state vectors from JPL Horizons |
| `scripts/02_solve_lamberts.py` | Solve Lambert transfers for each mission leg |
| `scripts/03_compute_flybys.py` | Compute gravity assist flyby ΔV and geometry |
| `scripts/04_plot_transfers.py` | Plot heliocentric transfer trajectories |
| `scripts/05_compute_orbital_insertion` | Calculate Jupiter Orbit Insertion ΔV |
| `scripts/06_verify_c3_vinf.py` | Verify launch C3 and arrival v∞ values |

Mission legs are configured in `config/mission_events.csv`.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Source

- `src/lambert.py` — Lambert solver
- `src/flybys.py` — flyby ΔV computation
- `src/request_jpl.py` — JPL Horizons API client
- `src/constants.py` — gravitational parameters and physical constants
