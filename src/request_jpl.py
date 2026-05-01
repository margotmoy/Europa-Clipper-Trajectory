"""
Wrapper for the JPL Horizons API.

Returns heliocentric ecliptic J2000 state vectors as numpy arrays.
"""

import re

import numpy as np
import requests

def query_horizons(body_id: str, start: str, stop: str) -> dict:
    """Query JPL Horizons for a heliocentric ecliptic J2000 state vector.

    Returns
    -------
    dict with keys: jd (float), date (str), r (ndarray km), v (ndarray km/s)
    """
    url = (
        f"https://ssd.jpl.nasa.gov/api/horizons.api"
        f"?format=text"
        f"&COMMAND={body_id}" 
        f"&CENTER=@sun" # heliocentric :) 
        f"&EPHEM_TYPE=VECTORS"
        f"&REF_PLANE=ECLIPTIC"
        f"&REF_SYSTEM=ICRF"
        f"&OUT_UNITS=KM-S"
        f"&VEC_TABLE=2"
        f"&VEC_LABELS=YES"
        f"&CSV_FORMAT=YES"
        f"&OBJ_DATA=NO"
        f"&START_TIME={start}"
        f"&STOP_TIME={stop}"
        f"&STEP_SIZE=1d"
    )
    resp = requests.get(url, timeout=30)
    if not resp.ok:
        print(f"  HTTP {resp.status_code}: {resp.text[:400]}")
        resp.raise_for_status()
    print(resp.text)
    return _parse_vectors(resp.text)


def _parse_vectors(text: str) -> dict:
    """Extract the state vector from a Horizons text response."""
    match = re.search(r"\$\$SOE\s*(.*?)\s*\$\$EOE", text, re.DOTALL) # This is where the state vector is kept
    if not match:
        raise ValueError(
            "Could not find $$SOE...$$EOE block in Horizons response.\n"
            "Response snippet:\n" + text[:800]
        )
    lines = [l.strip() for l in match.group(1).strip().splitlines() if l.strip()]
    parts = [p.strip() for p in lines[0].split(",")] # Pull out the state vector of the first epoch
    x,  y,  z  = float(parts[2]), float(parts[3]), float(parts[4])
    vx, vy, vz = float(parts[5]), float(parts[6]), float(parts[7])
    return {
        "jd":   float(parts[0]),
        "date": parts[1].strip(),
        "r":    np.array([x, y, z]),
        "v":    np.array([vx, vy, vz]),
    }
