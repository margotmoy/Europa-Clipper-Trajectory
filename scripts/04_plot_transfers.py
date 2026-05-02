"""
Plot Lambert transfer arcs in the ecliptic plane.

Run:
    python scripts/04_plot_transfers.py
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

AU_KM = 1.495978707e8
MU_SUN = 1.32712440018e11

BODY_COLORS = {
    "Earth":   "#4ea8de",
    "Mars":    "#e07b54",
    "Jupiter": "#c9a84c",
}

LEG_COLORS = {
    "E-M": "#4ea8de",
    "M-E": "#e07b54",
    "E-J": "#a678de",
}

ORBIT_RADII_AU = {
    "Earth":   1.0,
    "Mars":    1.524,
    "Jupiter": 5.204,
}


def lambert_arc_points(r1_vec, v1_vec, r2_vec, e, n=500):
    """Return (n, 3) positions tracing the Lambert arc from r1 to r2."""
    h_vec = np.cross(r1_vec, v1_vec)
    h_hat = h_vec / np.linalg.norm(h_vec)

    r1 = np.linalg.norm(r1_vec)
    e_vec = np.cross(v1_vec, h_vec) / MU_SUN - r1_vec / r1
    e_norm = np.linalg.norm(e_vec)
    e_hat = e_vec / e_norm if e_norm > 1e-10 else r1_vec / r1
    q_hat = np.cross(h_hat, e_hat)

    p = np.dot(h_vec, h_vec) / MU_SUN

    def true_anomaly(r_vec):
        r_hat = r_vec / np.linalg.norm(r_vec)
        return np.arctan2(np.dot(r_hat, q_hat), np.dot(r_hat, e_hat))

    nu1 = true_anomaly(r1_vec)
    nu2 = true_anomaly(r2_vec)

    if h_vec[2] >= 0:       # prograde
        if nu2 < nu1:
            nu2 += 2 * np.pi
    else:                   # retrograde
        if nu2 > nu1:
            nu2 -= 2 * np.pi

    nus = np.linspace(nu1, nu2, n)
    r_mags = p / (1 + e * np.cos(nus))
    return r_mags[:, None] * (np.cos(nus)[:, None] * e_hat + np.sin(nus)[:, None] * q_hat)


def main():
    transfers = pd.read_csv(PROJECT_ROOT / "data" / "processed_states" / "transfer_results.csv")
    ephem = pd.read_csv(PROJECT_ROOT / "data" / "processed_states" / "ephemerides.csv")

    ephem_map = {(row.body, row.date): row for _, row in ephem.iterrows()}

    fig, ax = plt.subplots(figsize=(13, 13))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Reference orbit circles
    theta = np.linspace(0, 2 * np.pi, 360)
    for body, r_au in ORBIT_RADII_AU.items():
        ax.plot(r_au * np.cos(theta), r_au * np.sin(theta),
                color=BODY_COLORS[body], lw=0.6, alpha=0.35, zorder=1)

    # Sun
    ax.plot(0, 0, "o", color="#FDB813", markersize=18, zorder=5,
            markeredgecolor="#444444", markeredgewidth=0.4)
    ax.annotate("Sun", (0, 0), xytext=(0.07, 0.07), textcoords="data",
                fontsize=9, color="#b87c00", zorder=6)

    plotted_bodies = set()

    for _, leg in transfers.iterrows():
        e1 = ephem_map[(leg["start_body"], leg["start_date"])]
        e2 = ephem_map[(leg["end_body"],   leg["end_date"])]

        r1 = np.array([e1.x_km, e1.y_km, e1.z_km])
        r2 = np.array([e2.x_km, e2.y_km, e2.z_km])
        v1 = np.array([leg["v1x_km_s"], leg["v1y_km_s"], leg["v1z_km_s"]])

        arc = lambert_arc_points(r1, v1, r2, leg["e"])
        color = LEG_COLORS.get(leg["leg"], "#ffffff")

        ax.plot(arc[:, 0] / AU_KM, arc[:, 1] / AU_KM,
                color=color, lw=1.8, label=leg["leg"], zorder=3)

        # Direction arrow at arc midpoint
        mid = len(arc) // 2
        dx = arc[mid + 5, 0] - arc[mid - 5, 0]
        dy = arc[mid + 5, 1] - arc[mid - 5, 1]
        norm = np.hypot(dx, dy)
        ax.annotate("",
                    xy=(arc[mid, 0] / AU_KM + dx / norm * 0.12,
                        arc[mid, 1] / AU_KM + dy / norm * 0.12),
                    xytext=(arc[mid, 0] / AU_KM, arc[mid, 1] / AU_KM),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4),
                    zorder=4)

        for body, pos, date in [(leg["start_body"], r1, leg["start_date"]),
                                 (leg["end_body"],   r2, leg["end_date"])]:
            if (body, date) not in plotted_bodies:
                c = BODY_COLORS.get(body, "#ffffff")
                x_au, y_au = pos[0] / AU_KM, pos[1] / AU_KM
                ax.plot(x_au, y_au, "o", color=c, markersize=9, zorder=7,
                        markeredgecolor="#333333", markeredgewidth=0.5)
                ax.annotate(f"{body}\n{date}", (x_au, y_au),
                            xytext=(9, 5), textcoords="offset points",
                            fontsize=8, color=c, zorder=8)
                plotted_bodies.add((body, date))

    ax.set_xlabel("X [AU]", color="#222222", fontsize=11)
    ax.set_ylabel("Y [AU]", color="#222222", fontsize=11)
    ax.set_title("Europa Mission — Lambert Transfer Arcs (Ecliptic Plane)",
                 color="#111111", fontsize=13, pad=14)
    ax.set_aspect("equal")
    ax.tick_params(colors="#444444")
    for spine in ax.spines.values():
        spine.set_edgecolor("#aaaaaa")
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.legend(facecolor="white", edgecolor="#aaaaaa", labelcolor="#111111",
              fontsize=10, loc="upper right")

    out_path = PROJECT_ROOT / "data" / "figures" / "lambert_transfers.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
