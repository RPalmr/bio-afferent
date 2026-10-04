"""Generate documentation figures (deterministic; Agg backend).

Run:  PYTHONPATH=. python3 docs/generate_figures.py
Outputs PNGs into docs/_static/.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bioafferent import AfferentEncoder, EncoderConfig, GolgiTendonOrgan, MuscleSpindle

DT = 0.001
OUT = "docs/_static"


def ramp_and_hold():
    n_base, n_ramp, n_hold = 500, 1100, 1100
    base = np.full(n_base, 0.95)
    ramp = 0.95 + 0.11 * DT * np.arange(1, n_ramp + 1)
    hold = np.full(n_hold, ramp[-1])
    length = np.concatenate([base, ramp, hold])
    velocity = np.concatenate([[0.0], np.diff(length) / DT])
    t = np.arange(len(length)) * DT
    return t, length, velocity


def fig_spindle():
    t, L, V = ramp_and_hold()
    sp = MuscleSpindle()
    series = {}
    for gd, gs, tag in (
        (0, 0, "no gamma"),
        (70, 0, r"dyn $\gamma$=70"),
        (0, 70, r"static $\gamma$=70"),
    ):
        st = sp.init_state(())
        ia, ii = [], []
        for length, vel in zip(L, V, strict=True):
            st, out = sp.step(st, float(length), float(vel), gd, gs, DT)
            ia.append(float(out.ia_rate))
            ii.append(float(out.ii_rate))
        series[tag] = (np.array(ia), np.array(ii))
    fig, ax = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
    ax[0].plot(t, L, color="black")
    ax[0].set_ylabel("length (L/L0)")
    for tag, (ia, ii) in series.items():
        ax[1].plot(t, ia, label=f"Ia {tag}")
        ax[2].plot(t, ii, label=f"II {tag}")
    ax[1].set_ylabel("Ia (Hz)")
    ax[2].set_ylabel("II (Hz)")
    ax[2].set_xlabel("time (s)")
    for a in ax[1:]:
        a.legend(fontsize=8, loc="upper left")
    fig.suptitle("Ramp-and-hold stretch under fusimotor drive")
    fig.tight_layout()
    fig.savefig(f"{OUT}/spindle_ramphold.png", dpi=110)
    plt.close(fig)


def fig_gto():
    gto = GolgiTendonOrgan()
    forces = np.linspace(0, 400, 81)
    rates = []
    for f in forces:
        st = gto.init_state(())
        for _ in range(3000):
            st, out = gto.step(st, float(f), DT)
        rates.append(float(out.ib_rate))
    fig, a = plt.subplots(figsize=(5.5, 3.8))
    a.plot(forces, rates, color="black")
    a.set_xlabel("tendon force (N)")
    a.set_ylabel("Ib rate (Hz)")
    a.set_title("GTO force-to-rate transduction (threshold + saturation)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/gto_curve.png", dpi=110)
    plt.close(fig)


def fig_spikes():
    import jax

    key = jax.random.PRNGKey(0)
    rate = np.concatenate([np.full(500, 15.0), np.full(500, 70.0)])
    fig, ax = plt.subplots(3, 1, figsize=(7, 4.5), sharex=True)
    for i, mode in enumerate(("poisson", "inhomogeneous", "gamma")):
        enc = AfferentEncoder(EncoderConfig(mode=mode))
        st = enc.init_state((12,))
        mat = []
        for r in rate:
            key, st, out = enc.step(key, st, np.full((12,), r, dtype=np.float32), DT)
            mat.append(np.asarray(out.spikes))
        mat = np.array(mat)
        for u in range(12):
            times = np.where(mat[:, u])[0] * DT
            ax[i].scatter(times, np.full_like(times, u), s=4, color="black")
        ax[i].set_ylabel(f"{mode}\nunit")
    ax[-1].set_xlabel("time (s)")
    fig.suptitle("Spike trains under a 15->70 Hz rate step (12 units)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/spikes_raster.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    fig_spindle()
    fig_gto()
    fig_spikes()
    print("wrote figures to", OUT)
