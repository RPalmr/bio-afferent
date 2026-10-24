"""Quickstart: bioafferent in under 25 lines.

1. Bare muscles with the `Afferents` facade (no states, keys, or configs).
2. Spike trains by passing an `EncoderConfig`.
3. Fusimotor control: static gamma drive raises resting discharge.
"""

from bioafferent import Afferents, EncoderConfig

# --- 1. two muscles, continuous firing rates (only mechanics needed) ---
aff = Afferents(n_muscles=2)
for _ in range(1000):
    out = aff.step(length=1.04, velocity=0.03, force=20.0)
print(f"Ia={out.ia_rate} II={out.ii_rate} Ib={out.ib_rate}")

# --- 2. same, plus Poisson spike trains ---
spiking = Afferents(n_muscles=2, encoder_config=EncoderConfig(mode="poisson"))
n_spikes = 0
for _ in range(1000):
    out = spiking.step(length=1.04, velocity=0.03, force=20.0, dt=0.001)
    n_spikes += int(out.spikes.sum())
print(f"spikes in 1 s: {n_spikes}")

# --- 3. static gamma drives resting discharge at short length ---
quiet_aff, driven_aff = Afferents(n_muscles=1), Afferents(n_muscles=1)
for _ in range(2000):
    quiet = quiet_aff.step(length=0.95, velocity=0.0, force=2.0, dt=0.001)
    driven = driven_aff.step(length=0.95, velocity=0.0, force=2.0, gamma_static=70.0, dt=0.001)
print(f"rest II={float(quiet.ii_rate[0]):.1f} Hz, with static gamma={float(driven.ii_rate[0]):.1f} Hz")
