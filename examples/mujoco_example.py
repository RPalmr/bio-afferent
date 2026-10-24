"""Real MuJoCo + bioafferent integration.

Loads the bundled `mujoco/testdata/model.xml` (real XML with actuators),
steps real physics, extracts actuator mechanics (`length`, `velocity`, `force`),
and feeds them into pure-JAX spindle/GTO models.

Requires: `pip install "bioafferent[mujoco]"`
"""

import jax.numpy as jnp
import mujoco
import numpy as np

from bioafferent import Afferents, EncoderConfig
from bioafferent.wrappers.mujoco import extract_mujoco_mechanics

# ------------------------------------------------------------------
# 1. Load a real MuJoCo model (bundled with mujoco package)
# ------------------------------------------------------------------
MODEL_PATH = mujoco.__file__.replace("__init__.py", "testdata/model.xml")
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)

# ------------------------------------------------------------------
# 2. Initialize bioafferent models (one per actuator)
# ------------------------------------------------------------------
raw_alen = np.asarray(data.actuator_length)
positive_ids = np.where(raw_alen > 0)[0]
if len(positive_ids) == 0:
    positive_ids = np.arange(len(raw_alen))

n_muscles = len(positive_ids)
rest_lengths = np.abs(raw_alen[positive_ids])
rest_lengths[rest_lengths < 1e-6] = 1.0  # guard against zero rest length

# Use the facade for simplicity; replace with functional API for JIT
aff = Afferents(
    n_muscles=n_muscles,
    encoder_config=EncoderConfig(mode="poisson"),
    seed=42,
)

# Storage for production output (.npz)
ia_log = []
ii_log = []
ib_log = []
spikes_log = []
step_times = []

# ------------------------------------------------------------------
# 3. Step real physics + bioafferent (real-time loop, no host transfer)
# ------------------------------------------------------------------
print(f"Model: {MODEL_PATH}")
print(f"Actuators selected: {positive_ids}")
print(f"Muscles tracked: {n_muscles}\n")

for step_idx in range(200):
    # Real physics step
    mujoco.mj_step(model, data)

    # Extract mechanics from real MuJoCo data
    mech = extract_mujoco_mechanics(
        data,
        actuator_ids=positive_ids,
        rest_lengths=rest_lengths,
    )

    # Bioafferent advance (pure JAX, compiled in real pipeline)
    rates = aff.step(
        length=jnp.array(np.clip(np.abs(mech.length), 0.01, None), dtype=jnp.float32),
        velocity=jnp.array(mech.velocity, dtype=jnp.float32),
        force=jnp.array(mech.force, dtype=jnp.float32),
        gamma_static=30.0,
        gamma_dynamic=25.0,
        dt=model.opt.timestep,
    )

    # Log arrays for production save
    ia_log.append(np.array(rates.ia_rate).flatten())
    ii_log.append(np.array(rates.ii_rate).flatten())
    ib_log.append(np.array(rates.ib_rate).flatten())
    spikes_log.append(np.array(rates.spikes) if rates.spikes is not None else None)
    step_times.append(float(data.time))

    if step_idx == 0:
        print("Initial afferent rates (Hz):")
        print(f"  Ia: {np.array(rates.ia_rate).flatten()}")
        print(f"  II: {np.array(rates.ii_rate).flatten()}")
        print(f"  Ib: {np.array(rates.ib_rate).flatten()}")

    if step_idx == 199:
        print(f"\nAfter 200 physics steps (t={data.time:.3f}s):")
        print(f"  Ia: {np.array(rates.ia_rate).flatten()}")
        print(f"  II: {np.array(rates.ii_rate).flatten()}")
        print(f"  Ib: {np.array(rates.ib_rate).flatten()}")
        if rates.spikes is not None:
            print(f"  Spikes (bool per afferent): {np.array(rates.spikes)}")

# ------------------------------------------------------------------
# 4. Save output (production standard: stdout + .npz)
# ------------------------------------------------------------------
np.savez(
    "mujoco_example_output.npz",
    ia=np.stack(ia_log),
    ii=np.stack(ii_log),
    ib=np.stack(ib_log),
    spikes=np.array(spikes_log) if spikes_log[0] is not None else np.array([]),
    time=np.array(step_times),
    model_path=MODEL_PATH,
)
print(f"\nSaved production output: mujoco_example_output.npz (steps={len(step_times)})")
