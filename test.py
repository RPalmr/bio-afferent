import jax.numpy as jnp
from bioafferent import MuscleSpindle, SpindleConfig

spindle = MuscleSpindle(SpindleConfig())
state = spindle.init_state(shape=())
for _ in range(1000):
    state, out = spindle.step(
        state, length=1.02, velocity=0.0, gamma_dynamic=0.3, gamma_static=0.2, dt=0.001
    )
print(float(out.ia_rate), float(out.ii_rate))