"""Basic spindle example: ramp-and-hold stretch with gamma drives."""

from bioafferent import MuscleSpindle, SpindleConfig

spindle = MuscleSpindle(SpindleConfig())
state = spindle.init_state(())

dt, t = 0.001, 0.0
print(f"{'t(s)':>6} {'L':>6} {'Ia(Hz)':>8} {'II(Hz)':>8}")
for i in range(2700):
    if t < 0.5:
        L, V = 0.95, 0.0
    elif t < 1.6:
        L, V = 0.95 + 0.11 * (t - 0.5), 0.11
    else:
        L, V = 1.071, 0.0
    state, out = spindle.step(state, L, V, gamma_dynamic=40.0, gamma_static=30.0, dt=dt)
    t += dt
    if i % 450 == 0:
        print(f"{t:6.2f} {L:6.3f} {float(out.ia_rate):8.1f} {float(out.ii_rate):8.1f}")
print("final:", float(out.ia_rate), float(out.ii_rate))
