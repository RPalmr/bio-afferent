"""GTO example: force steps and ramp."""

from bioafferent import GolgiTendonOrgan, GTOConfig

gto = GolgiTendonOrgan(GTOConfig())
state = gto.init_state(())

for i in range(2000):
    force = 0.0 if i < 500 else (50.0 if i < 1000 else 150.0)
    state, out = gto.step(state, force, dt=0.001)
    if i % 500 == 0:
        print(f"step {i}: force={force:6.1f} N -> Ib={float(out.ib_rate):6.1f} Hz")
print("final Ib:", float(out.ib_rate))
