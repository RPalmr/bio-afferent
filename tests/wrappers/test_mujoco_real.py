"""Real MuJoCo adapter integration (requires ``mujoco`` package installed)."""

import numpy as np
import pytest

from bioafferent.wrappers.mujoco import extract_mujoco_mechanics
from bioafferent.wrappers.mjx import default_mjx_models, mjx_afferent_step

mujoco = pytest.importorskip("mujoco", reason="mujo missing")

XML = """\
<mujoco>
  <option timestep="0.001"/>
  <worldbody>
    <body name="fascicle" pos="0 0 0">
      <joint name="stretch" type="slide" axis="1 0 0" range="0 0.4"
             stiffness="1000" springref="0.2" damping="5"/>
      <geom type="sphere" size="0.01" mass="0.1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="muscle" joint="stretch" ctrlrange="-100 100"/>
  </actuator>
</mujoco>
"""


def _run_sinusoid():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    data.qpos[0] = 0.2
    mujoco.mj_forward(model, data)
    spindle, gto = default_mjx_models()
    ss, gs = spindle.init_state((1,)), gto.init_state((1,))
    ia, ii, ib, vel, frc = [], [], [], [], []
    for i in range(4000):
        t = i * model.opt.timestep
        data.ctrl[0] = 20.0 + 20.0 * np.sin(2.0 * np.pi * t)
        mujoco.mj_step(model, data)
        mech = extract_mujoco_mechanics(data, rest_lengths=[0.2])
        ss, gs, aff = mjx_afferent_step(ss, gs, mech, spindle, gto, 40.0, 30.0)
        ia.append(float(aff.ia_rate[0]))
        ii.append(float(aff.ii_rate[0]))
        ib.append(float(aff.ib_rate[0]))
        vel.append(float(mech.velocity[0]))
        frc.append(float(mech.force[0]))
    return np.array(ia), np.array(ii), np.array(ib), np.array(vel), np.array(frc)


def test_real_mujoco_adaptive_responses():
    ia, ii, ib, vel, frc = _run_sinusoid()
    # Warm-up (first 1 s) excluded
    w = slice(1000, None)
    assert np.isfinite(ia[w]).all()
    assert np.isfinite(ib[w]).all()
    # Ia should be higher during stretch (positive velocity)
    assert ia[w][vel[w] > 0].mean() > ia[w][vel[w] < 0].mean()
    # Ib should correlate positively with force
    assert np.corrcoef(ib[w], frc[w])[0, 1] > 0.5
