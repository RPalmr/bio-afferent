"""Interactive MuJoCo visualization of the muscle model with bioafferent readouts."""

import time

import mujoco
import mujoco.viewer
import numpy as np

from bioafferent.wrappers.mujoco import extract_mujoco_mechanics
from bioafferent.wrappers.mjx import default_mjx_models, mjx_afferent_step

XML = r"""
<mujoco>
  <option timestep="0.001"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
  </visual>
  <worldbody>
    <body name="muscle" pos="0 0 0">
      <joint name="stretch" type="slide" axis="1 0 0" range="0 0.4"
             stiffness="1000" springref="0.2" damping="5"/>
      <geom name="muscle_vis" type="capsule" size="0.02 0.05" rgba="0.8 0.2 0.2 1"
            fromto="-0.05 0 0 0.05 0 0"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="muscle" joint="stretch" ctrlrange="-100 100"/>
  </actuator>
</mujoco>
"""


def main(duration_sec=5.0):
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    data.qpos[0] = 0.2  # slide joint initial rest length

if __name__ == "__main__":
    # Simpler direct interactive demo using the working model
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    data.qpos[0] = 0.2
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        spindle, gto = default_mjx_models()
        ss, gs = spindle.init_state((1,)), gto.init_state((1,))
        start = time.time()
        while viewer.is_running() and (time.time() - start) < 5.0:
            t = (time.time() - start) % 2.0  # 1 Hz sinusoid cycle
            data.ctrl[0] = 20.0 + 20.0 * np.sin(2.0 * np.pi * t)
            mujoco.mj_step(model, data)
            viewer.sync()

            mech = extract_mujoco_mechanics(data, rest_lengths=[0.2])
            ss, gs, aff = mjx_afferent_step(ss, gs, mech, spindle, gto, 40.0, 30.0)
            if int((time.time() - start) * 1000) % 500 == 0:
                print(f"t={(time.time()-start):.2f}s len={float(mech.length[0]):.3f} vel={float(mech.velocity[0]):.3f} force={float(mech.force[0]):.1f} | Ia={float(aff.ia_rate[0]):.1f} Hz II={float(aff.ii_rate[0]):.1f} Hz Ib={float(aff.ib_rate[0]):.1f} Hz")

            # Slow real-time rate
            time.sleep(0.001)

        viewer.close()
