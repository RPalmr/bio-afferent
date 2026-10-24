Bioafferent
===========

What it does
------------

Bioafferent converts mechanical muscle state — length, velocity, force — into
the sensory signals a nervous system would actually receive: muscle spindle
primary (Ia) and secondary (II) firing rates, Golgi tendon organ (Ib) firing
rates, and discrete spike trains encoded from those rates. It is a JAX-native
Python library built for physics simulators (MuJoCo/MJX) and
reinforcement-learning environments (Gymnasium), with a clean CPU-vectorized
core that runs independently of any simulator.

.. code-block:: python

   import jax.numpy as jnp
   from bioafferent import MuscleSpindle, SpindleConfig

   spindle = MuscleSpindle(SpindleConfig())
   state = spindle.init_state(shape=())
   for _ in range(1000):
       state, out = spindle.step(
           state, length=1.02, velocity=0.0,
           gamma_dynamic=0.3, gamma_static=0.2, dt=0.001,
       )
   print(float(out.ia_rate), float(out.ii_rate))

.. figure:: _static/spindle_ramphold.png
   :alt: Ia and II firing rates during a ramp-and-hold stretch
   :width: 100%

   Primary (Ia) afferents burst at stretch onset and adapt; secondary (II)
   afferents follow length. Dynamic gamma sharpens Ia velocity sensitivity;
   static gamma raises both channels.

Why it is needed
----------------

Controllers trained in simulation usually observe raw mechanics — joint
angles, velocities, torques — while animals observe the world through
proprioceptive afferents with their own dynamics: velocity-sensitive bursts
that adapt, length-following tonic discharge, fusimotor gain control, force
saturation, spiking noise. Policies, neuromechanical models, and spiking
controllers built on raw state therefore miss a layer of biological
computation that shapes real sensorimotor behavior: adaptation, gamma-driven
sensitivity shifts, and the distinct Ia/II/Ib channels the spinal cord
actually receives.

Bioafferent closes that gap with documented, literature-structured models
(Mileusnic et al. 2006 spindle architecture; Houk & Simon / Lin & Crago GTO
properties; Poisson and gamma-renewal spike encoding) engineered for
vectorized JAX simulation — so the same feedback dynamics run for one muscle
or ten thousand, inside `jit`, `vmap`, and `scan`, at a measured ~0.3 ms per
step for 1000 environments x 20 muscles on CPU.

.. toctree::
   :maxdepth: 2
   :caption: Guide

   usage
   architecture
   physiology
   performance
   api
   changelog

.. toctree::
   :maxdepth: 2
   :caption: Development

   todo

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
