API reference
=============

Facade
------

.. autoclass:: bioafferent.Afferents
   :members:

.. autoclass:: bioafferent.AfferentRates
   :members:

Models
------

.. autoclass:: bioafferent.MuscleSpindle
   :members: step, rollout, init_state

.. autoclass:: bioafferent.GolgiTendonOrgan
   :members: step, rollout, init_state

.. autoclass:: bioafferent.AfferentEncoder
   :members: step, init_state

Encoding and decoding
---------------------

.. autofunction:: bioafferent.population_encode

.. autofunction:: bioafferent.decode_rates

Configurations
--------------

.. autoclass:: bioafferent.SpindleConfig
   :members:

.. autoclass:: bioafferent.GTOConfig
   :members:

.. autoclass:: bioafferent.EncoderConfig
   :members:

.. autoclass:: bioafferent.IntegrationConfig
   :members:

States
------

.. autoclass:: bioafferent.SpindleState
   :members:

.. autoclass:: bioafferent.GTOState
   :members:

.. autoclass:: bioafferent.EncoderState
   :members:

Wrappers
--------

.. autofunction:: bioafferent.wrap

.. autoclass:: bioafferent.AfferentWrapper
   :members: reset, step

.. automodule:: bioafferent.wrappers.base
   :members: MechanicalState, MechanicalAdapter

.. automodule:: bioafferent.wrappers.mujoco
   :members: extract_mujoco_mechanics

.. automodule:: bioafferent.wrappers.mjx
   :members: extract_mjx_mechanics, mjx_afferent_step, default_mjx_models, MJXAfferents, mjx_rest_lengths

Exceptions
----------

.. autoexception:: bioafferent.BioafferentError
.. autoexception:: bioafferent.ConfigurationError
.. autoexception:: bioafferent.ParameterError
.. autoexception:: bioafferent.NumericalStabilityError
.. autoexception:: bioafferent.InvalidStateError
.. autoexception:: bioafferent.ShapeError
.. autoexception:: bioafferent.UnsupportedBackendError
