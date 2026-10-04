"""Unit tests: configuration validation."""

import pytest

from bioafferent import (
    ConfigurationError,
    EncoderConfig,
    GTOConfig,
    IntegrationConfig,
    ParameterError,
    SpindleConfig,
)


def test_spindle_defaults_valid():
    c = SpindleConfig()
    assert c.dtype == "float32"
    assert c.max_rate == 300.0


def test_spindle_rejects_bad_dtype():
    with pytest.raises(ConfigurationError):
        SpindleConfig(dtype="float16")


def test_spindle_rejects_bad_integrator():
    with pytest.raises(ConfigurationError):
        SpindleConfig(integrator="rk45")


def test_spindle_rejects_nonpositive_tau():
    with pytest.raises(ConfigurationError):
        SpindleConfig(tau_activation_dynamic=-0.1)


def test_spindle_rejects_bad_triplet():
    with pytest.raises((ParameterError, ConfigurationError)):
        SpindleConfig(stiffness_rest=(1.0, 2.0))  # type: ignore[arg-type]
    with pytest.raises((ParameterError, ConfigurationError)):
        SpindleConfig(damping_rest=(1.0, -2.0, 3.0))


def test_spindle_rejects_bad_gamma_scale():
    with pytest.raises(ConfigurationError):
        SpindleConfig(chain_static_scale=1.5)


def test_gto_defaults_valid():
    assert GTOConfig().baseline == 2.0


def test_gto_rejects_negative_threshold():
    with pytest.raises(ConfigurationError):
        GTOConfig(threshold=-1.0)


def test_gto_rejects_nonpositive_tau():
    with pytest.raises(ConfigurationError):
        GTOConfig(tau_force=0.0)


def test_encoder_rejects_bad_mode():
    with pytest.raises(ConfigurationError):
        EncoderConfig(mode="hawkes")


def test_encoder_rejects_bad_order():
    with pytest.raises(ConfigurationError):
        EncoderConfig(gamma_order=0)


def test_float64_requires_x64_enabled():
    import jax

    from bioafferent import GolgiTendonOrgan, MuscleSpindle

    if bool(jax.config.read("jax_enable_x64")):
        return  # guard only applies when x64 is disabled
    with pytest.raises(ConfigurationError, match="64-bit"):
        MuscleSpindle(SpindleConfig(dtype="float64"))
    with pytest.raises(ConfigurationError, match="64-bit"):
        GolgiTendonOrgan(GTOConfig(dtype="float64"))


def test_integration_config_rejects_bad_dt():
    with pytest.raises(ConfigurationError):
        IntegrationConfig(dt=0.0)
    with pytest.raises(ConfigurationError):
        IntegrationConfig(dt=-0.001)


def test_integration_config_rejects_bad_method():
    with pytest.raises(ConfigurationError):
        IntegrationConfig(method="rk45")
