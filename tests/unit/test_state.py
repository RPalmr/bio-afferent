"""Unit tests: state construction, shapes, pytree behavior."""

import jax
import jax.numpy as jnp

from bioafferent import (
    AfferentEncoder,
    EncoderState,
    GTOState,
    MuscleSpindle,
    SpindleState,
)


def test_spindle_state_shapes(spindle):
    s = spindle.init_state((2, 3))
    assert s.tension.shape == (2, 3, 3)
    assert s.activation_dynamic.shape == (2, 3)
    s0 = spindle.init_state(())
    assert s0.tension.shape == (3,)


def test_gto_state_shapes(gto):
    assert gto.init_state((5,)).force_filtered.shape == (5,)
    assert gto.init_state(()).force_filtered.shape == ()


def test_state_replace():
    s = MuscleSpindle().init_state((2,))
    new_t = jnp.ones((2, 3))
    s2 = s.replace(tension=new_t)
    assert bool(jnp.all(s2.tension == 1.0))
    assert bool(jnp.all(s.tension == 0.0))  # original untouched


def _roundtrip(state):
    leaves, treedef = jax.tree_util.tree_flatten(state)
    return jax.tree_util.tree_unflatten(treedef, leaves)


def test_spindle_pytree_roundtrip(spindle):
    s = spindle.init_state((2,))
    s2 = _roundtrip(s)
    assert isinstance(s2, SpindleState)
    assert s2.tension.shape == (2, 3)


def test_gto_pytree_roundtrip(gto):
    s2 = _roundtrip(gto.init_state((4,)))
    assert isinstance(s2, GTOState)


def test_encoder_pytree_roundtrip():
    s2 = _roundtrip(AfferentEncoder().init_state((4,)))
    assert isinstance(s2, EncoderState)


def test_state_passthrough_jit(spindle):
    s = spindle.init_state((2,))

    @jax.jit
    def f(st):
        return st.replace(tension=st.tension + 1.0)

    out = f(s)
    assert out.tension.shape == (2, 3)
    assert bool(jnp.all(out.tension == 1.0))
