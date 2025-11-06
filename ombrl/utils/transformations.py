import jax.numpy as jnp
import chex


def symlog(x: chex.Array):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x: chex.Array):
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))
