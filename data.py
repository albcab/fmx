from typing import Callable, Optional

import jax
import jax.numpy as jnp
import jax.random as random
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree
from jax.experimental.ode import odeint
from jax.scipy.stats.norm import logpdf as norm_logpdf

from fmx.types import PyTree, PRNGKey, FlowMatchingMethod


class flow_matching:
    """Conditional Flow Matching given target data

    Keyword arguments:
        vector_field_apply: Callable that computes the vector field, with signature (parameters, time, sample) -> (vector field)
        data: PyTree where each leaf has leading dimension (number_of_observations, ...)
        weights: Optional array of shape (number_of_observations,)
        sigma: Standard deviation of conditional probability path
        odeint: Callable that computes the numerical approximation of probability path, with signature (vector field, initial condition) -> (flow)
        time_gn: Callable that randomly generates times in [0, 1]
        reference_gn: Optional random generator of reference density

    Return: Conditional flow matching loss function generator
    """

    def __new__(
        cls,
        vector_field_apply: Callable,
        data: PyTree,
        weights: Optional[jax.Array] = None,
        sigma: float = 1e-4,
        odeint: Callable = lambda func, x0: odeint(
            func, x0, jnp.linspace(0.0, 1.0, 11)
        ),
        time_gn: Optional[Callable] = None,
        reference_gn: Optional[Callable] = None,
    ) -> FlowMatchingMethod:
        if time_gn is None:
            time_gn = random.uniform

        _, unravel_fn = ravel_pytree(tree_map(lambda s: s[0], data))
        data = jax.vmap(lambda s: ravel_pytree(s)[0])(data)
        num_obs, num_dim = data.shape

        if reference_gn is None:
            reference_gn = lambda key: random.normal(key, (num_dim,))

            def conditional_gn(rng_key, time, sample):
                sample, ref_sample = sample
                sd = 1.0 - (1.0 - sigma) * time
                cond_sample = time * sample + sd * ref_sample
                target_vector_field = sample - (1 - sigma) * ref_sample
                return cond_sample, target_vector_field

        else:

            def conditional_gn(rng_key, time, sample):
                sample, ref_sample = sample
                epsilon = random.normal(rng_key)
                cond_sample = time * sample + (1 - time) * ref_sample + sigma * epsilon
                target_vector_field = sample - ref_sample
                return cond_sample, target_vector_field

        def sample_gn(rng_key):
            key_sam, key_ref = random.split(rng_key)
            indx = random.choice(key_sam, num_obs, p=weights)
            sample = data[indx]
            ref_sample = reference_gn(key_ref)
            return sample, ref_sample

        def loss_generator(rng_key: PRNGKey, batch_size: int) -> Callable:
            def get_sample(rng_key):
                key_time, key_sample, key_cond = random.split(rng_key, 3)
                time = time_gn(key_time)
                sample = sample_gn(key_sample)
                cond_sample, target_vector_field = conditional_gn(
                    key_cond, time, sample
                )
                return time, cond_sample, target_vector_field

            keys = random.split(rng_key, batch_size)
            times, cond_samples, target_vector_fields = jax.vmap(get_sample)(keys)

            def loss_fn(vector_field_param: PyTree):
                approx_vector_fields = jax.vmap(
                    lambda time, sample: vector_field_apply(
                        vector_field_param, time, sample
                    )
                )(times, cond_samples)
                diffs = approx_vector_fields - target_vector_fields
                if weights is None:
                    # return jax.vmap(lambda diff: jnp.dot(diff, diff))(diffs).sum()
                    return jnp.square(diffs).sum()
                else:
                    # return jax.vmap(lambda weight, diff: weight * jnp.dot(diff, diff))(weights, diffs).sum()
                    return (weights * jnp.square(diffs).sum(axis=1)).sum()

            return loss_fn

        def sample_generator(
            rng_key: PRNGKey, vector_field_param: PyTree, num_samples: int
        ) -> PyTree:
            vector_field = lambda x, time: vector_field_apply(
                vector_field_param, time, x
            )

            def one_sample(key: PRNGKey):
                x0 = reference_gn(key)
                flow = odeint(vector_field, x0)
                return unravel_fn(flow[-1])

            keys = random.split(rng_key, num_samples)
            return jax.vmap(one_sample)(keys)

        def logprob_generator(vector_field_param: PyTree):
            def augmented_vector_field(x_ldj, time):
                x, _ = x_ldj
                time = 1.0 - time
                dx = vector_field_apply(vector_field_param, time, x)
                jacobian = jax.jacfwd(
                    lambda x: vector_field_apply(vector_field_param, time, x)
                )(x)
                dldj = jnp.einsum("ii", jacobian)
                return dx, dldj

            def logprob_fn(target_sample: PyTree):
                sample = ravel_pytree(target_sample)[0]
                inv_flow, ldj_flow = odeint(augmented_vector_field, (sample, 0.0))
                return norm_logpdf(inv_flow[-1]).sum() - ldj_flow[-1]

            return logprob_fn

        return FlowMatchingMethod(loss_generator, sample_generator, logprob_generator)
