from typing import NamedTuple, Callable, Union, Tuple, Optional

import jax
import jax.random as random
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree
from jax.experimental.ode import odeint

from fmx.types import PyTree, PRNGKey


class FlowMatchingState(NamedTuple):
    field_parameters: PyTree
    time: float
    observation: Union[PyTree, Tuple[PyTree]]
    conditional_observation: PyTree
    loss: float


class flow_matching:
    """Conditional Flow Matching given samples from the target distribution

    Keyword arguments:
        vector_field_apply: Callable with signature (parameters, time, sample) -> (vector field)
        samples: PyTree where each leaf has leading dimension (number_of_observations, ...)
        weights: Optional array of shape (number_of_observations,)
        sigma: Standard deviation of conditional probability path
        time_gn: Callable that randomly generates times in [0, 1]
        reference_gn: Optional random generator of reference density

    Return: Conditional flow matching loss function generator
    """

    def __new__(
        cls,
        vector_field_apply: Callable,
        samples: PyTree,
        weights: Optional[jax.Array] = None,
        sigma: float = 1e-4,
        odeint: Callable = lambda func, x0: odeint(func, x0, jax.numpy.arange(0, 1, .1)),
        time_gn: Optional[Callable] = None,
        reference_gn: Optional[Callable] = None,
    ) -> Callable:
        if time_gn is None:
            time_gn = random.uniform
        # num_obs_all, _ = ravel_pytree(tree_map(lambda s: s.shape[0], samples))
        # # num_obs_all = samples.shape[0]
        # if num_obs_all[0] != num_obs_all.mean():
        #     raise ("Leafs of samples PyTree have unequal leading dimension.")
        # num_obs = num_obs_all[0]
        # num_dim = jax.numpy.sum(ravel_pytree(tree_map(lambda s: s.reshape(num_obs, -1).shape[1], samples))[0])

        _, unravel_fn = ravel_pytree(tree_map(lambda s: s[0], samples))
        samples = jax.vmap(lambda s: ravel_pytree(s)[0])(samples)
        num_obs, num_dim = samples.shape

        if reference_gn is None:
            reference_gn = lambda key: random.normal(key, (num_dim,))

            # def sample_gn(rng_key):
            #     indx = random.choice(rng_key, num_obs, p=weights)
            #     # sample = tree_map(lambda s: s[indx], samples)
            #     sample = samples[indx]
            #     return sample

            # def conditional_gn(rng_key, time, sample):
            #     epsilon = random.normal(rng_key)
            #     sd = 1.0 - (1.0 - sigma) * time
            #     cond_sample = time * sample + sd * epsilon
            #     target_vector_field = (sample - (1. - sigma) * cond_sample) / (
            #         1. - (1. - sigma) * time
            #     )
            #     return cond_sample, target_vector_field

            def conditional_gn(rng_key, time, sample):
                sample, ref_sample = sample
                sd = 1.0 - (1.0 - sigma) * time
                cond_sample = time * sample + sd * ref_sample
                target_vector_field = (sample - (1. - sigma) * cond_sample) / (
                    1. - (1. - sigma) * time
                )
                target_vector_field = sample - (1 - sigma) * ref_sample
                return cond_sample, target_vector_field

        else:

            # def sample_gn(rng_key):
            #     key_sam, key_ref = random.split(rng_key)
            #     indx = random.choice(key_sam, num_obs, p=weights)
            #     # sample = tree_map(lambda s: s[indx], samples)
            #     sample = samples[indx]
            #     ref_sample = reference_gn(key_ref)
            #     return sample, ref_sample

            def conditional_gn(rng_key, time, sample):
                sample, ref_sample = sample
                epsilon = random.normal(rng_key)
                cond_sample = time * sample + (1 - time) * ref_sample + sigma * epsilon
                target_vector_field = sample - ref_sample
                return cond_sample, target_vector_field
            
        def sample_gn(rng_key):
            key_sam, key_ref = random.split(rng_key)
            indx = random.choice(key_sam, num_obs, p=weights)
            # sample = tree_map(lambda s: s[indx], samples)
            sample = samples[indx]
            ref_sample = reference_gn(key_ref)
            return sample, ref_sample

        #TODO: Make non uniform weights work
        if weights is not None:
            if num_obs != weights.shape[0]:
                raise (
                    "Number of observations in samples PyTree and weights do not match."
                )

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

            def loss(vector_field_param: PyTree):
                approx_vector_fields = jax.vmap(
                    lambda time, sample: vector_field_apply(
                        vector_field_param, time, sample
                    )
                )(times, cond_samples)
                diffs = approx_vector_fields - target_vector_fields
                # norms = jax.vmap(lambda weight, diff: weight * jax.numpy.dot(diff, diff))(weights, diffs)
                # return jax.numpy.sum(norms)
                return jax.numpy.sum(diffs * diffs)

            return loss
        
        def samples_generator(rng_key: PRNGKey, vector_field_param: PyTree, num_samples: int) -> PyTree:
            vector_field = lambda x0, time: vector_field_apply(vector_field_param, time, x0)
            def one_sample(key: PRNGKey):
                x0 = reference_gn(key)
                flow = odeint(vector_field, x0)
                return unravel_fn(flow[-1])
            keys = random.split(rng_key, num_samples)
            return jax.vmap(one_sample)(keys)

        return loss_generator, samples_generator
