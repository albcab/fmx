from typing import Any, Iterable, Mapping, Union, NamedTuple, Callable

import jax

#: JAX PyTrees
PyTree = Union[jax.Array, Iterable["PyTree"], Mapping[Any, "PyTree"]]

#: JAX PRNGKey
PRNGKey = jax.random.PRNGKeyArray


class FlowMatchingMethod(NamedTuple):
    """A trio of functions for learning a vector field using flow matching

    get_loss: Generates a loss function given a batch size
    sample: Generates samples given learned parameters
    get_logporb: Generates a log density function given learned parameters
    """

    get_loss: Callable
    sample: Callable
    get_logprob: Callable
