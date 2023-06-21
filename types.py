from typing import Any, Iterable, Mapping, Union

import jax

#: JAX PyTrees
PyTree = Union[jax.Array, Iterable["PyTree"], Mapping[Any, "PyTree"]]

#: JAX PRNGKey
PRNGKey = jax.random.PRNGKeyArray
