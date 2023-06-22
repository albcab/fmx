# Lightweight Conditional Flow Matching using JAX

Use the library by cloning the repository to your working folder:
```bash
git clone https://github.com/albcab/fmx.git
```

## Supported algorithms

Learning the two moons 2D density as a running example, using [haiku](https://github.com/deepmind/dm-haiku) to build the neural network and [optax](https://github.com/deepmind/optax) to optimize:

```python
import jax

from sklearn.datasets import make_moons
import haiku as hk
import optax

init_W = hk.initializers.VarianceScaling(0.1)
init_b = hk.initializers.RandomNormal(0.01)

def mlp_generator(out_dim, hidden_dims, non_linearity=jax.nn.relu):
    layers = []
    for out in hidden_dims:
        layers.append(hk.Linear(out, w_init=init_W, b_init=init_b))
        layers.append(non_linearity)
    layers.append(hk.Linear(out_dim, w_init=init_W, b_init=init_b))
    return hk.Sequential(layers)

def mlp_vector_field(time, sample):
    (out_dim,) = sample.shape
    mlp = mlp_generator(out_dim, hidden_dims=[256] * 3)
    input = jax.numpy.hstack([time, sample])
    return mlp(input)

vector_field = hk.without_apply_rng(hk.transform(mlp_vector_field))

data, _ = make_moons(4096, noise=0.05)
data = jax.numpy.array(data)

rng_key = jax.random.PRNGKey(0)
rng_key, key_init = jax.random.split(rng_key)
vector_field_params = vector_field.init(key_init, 0.0, data[0])

optim = optax.adamw(1e-3)
optim_state = optim.init(vector_field_params)

epochs, batch_size = 4096, 256 
```

### Lipman et al. [Flow matching for generative modeling](https://arxiv.org/abs/2210.02747). 2022.

Load and initialize the generating functions of the method:
```python
from fmx.data import flow_matching

fmx = flow_matching(vector_field.apply, data)
"""Keyword arguments include:
    vector_field_apply: Callable that computes the vector field, with signature (parameters, time, sample) -> (vector field)
    samples: PyTree where each leaf has leading dimension (number_of_observations, ...)
    sigma: Standard deviation of conditional probability path
    odeint: Callable that computes the numerical approximation of probability path, with signature (vector field, initial condition) -> (flow)
"""
```

Learn the parameters of the vector field:
```python
def step_fn(carry, key):
    vector_field_params, optim_state = carry
    fm_loss = fmx.get_loss(key, batch_size)
    loss_value, grads = jax.value_and_grad(fm_loss)(vector_field_params)
    updates, optim_state = optim.update(grads, optim_state, vector_field_params)
    vector_field_params = optax.apply_updates(vector_field_params, updates)
    return (vector_field_params, optim_state), loss_value

keys = jax.random.split(rng_key, epochs)
(vector_field_params, optim_state), loss_values = jax.lax.scan(
    step_fn, (vector_field_params, optim_state), keys
)
```

Generate new samples given the learned parameters:
```python
(key_samples,) = jax.random.split(keys[0], 1)
num_samples = 4096
samples = fmx.sample(key_samples, vector_field_params, num_samples)
```

Evaluate the learned log density of the data:
```python
logprob_fn = fmx.get_logprob(vector_field_params)
logprob_data = jax.vmap(logprob_fn)(data)
```

### Tong et al. [Conditional flow matching: Simulation-free dynamic optimal transport](https://arxiv.org/abs/2302.00482). 2023.

***Only supports simplified conditional flow matching***

Same as above but adding the argument `reference_gn: Optional random generator of reference density`:
```python
from fmx.data import flow_matching

reference_gn = lambda key: jax.random.normal(key, (2,))
fmx = flow_matching(vector_field.apply, data, reference_gn=reference_gn)
```