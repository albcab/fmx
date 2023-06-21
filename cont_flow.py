import jax

import haiku as hk

init_W = hk.initializers.VarianceScaling(0.1)
init_b = hk.initializers.RandomNormal(0.01)


def mlp_generator(out_dim, hidden_dims=[64, 64], non_linearity=jax.nn.relu, norm=False):
    layers = []
    if norm:
        layers.append(hk.LayerNorm(-1, True, True))
    for out in hidden_dims:
        layers.append(hk.Linear(out, w_init=init_W, b_init=init_b))
        layers.append(non_linearity)
        if norm:
            layers.append(hk.LayerNorm(-1, True, True))
    layers.append(hk.Linear(out_dim, w_init=init_W, b_init=init_b))
    return hk.Sequential(layers)


def mlp_vector_field(time, sample):
    (out_dim,) = sample.shape
    mlp = mlp_generator(out_dim, hidden_dims=[256] * 3)
    input = jax.numpy.hstack([time, sample])
    return mlp(input)


vector_field = hk.without_apply_rng(hk.transform(mlp_vector_field))

from fmx.samples import flow_matching
import jax.random as random
from sklearn.datasets import make_moons
import optax
import matplotlib.pyplot as plt


def run(rng_key, optim, epochs, batch_size):
    samples, _ = make_moons(4096, noise=0.05)
    samples = jax.numpy.array(samples)
    fm_loss_gn, samples_gn = flow_matching(vector_field.apply, samples)#, reference_gn=lambda key: random.normal(key, (2,)))
    rng_key, key_init = random.split(rng_key)
    vector_field_params = vector_field.init(key_init, 0.0, samples[0])
    optim_state = optim.init(vector_field_params)

    def step_fn(carry, key):
        vector_field_params, optim_state = carry
        fm_loss = fm_loss_gn(key, batch_size)
        loss_value, grads = jax.value_and_grad(fm_loss)(vector_field_params)
        updates, optim_state = optim.update(grads, optim_state, vector_field_params)
        vector_field_params = optax.apply_updates(vector_field_params, updates)
        return (vector_field_params, optim_state), loss_value

    keys = random.split(rng_key, epochs)
    (vector_field_params, optim_state), loss_values = jax.lax.scan(
        step_fn, (vector_field_params, optim_state), keys
    )
    key_samples, = random.split(keys[0], 1)
    x = samples_gn(key_samples, vector_field_params, 4096)
    plt.hist2d(x[:, 0], x[:, 1], bins=64)
    plt.show()
    return loss_values
