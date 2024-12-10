import math
from functools import reduce

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from numpyro.infer import MCMC, NUTS
from opacus.accountants.utils import get_noise_multiplier
from scipy import stats
from tqdm import tqdm

from model import per_example_kl_divergence, get_positive_definite_matrix, parameter_based_inference_model, \
    gradient_based_inference_model, \
    log_likelihood, get_cholesky_lower, per_example_kl_divergence_with_exact_entropy, \
    gradient_based_inference_model_posterior_laplace_approximation

EXPERIMENT_NAMES = ['logistic_regression', 'gamma_exponential', 'beta_bernoulli', 'beta_binomial',
                    'dirichlet_categorical', 'linear_regression10d', 'linear_regression']

DEBUG_MODE_ON = False


def train_variational_inference_model_dpsgd(xs,
                                            ys,
                                            theta_dimension,
                                            per_example_loss_function,
                                            training_iterations,
                                            target_epsilon,
                                            target_delta,
                                            dpsgd_initial_mu_value,
                                            dpsgd_initial_pre_transformed_covariance_value,
                                            mu_learning_rate,
                                            pre_transformed_covariance_learning_rate,
                                            mu_learning_rate_v,
                                            pre_cov_learning_rate_v,
                                            clipping_threshold,
                                            mu_gradient_scale,
                                            pre_transformed_cov_gradient_scale,
                                            sampling_rate,
                                            is_covariance_diagonal,
                                            mc_integration_samples_count,
                                            prng_key,
                                            noise_scale=None,
                                            show_progress_bar=False):
    examples_count = xs.shape[0]
    if is_covariance_diagonal:
        pre_transformed_covariance_components = theta_dimension
    else:
        pre_transformed_covariance_components = theta_dimension * (theta_dimension + 1) // 2

    phi = {
        'mu': jnp.ones((theta_dimension,), jnp.float32) * dpsgd_initial_mu_value,
        'pre_transformed_covariance': jnp.ones((pre_transformed_covariance_components,),
                                               jnp.float32) * dpsgd_initial_pre_transformed_covariance_value
    }

    mu_trace = jnp.zeros((training_iterations, theta_dimension), dtype=jnp.float32)
    pre_transformed_covariance_trace = jnp.zeros((training_iterations,
                                                  pre_transformed_covariance_components),
                                                 dtype=jnp.float32)
    mu_perturbed_gradient_trace = jnp.zeros((training_iterations, theta_dimension), dtype=jnp.float32)
    mu_max_grad_norm_trace = jnp.zeros((training_iterations, 1), dtype=jnp.float32)
    mu_mean_grad_norm_trace = jnp.zeros((training_iterations, 1), dtype=jnp.float32)
    pre_cov_max_grad_norm_trace = jnp.zeros((training_iterations, 1), dtype=jnp.float32)
    pre_cov_mean_grad_norm_trace = jnp.zeros((training_iterations, 1), dtype=jnp.float32)

    mu_max_abs_grad_trace = jnp.zeros((training_iterations, theta_dimension), dtype=jnp.float32)
    mu_mean_abs_grad_trace = jnp.zeros((training_iterations, theta_dimension), dtype=jnp.float32)
    pre_cov_max_abs_grad_norm_trace = jnp.zeros((training_iterations, pre_transformed_covariance_components),
                                                   dtype=jnp.float32)
    pre_cov_mean_abs_grad_norm_trace = jnp.zeros((training_iterations, pre_transformed_covariance_components),
                                                 dtype=jnp.float32)

    pre_transformed_covariance_perturbed_gradient_trace = jnp.zeros((training_iterations,
                                                                     pre_transformed_covariance_components),
                                                                    dtype=jnp.float32)

    if noise_scale is None:
        dpsgd_noise_scale = get_noise_multiplier(target_epsilon=target_epsilon,
                                                 target_delta=target_delta,
                                                 sample_rate=sampling_rate,
                                                 accountant='prv',
                                                 steps=training_iterations)
    else:
        dpsgd_noise_scale = noise_scale
    print(dpsgd_noise_scale)
    # dpsgd_noise_scale = 0.0
    phi_dimension = theta_dimension + pre_transformed_covariance_components

    if mu_learning_rate is None or pre_transformed_covariance_learning_rate is None:
        learning_rates_common_factor = 1 / (
                math.sqrt(training_iterations * phi_dimension) * max(dpsgd_noise_scale, 1e-9) * clipping_threshold)
    else:
        learning_rates_common_factor = None
    if mu_learning_rate is None:
        mu_learning_rate = mu_learning_rate_v * mu_gradient_scale * learning_rates_common_factor
    if pre_transformed_covariance_learning_rate is None:
        pre_transformed_covariance_learning_rate = pre_cov_learning_rate_v * pre_transformed_cov_gradient_scale * learning_rates_common_factor
    # dpsgd_noise_scale = 0.0
    # print(mu_learning_rate)
    # print(pre_transformed_covariance_learning_rate)
    # optimizer1 = optax.adam(mu_learning_rate)
    # optimizer2 = optax.adam(pre_transformed_covariance_learning_rate)
    # optimizer_state1 = optimizer1.init(phi['mu'])
    # optimizer_state2 = optimizer2.init(phi['pre_transformed_covariance'])

    # failure_probability = 1e-9
    # logical_batch_size = 1
    # batch_size_distribution = binom(examples_count, sampling_rate)
    # while logical_batch_size < examples_count:
    #     current_probability = batch_size_distribution.sf(logical_batch_size)
    #     if current_probability < failure_probability:
    #         break
    #     logical_batch_size += 1

    optimizer_state1 = None
    optimizer_state2 = None

    def update_params(t, args):
        (phi_,
         mu_trace_,
         pre_transformed_covariance_trace_,
         mu_perturbed_gradient_trace_,
         pre_transformed_covariance_perturbed_gradient_trace_,
         mu_max_grad_norm_trace_,
         mu_mean_grad_norm_trace_,
         pre_cov_max_grad_norm_trace_,
         pre_cov_mean_grad_norm_trace_,
         mu_max_abs_grad_trace_,
         mu_mean_abs_grad_trace_,
         pre_cov_max_abs_grad_norm_trace_,
         pre_cov_mean_abs_grad_norm_trace_,
         optimizer_state1_,
         optimizer_state2_) = args
        current_prng_key = jax.random.fold_in(prng_key, t)
        gradient_prng_key, mu_dpsgd_prng_key, covariance_dpsgd_prng_key, batch_prng_key = jax.random.split(
            current_prng_key, 4)

        mu_trace_ = mu_trace_.at[t].set(phi_['mu'])
        pre_transformed_covariance_trace_ = pre_transformed_covariance_trace_.at[t].set(
            phi_['pre_transformed_covariance'])
        mc_integration_standard_normal_samples = jax.random.normal(gradient_prng_key,
                                                                   (mc_integration_samples_count, theta_dimension, 1),
                                                                   dtype=jnp.float32)

        def grad_kl_divergence(x, y):
            return jax.grad(
                per_example_loss_function,
                argnums=2)(x,
                           y,
                           phi_,
                           examples_count,
                           mc_integration_standard_normal_samples,
                           is_diagonal_matrix=is_covariance_diagonal)

        gradients = jax.vmap(grad_kl_divergence, in_axes=(0, 0), out_axes=0)(xs, ys)

        scaled_gradients = {
            'mu': gradients['mu'] * mu_gradient_scale,
            'pre_transformed_covariance': gradients['pre_transformed_covariance'] * pre_transformed_cov_gradient_scale
        }

        gradients = scaled_gradients
        mu_grad_norms = jnp.linalg.norm(gradients['mu'], ord=2, axis=-1)
        mu_max_grad_norm_trace_ = mu_max_grad_norm_trace_.at[t].set(jnp.max(mu_grad_norms))
        mu_mean_grad_norm_trace_ = mu_mean_grad_norm_trace_.at[t].set(jnp.mean(mu_grad_norms))
        pre_cov_grad_norms = jnp.linalg.norm(gradients['pre_transformed_covariance'], ord=2, axis=-1)
        pre_cov_max_grad_norm_trace_ = pre_cov_max_grad_norm_trace_.at[t].set(jnp.max(pre_cov_grad_norms))
        pre_cov_mean_grad_norm_trace_ = pre_cov_mean_grad_norm_trace_.at[t].set(jnp.mean(pre_cov_grad_norms))
        mu_max_abs_grad_trace_ = mu_max_abs_grad_trace_.at[t].set(jnp.max(jnp.abs(gradients['mu']), axis=0))
        mu_mean_abs_grad_trace_ = mu_mean_abs_grad_trace_.at[t].set(jnp.mean(jnp.abs(gradients['mu']), axis=0))
        pre_cov_max_abs_grad_norm_trace_ = pre_cov_max_abs_grad_norm_trace_.at[t].set(
            jnp.max(jnp.abs(gradients['pre_transformed_covariance']), axis=0))
        pre_cov_mean_abs_grad_norm_trace_ = pre_cov_mean_abs_grad_norm_trace_.at[t].set(
            jnp.mean(jnp.abs(gradients['pre_transformed_covariance']), axis=0))

        concatenated_gradients = jnp.concatenate(
            [scaled_gradients['mu'], scaled_gradients['pre_transformed_covariance']],
            axis=-1)
        gradient_norms = jnp.linalg.norm(concatenated_gradients, ord=2, axis=-1)
        clipping_factors = jnp.minimum(1.0, clipping_threshold / gradient_norms).reshape(-1, 1)

        clipped_gradients = jax.tree_map(lambda x: clipping_factors * x, {
            'mu': scaled_gradients['mu'],
            'pre_transformed_covariance': scaled_gradients['pre_transformed_covariance']
        })

        batch_mask = jax.random.bernoulli(batch_prng_key, sampling_rate, shape=(examples_count,))
        masked_gradients = jax.tree_util.tree_map(
            lambda x: jnp.where(batch_mask.reshape((examples_count,) + (1,) * (x.ndim - 1)), x,
                                jnp.zeros(x.shape[1:])),
            {
                'mu': clipped_gradients['mu'],
                'pre_transformed_covariance': clipped_gradients['pre_transformed_covariance']
            })

        summed_clipped_gradients = jax.tree_map(lambda x: jnp.sum(x, axis=0), {
            'mu': masked_gradients['mu'],
            'pre_transformed_covariance': masked_gradients['pre_transformed_covariance']
        })

        mu_noise = clipping_threshold * dpsgd_noise_scale * jax.random.normal(mu_dpsgd_prng_key, (theta_dimension,))
        pre_transformed_covariance_noise = clipping_threshold * dpsgd_noise_scale * jax.random.normal(
            covariance_dpsgd_prng_key,
            (pre_transformed_covariance_components,))

        perturbed_gradients = {
            'mu': (summed_clipped_gradients['mu'] + mu_noise) / mu_gradient_scale,
            'pre_transformed_covariance': (summed_clipped_gradients['pre_transformed_covariance']
                                           + pre_transformed_covariance_noise) / pre_transformed_cov_gradient_scale
        }

        mu_perturbed_gradient_trace_ = mu_perturbed_gradient_trace_.at[t].set(perturbed_gradients['mu'])
        pre_transformed_covariance_perturbed_gradient_trace_ = \
            pre_transformed_covariance_perturbed_gradient_trace_.at[t].set(
                perturbed_gradients['pre_transformed_covariance']
            )

        phi_ = {
            'mu': phi_['mu'] - mu_learning_rate * perturbed_gradients['mu'],
            'pre_transformed_covariance': phi_[
                                              'pre_transformed_covariance'] - pre_transformed_covariance_learning_rate *
                                          perturbed_gradients[
                                              'pre_transformed_covariance']
        }

        # updates1, optimizer_state1_ = optimizer1.update(perturbed_gradients['mu'],
        #                                                 optimizer_state1_)
        # updates2, optimizer_state2_ = optimizer2.update(perturbed_gradients['pre_transformed_covariance'],
        #                                                 optimizer_state2_)
        # phi_ = {
        #     'mu': optax.apply_updates(phi_['mu'], updates1),
        #     'pre_transformed_covariance': optax.apply_updates(phi_['pre_transformed_covariance'], updates2)
        # }

        return (phi_,
                mu_trace_,
                pre_transformed_covariance_trace_,
                mu_perturbed_gradient_trace_,
                pre_transformed_covariance_perturbed_gradient_trace_,
                mu_max_grad_norm_trace_,
                mu_mean_grad_norm_trace_,
                pre_cov_max_grad_norm_trace_,
                pre_cov_mean_grad_norm_trace_,
                mu_max_abs_grad_trace_,
                mu_mean_abs_grad_trace_,
                pre_cov_max_abs_grad_norm_trace_,
                pre_cov_mean_abs_grad_norm_trace_,
                optimizer_state1_,
                optimizer_state2_)

    if show_progress_bar:
        runs = training_iterations // 100
        remainder_iterations = training_iterations % 100

        progress_bar_updates = runs
        if remainder_iterations > 0:
            progress_bar_updates += 1

        progress_bar = tqdm(total=progress_bar_updates)
        for run in range(runs):
            (phi,
             mu_trace,
             pre_transformed_covariance_trace,
             mu_perturbed_gradient_trace,
             pre_transformed_covariance_perturbed_gradient_trace,
             mu_max_grad_norm_trace,
             mu_mean_grad_norm_trace,
             pre_cov_max_grad_norm_trace,
             pre_cov_mean_grad_norm_trace,
             mu_max_abs_grad_trace,
             mu_mean_abs_grad_trace,
             pre_cov_max_abs_grad_norm_trace,
             pre_cov_mean_abs_grad_norm_trace,
             optimizer_state1,
             optimizer_state2) = jax.lax.fori_loop(run * 100,
                                                   (run + 1) * 100,
                                                   update_params,
                                                   (phi,
                                                    mu_trace,
                                                    pre_transformed_covariance_trace,
                                                    mu_perturbed_gradient_trace,
                                                    pre_transformed_covariance_perturbed_gradient_trace,
                                                    mu_max_grad_norm_trace,
                                                    mu_mean_grad_norm_trace,
                                                    pre_cov_max_grad_norm_trace,
                                                    pre_cov_mean_grad_norm_trace,
                                                    mu_max_abs_grad_trace,
                                                    mu_mean_abs_grad_trace,
                                                    pre_cov_max_abs_grad_norm_trace,
                                                    pre_cov_mean_abs_grad_norm_trace,
                                                    optimizer_state1,
                                                    optimizer_state2))
            progress_bar.update()

        if remainder_iterations > 0:
            (phi,
             mu_trace,
             pre_transformed_covariance_trace,
             mu_perturbed_gradient_trace,
             pre_transformed_covariance_perturbed_gradient_trace,
             mu_max_grad_norm_trace,
             mu_mean_grad_norm_trace,
             pre_cov_max_grad_norm_trace,
             pre_cov_mean_grad_norm_trace,
             mu_max_abs_grad_trace,
             mu_mean_abs_grad_trace,
             pre_cov_max_abs_grad_norm_trace,
             pre_cov_mean_abs_grad_norm_trace,
             optimizer_state1,
             optimizer_state2) = jax.lax.fori_loop(runs * 100,
                                                   runs * 100 + remainder_iterations,
                                                   update_params,
                                                   (phi,
                                                    mu_trace,
                                                    pre_transformed_covariance_trace,
                                                    mu_perturbed_gradient_trace,
                                                    pre_transformed_covariance_perturbed_gradient_trace,
                                                    mu_max_grad_norm_trace,
                                                    mu_mean_grad_norm_trace,
                                                    pre_cov_max_grad_norm_trace,
                                                    pre_cov_mean_grad_norm_trace,
                                                    mu_max_abs_grad_trace,
                                                    mu_mean_abs_grad_trace,
                                                    pre_cov_max_abs_grad_norm_trace,
                                                    pre_cov_mean_abs_grad_norm_trace,
                                                    optimizer_state1,
                                                    optimizer_state2))
            progress_bar.update()
        progress_bar.close()
    else:
        if not DEBUG_MODE_ON:
            (phi,
             mu_trace,
             pre_transformed_covariance_trace,
             mu_perturbed_gradient_trace,
             pre_transformed_covariance_perturbed_gradient_trace,
             mu_max_grad_norm_trace,
             mu_mean_grad_norm_trace,
             pre_cov_max_grad_norm_trace,
             pre_cov_mean_grad_norm_trace,
             mu_max_abs_grad_trace,
             mu_mean_abs_grad_trace,
             pre_cov_max_abs_grad_norm_trace,
             pre_cov_mean_abs_grad_norm_trace,
             optimizer_state1,
             optimizer_state2) = jax.lax.fori_loop(0,
                                                   training_iterations,
                                                   update_params,
                                                   (phi,
                                                    mu_trace,
                                                    pre_transformed_covariance_trace,
                                                    mu_perturbed_gradient_trace,
                                                    pre_transformed_covariance_perturbed_gradient_trace,
                                                    mu_max_grad_norm_trace,
                                                    mu_mean_grad_norm_trace,
                                                    pre_cov_max_grad_norm_trace,
                                                    pre_cov_mean_grad_norm_trace,
                                                    mu_max_abs_grad_trace,
                                                    mu_mean_abs_grad_trace,
                                                    pre_cov_max_abs_grad_norm_trace,
                                                    pre_cov_mean_abs_grad_norm_trace,
                                                    optimizer_state1,
                                                    optimizer_state2))
        else:
            for t in range(training_iterations):
                (phi,
                 mu_trace,
                 pre_transformed_covariance_trace,
                 mu_perturbed_gradient_trace,
                 pre_transformed_covariance_perturbed_gradient_trace,
                 mu_max_grad_norm_trace,
                 mu_mean_grad_norm_trace,
                 pre_cov_max_grad_norm_trace,
                 pre_cov_mean_grad_norm_trace,
                 mu_max_abs_grad_trace,
                 mu_mean_abs_grad_trace,
                 pre_cov_max_abs_grad_norm_trace,
                 pre_cov_mean_abs_grad_norm_trace,
                 optimizer_state1,
                 optimizer_state2) = update_params(t, (phi,
                                                       mu_trace,
                                                       pre_transformed_covariance_trace,
                                                       mu_perturbed_gradient_trace,
                                                       pre_transformed_covariance_perturbed_gradient_trace,
                                                       mu_max_grad_norm_trace,
                                                       mu_mean_grad_norm_trace,
                                                       pre_cov_max_grad_norm_trace,
                                                       pre_cov_mean_grad_norm_trace,
                                                       mu_max_abs_grad_trace,
                                                       mu_mean_abs_grad_trace,
                                                       pre_cov_max_abs_grad_norm_trace,
                                                       pre_cov_mean_abs_grad_norm_trace,
                                                       optimizer_state1,
                                                       optimizer_state2))

    # plt.plot(pre_cov_max_grad_norm_trace)
    # plt.show()
    # plt.plot(mu_max_grad_norm_trace)
    # plt.show()
    print_debug_info(mu_max_grad_norm_trace,
                     mu_mean_grad_norm_trace,
                     mu_max_abs_grad_trace,
                     mu_mean_abs_grad_trace,
                     pre_cov_max_grad_norm_trace,
                     pre_cov_mean_grad_norm_trace,
                     pre_cov_max_abs_grad_norm_trace,
                     pre_cov_mean_abs_grad_norm_trace)

    return (phi,
            mu_trace,
            pre_transformed_covariance_trace,
            mu_perturbed_gradient_trace,
            pre_transformed_covariance_perturbed_gradient_trace,
            dpsgd_noise_scale,
            mu_max_grad_norm_trace,
            mu_mean_grad_norm_trace,
            pre_cov_max_grad_norm_trace,
            pre_cov_mean_grad_norm_trace,
            mu_max_abs_grad_trace,
            mu_mean_abs_grad_trace,
            pre_cov_max_abs_grad_norm_trace,
            pre_cov_mean_abs_grad_norm_trace,
            learning_rates_common_factor,
            mu_learning_rate,
            pre_transformed_covariance_learning_rate)


def print_debug_info(mu_max_grad_norm_trace,
                     mu_mean_grad_norm_trace,
                     mu_max_abs_grad_trace,
                     mu_mean_abs_grad_trace,
                     pre_cov_max_grad_norm_trace,
                     pre_cov_mean_grad_norm_trace,
                     pre_cov_max_abs_grad_norm_trace,
                     pre_cov_mean_abs_grad_norm_trace):
    # burn_in = 2000
    burn_in = int(0.2 * mu_max_grad_norm_trace.shape[0])
    # plt.plot(mu_max_abs_grad_trace[burn_in:])
    # plt.show()
    # plt.plot(pre_cov_max_abs_grad_norm_trace[burn_in:])
    # plt.show()

    print('max max grad norms')
    print(jnp.max(mu_max_grad_norm_trace[burn_in:]))
    print(jnp.max(pre_cov_max_grad_norm_trace[burn_in:]))
    print('mean max grad norms')
    print(jnp.mean(mu_max_grad_norm_trace[burn_in:]))
    print(jnp.mean(pre_cov_max_grad_norm_trace[burn_in:]))
    print('---------------')
    print('max mean grad norms')
    print(jnp.max(mu_mean_grad_norm_trace[burn_in:]))
    print(jnp.max(pre_cov_mean_grad_norm_trace[burn_in:]))
    print('mean mean grad norms')
    print(jnp.mean(mu_mean_grad_norm_trace[burn_in:]))
    print(jnp.mean(pre_cov_mean_grad_norm_trace[burn_in:]))
    print('---------------')
    print('max max abs grads')
    max_mu_max_abs_grads = jnp.max(mu_max_abs_grad_trace[burn_in:], axis=0)
    print(max_mu_max_abs_grads)
    max_pre_cov_max_abs_grads = jnp.max(pre_cov_max_abs_grad_norm_trace[burn_in:], axis=0)
    print(max_pre_cov_max_abs_grads)
    print('---------------')
    print('max mean abs grads')
    max_mu_mean_abs_grads = jnp.max(mu_mean_abs_grad_trace[burn_in:], axis=0)
    print(max_mu_mean_abs_grads)
    max_pre_cov_mean_abs_grads = jnp.max(pre_cov_mean_abs_grad_norm_trace[burn_in:], axis=0)
    print(max_pre_cov_mean_abs_grads)
    print('---------------')
    print('mean max abs grads')
    mean_mu_max_abs_grads = jnp.mean(mu_max_abs_grad_trace[burn_in:], axis=0)
    print(mean_mu_max_abs_grads)
    mean_pre_cov_max_abs_grads = jnp.mean(pre_cov_max_abs_grad_norm_trace[burn_in:], axis=0)
    print(mean_pre_cov_max_abs_grads)
    print('---------------')
    print('mean mean abs grads')
    mean_mu_mean_abs_grads = jnp.mean(mu_mean_abs_grad_trace[burn_in:], axis=0)
    print(mean_mu_mean_abs_grads)
    mean_pre_cov_mean_abs_grads = jnp.mean(pre_cov_mean_abs_grad_norm_trace[burn_in:], axis=0)
    print(mean_pre_cov_mean_abs_grads)
    print('=========================')

    # count = max_mu_max_abs_grads.shape[0] + max_pre_cov_max_abs_grads.shape[0]
    # max_of_max = jnp.max(mu_max_abs_grad_trace[burn_in:], axis=0)
    # print(max_of_max)
    # print(max_pre_cov_max_abs_grads)
    # print(jnp.max(max_of_max) / max_of_max)
    # print(jnp.max(max_of_max) / max_pre_cov_max_abs_grads)
    # print(jnp.max(max_of_max) * jnp.sqrt(count))
    # ===============================


#
# def mcmc_pre_conditioned_samples_to_hessians(mcmc,
#                                              matrix_dimension,
#                                              is_hessian_positive_definite,
#                                              is_hessian_diagonal_matrix):
#     hessian_diagonal_mu_samples = mcmc.get_samples()['hessian_diagonal_mu']
#     hessian_diagonal_pre_conditioned_covariance_samples = mcmc.get_samples()[
#         'hessian_diagonal_pre_conditioned_covariance']
#     if is_hessian_diagonal_matrix:
#         hessian_strictly_lower_samples = None
#     else:
#         hessian_strictly_lower_samples = mcmc.get_samples()['hessian_strictly_lower']
#
#     hessian_samples = []
#     for i in range(hessian_diagonal_mu_samples.shape[0]):
#         hessian_diagonal_mu = hessian_diagonal_mu_samples[i]
#         hessian_diagonal_pre_conditioned_covariance = hessian_diagonal_pre_conditioned_covariance_samples[i]
#         hessian_diagonal = jnp.concatenate([hessian_diagonal_mu,
#                                             hessian_diagonal_pre_conditioned_covariance], axis=-1)
#
#         if is_hessian_diagonal_matrix:
#             hessian_lower_vector = hessian_diagonal
#         else:
#             hessian_strictly_lower = hessian_strictly_lower_samples[i]
#
#             hessian_lower = jnp.diagonal(hessian_diagonal) + jnp.eye(matrix_dimension, dtype=jnp.float32).at[
#                 jnp.tril_indices(matrix_dimension, -1)].set(hessian_strictly_lower)
#
#             hessian_lower_vector = hessian_lower[jnp.tril_indices(matrix_dimension, 0)]
#
#         hessian_samples.append(convert_vector_to_symmetric_matrix(hessian_lower_vector,
#                                                                   matrix_dimension,
#                                                                   is_hessian_diagonal_matrix,
#                                                                   is_hessian_positive_definite))
#     return jnp.array(hessian_samples)

def mcmc_pre_conditioned_samples_to_hessians(mcmc,
                                             matrix_dimension,
                                             is_hessian_positive_definite,
                                             is_hessian_diagonal_matrix):
    # hessian_diagonal_mu_samples = mcmc.get_samples()['hessian_diagonal_mu']
    # hessian_diagonal_pre_conditioned_covariance_samples = mcmc.get_samples()[
    #     'hessian_diagonal_pre_conditioned_covariance']
    # if is_hessian_diagonal_matrix:
    #     hessian_strictly_lower_samples = None
    # else:
    #     hessian_strictly_lower_samples = mcmc.get_samples()['hessian_strictly_lower']
    #
    # hessian_samples = []
    # for i in range(hessian_diagonal_mu_samples.shape[0]):
    #     hessian_diagonal_mu = hessian_diagonal_mu_samples[i]
    #     hessian_diagonal_pre_conditioned_covariance = hessian_diagonal_pre_conditioned_covariance_samples[i]
    #     hessian_diagonal = jnp.concatenate([hessian_diagonal_mu,
    #                                         hessian_diagonal_pre_conditioned_covariance], axis=-1)
    #
    #     if is_hessian_diagonal_matrix:
    #         hessian_lower_vector = hessian_diagonal
    #     else:
    #         hessian_strictly_lower = hessian_strictly_lower_samples[i]
    #
    #         hessian_lower = jnp.diagonal(hessian_diagonal) + jnp.eye(matrix_dimension, dtype=jnp.float32).at[
    #             jnp.tril_indices(matrix_dimension, -1)].set(hessian_strictly_lower)
    #
    #         hessian_lower_vector = hessian_lower[jnp.tril_indices(matrix_dimension, 0)]
    #
    #     hessian_samples.append(convert_vector_to_symmetric_matrix(hessian_lower_vector,
    #                                                               matrix_dimension,
    #                                                               is_hessian_diagonal_matrix,
    #                                                               is_hessian_positive_definite))
    # return jnp.array(hessian_samples)

    hessian_diagonal_samples = mcmc.get_samples()['hessian_diagonal']
    return jnp.array([jnp.diag(hessian_diagonal) for hessian_diagonal in hessian_diagonal_samples])


def sample_mcmc_parameter_based_inference_model(phi_trace,
                                                theta_dimension,
                                                burn_in,
                                                dpsgd_noise_scale,
                                                dpsgd_clipping_threshold,
                                                dpsgd_gradient_scale_vector,
                                                dpsgd_learning_rates_vector,
                                                hessian_prior_mean,
                                                hessian_prior_std,
                                                phi_prior_mean,
                                                phi_prior_std,
                                                mcmc_num_warmup,
                                                mcmc_num_samples,
                                                mcmc_num_chains,
                                                prng_key,
                                                is_hessian_positive_definite=True,
                                                is_hessian_diagonal_matrix=True):
    kernel = NUTS(parameter_based_inference_model)
    mcmc = MCMC(
        kernel,
        num_warmup=mcmc_num_warmup,
        num_samples=mcmc_num_samples,
        num_chains=mcmc_num_chains,
        progress_bar=False
    )
    mcmc.run(
        prng_key,
        phi_trace[burn_in:],
        theta_dimension,
        dpsgd_noise_scale,
        dpsgd_clipping_threshold,
        dpsgd_gradient_scale_vector,
        dpsgd_learning_rates_vector,
        hessian_prior_mean,
        hessian_prior_std,
        phi_prior_mean,
        phi_prior_std,
        is_hessian_positive_definite,
        is_hessian_diagonal_matrix
    )
    hessian_samples = mcmc_pre_conditioned_samples_to_hessians(mcmc,
                                                               phi_trace.shape[-1],
                                                               is_hessian_positive_definite,
                                                               is_hessian_diagonal_matrix)
    return mcmc.get_samples()['optimal_phi'], hessian_samples


def sample_mcmc_gradient_based_inference_model(phi_trace,
                                               gradient_trace,
                                               theta_dimension,
                                               burn_in,
                                               sampling_rate,
                                               dpsgd_noise_scale,
                                               dpsgd_clipping_threshold,
                                               dpsgd_gradient_scale_vector,
                                               hessian_prior_mean,
                                               hessian_prior_std,
                                               phi_prior_mean,
                                               phi_prior_std,
                                               mcmc_num_warmup,
                                               mcmc_num_samples,
                                               mcmc_num_chains,
                                               prng_key,
                                               add_subsampling_noise=False,
                                               is_hessian_positive_definite=True,
                                               is_hessian_diagonal_matrix=True):
    kernel = NUTS(gradient_based_inference_model)
    mcmc = MCMC(
        kernel,
        num_warmup=mcmc_num_warmup,
        num_samples=mcmc_num_samples,
        num_chains=mcmc_num_chains,
        progress_bar=False
    )
    mcmc.run(
        prng_key,
        phi_trace[burn_in:],
        gradient_trace[burn_in:],
        theta_dimension,
        sampling_rate,
        dpsgd_noise_scale,
        dpsgd_clipping_threshold,
        dpsgd_gradient_scale_vector,
        hessian_prior_mean,
        hessian_prior_std,
        phi_prior_mean,
        phi_prior_std,
        add_subsampling_noise,
        is_hessian_positive_definite,
        is_hessian_diagonal_matrix
    )
    hessian_samples = mcmc_pre_conditioned_samples_to_hessians(mcmc,
                                                               phi_trace.shape[-1],
                                                               is_hessian_positive_definite,
                                                               is_hessian_diagonal_matrix)
    return mcmc.get_samples()['optimal_phi'], hessian_samples


def perform_dpsgd(experiment_args, experiment_tracker):
    (experiment_tracker.last_phi,
     experiment_tracker.mu_trace,
     experiment_tracker.pre_transformed_covariance_trace,
     experiment_tracker.mu_perturbed_gradient_trace,
     experiment_tracker.pre_transformed_covariance_perturbed_gradient_trace,
     experiment_tracker.dpsgd_noise_scale,
     experiment_tracker.mu_median_grad_norm_trace,
     experiment_tracker.mu_mean_grad_norm_trace,
     experiment_tracker.pre_cov_median_grad_norm_trace,
     experiment_tracker.pre_cov_mean_grad_norm_trace,
     experiment_tracker.mu_median_abs_grad_trace,
     experiment_tracker.mu_mean_abs_grad_trace,
     experiment_tracker.pre_cov_median_abs_grad_norm_trace,
     experiment_tracker.pre_cov_mean_abs_grad_norm_trace,
     experiment_tracker.learning_rates_common_factor,
     experiment_tracker.mu_learning_rate,
     experiment_tracker.pre_transformed_covariance_learning_rate) = train_variational_inference_model_dpsgd(
        experiment_args.train_xs,
        experiment_args.train_ys,
        experiment_args.theta_dimension,
        experiment_args.dpsgd_per_example_loss_function,
        experiment_args.training_iterations,
        experiment_args.target_epsilon,
        experiment_args.target_delta,
        experiment_args.dpsgd_initial_mu_value,
        experiment_args.dpsgd_initial_pre_transformed_covariance_value,
        experiment_args.dpsgd_mu_learning_rate,
        experiment_args.dpsgd_pre_transformed_covariance_learning_rate,
        experiment_args.mu_learning_rate_v,
        experiment_args.pre_cov_learning_rate_v,
        experiment_args.clipping_threshold,
        experiment_args.mu_gradient_scale,
        experiment_args.pre_transformed_covariance_gradient_scale,
        experiment_args.sampling_rate,
        experiment_args.is_covariance_diagonal_matrix,
        experiment_args.kl_divergence_mc_integration_samples_count,
        experiment_args.algorithm_prng_key,
        experiment_args.custom_noise_scale)

    experiment_tracker.last_mu = experiment_tracker.last_phi['mu']
    experiment_tracker.last_covariance = get_positive_definite_matrix(
        experiment_tracker.last_phi['pre_transformed_covariance'],
        experiment_args.theta_dimension,
        experiment_args.is_covariance_diagonal_matrix)

    experiment_tracker.phi_trace = jnp.concatenate([experiment_tracker.mu_trace,
                                                    experiment_tracker.pre_transformed_covariance_trace],
                                                   axis=-1)
    experiment_tracker.gradient_trace = jnp.concatenate([experiment_tracker.mu_perturbed_gradient_trace,
                                                         experiment_tracker.pre_transformed_covariance_perturbed_gradient_trace],
                                                        axis=-1)

    return experiment_tracker


def perform_mcmc_sampling(experiment_args, experiment_tracker):
    experiment_tracker.mcmc_burn_in = int(experiment_args.trace_burn_in_percentage *
                                          experiment_tracker.phi_trace.shape[0])
    theta_dimension = experiment_args.theta_dimension

    mu_gradient_scale_vector = jnp.ones((theta_dimension,), dtype=jnp.float32) * experiment_args.mu_gradient_scale
    pre_transformed_covariance_gradient_scale_vector = jnp.ones(
        (experiment_tracker.pre_transformed_covariance_trace.shape[-1],),
        dtype=jnp.float32) * experiment_args.pre_transformed_covariance_gradient_scale
    gradient_scale_vector = jnp.concatenate(
        [mu_gradient_scale_vector, pre_transformed_covariance_gradient_scale_vector], axis=0)

    if experiment_args.mcmc_inference_model == 'parameter_based':
        mu_learning_rates_vector = jnp.ones((theta_dimension,),
                                            dtype=jnp.float32) * experiment_args.dpsgd_mu_learning_rate
        pre_transformed_covariance_rates_vector = jnp.ones(
            (experiment_tracker.pre_transformed_covariance_trace.shape[-1],),
            dtype=jnp.float32) * experiment_args.dpsgd_pre_transformed_covariance_learning_rate
        learning_rates_vector = jnp.concatenate([mu_learning_rates_vector, pre_transformed_covariance_rates_vector],
                                                axis=0)

        (experiment_tracker.optimal_phi_samples,
         experiment_tracker.hessian_samples) = sample_mcmc_parameter_based_inference_model(
            experiment_tracker.phi_trace,
            experiment_args.theta_dimension,
            experiment_tracker.mcmc_burn_in,
            experiment_tracker.dpsgd_noise_scale,
            experiment_args.clipping_threshold,
            gradient_scale_vector,
            learning_rates_vector,
            experiment_tracker.hessian_prior_mean,
            experiment_tracker.hessian_prior_std,
            experiment_tracker.phi_prior_mean,
            experiment_tracker.phi_prior_std,
            experiment_args.mcmc_num_warmup,
            experiment_args.mcmc_num_samples,
            experiment_args.mcmc_num_chains,
            experiment_args.algorithm_prng_key
        )

        experiment_tracker.mu_optimal_phi_samples = experiment_tracker.optimal_phi_samples[:,
                                                    :experiment_args.theta_dimension]
        experiment_tracker.pre_transformed_covariance_optimal_phi_samples = experiment_tracker.optimal_phi_samples[:,
                                                                            experiment_args.theta_dimension:]
    elif experiment_args.mcmc_inference_model == 'gradient_based':
        (experiment_tracker.optimal_phi_samples,
         experiment_tracker.hessian_samples) = sample_mcmc_gradient_based_inference_model(
            experiment_tracker.phi_trace,
            experiment_tracker.gradient_trace,
            experiment_args.theta_dimension,
            experiment_tracker.mcmc_burn_in,
            experiment_args.sampling_rate,
            experiment_tracker.dpsgd_noise_scale,
            experiment_args.clipping_threshold,
            gradient_scale_vector,
            experiment_tracker.hessian_prior_mean,
            experiment_tracker.hessian_prior_std,
            experiment_tracker.phi_prior_mean,
            experiment_tracker.phi_prior_std,
            experiment_args.mcmc_num_warmup,
            experiment_args.mcmc_num_samples,
            experiment_args.mcmc_num_chains,
            experiment_args.algorithm_prng_key,
            experiment_args.inference_add_subsampling_noise)

        experiment_tracker.mu_optimal_phi_samples = experiment_tracker.optimal_phi_samples[:,
                                                    :experiment_args.theta_dimension]
        experiment_tracker.pre_transformed_covariance_optimal_phi_samples = experiment_tracker.optimal_phi_samples[:,
                                                                            experiment_args.theta_dimension:]
    else:
        raise ValueError(f'Unrecognized mcmc_inference_model argument: {experiment_args.mcmc_inference_model}')
    return experiment_tracker


def perform_laplace_approximation(experiment_args, experiment_tracker):
    mu_gradient_scale_vector = jnp.ones((experiment_args.theta_dimension,),
                                        dtype=jnp.float32) * experiment_args.mu_gradient_scale
    pre_transformed_covariance_gradient_scale_vector = jnp.ones(
        (experiment_tracker.pre_transformed_covariance_trace.shape[-1],),
        dtype=jnp.float32) * experiment_args.pre_transformed_covariance_gradient_scale
    gradient_scale_vector = jnp.concatenate([mu_gradient_scale_vector,
                                             pre_transformed_covariance_gradient_scale_vector], axis=0)

    perturbed_gradient_trace = jnp.concatenate([experiment_tracker.mu_perturbed_gradient_trace,
                                                experiment_tracker.pre_transformed_covariance_perturbed_gradient_trace],
                                               axis=-1)
    (experiment_tracker.laplace_approximation_posterior_mode,
     covariance_matrix1,
     covariance_matrix2,
     covariance_matrix3,
     experiment_tracker.laplace_parameter_trace) = gradient_based_inference_model_posterior_laplace_approximation(
        experiment_tracker.phi_trace,
        perturbed_gradient_trace,
        experiment_args.sampling_rate,
        experiment_tracker.dpsgd_noise_scale,
        experiment_args.clipping_threshold,
        gradient_scale_vector,
        experiment_tracker.hessian_prior_mean,
        experiment_tracker.hessian_prior_std,
        experiment_tracker.phi_prior_mean,
        experiment_tracker.phi_prior_std,
        experiment_args.trace_burn_in_percentage,
        experiment_args.inference_add_subsampling_noise,
        optimization_iterations=experiment_args.laplace_approximation_iterations,
        optimization_learning_rate=experiment_args.laplace_approximation_learning_rate,
        optimization_apply_trace_averaging=experiment_args.laplace_approximation_trace_averaging,
        optimization_trace_averaging_burn_in=experiment_args.laplace_approximation_trace_averaging_burn_in)

    covariance_matrix = covariance_matrix1
    if jnp.isnan(covariance_matrix).any():
        covariance_matrix = covariance_matrix2
    if jnp.isnan(covariance_matrix).any():
        covariance_matrix = covariance_matrix2
    # experiment_tracker.laplace_approximation_covariance = covariance_matrix

    # optimal_phi_mode = experiment_tracker.experiment_tracker.laplace_approximation_posterior_mode['optimal_phi']
    # experiment_tracker.laplace_approximation_optimal_phi_posterior_mean = optimal_phi_mode
    # optimal_phi_dims = optimal_phi_mode.shape[0]
    # hessian_dims = experiment_tracker.experiment_tracker.laplace_approximation_posterior_mode['hessian'].shape[0]
    # # from wikipedia
    # cov_11 = covariance_matrix[:optimal_phi_dims, :optimal_phi_dims]
    # cov_12 = covariance_matrix[:optimal_phi_dims, optimal_phi_dims: optimal_phi_dims + hessian_dims]
    # cov_21 = covariance_matrix[optimal_phi_dims: optimal_phi_dims + hessian_dims, :optimal_phi_dims]
    # cov_22 = covariance_matrix[optimal_phi_dims: optimal_phi_dims + hessian_dims,
    #          optimal_phi_dims: optimal_phi_dims + hessian_dims]
    # optimal_phi_covariance = cov_11 - cov_12 @ cov_22 @ cov_21

    # experiment_tracker.laplace_approximation_posterior_covariance = optimal_phi_covariance
    experiment_tracker.laplace_approximation_posterior_covariance = covariance_matrix
    return experiment_tracker


def gaussian_mixture_ppf(normal_distributions,
                         search_interval,
                         percentile,
                         epsilon=1e-6,
                         max_iterations=1000):
    a, b = search_interval

    a_cdf = np.mean(normal_distributions.cdf(a))
    if abs(a_cdf - percentile) < epsilon:
        return a

    b_cdf = np.mean(normal_distributions.cdf(b))
    if abs(b_cdf - percentile) < epsilon:
        return b

    x = (a + b) / 2
    for i in range(max_iterations):
        x_cdf = np.mean(normal_distributions.cdf(x))
        if abs(x_cdf - percentile) < epsilon:
            return x

        if x_cdf > percentile:
            b = x
        else:
            a = x
        x = (a + b) / 2

    return x


def is_within_gaussian_mixture(theta_true,
                               distributions,
                               confidence_alpha,
                               epsilon=1e-6):
    theta_dimension = theta_true.shape[0]

    percentile1 = (1 - math.pow(1 - confidence_alpha, 1.0 / theta_dimension)) / 2
    percentile2 = 1 - percentile1

    separate_intervals = []
    for i in range(theta_dimension):
        dist = distributions[i]
        separate_intervals.append(jnp.concatenate([jnp.expand_dims(dist.icdf(percentile1), -1),
                                                   jnp.expand_dims(dist.icdf(percentile2), -1)],
                                                  axis=-1))

    separate_intervals = jnp.array(separate_intervals, np.float32)
    min_as = jnp.min(separate_intervals[:, :, 0], axis=-1) - epsilon
    max_bs = jnp.max(separate_intervals[:, :, 1], axis=-1) + epsilon
    search_intervals = [(min_as[i], max_bs[i]) for i in range(theta_dimension)]
    credible_intervals = [(gaussian_mixture_ppf(distributions[i], search_intervals[i], percentile1),
                           gaussian_mixture_ppf(distributions[i], search_intervals[i], percentile2))
                          for i in range(theta_dimension)]

    is_within = [credible_intervals[i][0] <= theta_true[i] <= credible_intervals[i][1]
                 for i in range(theta_dimension)]
    all_within = reduce(lambda x, y: x and y, is_within, True)

    return all_within, is_within, credible_intervals


def calculate_estimates_from_mcmc_samples(optimal_phi_samples,
                                          is_covariance_diagonal_matrix,
                                          theta_dimension):
    mean_optimal_phi = np.mean(optimal_phi_samples, axis=0)
    std_optimal_phi = np.std(optimal_phi_samples, axis=0)
    estimated_mu = mean_optimal_phi[:theta_dimension]
    mu_optimal_phi_trace = optimal_phi_samples[:, :theta_dimension]
    n = mu_optimal_phi_trace.shape[0]

    if is_covariance_diagonal_matrix:
        mu_sample_covariance = np.diag(std_optimal_phi[:theta_dimension] ** 2)
    else:
        zero_centered_samples = mu_optimal_phi_trace - estimated_mu.reshape(1, -1)
        zero_centered_samples1 = zero_centered_samples.reshape(-1, theta_dimension, 1)
        zero_centered_samples2 = zero_centered_samples.reshape(-1, 1, theta_dimension)
        mu_sample_covariance = zero_centered_samples1 * zero_centered_samples2
        mu_sample_covariance = np.sum(mu_sample_covariance, axis=0)
        mu_sample_covariance /= (n - 1)

    covariance_optimal_phi_trace = np.zeros((optimal_phi_samples.shape[0], theta_dimension, theta_dimension))
    for i in range(optimal_phi_samples.shape[0]):
        covariance_optimal_phi_trace[i] = get_positive_definite_matrix(optimal_phi_samples[i, theta_dimension:],
                                                                       theta_dimension,
                                                                       is_covariance_diagonal_matrix)
    mean_covariance = np.mean(covariance_optimal_phi_trace, axis=0)
    estimated_covariance = mean_covariance + mu_sample_covariance

    return estimated_mu, estimated_covariance, mu_sample_covariance, mean_covariance, covariance_optimal_phi_trace


def calculate_estimates_from_laplace_approximation(posterior_mode,
                                                   posterior_covariance,
                                                   theta_dimension):
    optimal_phi = posterior_mode['optimal_phi']
    estimated_mu = optimal_phi[:theta_dimension]
    estimated_pre_transformed_conditioned_covariance = optimal_phi[theta_dimension:]
    conditioned_covariance = get_positive_definite_matrix(estimated_pre_transformed_conditioned_covariance,
                                                          theta_dimension,
                                                          True)

    mu_covariance = jnp.diag(jnp.diagonal(posterior_covariance)[:theta_dimension])
    estimated_covariance = mu_covariance + conditioned_covariance

    return estimated_mu, estimated_covariance


def is_within_confidence_region(theta_true,
                                estimated_mu,
                                estimated_covariance,
                                confidence_alpha):
    zero_centered = theta_true - estimated_mu
    squared_mahalanobis_distance = zero_centered @ np.linalg.inv(estimated_covariance) @ zero_centered.T
    test_radius = stats.chi2(df=theta_true.shape[-1]).ppf(1 - confidence_alpha)
    result = squared_mahalanobis_distance < test_radius
    return result, squared_mahalanobis_distance, test_radius


def calculate_parameter_log_likelihood(xs, ys, theta):
    def log_likelihood_of_data(xs_, ys_):
        return log_likelihood(xs_, ys_, theta)

    return jnp.sum(jax.vmap(log_likelihood_of_data)(xs, ys))


def sample_parameter_and_calculate_log_likelihood(xs,
                                                  ys,
                                                  theta_estimated_mu,
                                                  theta_estimated_covariance_cholesky_lower,
                                                  prng_key):
    normal_samples = jax.random.normal(prng_key, (theta_estimated_mu.shape[0], 1), jnp.float32)
    w = theta_estimated_mu.reshape(theta_estimated_mu.shape[0],
                                   1) + theta_estimated_covariance_cholesky_lower @ normal_samples
    w = w.reshape(theta_estimated_mu.shape[0])
    return calculate_parameter_log_likelihood(xs, ys, w)


def calculate_parameters_kl_divergence(kl_divergence_objective,
                                       xs,
                                       ys,
                                       mu,
                                       pre_transformed_covariance,
                                       mc_integration_samples_count,
                                       prng_key,
                                       is_covariance_diagonal_matrix,
                                       transform_covariance_matrix=True):
    mc_integration_standard_normal_samples = jax.random.normal(prng_key,
                                                               (mc_integration_samples_count, mu.shape[0], 1),
                                                               jnp.float32)
    phi = {'mu': mu, 'pre_transformed_covariance': pre_transformed_covariance}

    def kl_divergence_of_data(x, y):
        if kl_divergence_objective == 'kl_divergence':
            return per_example_kl_divergence(x,
                                             y,
                                             phi,
                                             xs.shape[0],
                                             mc_integration_standard_normal_samples,
                                             is_covariance_diagonal_matrix,
                                             transform_covariance_matrix)
        elif kl_divergence_objective == 'kl_divergence_with_exact_entropy':
            return per_example_kl_divergence_with_exact_entropy(x,
                                                                y,
                                                                phi,
                                                                xs.shape[0],
                                                                mc_integration_standard_normal_samples,
                                                                is_covariance_diagonal_matrix,
                                                                transform_covariance_matrix)

    return jnp.sum(jax.vmap(kl_divergence_of_data, in_axes=(0, 0), out_axes=0)(xs, ys))


def calculate_average_log_likelihood(xs,
                                     ys,
                                     phi_estimated_mu,
                                     phi_estimated_covariance_cholesky_lower,
                                     samples_count,
                                     prng_key):
    def sample_parameter_and_calculate_log_likelihood_partial(prng_key_):
        return sample_parameter_and_calculate_log_likelihood(xs,
                                                             ys,
                                                             phi_estimated_mu,
                                                             phi_estimated_covariance_cholesky_lower,
                                                             prng_key_)

    prng_keys = jax.random.split(prng_key, samples_count)
    return jax.vmap(sample_parameter_and_calculate_log_likelihood_partial)(prng_keys).mean()


def calculate_estimates_and_confidence_region(experiment_args, experiment_tracker):
    if experiment_args.use_mcmc_sampling:
        (experiment_tracker.estimated_mu,
         experiment_tracker.estimated_covariance,
         experiment_tracker.mu_sample_covariance,
         experiment_tracker.mean_covariance,
         experiment_tracker.covariance_optimal_phi_trace) = calculate_estimates_from_mcmc_samples(
            experiment_tracker.optimal_phi_samples,
            experiment_args.is_covariance_diagonal_matrix,
            experiment_args.theta_dimension
        )
    else:
        experiment_tracker.estimated_mu, experiment_tracker.estimated_covariance = calculate_estimates_from_laplace_approximation(
            experiment_tracker.laplace_approximation_posterior_mode,
            experiment_tracker.laplace_approximation_posterior_covariance,
            experiment_args.theta_dimension)

    theta_true = experiment_args.theta_transform(experiment_args.theta_true)

    (experiment_tracker.is_within_confidence_region_result,
     experiment_tracker.squared_mahalanobis_distance,
     experiment_tracker.test_radius) = is_within_confidence_region(theta_true,
                                                                   experiment_tracker.estimated_mu,
                                                                   experiment_tracker.estimated_covariance,
                                                                   experiment_args.confidence_alpha)
    if experiment_args.is_covariance_diagonal_matrix:
        experiment_tracker.estimated_covariance_cholesky_lower = jnp.diag(
            jnp.sqrt(jnp.diagonal(experiment_tracker.estimated_covariance)))
    else:
        experiment_tracker.estimated_covariance_cholesky_lower = get_cholesky_lower(
            experiment_tracker.estimated_covariance)

    # experiment_tracker.average_log_likelihood = calculate_average_log_likelihood(
    #     experiment_args.validation_xs,
    #     experiment_args.validation_ys,
    #     experiment_tracker.estimated_mu,
    #     experiment_tracker.estimated_covariance_cholesky_lower,
    #     experiment_args.estimated_parameter_log_likelihood_samples,
    #     experiment_args.algorithm_prng_key
    # )
    #
    # experiment_tracker.parameters_kl_divergence = calculate_parameters_kl_divergence(
    #     'kl_divergence',
    #     experiment_args.validation_xs,
    #     experiment_args.validation_ys,
    #     experiment_tracker.estimated_mu,
    #     experiment_tracker.estimated_covariance_cholesky_lower,
    #     experiment_args.kl_divergence_mc_integration_samples_count,
    #     experiment_args.algorithm_prng_key,
    #     experiment_args.is_covariance_diagonal_matrix,
    #     transform_covariance_matrix=False
    # )

    return experiment_tracker


def calculate_hessian_prior(experiment_args, experiment_tracker):
    burn_in = int(experiment_tracker.phi_trace.shape[0] * experiment_args.trace_burn_in_percentage)
    phi_trace_last = experiment_tracker.phi_trace[burn_in:]
    gradient_trace_last = experiment_tracker.gradient_trace[burn_in:]
    mu_phi = jnp.mean(phi_trace_last, axis=0)
    mu_gradient_scale_vector = jnp.ones((experiment_args.theta_dimension,),
                                        dtype=jnp.float32) * experiment_args.mu_gradient_scale
    pre_transformed_covariance_gradient_scale_vector = jnp.ones(
        (experiment_tracker.pre_transformed_covariance_trace.shape[-1],),
        dtype=jnp.float32) * experiment_args.pre_transformed_covariance_gradient_scale
    gradient_scale_vector = jnp.concatenate([mu_gradient_scale_vector,
                                             pre_transformed_covariance_gradient_scale_vector], axis=0)
    experiment_tracker.hessian_prior_mean = jnp.sum(gradient_trace_last * (phi_trace_last - mu_phi), axis=0) / jnp.sum(
        jnp.square(phi_trace_last - mu_phi), axis=0)
    experiment_tracker.hessian_prior_mean = jnp.abs(
        experiment_tracker.hessian_prior_mean) / experiment_args.sampling_rate
    experiment_tracker.hessian_prior_std = experiment_tracker.dpsgd_noise_scale * experiment_args.clipping_threshold / (
            jnp.sqrt(jnp.sum(jnp.square(phi_trace_last - mu_phi),
                             axis=0)) * gradient_scale_vector) / experiment_args.sampling_rate
    return experiment_tracker


def calculate_hessian_and_optimal_phi_priors(experiment_args, experiment_tracker):
    burn_in = int(experiment_tracker.phi_trace.shape[0] * experiment_args.trace_burn_in_percentage)
    phi_trace = experiment_tracker.phi_trace[burn_in:]
    experiment_tracker.phi_prior_mean = jnp.mean(phi_trace, axis=0)
    experiment_tracker.phi_prior_std = jnp.std(phi_trace, axis=0)
    return calculate_hessian_prior(experiment_args, experiment_tracker)
