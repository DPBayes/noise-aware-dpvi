import math
import os.path
import time
from argparse import ArgumentParser

import jax
import jax.numpy as jnp
import joblib
import numpy as np
import numpyro
from tqdm import tqdm

from model import get_positive_definite_matrix, per_example_kl_divergence, per_example_kl_divergence_gamma_exponential, \
    per_example_kl_divergence_beta_binomial, logistic_regression_theta_transform, gamma_exponential_theta_transform, \
    beta_binomial_theta_transform, per_example_kl_divergence_beta_bernoulli, \
    per_example_kl_divergence_dirichlet_categorical_full, \
    dirichlet_categorical_theta_transform_full, per_example_kl_divergence_linear_regression, \
    linear_regression_theta_transform
from noise_aware_dpsgd import calculate_estimates_from_mcmc_samples, perform_mcmc_sampling, \
    perform_dpsgd, calculate_hessian_and_optimal_phi_priors, perform_laplace_approximation, \
    calculate_estimates_from_laplace_approximation, EXPERIMENT_NAMES
from utils import DynamicAttributes, generate_data, generate_data_exponential, generate_data_binomial, \
    generate_data_bernoulli, generate_data_categorical, generate_data_linear_regression, \
    generate_data_linear_regression10d


def set_default_args_logistic_regression(experiment_args):
    experiment_args.random_seed = 984
    experiment_args.data_random_seed = 1234
    experiment_args.data_size = 10000
    experiment_args.training_data_size = 5000
    experiment_args.validation_data_size = 5000
    experiment_args.training_iterations = 10000
    experiment_args.target_epsilon = 0.25
    experiment_args.target_delta = 10 ** -5
    experiment_args.dpsgd_initial_mu_value = 0.0
    experiment_args.dpsgd_initial_pre_transformed_covariance_value = -6.0
    experiment_args.sampling_rate = 0.1
    experiment_args.clipping_threshold = 2.0
    experiment_args.mu_gradient_scale = 1.0
    experiment_args.pre_transformed_covariance_gradient_scale = 100.0
    experiment_args.dpsgd_mu_learning_rate = None
    experiment_args.dpsgd_pre_transformed_covariance_learning_rate = None
    experiment_args.mu_learning_rate_v = math.sqrt(2)
    experiment_args.pre_cov_learning_rate_v = math.sqrt(2)
    experiment_args.is_covariance_diagonal_matrix = True
    experiment_args.kl_divergence_mc_integration_samples_count = 10
    experiment_args.mcmc_inference_model = 'gradient_based'
    experiment_args.dpsgd_per_example_loss_function = per_example_kl_divergence
    experiment_args.use_mcmc_sampling = False
    experiment_args.mcmc_num_warmup = 1000
    experiment_args.mcmc_num_samples = 4000
    experiment_args.mcmc_num_chains = 1
    experiment_args.trace_burn_in_percentage = 0.2
    experiment_args.mcmc_mu_normal_prior_std = 10.0
    experiment_args.mcmc_covariance_normal_prior_std = 1.0
    experiment_args.custom_noise_scale = None
    experiment_args.inference_add_subsampling_noise = True
    experiment_args.theta_transform = logistic_regression_theta_transform
    experiment_args.laplace_approximation_learning_rate = 1.0
    experiment_args.laplace_approximation_iterations = 100000
    experiment_args.laplace_approximation_trace_averaging = True
    experiment_args.laplace_approximation_trace_averaging_burn_in = 0.8
    return experiment_args


def set_default_args_gamma_exponential(experiment_args):
    experiment_args.random_seed = 984
    experiment_args.data_random_seed = 1234
    experiment_args.data_size = 10000
    experiment_args.training_data_size = 5000
    experiment_args.validation_data_size = 5000
    experiment_args.training_iterations = 10000
    experiment_args.target_epsilon = 1.0
    experiment_args.target_delta = 10 ** -5
    experiment_args.sampling_rate = 0.1
    experiment_args.dpsgd_initial_mu_value = 0.0
    experiment_args.dpsgd_initial_pre_transformed_covariance_value = -4.0
    experiment_args.clipping_threshold = 2.0
    experiment_args.mu_gradient_scale = 1.0
    experiment_args.pre_transformed_covariance_gradient_scale = 100
    experiment_args.dpsgd_mu_learning_rate = None
    experiment_args.dpsgd_pre_transformed_covariance_learning_rate = None
    experiment_args.mu_learning_rate_v = math.sqrt(2)
    experiment_args.pre_cov_learning_rate_v = math.sqrt(2)
    experiment_args.is_covariance_diagonal_matrix = True
    experiment_args.kl_divergence_mc_integration_samples_count = 10
    experiment_args.mcmc_inference_model = 'gradient_based'
    experiment_args.dpsgd_per_example_loss_function = per_example_kl_divergence_gamma_exponential
    experiment_args.use_mcmc_sampling = True
    experiment_args.mcmc_num_warmup = 1000
    experiment_args.mcmc_num_samples = 4000
    experiment_args.mcmc_num_chains = 1
    experiment_args.trace_burn_in_percentage = 0.2
    experiment_args.mcmc_mu_normal_prior_std = 10.0
    experiment_args.mcmc_covariance_normal_prior_std = 1.0
    experiment_args.custom_noise_scale = None
    experiment_args.inference_add_subsampling_noise = True
    experiment_args.theta_transform = gamma_exponential_theta_transform
    experiment_args.laplace_approximation_learning_rate = 1.0
    experiment_args.laplace_approximation_iterations = 100000
    experiment_args.laplace_approximation_trace_averaging = True
    experiment_args.laplace_approximation_trace_averaging_burn_in = 0.8
    return experiment_args


def set_default_args_beta_bernoulli(experiment_args):
    experiment_args.random_seed = 984
    experiment_args.data_random_seed = 1234
    experiment_args.data_size = 10000
    experiment_args.training_data_size = 5000
    experiment_args.validation_data_size = 5000
    experiment_args.training_iterations = 10000
    experiment_args.target_epsilon = 1.0
    experiment_args.target_delta = 10 ** -5
    experiment_args.dpsgd_initial_mu_value = 0.0
    experiment_args.dpsgd_initial_pre_transformed_covariance_value = -4.0
    experiment_args.sampling_rate = 0.1
    experiment_args.clipping_threshold = 2.0
    experiment_args.mu_gradient_scale = 1.0
    experiment_args.pre_transformed_covariance_gradient_scale = 120.0
    experiment_args.dpsgd_mu_learning_rate = None
    experiment_args.dpsgd_pre_transformed_covariance_learning_rate = None
    experiment_args.mu_learning_rate_v = math.sqrt(2)
    experiment_args.pre_cov_learning_rate_v = math.sqrt(2)
    experiment_args.is_covariance_diagonal_matrix = True
    experiment_args.kl_divergence_mc_integration_samples_count = 10
    experiment_args.mcmc_inference_model = 'gradient_based'
    experiment_args.dpsgd_per_example_loss_function = per_example_kl_divergence_beta_bernoulli
    experiment_args.use_mcmc_sampling = True
    experiment_args.mcmc_num_warmup = 1000
    experiment_args.mcmc_num_samples = 4000
    experiment_args.mcmc_num_chains = 1
    experiment_args.trace_burn_in_percentage = 0.2
    experiment_args.mcmc_mu_normal_prior_std = 10.0
    experiment_args.mcmc_covariance_normal_prior_std = 1.0
    experiment_args.custom_noise_scale = None
    experiment_args.inference_add_subsampling_noise = True
    experiment_args.theta_transform = beta_binomial_theta_transform
    experiment_args.laplace_approximation_learning_rate = 1.0
    experiment_args.laplace_approximation_iterations = 100000
    experiment_args.laplace_approximation_trace_averaging = True
    experiment_args.laplace_approximation_trace_averaging_burn_in = 0.8
    return experiment_args


def set_default_args_beta_binomial(experiment_args):
    experiment_args.random_seed = 984
    experiment_args.data_random_seed = 1234
    experiment_args.data_size = 10000
    experiment_args.training_data_size = 5000
    experiment_args.validation_data_size = 5000
    experiment_args.training_iterations = 10000
    experiment_args.target_epsilon = 1.0
    experiment_args.target_delta = 10 ** -5
    experiment_args.dpsgd_initial_mu_value = 0.0
    experiment_args.dpsgd_initial_pre_transformed_covariance_value = -4.0
    experiment_args.sampling_rate = 0.1
    experiment_args.clipping_threshold = 50
    experiment_args.mu_gradient_scale = 1.0
    experiment_args.pre_transformed_covariance_gradient_scale = 5000
    experiment_args.dpsgd_mu_learning_rate = None
    experiment_args.dpsgd_pre_transformed_covariance_learning_rate = None
    experiment_args.mu_learning_rate_v = math.sqrt(2)
    experiment_args.pre_cov_learning_rate_v = math.sqrt(2)
    experiment_args.is_covariance_diagonal_matrix = True
    experiment_args.kl_divergence_mc_integration_samples_count = 10
    experiment_args.mcmc_inference_model = 'gradient_based'
    experiment_args.dpsgd_per_example_loss_function = per_example_kl_divergence_beta_binomial
    experiment_args.use_mcmc_sampling = True
    experiment_args.mcmc_num_warmup = 1000
    experiment_args.mcmc_num_samples = 4000
    experiment_args.mcmc_num_chains = 1
    experiment_args.trace_burn_in_percentage = 0.2
    experiment_args.mcmc_mu_normal_prior_std = 10.0
    experiment_args.mcmc_covariance_normal_prior_std = 1.0
    experiment_args.custom_noise_scale = None
    experiment_args.inference_add_subsampling_noise = True
    experiment_args.theta_transform = beta_binomial_theta_transform
    experiment_args.laplace_approximation_learning_rate = 1.0
    experiment_args.laplace_approximation_iterations = 100000
    experiment_args.laplace_approximation_trace_averaging = True
    experiment_args.laplace_approximation_trace_averaging_burn_in = 0.8
    return experiment_args


def set_default_args_dirichlet_categorical(experiment_args):
    experiment_args.random_seed = 984
    experiment_args.data_random_seed = 1234
    experiment_args.data_size = 10000
    experiment_args.training_data_size = 5000
    experiment_args.validation_data_size = 5000
    experiment_args.training_iterations = 10000
    experiment_args.target_epsilon = 1.0
    experiment_args.target_delta = 10 ** -5
    experiment_args.dpsgd_initial_mu_value = 0.0
    experiment_args.dpsgd_initial_pre_transformed_covariance_value = -6.0
    experiment_args.sampling_rate = 0.1
    experiment_args.clipping_threshold = math.sqrt(2)
    experiment_args.mu_gradient_scale = 1.0
    experiment_args.pre_transformed_covariance_gradient_scale = 100.0
    experiment_args.dpsgd_mu_learning_rate = None
    experiment_args.dpsgd_pre_transformed_covariance_learning_rate = None
    experiment_args.mu_learning_rate_v = math.sqrt(2)
    experiment_args.pre_cov_learning_rate_v = math.sqrt(2)
    experiment_args.is_covariance_diagonal_matrix = True
    experiment_args.kl_divergence_mc_integration_samples_count = 10
    experiment_args.mcmc_inference_model = 'gradient_based'
    experiment_args.dpsgd_per_example_loss_function = per_example_kl_divergence_dirichlet_categorical_full
    experiment_args.use_mcmc_sampling = True
    experiment_args.mcmc_num_warmup = 1000
    experiment_args.mcmc_num_samples = 4000
    experiment_args.mcmc_num_chains = 1
    experiment_args.trace_burn_in_percentage = 0.2
    experiment_args.mcmc_mu_normal_prior_std = 10.0
    experiment_args.mcmc_covariance_normal_prior_std = 1.0
    experiment_args.custom_noise_scale = None
    experiment_args.inference_add_subsampling_noise = True
    experiment_args.theta_transform = dirichlet_categorical_theta_transform_full
    experiment_args.laplace_approximation_learning_rate = 1.0
    experiment_args.laplace_approximation_iterations = 100000
    experiment_args.laplace_approximation_trace_averaging = True
    experiment_args.laplace_approximation_trace_averaging_burn_in = 0.8
    return experiment_args


"""
For higher epsilon this was better: (For laplace's approximation though the default in the function was better for all epsilons)
 'dpsgd_initial_mu_value': Array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1., -4.],      dtype=float32),
 'dpsgd_initial_pre_transformed_covariance_value': -6.0,
 'sampling_rate': 0.1,
 'clipping_threshold': 260.0,
 'mu_gradient_scale': Array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], dtype=float32),
 'pre_transformed_covariance_gradient_scale': Array([1500., 1500., 1500., 1500., 1500., 1500., 1500., 1500., 1500.,
        1500., 1500., 1500.], dtype=float32),
 'dpsgd_mu_learning_rate': None,
 'dpsgd_pre_transformed_covariance_learning_rate': None,
 'mu_learning_rate_v': Array([1.4142135 , 1.4142135 , 1.4142135 , 1.4142135 , 1.4142135 ,
        1.4142135 , 1.4142135 , 1.4142135 , 1.4142135 , 1.4142135 ,
        1.4142135 , 0.20203051], dtype=float32),
 'pre_cov_learning_rate_v': 5.656854249492381,
"""


#
def set_default_args_linear_regression10d(experiment_args):
    experiment_args.random_seed = 984
    experiment_args.data_random_seed = 1234
    experiment_args.data_size = 10000
    experiment_args.training_data_size = 5000
    experiment_args.validation_data_size = 5000
    experiment_args.training_iterations = 10000
    experiment_args.target_epsilon = 1.0
    experiment_args.target_delta = 10 ** -5
    experiment_args.dpsgd_initial_mu_value = jnp.concatenate(
        [jnp.ones((11,), dtype=jnp.float32), jnp.array([-4.0], dtype=jnp.float32)])
    experiment_args.dpsgd_initial_pre_transformed_covariance_value = -6.0
    experiment_args.sampling_rate = 0.1
    experiment_args.clipping_threshold = 260.0  # 260, 350, 500
    experiment_args.mu_gradient_scale = jnp.concatenate(
        [jnp.ones((10,), dtype=jnp.float32) * 1.0,
         jnp.array([1.0, 1.0], jnp.float32)])
    experiment_args.pre_transformed_covariance_gradient_scale = jnp.concatenate(
        [jnp.ones((10,), dtype=jnp.float32) * 1500.0,
         jnp.array([1500.0, 1500.0], jnp.float32)])
    # experiment_args.pre_transformed_covariance_gradient_scale = 1000.0
    experiment_args.dpsgd_mu_learning_rate = None
    experiment_args.dpsgd_pre_transformed_covariance_learning_rate = None
    # experiment_args.mu_learning_rate_v = jnp.concatenate([jnp.ones((10,), dtype=jnp.float32) * math.sqrt(2),
    #                                                       jnp.array([math.sqrt(2), math.sqrt(2) / 14],
    #                                                                 jnp.float32)]) * 4
    experiment_args.mu_learning_rate_v = jnp.concatenate([jnp.ones((10,), dtype=jnp.float32) * math.sqrt(2),
                                                          jnp.array([math.sqrt(2), math.sqrt(2) / 28],
                                                                    jnp.float32)]) * 4
    # experiment_args.pre_cov_learning_rate_v = math.sqrt(2)
    # experiment_args.mu_learning_rate_v = math.sqrt(2)
    experiment_args.pre_cov_learning_rate_v = math.sqrt(2) * 4
    experiment_args.is_covariance_diagonal_matrix = True
    experiment_args.kl_divergence_mc_integration_samples_count = 10
    experiment_args.mcmc_inference_model = 'gradient_based'
    experiment_args.dpsgd_per_example_loss_function = per_example_kl_divergence_linear_regression
    experiment_args.use_mcmc_sampling = True
    experiment_args.mcmc_num_warmup = 1000
    experiment_args.mcmc_num_samples = 4000
    experiment_args.mcmc_num_chains = 1
    experiment_args.trace_burn_in_percentage = 0.2
    experiment_args.mcmc_mu_normal_prior_std = 10.0
    experiment_args.mcmc_covariance_normal_prior_std = 1.0
    experiment_args.custom_noise_scale = None
    experiment_args.inference_add_subsampling_noise = True
    experiment_args.theta_transform = linear_regression_theta_transform
    experiment_args.laplace_approximation_learning_rate = 1.0
    experiment_args.laplace_approximation_iterations = 100000
    experiment_args.laplace_approximation_trace_averaging = True
    experiment_args.laplace_approximation_trace_averaging_burn_in = 0.8
    return experiment_args


def set_default_args_linear_regression(experiment_args):
    experiment_args.random_seed = 984
    experiment_args.data_random_seed = 1234
    experiment_args.data_size = 10000
    experiment_args.training_data_size = 5000
    experiment_args.validation_data_size = 5000
    experiment_args.training_iterations = 10000
    experiment_args.target_epsilon = 1.0
    experiment_args.target_delta = 10 ** -5
    experiment_args.dpsgd_initial_mu_value = jnp.concatenate(
        [jnp.ones((11,), dtype=jnp.float32), jnp.array([-4.0], dtype=jnp.float32)])
    experiment_args.dpsgd_initial_pre_transformed_covariance_value = -6.0
    experiment_args.sampling_rate = 0.1
    experiment_args.clipping_threshold = 260.0  # 260, 350, 500
    experiment_args.mu_gradient_scale = jnp.concatenate(
        [jnp.ones((10,), dtype=jnp.float32) * 1.0,
         jnp.array([1.0, 1.0], jnp.float32)])
    experiment_args.pre_transformed_covariance_gradient_scale = jnp.concatenate(
        [jnp.ones((10,), dtype=jnp.float32) * 1500.0,
         jnp.array([1500.0, 750.0], jnp.float32)])
    # experiment_args.pre_transformed_covariance_gradient_scale = 1000.0
    experiment_args.dpsgd_mu_learning_rate = None
    experiment_args.dpsgd_pre_transformed_covariance_learning_rate = None
    experiment_args.mu_learning_rate_v = jnp.concatenate([jnp.ones((10,), dtype=jnp.float32) * math.sqrt(2),
                                                          jnp.array([math.sqrt(2), math.sqrt(2) / 3],
                                                                    jnp.float32)])
    # experiment_args.pre_cov_learning_rate_v = math.sqrt(2)
    # experiment_args.mu_learning_rate_v = math.sqrt(2)
    experiment_args.pre_cov_learning_rate_v = math.sqrt(2) * 4
    experiment_args.is_covariance_diagonal_matrix = True
    experiment_args.kl_divergence_mc_integration_samples_count = 10
    experiment_args.mcmc_inference_model = 'gradient_based'
    experiment_args.dpsgd_per_example_loss_function = per_example_kl_divergence_linear_regression
    experiment_args.use_mcmc_sampling = True
    experiment_args.mcmc_num_warmup = 1000
    experiment_args.mcmc_num_samples = 4000
    experiment_args.mcmc_num_chains = 1
    experiment_args.trace_burn_in_percentage = 0.2
    experiment_args.mcmc_mu_normal_prior_std = 10.0
    experiment_args.mcmc_covariance_normal_prior_std = 1.0
    experiment_args.custom_noise_scale = None
    experiment_args.inference_add_subsampling_noise = True
    experiment_args.theta_transform = linear_regression_theta_transform
    experiment_args.laplace_approximation_learning_rate = 1.0
    experiment_args.laplace_approximation_iterations = 100000
    experiment_args.laplace_approximation_trace_averaging = True
    experiment_args.laplace_approximation_trace_averaging_burn_in = 0.8
    return experiment_args


def set_linear_regression_alignment_params(experiment_args):
    experiment_args.mu_gradient_scale = jnp.concatenate(
        [jnp.ones((1,), dtype=jnp.float32) * 1.0,
         jnp.array([1.0, 10.0], jnp.float32)])
    experiment_args.pre_transformed_covariance_gradient_scale = jnp.concatenate(
        [jnp.ones((1,), dtype=jnp.float32) * 1500.0,
         jnp.array([1500.0, 2500.0], jnp.float32)])
    return experiment_args


def set_linear_regression_alignment_params10d(experiment_args):
    experiment_args.mu_gradient_scale = 1000.0
    return experiment_args


def parse_program_args():
    parser = ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--target_epsilon', type=float, default=1.0, help='DP Epsilon')
    parser.add_argument('--task_id', type=int, default=0, help='Task ID')
    parser.add_argument('--theta_random_seed', type=int, default=666, help='Task ID')
    parser.add_argument('--debug_output_directory', type=str, help='Debug Output Directory')
    parser.add_argument('--output_directory', type=str, help='Output Directory')
    parser.add_argument('--use_mcmc_sampling', type=bool, default=False, help='Use MCMC Sampling')
    parser.add_argument('--experiment_name', type=str, default='logistic_regression',
                        help='One of the following: ' + ','.join(EXPERIMENT_NAMES))
    program_args, _ = parser.parse_known_args()
    return program_args


def perform_experiment(program_args, task_id):
    experiment_args = DynamicAttributes()
    experiment_name = program_args.experiment_name
    experiment_args.experiment_name = experiment_name
    if experiment_name == 'logistic_regression':
        experiment_args = set_default_args_logistic_regression(experiment_args)
        theta_true = numpyro.distributions.MultivariateNormal(jnp.zeros(3), jnp.eye(3)).sample(
            key=jax.random.PRNGKey(program_args.theta_random_seed + task_id))
        experiment_args.data_generator_function = generate_data
    elif experiment_name == 'gamma_exponential':
        experiment_args = set_default_args_gamma_exponential(experiment_args)
        theta_true = numpyro.distributions.Gamma(8.0, 2.0).sample(
            key=jax.random.PRNGKey(program_args.theta_random_seed + task_id),
            sample_shape=(1,))
        experiment_args.data_generator_function = generate_data_exponential
    elif experiment_name == 'beta_bernoulli':
        experiment_args = set_default_args_beta_bernoulli(experiment_args)
        theta_true = numpyro.distributions.Beta(10.0, 10.0).sample(
            key=jax.random.PRNGKey(program_args.theta_random_seed + task_id),
            sample_shape=(1,))
        experiment_args.data_generator_function = generate_data_bernoulli
    elif experiment_name == 'beta_binomial':
        experiment_args = set_default_args_beta_binomial(experiment_args)
        theta_true = numpyro.distributions.Beta(10.0, 10.0).sample(
            key=jax.random.PRNGKey(program_args.theta_random_seed + task_id),
            sample_shape=(1,))
        experiment_args.data_generator_function = generate_data_binomial
    elif experiment_name == 'dirichlet_categorical':
        experiment_args.data_generator_function = generate_data_categorical
        theta_true = numpyro.distributions.Dirichlet(np.ones((3,), np.float32) * 5.0).sample(
            key=jax.random.PRNGKey(program_args.theta_random_seed + task_id),
            sample_shape=(1,))[0, 0:-1]
        experiment_args.theta_true = theta_true
        experiment_args = set_default_args_dirichlet_categorical(experiment_args)
    elif experiment_name == 'linear_regression':
        experiment_args.data_generator_function = generate_data_linear_regression
        theta_w_prng_key, sigma_squared_prng_key = jax.random.split(
            jax.random.PRNGKey(program_args.theta_random_seed + task_id))
        sigma_squared = numpyro.distributions.InverseGamma(20.0, 0.5).sample(key=sigma_squared_prng_key)
        dims = 2
        theta_w_cov = sigma_squared * jnp.linalg.inv(jnp.eye(dims) * dims / 40.0)
        theta_w = numpyro.distributions.MultivariateNormal(jnp.zeros((dims,), dtype=jnp.float32), theta_w_cov).sample(
            key=theta_w_prng_key)
        theta_true = jnp.concatenate([theta_w, jnp.array([sigma_squared], dtype=jnp.float32)],
                                     axis=0)
        experiment_args = set_default_args_linear_regression(experiment_args)
        experiment_args.target_epsilon = program_args.target_epsilon
        # experiment_args = set_linear_regression_alignment_params(experiment_args)
    elif experiment_name == 'linear_regression10d':
        experiment_args.data_generator_function = generate_data_linear_regression10d
        theta_w_prng_key, sigma_squared_prng_key = jax.random.split(
            jax.random.PRNGKey(program_args.theta_random_seed + task_id))
        sigma_squared = numpyro.distributions.InverseGamma(20.0, 0.5).sample(key=sigma_squared_prng_key)
        dims = 11
        theta_w_cov = sigma_squared * jnp.linalg.inv(jnp.eye(dims) * (dims - 1) / 40.0)
        theta_w = numpyro.distributions.MultivariateNormal(jnp.zeros((dims,), dtype=jnp.float32), theta_w_cov).sample(
            key=theta_w_prng_key)
        theta_true = jnp.concatenate([theta_w, jnp.array([sigma_squared], dtype=jnp.float32)],
                                     axis=0)
        experiment_args = set_default_args_linear_regression10d(experiment_args)
        experiment_args.target_epsilon = program_args.target_epsilon
    else:
        raise ValueError('Unknown experiment name')
    experiment_args.theta_random_seed = program_args.theta_random_seed
    experiment_args.use_mcmc_sampling = program_args.use_mcmc_sampling
    experiment_args.target_epsilon = program_args.target_epsilon
    experiment_args.task_id = task_id
    experiment_args.data_random_seed += task_id
    experiment_args.random_seed += task_id
    experiment_args.theta_true = theta_true
    experiment_tracker, experiment_tracker_without_debug_info = perform_coverage_test(experiment_args)
    inference_method = 'laplace'
    if experiment_args.use_mcmc_sampling:
        inference_method = 'mcmc'
    joblib.dump(experiment_tracker,
                os.path.join(program_args.debug_output_directory,
                             f'experiment_{experiment_name}_{inference_method}_{program_args.target_epsilon}_{program_args.task_id}.pkl'),
                compress=('xz', 9))
    joblib.dump(experiment_tracker_without_debug_info,
                os.path.join(program_args.output_directory,
                             f'experiment_{experiment_name}_{inference_method}_{program_args.target_epsilon}_{program_args.task_id}.pkl'),
                compress=('xz', 9))


def perform_coverage_test(experiment_args):
    experiment_args.algorithm_prng_key = jax.random.PRNGKey(experiment_args.random_seed)
    experiment_args.data_prng_key = jax.random.PRNGKey(experiment_args.data_random_seed)

    experiment_args.xs, experiment_args.ys = experiment_args.data_generator_function(experiment_args.theta_true,
                                                                                     prng_key=experiment_args.data_prng_key,
                                                                                     N=experiment_args.data_size)

    experiment_args.train_xs = experiment_args.xs[:experiment_args.training_data_size]
    experiment_args.train_ys = experiment_args.ys[:experiment_args.training_data_size]
    experiment_args.validation_xs = experiment_args.xs[experiment_args.training_data_size:]
    experiment_args.validation_ys = experiment_args.ys[experiment_args.training_data_size:]
    experiment_args.theta_dimension = experiment_args.theta_true.shape[0]

    experiment_tracker = DynamicAttributes()
    experiment_tracker_without_debug_info = DynamicAttributes()

    experiment_tracker.experiment_args = experiment_args
    experiment_tracker_without_debug_info.experiment_args = experiment_args
    experiment_tracker = perform_dpsgd(experiment_args, experiment_tracker)
    experiment_tracker_without_debug_info.mu_trace = experiment_tracker.mu_trace
    experiment_tracker_without_debug_info.phi_trace = experiment_tracker.phi_trace
    experiment_tracker_without_debug_info.dpsgd_noise_scale = experiment_tracker.dpsgd_noise_scale

    experiment_tracker = calculate_hessian_and_optimal_phi_priors(experiment_args, experiment_tracker)
    inference_time = time.time()
    if experiment_args.use_mcmc_sampling:
        experiment_tracker = perform_mcmc_sampling(experiment_args, experiment_tracker)
    else:
        experiment_tracker = perform_laplace_approximation(experiment_args, experiment_tracker)
    inference_time = time.time() - inference_time
    experiment_tracker.inference_time = inference_time
    experiment_tracker_without_debug_info.inference_time = inference_time

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
        experiment_tracker_without_debug_info.optimal_phi_samples = experiment_tracker.optimal_phi_samples
        experiment_tracker_without_debug_info.estimated_mu = experiment_tracker.estimated_mu
        experiment_tracker_without_debug_info.estimated_covariance = experiment_tracker.estimated_covariance
    else:
        experiment_tracker.estimated_mu, experiment_tracker.estimated_covariance = calculate_estimates_from_laplace_approximation(
            experiment_tracker.laplace_approximation_posterior_mode,
            experiment_tracker.laplace_approximation_posterior_covariance,
            experiment_args.theta_dimension)
        experiment_tracker_without_debug_info.estimated_mu = experiment_tracker.estimated_mu
        experiment_tracker_without_debug_info.estimated_covariance = experiment_tracker.estimated_covariance

    experiment_tracker.theta_stds = jnp.sqrt(jnp.diagonal(experiment_tracker.estimated_covariance))
    experiment_tracker.theta_means = experiment_tracker.estimated_mu
    experiment_tracker_without_debug_info.theta_stds = experiment_tracker.theta_stds
    experiment_tracker_without_debug_info.theta_means = experiment_tracker.theta_means

    if experiment_args.use_mcmc_sampling:
        theta_dimension = experiment_args.theta_dimension
        covariance_optimal_phi_samples = np.zeros((experiment_tracker.optimal_phi_samples.shape[0], theta_dimension),
                                                  np.float32)
        for i in range(experiment_tracker.optimal_phi_samples.shape[0]):
            covariance_optimal_phi_samples[i] = np.sqrt(np.diagonal(get_positive_definite_matrix(
                experiment_tracker.optimal_phi_samples[i, theta_dimension:],
                theta_dimension,
                True)))
        experiment_tracker.covariance_optimal_phi_samples = covariance_optimal_phi_samples
        experiment_tracker_without_debug_info.covariance_optimal_phi_samples = covariance_optimal_phi_samples

    return experiment_tracker, experiment_tracker_without_debug_info


def local_main():
    program_args = DynamicAttributes()
    program_args.target_epsilon = 0.1
    program_args.theta_random_seed = 666
    program_args.debug_output_directory = './temp/output-debug'
    if not os.path.isdir(program_args.debug_output_directory):
        os.makedirs(program_args.debug_output_directory)
    program_args.output_directory = './temp/output'
    if not os.path.isdir(program_args.output_directory):
        os.makedirs(program_args.output_directory)
    program_args.use_mcmc_sampling = False
    program_args.experiment_name = 'linear_regression10d'

    progress_bar = tqdm(total=20)
    for task_id in range(1, 21):
        program_args.task_id = task_id
        perform_experiment(program_args, task_id)
        progress_bar.update()
    progress_bar.close()


def main():
    program_args = parse_program_args()
    task_id = program_args.task_id
    perform_experiment(program_args, task_id)


if __name__ == '__main__':
    main()
