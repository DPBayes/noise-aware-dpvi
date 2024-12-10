import os
import types

import jax.numpy as jnp
import jax.random
import joblib
import numpy as np
import numpyro.distributions as dist
from matplotlib import pyplot as plt
from scipy import stats


def generate_data(weights, prng_key, min_x=-1, max_x=1, N=1000):
    xs_prng_key, ys_prng_key = jax.random.split(prng_key, 2)
    xs = dist.Uniform(low=min_x, high=max_x).sample(key=xs_prng_key,
                                                    sample_shape=(N, weights.shape[0]))
    logits = np.sum(xs * weights, axis=-1)
    ys = 1.0 / (1.0 + np.exp(-1.0 * logits))
    ys = dist.Bernoulli(probs=ys).sample(key=ys_prng_key)
    return xs, ys


def generate_data_exponential(exp_lambda, prng_key, N=5000):
    return (dist.Exponential(exp_lambda).sample(key=prng_key,
                                                sample_shape=(N,)),
            jnp.zeros((N,), dtype=jnp.float32))


def generate_data_bernoulli(prob, prng_key, N=5000):
    return (dist.Bernoulli(prob).sample(key=prng_key,
                                        sample_shape=(N,)),
            jnp.zeros((N,), dtype=jnp.float32))


def generate_data_binomial(prob, prng_key, trials=1000, N=5000):
    return (dist.Binomial(trials, prob).sample(key=prng_key,
                                               sample_shape=(N,)),
            jnp.zeros((N,), dtype=jnp.float32))


def generate_data_categorical(probs, prng_key, N=5000):
    full_probs = jnp.concatenate([probs, 1 - jnp.sum(probs, keepdims=True)])
    return (dist.Categorical(full_probs).sample(key=prng_key,
                                                sample_shape=(N,)),
            jnp.zeros((N,), dtype=jnp.float32))


def generate_data_categorical_full(probs, prng_key, N=5000):
    return (dist.Categorical(probs).sample(key=prng_key,
                                           sample_shape=(N,)),
            jnp.zeros((N,), dtype=jnp.float32))


def generate_data_linear_regression10d(theta,
                                       prng_key,
                                       # data_params=(jnp.array([0.0, 0.0]), jnp.eye(2, dtype=jnp.float32)),
                                       # data_params=(jnp.array([0.0]), jnp.eye(1, dtype=jnp.float32)),
                                       data_params=(
                                               jnp.zeros((10,), dtype=jnp.float32), jnp.eye(10, dtype=jnp.float32)),
                                       N=5000):
    # dim_theta = dim_data + 1 (bias) + 1 (sigma_squared)
    # data_params: mu_x, cov_x (has to be of dimension dim_theta - 2)
    xs_prng_key, ys_prng_key = jax.random.split(prng_key, 2)
    # data_cov = stats.invwishart(50, np.eye(10)).rvs(size=1)
    data_cov = data_params[1]
    xs = dist.MultivariateNormal(data_params[0], data_cov).sample(prng_key, sample_shape=(N,))
    # xs = dist.Uniform(-1, 1).sample(xs_prng_key, sample_shape=(N, 10))
    xs = jnp.concatenate([xs, jnp.ones((N, 1), dtype=jnp.float32)], axis=-1)
    theta_weights = theta[:-1].reshape(1, -1)
    theta_sigma_squared = theta[-1]
    sigma_noise = dist.Normal(0.0, jnp.sqrt(theta_sigma_squared)).sample(ys_prng_key, sample_shape=(N,))
    ys = jnp.sum(theta_weights * xs, axis=-1) + sigma_noise

    # plt.scatter(xs[:, 0], ys)
    # plt.scatter(xs[:, 0], jnp.sum(theta_weights * xs, axis=-1))
    # plt.show()
    return xs, ys


def generate_data_linear_regression(theta,
                                    prng_key,
                                    # data_params=(jnp.array([0.0, 0.0]), jnp.eye(2, dtype=jnp.float32)),
                                    # data_params=(jnp.array([0.0]), jnp.eye(1, dtype=jnp.float32)),
                                    data_params=(jnp.zeros((1,), dtype=jnp.float32), jnp.eye(1, dtype=jnp.float32)),
                                    N=5000):
    # dim_theta = dim_data + 1 (bias) + 1 (sigma_squared)
    # data_params: mu_x, cov_x (has to be of dimension dim_theta - 2)
    xs_prng_key, ys_prng_key = jax.random.split(prng_key, 2)
    xs = dist.MultivariateNormal(data_params[0], data_params[1]).sample(prng_key, sample_shape=(N,))
    # xs = dist.Uniform(-1, 1).sample(prng_key, sample_shape=(N, 1))
    xs = jnp.concatenate([xs, jnp.ones((N, 1), dtype=jnp.float32)], axis=-1)
    theta_weights = theta[:-1].reshape(1, -1)
    theta_sigma_squared = theta[-1]
    sigma_noise = dist.Normal(0.0, jnp.sqrt(theta_sigma_squared)).sample(ys_prng_key, sample_shape=(N,))
    ys = jnp.sum(theta_weights * xs, axis=-1) + sigma_noise

    # plt.scatter(xs[:, 0], ys)
    # plt.scatter(xs[:, 0], jnp.sum(theta_weights * xs, axis=-1))
    # plt.show()
    return xs, ys


# def generate_data(weights, prng_key, min_x=-1, max_x=1, N=1000):
#     slack = 0
#     xs1 = dist.Uniform(low=min_x, high=-slack).sample(key=prng_key,
#                                                       sample_shape=(N // 2, weights.shape[0]))
#     xs2 = dist.Uniform(low=slack, high=max_x).sample(key=prng_key,
#                                                      sample_shape=(N // 2, weights.shape[0]))
#     xs = np.concatenate([xs1, xs2], axis=0)
#     logits = np.sum(xs * weights, axis=-1)
#     ys = 1.0 / (1.0 + np.exp(-1.0 * logits))
#     ys = dist.Bernoulli(probs=ys).sample(key=prng_key)
#     return xs, ys


def plot_trace(trace,
               results_path,
               file_name,
               figure_height):
    n = trace.shape[1]
    fig, axs = plt.subplots(n, 1)
    fig.set_figheight(figure_height)
    if n > 1:
        for i in range(n):
            axs[i].plot(trace[:, i])
    else:
        axs.plot(trace)
    plt.savefig(os.path.join(results_path, file_name + '.png'))
    plt.close('all')


def plot_trace_show(trace,
                    figure_height):
    n = trace.shape[1]
    fig, axs = plt.subplots(n, 1)
    fig.set_figheight(figure_height)
    if n > 1:
        for i in range(n):
            axs[i].plot(trace[:, i])
    else:
        axs.plot(trace)
    plt.show()


def plot_distribution(trace,
                      results_path,
                      file_name,
                      figure_height):
    if len(trace.shape) == 1:
        n = 1
    else:
        n = trace.shape[1]
    fig, axs = plt.subplots(n, 1)
    fig.set_figheight(figure_height)
    if n > 1:
        for i in range(n):
            x = trace[:, i]
            counts, bins = np.histogram(x, bins='auto')
            axs[i].stairs(counts, bins)
    else:
        counts, bins = np.histogram(trace, bins='auto')
        axs.stairs(counts, bins)
    plt.savefig(os.path.join(results_path, file_name + '.png'))
    plt.close('all')


def display_mcmc_estimation_results(mu_optimal_phi_samples,
                                    pre_transformed_cov_optimal_phi_samples,
                                    cov_opt_trace_flattened,
                                    estimated_mu,
                                    estimated_covariance,
                                    mean_cov,
                                    mu_sample_covariance,
                                    results_path,
                                    main_log_path,
                                    experiment_name,
                                    save_plots,
                                    figure_height):
    if save_plots:
        plot_trace(mu_optimal_phi_samples,
                   results_path,
                   f'{experiment_name}_mu_mcmc_trace',
                   figure_height)
        plot_trace(pre_transformed_cov_optimal_phi_samples,
                   results_path,
                   f'{experiment_name}_pre_transformed_covariance_mcmc_trace',
                   figure_height)
        # plot_distribution(mu_optimal_phi_samples,
        #                   results_path,
        #                   f'{experiment_name}_mu_mcmc_distribution',
        #                   figure_height)
        # plot_distribution(pre_transformed_cov_optimal_phi_samples,
        #                   results_path,
        #                   f'{experiment_name}_pre_transformed_covariance_mcmc_distribution',
        #                   figure_height)
    update_log(main_log_path, 'estimated mu:\n')
    update_log(main_log_path, f'{estimated_mu}\n')
    update_log(main_log_path, '------------------------------\n')
    update_log(main_log_path, 'estimated cov:\n')
    update_log(main_log_path, 'E[Cov[param]]:\n')
    update_log(main_log_path, f'{mean_cov}\n')
    update_log(main_log_path, 'Cov[E[param]]:\n')
    update_log(main_log_path, f'{mu_sample_covariance}\n')
    update_log(main_log_path, 'total variation:\n')
    update_log(main_log_path, f'{estimated_covariance}\n')
    update_log(main_log_path, '------------------------------\n')


def update_log(file_path, text):
    with open(file_path, 'a') as file:
        file.write(text)


def display_laplace_approximation_results(estimated_mu,
                                          estimated_covariance,
                                          main_log_path):
    update_log(main_log_path, 'estimated mu:\n')
    update_log(main_log_path, f'{estimated_mu}\n')
    update_log(main_log_path, '------------------------------\n')
    update_log(main_log_path, 'estimated cov:\n')
    # update_log(main_log_path, 'E[Cov[param]]:\n')
    # update_log(main_log_path, f'{mean_cov}\n')
    # update_log(main_log_path, 'Cov[E[param]]:\n')
    # update_log(main_log_path, f'{mu_sample_covariance}\n')
    update_log(main_log_path, 'total variation:\n')
    update_log(main_log_path, f'{estimated_covariance}\n')
    update_log(main_log_path, '------------------------------\n')


def log_results(experiment_args,
                experiment_tracker,
                results_path,
                main_log_path,
                inference_off=False,
                save_plots=False,
                figure_height=8):
    update_log(main_log_path, 'real last_params:\n')
    update_log(main_log_path, f'{experiment_args.theta_transform(experiment_args.theta_true)}\n')
    update_log(main_log_path, '------------------------------\n')
    # update_log(main_log_path, 'log-likelihood of real parameter:\n')
    # update_log(main_log_path, f'{experiment_tracker.real_parameter_log_likelihood}\n')
    # update_log(main_log_path, '------------------------------\n')
    # update_log(main_log_path, 'mle logistic regression estimation:\n')
    # update_log(main_log_path, f'{experiment_tracker.mle_logistic_regression_estimation}\n')
    # update_log(main_log_path, '------------------------------\n')
    update_log(main_log_path, 'last params:\n')
    update_log(main_log_path, f'{experiment_tracker.last_phi}\n')
    update_log(main_log_path, '------------------------------\n')
    update_log(main_log_path, 'last mu:\n')
    update_log(main_log_path, f"{experiment_tracker.last_phi['mu']}\n")
    update_log(main_log_path, '------------------------------\n')
    update_log(main_log_path, 'last cov:\n')
    update_log(main_log_path, f'{experiment_tracker.last_covariance}\n')
    update_log(main_log_path, '------------------------------\n')
    # update_log(main_log_path, '------------------------------\n')
    update_log(main_log_path, 'mean mu:\n')
    update_log(main_log_path,
               f"{jnp.mean(experiment_tracker.phi_trace[2000:, :experiment_args.theta_dimension], axis=0)}\n")
    update_log(main_log_path, '------------------------------\n')
    update_log(main_log_path, 'mean cov:\n')
    update_log(main_log_path,
               f'{jnp.mean(experiment_tracker.phi_trace[2000:, experiment_args.theta_dimension:], axis=0)}\n')
    update_log(main_log_path, '------------------------------\n')

    if save_plots:
        plot_trace(experiment_tracker.mu_trace,
                   results_path,
                   'mu_trace',
                   figure_height)
        plot_trace(experiment_tracker.mu_perturbed_gradient_trace,
                   results_path,
                   'mu_grad_trace',
                   figure_height)
        plot_trace(experiment_tracker.mu_median_grad_norm_trace[2000:],
                   results_path,
                   'mu_median_grad_norm_trace',
                   figure_height)
        plot_trace(experiment_tracker.mu_mean_grad_norm_trace[2000:],
                   results_path,
                   'mu_mean_grad_norm_trace',
                   figure_height)
        plot_trace(experiment_tracker.mu_median_abs_grad_trace[2000:],
                   results_path,
                   'mu_median_abs_grad_trace',
                   figure_height)
        plot_trace(experiment_tracker.mu_mean_abs_grad_trace[2000:],
                   results_path,
                   'mu_mean_abs_grad_trace',
                   figure_height)

    if save_plots:
        plot_trace(experiment_tracker.pre_transformed_covariance_trace,
                   results_path,
                   'pre_transformed_covariance_trace',
                   figure_height)
        plot_trace(experiment_tracker.pre_transformed_covariance_perturbed_gradient_trace,
                   results_path,
                   'pre_transformed_covariance_perturbed_gradient_trace',
                   figure_height)
        plot_trace(experiment_tracker.pre_cov_median_grad_norm_trace[2000:],
                   results_path,
                   'pre_cov_median_grad_norm_trace',
                   figure_height)
        plot_trace(experiment_tracker.pre_cov_mean_grad_norm_trace[2000:],
                   results_path,
                   'pre_cov_mean_grad_norm_trace',
                   figure_height)
        plot_trace(experiment_tracker.pre_cov_median_abs_grad_norm_trace[2000:],
                   results_path,
                   'pre_cov_median_abs_grad_norm_trace',
                   figure_height)
        plot_trace(experiment_tracker.pre_cov_mean_abs_grad_norm_trace[2000:],
                   results_path,
                   'pre_cov_mean_abs_grad_norm_trace',
                   figure_height)

    if inference_off:
        return
    update_log(main_log_path,
               '================================================================================\n')
    if experiment_args.use_mcmc_sampling:
        display_mcmc_estimation_results(experiment_tracker.mu_optimal_phi_samples,
                                        experiment_tracker.pre_transformed_covariance_optimal_phi_samples,
                                        experiment_tracker.covariance_optimal_phi_trace,
                                        experiment_tracker.estimated_mu,
                                        experiment_tracker.estimated_covariance,
                                        experiment_tracker.mean_covariance,
                                        experiment_tracker.mu_sample_covariance,
                                        results_path,
                                        main_log_path,
                                        experiment_args.mcmc_inference_model,
                                        save_plots,
                                        figure_height)
    else:
        display_laplace_approximation_results(experiment_tracker.estimated_mu,
                                              experiment_tracker.estimated_covariance,
                                              main_log_path)
    update_log(main_log_path, f'confidence alpha: {experiment_args.confidence_alpha}\n')
    update_log(main_log_path,
               f'squared mahalanobis distance: {experiment_tracker.squared_mahalanobis_distance}\n')
    update_log(main_log_path, f'test radius: {experiment_tracker.test_radius}\n')
    if experiment_tracker.is_within_confidence_region_result:
        decision = 'is'
    else:
        decision = 'is not'
    update_log(main_log_path,
               f'theta_true = {experiment_args.theta_true} {decision} within the confidence region determined by'
               f' the estimated mean and variance\n')
    update_log(main_log_path, '------------------------------\n')
    # update_log(main_log_path,
    #            f'average log-likelihood of estimated parameter: {experiment_tracker.average_log_likelihood}\n')
    # update_log(main_log_path, '------------------------------\n')
    # update_log(main_log_path, 'parameters kl-divergence:\n')
    # update_log(main_log_path, f'{experiment_tracker.parameters_kl_divergence}\n')


class DynamicAttributes:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self):
        return self.__dict__

    def pretty_print(self):
        print('{')
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray) or isinstance(value, jnp.ndarray) or isinstance(value, types.FunctionType):
                continue
            print(f'    {key}: {value},')
        print('}')


def find_file_name_with_same_arguments_dict(file_name_arguments_dict_path, current_arguments_dict):
    file_name_with_same_arguments = None
    if os.path.isfile(file_name_arguments_dict_path):
        file_name_arguments_dict = joblib.load(file_name_arguments_dict_path)

        for file_name in file_name_arguments_dict:
            arguments_dict = file_name_arguments_dict[file_name]

            if len(arguments_dict) != len(current_arguments_dict):
                continue

            are_equal = True
            for key in arguments_dict:
                if key not in current_arguments_dict:
                    are_equal = False
                    break

                if type(arguments_dict[key]) is not type(current_arguments_dict[key]):
                    are_equal = False
                    break

                if isinstance(arguments_dict[key], float):
                    are_equal = abs(arguments_dict[key] - current_arguments_dict[key]) < 1e-6
                    if not are_equal:
                        break
                else:
                    are_equal = arguments_dict[key] == current_arguments_dict[key]
                    if not are_equal:
                        break

            if are_equal:
                file_name_with_same_arguments = file_name
                break
    return file_name_with_same_arguments


def update_file_name_arguments_dict(file_name_arguments_dict_path, file_name, arguments_dict):
    if os.path.isfile(file_name_arguments_dict_path):
        file_name_arguments_dict = joblib.load(file_name_arguments_dict_path)
        file_name_arguments_dict[file_name] = arguments_dict
        joblib.dump(file_name_arguments_dict, file_name_arguments_dict_path)
    else:
        file_name_arguments_dict = {file_name: arguments_dict}
        joblib.dump(file_name_arguments_dict, file_name_arguments_dict_path)


def print_latex(matrix, decimals=2):
    print('\\begin{bmatrix}')
    for i in range(matrix.shape[0]):
        row = ''
        for j in range(matrix.shape[1]):
            if i == j:
                row += '\\mathbf{' + str(round(float(matrix[i, j]), decimals)) + '} & '
            else:
                row += f'{round(float(matrix[i, j]), decimals)} & '
        row = row[:-2] + '\\\\'
        print(row)
    print('\\end{bmatrix}')


if __name__ == '__main__':
    generate_data_linear_regression(jnp.array([2.0, -1.0, 1.0], jnp.float32), jax.random.PRNGKey(234))
