import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import lax
from jax.scipy.linalg import cho_factor, cho_solve
from jax.scipy.optimize import minimize
from numpyro import distributions as dist
from numpyro.primitives import sample, plate


def logistic_regression_theta_transform(theta):
    return theta


def gamma_exponential_theta_transform(theta):
    return np.log(np.exp(theta) - 1)


def beta_binomial_theta_transform(theta):
    return np.log(theta / (1 - theta))


def dirichlet_categorical_theta_transform(theta):
    return np.log(theta / (1 - theta))


def dirichlet_categorical_theta_transform_full(theta):
    return np.log(theta / (1.0 - np.sum(theta)))


def linear_regression_theta_transform(theta):
    return np.concatenate([theta[:-1], np.log(np.exp(theta[-1:]) - 1)], axis=0)


def convert_to_lower_triangular_matrix(vector,
                                       matrix_dimension,
                                       is_diagonal_matrix):
    if is_diagonal_matrix:
        return jnp.diag(vector)
    return jnp.eye(matrix_dimension, dtype=jnp.float32).at[jnp.tril_indices(matrix_dimension, 0)].set(vector)


def get_conditioned_cholesky_lower_triangular_matrix(pre_conditioned_lower_triangular_matrix,
                                                     matrix_dimension,
                                                     is_diagonal_matrix):
    if is_diagonal_matrix:
        return jnp.diag(jax.nn.softplus(pre_conditioned_lower_triangular_matrix))
    else:
        matrix = convert_to_lower_triangular_matrix(pre_conditioned_lower_triangular_matrix,
                                                    matrix_dimension,
                                                    False)
        strictly_lower = jnp.tril(matrix, -1)
        return strictly_lower + jnp.diag(jax.nn.softplus(jnp.diagonal(matrix)))


def get_positive_definite_matrix(pre_conditioned_lower_triangular_matrix,
                                 matrix_dimension,
                                 is_diagonal_matrix):
    covariance_cholesky_lower = get_conditioned_cholesky_lower_triangular_matrix(
        pre_conditioned_lower_triangular_matrix,
        matrix_dimension,
        is_diagonal_matrix)
    return covariance_cholesky_lower @ covariance_cholesky_lower.T


def get_cholesky_lower(matrix):
    return jnp.linalg.cholesky(matrix, upper=False)


def lower_triangular_matrix_inverse(matrix):
    strictly_lower = jnp.tril(matrix, -1)
    diagonal = jnp.diagonal(matrix)
    matrix_dimension = matrix.shape[0]
    matrix_inverse = jnp.eye(matrix_dimension, dtype=jnp.float32)
    diagonal_inverse = jnp.eye(matrix_dimension, dtype=jnp.float32) / diagonal
    current_pow = diagonal_inverse @ strictly_lower
    for i in range(1, matrix_dimension):
        matrix_inverse += (-1) ** i * current_pow
        current_pow = current_pow @ diagonal_inverse @ strictly_lower
    return matrix_inverse @ diagonal_inverse


def log_likelihood(x, y, theta):
    logit = jnp.sum(x * theta)
    return y * jax.nn.log_sigmoid(logit) + (1 - y) * jax.nn.log_sigmoid(-logit)


def log_prior(theta):
    return -0.5 * (theta.shape[0] * jnp.log(2 * jnp.pi) + jnp.sum(theta * theta))


def log_variational(mu,
                    pre_transformed_covariance_matrix,
                    theta,
                    is_covariance_diagonal_matrix,
                    transform_covariance_matrix=True):
    theta_dimension = mu.shape[0]
    zero_centered = (theta - mu).reshape(theta_dimension, 1)
    if transform_covariance_matrix:
        cholesky_lower = get_conditioned_cholesky_lower_triangular_matrix(pre_transformed_covariance_matrix,
                                                                          theta_dimension,
                                                                          is_covariance_diagonal_matrix)
    else:
        cholesky_lower = pre_transformed_covariance_matrix
    cholesky_lower_inverse = lower_triangular_matrix_inverse(cholesky_lower)
    mahalanobis_distance = jnp.transpose(zero_centered) @ (cholesky_lower_inverse.T
                                                           @ cholesky_lower_inverse) @ zero_centered
    return -0.5 * (theta_dimension
                   * jnp.log(2 * jnp.pi)
                   + 2 * jnp.sum(jnp.log(jnp.diagonal(cholesky_lower)))
                   + mahalanobis_distance)


def exact_negative_entropy(pre_transformed_covariance_matrix,
                           theta_dimension,
                           is_covariance_diagonal_matrix,
                           transform_covariance_matrix):
    if transform_covariance_matrix:
        cholesky_lower = get_conditioned_cholesky_lower_triangular_matrix(pre_transformed_covariance_matrix,
                                                                          theta_dimension,
                                                                          is_covariance_diagonal_matrix)
    else:
        cholesky_lower = convert_to_lower_triangular_matrix(pre_transformed_covariance_matrix,
                                                            theta_dimension,
                                                            is_covariance_diagonal_matrix)
    result = (theta_dimension
              * jnp.log(2 * jnp.pi * jnp.e)
              + 2 * jnp.sum(jnp.log(jnp.diagonal(cholesky_lower))))
    return -0.5 * result


def sample_from_variational_distribution_cholesky(mu,
                                                  pre_transformed_covariance_matrix,
                                                  mc_integration_standard_normal_samples,
                                                  is_covariance_diagonal_matrix,
                                                  transform_covariance_matrix=True):
    theta_dimension = mu.shape[0]
    if transform_covariance_matrix:
        cholesky_lower = get_conditioned_cholesky_lower_triangular_matrix(pre_transformed_covariance_matrix,
                                                                          theta_dimension,
                                                                          is_covariance_diagonal_matrix)
    else:
        cholesky_lower = pre_transformed_covariance_matrix
    result = mu.reshape(theta_dimension, 1) + cholesky_lower @ mc_integration_standard_normal_samples
    return result.reshape(-1, theta_dimension)


def per_example_kl_divergence_with_exact_entropy(x,
                                                 y,
                                                 phi,
                                                 examples_count,
                                                 mc_integration_standard_normal_samples,
                                                 is_diagonal_matrix,
                                                 transform_covariance_matrix=True):
    mu = phi['mu']
    theta_dimension = mu.shape[0]
    pre_transformed_covariance = phi['pre_transformed_covariance']
    theta_samples = sample_from_variational_distribution_cholesky(mu,
                                                                  pre_transformed_covariance,
                                                                  mc_integration_standard_normal_samples,
                                                                  is_diagonal_matrix,
                                                                  transform_covariance_matrix)

    def log_likelihood_with_data(theta_):
        return log_likelihood(x, y, theta_)

    average_log_likelihood = jax.vmap(log_likelihood_with_data,
                                      in_axes=0,
                                      out_axes=0)(theta_samples).mean()
    average_log_prior = jax.vmap(log_prior,
                                 in_axes=0,
                                 out_axes=0)(theta_samples).mean()
    entropy = exact_negative_entropy(pre_transformed_covariance,
                                     theta_dimension,
                                     is_diagonal_matrix,
                                     transform_covariance_matrix)
    return entropy / examples_count - average_log_likelihood - average_log_prior / examples_count


def per_example_kl_divergence(x,
                              y,
                              phi,
                              examples_count,
                              mc_integration_standard_normal_samples,
                              is_diagonal_matrix,
                              transform_covariance_matrix=True):
    mu = phi['mu']
    pre_transformed_covariance = phi['pre_transformed_covariance']
    theta_samples = sample_from_variational_distribution_cholesky(mu,
                                                                  pre_transformed_covariance,
                                                                  mc_integration_standard_normal_samples,
                                                                  is_diagonal_matrix,
                                                                  transform_covariance_matrix)

    def log_likelihood_with_data(theta_):
        return log_likelihood(x, y, theta_)

    def log_variational_with_mu_covariance(theta_):
        return log_variational(mu,
                               pre_transformed_covariance,
                               theta_,
                               is_diagonal_matrix,
                               transform_covariance_matrix)

    average_log_likelihood = jax.vmap(log_likelihood_with_data,
                                      in_axes=0,
                                      out_axes=0)(theta_samples).mean()
    average_log_prior = jax.vmap(log_prior,
                                 in_axes=0,
                                 out_axes=0)(theta_samples).mean()
    average_log_variational = jax.vmap(log_variational_with_mu_covariance,
                                       in_axes=0,
                                       out_axes=0)(theta_samples).mean()

    return average_log_variational / examples_count - average_log_likelihood - average_log_prior / examples_count


def per_example_kl_divergence_gamma_exponential(x,
                                                y,
                                                phi,
                                                examples_count,
                                                mc_integration_standard_normal_samples,
                                                is_diagonal_matrix,
                                                transform_covariance_matrix=True):
    mu = phi['mu']
    pre_transformed_covariance = phi['pre_transformed_covariance']
    theta_samples = sample_from_variational_distribution_cholesky(mu,
                                                                  pre_transformed_covariance,
                                                                  mc_integration_standard_normal_samples,
                                                                  is_diagonal_matrix,
                                                                  transform_covariance_matrix)

    def log_likelihood_with_data(theta_):
        return dist.Exponential(jax.nn.softplus(theta_)[0]).log_prob(x)

    def log_variational_with_mu_covariance(theta_):
        return log_variational(mu,
                               pre_transformed_covariance,
                               theta_,
                               is_diagonal_matrix,
                               transform_covariance_matrix)

    def log_prior_gamma(theta_):
        return dist.Gamma(8.0, 2.0).log_prob(jax.nn.softplus(theta_)) + jax.nn.log_sigmoid(theta_)

    average_log_likelihood = jax.vmap(log_likelihood_with_data,
                                      in_axes=0,
                                      out_axes=0)(theta_samples).mean()
    average_log_prior = jax.vmap(log_prior_gamma,
                                 in_axes=0,
                                 out_axes=0)(theta_samples).mean()
    average_log_variational = jax.vmap(log_variational_with_mu_covariance,
                                       in_axes=0,
                                       out_axes=0)(theta_samples).mean()

    return average_log_variational / examples_count - average_log_likelihood - average_log_prior / examples_count


def per_example_kl_divergence_beta_bernoulli(x,
                                             y,
                                             phi,
                                             examples_count,
                                             mc_integration_standard_normal_samples,
                                             is_diagonal_matrix,
                                             transform_covariance_matrix=True):
    mu = phi['mu']
    pre_transformed_covariance = phi['pre_transformed_covariance']
    theta_samples = sample_from_variational_distribution_cholesky(mu,
                                                                  pre_transformed_covariance,
                                                                  mc_integration_standard_normal_samples,
                                                                  is_diagonal_matrix,
                                                                  transform_covariance_matrix)

    def log_likelihood_with_data(theta_):
        return x * jax.nn.log_sigmoid(theta_) + (1 - x) * jax.nn.log_sigmoid(-theta_)

    def log_variational_with_mu_covariance(theta_):
        return log_variational(mu,
                               pre_transformed_covariance,
                               theta_,
                               is_diagonal_matrix,
                               transform_covariance_matrix)

    def log_prior_beta(theta_, alpha=10.0, beta=10.0):
        term1 = (alpha - 1.0) * jax.nn.log_sigmoid(theta_) + (beta - 1.0) * jax.nn.log_sigmoid(-theta_)
        term2 = lax.lgamma(alpha) + lax.lgamma(beta) - lax.lgamma(alpha + beta)
        transform_derivative = jax.nn.log_sigmoid(theta_) + jax.nn.log_sigmoid(-theta_)
        return term1 - term2 + transform_derivative

    average_log_likelihood = jax.vmap(log_likelihood_with_data,
                                      in_axes=0,
                                      out_axes=0)(theta_samples).mean()
    average_log_prior = jax.vmap(log_prior_beta,
                                 in_axes=0,
                                 out_axes=0)(theta_samples).mean()
    average_log_variational = jax.vmap(log_variational_with_mu_covariance,
                                       in_axes=0,
                                       out_axes=0)(theta_samples).mean()

    return average_log_variational / examples_count - average_log_likelihood - average_log_prior / examples_count


def per_example_kl_divergence_beta_binomial(x,
                                            y,
                                            phi,
                                            examples_count,
                                            mc_integration_standard_normal_samples,
                                            is_diagonal_matrix,
                                            transform_covariance_matrix=True):
    mu = phi['mu']
    pre_transformed_covariance = phi['pre_transformed_covariance']
    theta_samples = sample_from_variational_distribution_cholesky(mu,
                                                                  pre_transformed_covariance,
                                                                  mc_integration_standard_normal_samples,
                                                                  is_diagonal_matrix,
                                                                  transform_covariance_matrix)

    def log_likelihood_with_data(theta_, trials=1000):
        term1 = x * jax.nn.log_sigmoid(theta_) + (trials - x) * jax.nn.log_sigmoid(-theta_)
        term2 = lax.lgamma((trials + 1) * 1.0)
        term3 = lax.lgamma((x + 1) * 1.0)
        term4 = lax.lgamma((trials - x + 1) * 1.0)
        return term1 + term2 - term3 - term4

    def log_variational_with_mu_covariance(theta_):
        return log_variational(mu,
                               pre_transformed_covariance,
                               theta_,
                               is_diagonal_matrix,
                               transform_covariance_matrix)

    def log_prior_beta(theta_, alpha=10.0, beta=10.0):
        term1 = (alpha - 1.0) * jax.nn.log_sigmoid(theta_) + (beta - 1.0) * jax.nn.log_sigmoid(-theta_)
        term2 = lax.lgamma(alpha) + lax.lgamma(beta) - lax.lgamma(alpha + beta)
        transform_derivative = jax.nn.log_sigmoid(theta_) + jax.nn.log_sigmoid(-theta_)
        return term1 - term2 + transform_derivative

    average_log_likelihood = jax.vmap(log_likelihood_with_data,
                                      in_axes=0,
                                      out_axes=0)(theta_samples).mean()
    average_log_prior = jax.vmap(log_prior_beta,
                                 in_axes=0,
                                 out_axes=0)(theta_samples).mean()
    average_log_variational = jax.vmap(log_variational_with_mu_covariance,
                                       in_axes=0,
                                       out_axes=0)(theta_samples).mean()

    return average_log_variational / examples_count - average_log_likelihood - average_log_prior / examples_count


def stable_sigmoid(x):
    return jnp.where(
        x >= 0.0,
        1.0 / (1.0 + jnp.exp(-x)),
        jnp.exp(x) / (1.0 + jnp.exp(x))
    )


def per_example_kl_divergence_dirichlet_categorical(x,
                                                    y,
                                                    phi,
                                                    examples_count,
                                                    mc_integration_standard_normal_samples,
                                                    is_diagonal_matrix,
                                                    transform_covariance_matrix=True):
    mu = phi['mu']
    pre_transformed_covariance = phi['pre_transformed_covariance']
    theta_samples = sample_from_variational_distribution_cholesky(mu,
                                                                  pre_transformed_covariance,
                                                                  mc_integration_standard_normal_samples,
                                                                  is_diagonal_matrix,
                                                                  transform_covariance_matrix)

    def calculate_log_constrained_full_theta(theta_):
        last_theta = jnp.maximum(1.0 - jnp.sum(stable_sigmoid(theta_), keepdims=True), 1e-37)
        return jnp.concatenate([jax.nn.log_sigmoid(theta_), jnp.log(last_theta)])

    def log_likelihood_with_data(theta_):
        return jnp.sum(jax.nn.one_hot(x, theta_.shape[-1] + 1) * calculate_log_constrained_full_theta(theta_))

    def log_variational_with_mu_covariance(theta_):
        return log_variational(mu,
                               pre_transformed_covariance,
                               theta_,
                               is_diagonal_matrix,
                               transform_covariance_matrix)

    def log_prior_dirichlet(theta_, alphas=jnp.diagonal(jnp.eye(3, dtype=jnp.float32)) * 5.0):
        term1 = jnp.sum((alphas - 1) * calculate_log_constrained_full_theta(theta_))
        term2 = jnp.sum(lax.lgamma(alphas)) - lax.lgamma(jnp.sum(alphas))
        transform_derivative = jnp.sum(jax.nn.log_sigmoid(theta_) + jax.nn.log_sigmoid(-theta_))
        return term1 - term2 + transform_derivative

    average_log_likelihood = jax.vmap(log_likelihood_with_data,
                                      in_axes=0,
                                      out_axes=0)(theta_samples).mean()
    average_log_prior = jax.vmap(log_prior_dirichlet,
                                 in_axes=0,
                                 out_axes=0)(theta_samples).mean()
    average_log_variational = jax.vmap(log_variational_with_mu_covariance,
                                       in_axes=0,
                                       out_axes=0)(theta_samples).mean()

    return average_log_variational / examples_count - average_log_likelihood - average_log_prior / examples_count


def per_example_kl_divergence_dirichlet_categorical_full(x,
                                                         y,
                                                         phi,
                                                         examples_count,
                                                         mc_integration_standard_normal_samples,
                                                         is_diagonal_matrix,
                                                         transform_covariance_matrix=True):
    mu = phi['mu']
    pre_transformed_covariance = phi['pre_transformed_covariance']
    theta_samples = sample_from_variational_distribution_cholesky(mu,
                                                                  pre_transformed_covariance,
                                                                  mc_integration_standard_normal_samples,
                                                                  is_diagonal_matrix,
                                                                  transform_covariance_matrix)

    def log_likelihood_with_data(theta_):
        theta_full = jnp.concatenate([theta_, jnp.zeros((1,), dtype=jnp.float32)])
        return jnp.sum(jax.nn.one_hot(x, theta_.shape[-1] + 1) * jax.nn.log_softmax(theta_full))

    def log_variational_with_mu_covariance(theta_):
        return log_variational(mu,
                               pre_transformed_covariance,
                               theta_,
                               is_diagonal_matrix,
                               transform_covariance_matrix)

    def softmax_without_last_theta(theta_):
        theta_full = jnp.concatenate([theta_, jnp.zeros((1,), dtype=jnp.float32)])
        return jax.nn.softmax(theta_full)[:-1]

    def log_prior_dirichlet(theta_, alphas=jnp.diagonal(jnp.eye(3, dtype=jnp.float32)) * 5.0):
        theta_full = jnp.concatenate([theta_, jnp.zeros((1,), dtype=jnp.float32)])
        term1 = jnp.sum((alphas - 1) * jax.nn.log_softmax(theta_full))
        term2 = jnp.sum(lax.lgamma(alphas)) - lax.lgamma(jnp.sum(alphas))
        transform_derivative = jnp.linalg.det(jax.jacfwd(softmax_without_last_theta)(theta_))
        return term1 - term2 + jnp.log(jnp.abs(transform_derivative))

    average_log_likelihood = jax.vmap(log_likelihood_with_data,
                                      in_axes=0,
                                      out_axes=0)(theta_samples).mean()
    average_log_prior = jax.vmap(log_prior_dirichlet,
                                 in_axes=0,
                                 out_axes=0)(theta_samples).mean()
    average_log_variational = jax.vmap(log_variational_with_mu_covariance,
                                       in_axes=0,
                                       out_axes=0)(theta_samples).mean()

    return average_log_variational / examples_count - average_log_likelihood - average_log_prior / examples_count


def per_example_kl_divergence_linear_regression(x,
                                                y,
                                                phi,
                                                examples_count,
                                                mc_integration_standard_normal_samples,
                                                is_diagonal_matrix,
                                                transform_covariance_matrix=True):
    mu = phi['mu']
    pre_transformed_covariance = phi['pre_transformed_covariance']
    theta_samples = sample_from_variational_distribution_cholesky(mu,
                                                                  pre_transformed_covariance,
                                                                  mc_integration_standard_normal_samples,
                                                                  is_diagonal_matrix,
                                                                  transform_covariance_matrix)

    def log_likelihood_with_data(theta_):
        theta_w = theta_[:-1]
        pre_sigma_squared = theta_[-1]
        sigma_squared = jax.nn.softplus(pre_sigma_squared)
        # sigma_squared = jnp.exp(pre_sigma_squared)
        term1 = -0.5 * jnp.log(sigma_squared)  # + pre_sigma_squared
        term2 = -0.5 * jnp.square(jnp.sum(theta_w * x) - y) / sigma_squared
        return term1 + term2
        # return dist.Normal(jnp.sum(theta_w * x), jnp.sqrt(sigma_squared)).log_prob(y)

    def log_variational_with_mu_covariance(theta_):
        return log_variational(mu,
                               pre_transformed_covariance,
                               theta_,
                               is_diagonal_matrix,
                               transform_covariance_matrix)

    def log_prior_linear_regression(theta_):
        theta_w = theta_[:-1]
        pre_sigma_squared = theta_[-1]
        sigma_squared = jax.nn.softplus(pre_sigma_squared)
        # sigma_squared = jnp.exp(pre_sigma_squared)
        # pre_sigma_squared_log_prior = dist.InverseGamma(20, 0.5).log_prob(
        #     sigma_squared) + jax.nn.log_sigmoid(pre_sigma_squared)  # + pre_sigma_squared
        num_weights = theta_.shape[0] - 1
        theta_w_cov = sigma_squared * jnp.linalg.inv(jnp.eye(num_weights) * (num_weights - 1) / 40.0)
        theta_w_mean = jnp.diagonal(jnp.eye(num_weights)) * 0.0
        # theta_w_log_prior = multivariate_normal_logpdf(theta_w, theta_w_mean, theta_w_cov)
        # return pre_sigma_squared_log_prior + theta_w_log_prior
        # return 0.0
        # return dist.InverseGamma(20.0, 3.0).log_prob(sigma_squared) + jax.nn.log_sigmoid(
        #     pre_sigma_squared) + dist.MultivariateNormal(theta_w_mean, theta_w_cov).log_prob(theta_w)
        return dist.InverseGamma(20.0, 0.5).log_prob(sigma_squared) + jax.nn.log_sigmoid(
            pre_sigma_squared) + dist.MultivariateNormal(theta_w_mean, theta_w_cov).log_prob(theta_w)

    average_log_likelihood = jax.vmap(log_likelihood_with_data,
                                      in_axes=0,
                                      out_axes=0)(theta_samples).mean()
    average_log_prior = jax.vmap(log_prior_linear_regression,
                                 in_axes=0,
                                 out_axes=0)(theta_samples).mean()
    average_log_variational = jax.vmap(log_variational_with_mu_covariance,
                                       in_axes=0,
                                       out_axes=0)(theta_samples).mean()

    return average_log_variational / examples_count - average_log_likelihood - average_log_prior / examples_count


def convert_vector_to_symmetric_matrix(lower_triangular_matrix,
                                       matrix_dimension,
                                       is_diagonal_matrix,
                                       is_positive_definite):
    if is_positive_definite:
        return get_positive_definite_matrix(lower_triangular_matrix,
                                            matrix_dimension,
                                            is_diagonal_matrix)
    else:
        return vector_to_symmetric_matrix(lower_triangular_matrix,
                                          matrix_dimension,
                                          is_diagonal_matrix)


def vector_to_symmetric_matrix(vector, matrix_dimension, is_diagonal_matrix):
    matrix = convert_to_lower_triangular_matrix(vector,
                                                matrix_dimension,
                                                is_diagonal_matrix)
    if is_diagonal_matrix:
        return matrix

    diagonal = jnp.diagonal(matrix)
    strictly_lower = jnp.tril(matrix, -1)
    return strictly_lower + strictly_lower.T + jnp.diag(diagonal)


def sample_hessian_matrix(matrix_dimension,
                          theta_dimension,
                          hessian_prior_mean,
                          hessian_prior_std,
                          is_hessian_positive_definite,
                          is_hessian_diagonal_matrix):
    # hessian_diagonal_mu = sample('hessian_diagonal_mu',
    #                              dist.Normal(400.0, 20.0),
    #                              sample_shape=(theta_dimension,))
    # hessian_diagonal_pre_conditioned_covariance = sample('hessian_diagonal_pre_conditioned_covariance',
    #                                                      dist.Normal(40.0, 2.0),
    #                                                      sample_shape=(theta_dimension,))
    #
    # hessian_diagonal = jnp.concatenate([hessian_diagonal_mu,
    #                                     hessian_diagonal_pre_conditioned_covariance], axis=-1)
    #
    # if is_hessian_diagonal_matrix:
    #     hessian_lower_vector = hessian_diagonal
    # else:
    #     hessian_strictly_lower = sample('hessian_strictly_lower',
    #                                     dist.Normal(0.0, 1.0),
    #                                     sample_shape=(matrix_dimension * (matrix_dimension - 1) // 2,))
    #     hessian_lower = jnp.diagonal(hessian_diagonal) + jnp.eye(matrix_dimension, dtype=jnp.float32).at[
    #         jnp.tril_indices(matrix_dimension, -1)].set(hessian_strictly_lower)
    #     hessian_lower_vector = hessian_lower[jnp.tril_indices(matrix_dimension, 0)]
    hessian_diagonal = sample('hessian_diagonal',
                              dist.Normal(hessian_prior_mean, hessian_prior_std))
    # common_term = jnp.log(1 + hessian_prior_std ** 2 / (hessian_prior_mean ** 2 + 1e-7))
    # hessian_diagonal = sample('hessian_diagonal',
    #                           dist.LogNormal(jnp.log(1 + hessian_prior_mean) - 0.5 * common_term,
    #                                          jnp.sqrt(common_term)))

    hessian_diagonal = jax.nn.softplus(hessian_diagonal)
    # return convert_vector_to_symmetric_matrix(hessian_lower_vector,
    #                                           matrix_dimension,
    #                                           is_hessian_diagonal_matrix,
    #                                           is_hessian_positive_definite)
    return jnp.diag(hessian_diagonal)


def parameter_based_inference_model(phi_trace,
                                    theta_dimension,
                                    dpsgd_noise_scale,
                                    dpsgd_clipping_threshold,
                                    dpsgd_gradient_scale_vector,
                                    dpsgd_learning_rates_vector,
                                    hessian_prior_mean,
                                    hessian_prior_std,
                                    phi_prior_mean,
                                    phi_prior_std,
                                    is_hessian_positive_definite=True,
                                    is_hessian_diagonal_matrix=True):
    phi_dimension = phi_trace.shape[-1]
    hessian = sample_hessian_matrix(phi_dimension,
                                    theta_dimension,
                                    hessian_prior_mean,
                                    hessian_prior_std,
                                    is_hessian_positive_definite,
                                    is_hessian_diagonal_matrix)
    optimal_phi = sample('optimal_phi',
                         dist.Normal(phi_prior_mean, phi_prior_std))
    noise_variance = (dpsgd_noise_scale
                      * dpsgd_clipping_threshold
                      * dpsgd_learning_rates_vector
                      / dpsgd_gradient_scale_vector) ** 2
    previous_phis = jnp.expand_dims(phi_trace[:-1], -1)
    with plate('phi_trace_plate', phi_trace.shape[0] - 1):
        gradients = hessian @ (previous_phis - jnp.expand_dims(optimal_phi, -1))
        mean = previous_phis - np.expand_dims(dpsgd_learning_rates_vector, -1) * gradients
        mean = mean.reshape(-1, phi_dimension)
        sample(
            'next_phis',
            dist.MultivariateNormal(mean, noise_variance * jnp.eye(phi_dimension, dtype=jnp.float32)),
            obs=phi_trace[1:]
        )


def gradient_based_inference_model(phi_trace,
                                   gradient_trace,
                                   theta_dimension,
                                   sampling_rate,
                                   dpsgd_noise_scale,
                                   dpsgd_clipping_threshold_vector,
                                   dpsgd_gradient_scale_vector,
                                   hessian_prior_mean,
                                   hessian_prior_std,
                                   phi_prior_mean,
                                   phi_prior_std,
                                   add_subsampling_noise=False,
                                   is_hessian_positive_definite=True,
                                   is_hessian_diagonal_matrix=True):
    phi_dimension = phi_trace.shape[-1]
    hessian = sample_hessian_matrix(phi_dimension,
                                    theta_dimension,
                                    hessian_prior_mean,
                                    hessian_prior_std,
                                    is_hessian_positive_definite,
                                    is_hessian_diagonal_matrix)
    optimal_phi = sample('optimal_phi',
                         dist.Normal(phi_prior_mean, 1))
    noise_variance = (dpsgd_noise_scale
                      * dpsgd_clipping_threshold_vector
                      / dpsgd_gradient_scale_vector) ** 2

    if add_subsampling_noise:
        subsampling_noise = sample('subsampling_noise',
                                   dist.MultivariateNormal(jnp.zeros((phi_dimension,), dtype=jnp.float32),
                                                           jnp.eye(phi_dimension, dtype=jnp.float32)))
    else:
        subsampling_noise = jnp.zeros((phi_dimension,), dtype=jnp.float32)

    subsampling_noise *= jnp.eye(phi_dimension, dtype=jnp.float32)
    subsampling_noise = subsampling_noise @ subsampling_noise.T

    with plate('phi_trace_plate', phi_trace.shape[0]):
        mean = sampling_rate * hessian @ jnp.expand_dims(phi_trace - optimal_phi, -1)
        mean = mean.reshape(-1, phi_dimension)
        sample(
            'gradients',
            dist.MultivariateNormal(mean,
                                    subsampling_noise + noise_variance * jnp.eye(phi_dimension, dtype=jnp.float32)),
            obs=gradient_trace
        )


def multivariate_normal_logpdf(x, mean, cov, eps=1e-7, is_diagonal_matrix=True):
    # this is a numerically stable implementation of multivariate normal log pdf using quadratic form evaluation with
    # Cholesky decomposition because numpyro distributions implementation is not stable
    n = mean.shape[0]
    diff = x - mean

    if not is_diagonal_matrix:
        lower_triangular_matrix, is_lower = cho_factor(cov + eps * jnp.eye(n))
        cov_quad_inv = cho_solve((lower_triangular_matrix, is_lower), diff)
        log_det_cov = 2 * jnp.sum(jnp.log(jnp.diag(lower_triangular_matrix)))
        quad_form = jnp.dot(diff, cov_quad_inv)
        log_prob = -0.5 * (n * jnp.log(2 * jnp.pi) + log_det_cov + quad_form)
    else:
        log_det_cov = jnp.sum(jnp.log(jnp.maximum(jnp.diagonal(cov), eps)))
        log_exp_term = jnp.sum(jnp.square(diff) / jnp.maximum(jnp.diagonal(cov), eps))
        log_prob = -0.5 * (n * jnp.log(2 * jnp.pi) + log_det_cov + log_exp_term)

    return log_prob


def gradient_based_inference_model_log_prior(params,
                                             phi_dimension,
                                             hessian_components,
                                             hessian_prior_mean,
                                             hessian_prior_std,
                                             optimal_phi_prior_mean,
                                             optimal_phi_prior_std,
                                             add_subsampling_noise):
    hessian_params = params[phi_dimension:phi_dimension + hessian_components]
    hessian_prior_log_prob = multivariate_normal_logpdf(hessian_params,
                                                        hessian_prior_mean,
                                                        jnp.square(hessian_prior_std) * jnp.eye(
                                                            hessian_params.shape[-1],
                                                            dtype=jnp.float32))
    phi_prior_log_prob = multivariate_normal_logpdf(params[:phi_dimension],
                                                    optimal_phi_prior_mean,
                                                    jnp.eye(phi_dimension, dtype=jnp.float32))

    if add_subsampling_noise:
        subsampling_prior_log_prob = multivariate_normal_logpdf(params[phi_dimension + hessian_components:],
                                                                jnp.zeros((phi_dimension,), dtype=jnp.float32),
                                                                jnp.eye(phi_dimension, dtype=jnp.float32))
    else:
        subsampling_prior_log_prob = 0.0

    return hessian_prior_log_prob + phi_prior_log_prob + subsampling_prior_log_prob


def gradient_based_inference_model_log_likelihood(phi,
                                                  perturbed_gradient,
                                                  params,
                                                  noise_variance,
                                                  sampling_rate,
                                                  add_subsampling_noise,
                                                  is_hessian_diagonal_matrix=True):
    phi_shape = phi.shape
    phi_dimension = phi_shape[-1]
    if add_subsampling_noise:
        hessian_components = params.shape[-1] - phi_dimension * 2
    else:
        hessian_components = params.shape[-1] - phi_dimension

    hessian_params = jax.nn.softplus(params[phi_dimension: phi_dimension + hessian_components])
    hessian = vector_to_symmetric_matrix(hessian_params,
                                         phi_dimension,
                                         is_hessian_diagonal_matrix)
    mean = jnp.reshape(sampling_rate * hessian @ jnp.expand_dims(phi - params[:phi_dimension], -1), phi_shape)

    if add_subsampling_noise:
        subsampling_noise = params[phi_dimension + hessian_components:] * jnp.eye(phi_dimension, dtype=jnp.float32)
        subsampling_noise = subsampling_noise @ subsampling_noise.T
    else:
        subsampling_noise = 0.0

    return multivariate_normal_logpdf(perturbed_gradient, mean, noise_variance + subsampling_noise) + jnp.sum(
        jax.nn.log_sigmoid(hessian_params))


def svd_inverse(hessian, eps=1e-7):
    U, S, Vt = jnp.linalg.svd(hessian)
    S_inv = jnp.diag(1.0 / jnp.maximum(S, eps))
    return Vt.T @ S_inv @ U.T


def gradient_based_inference_model_posterior_laplace_approximation(phi_trace,
                                                                   perturbed_gradient_trace,
                                                                   sampling_rate,
                                                                   dpsgd_noise_scale,
                                                                   dpsgd_clipping_threshold_vector,
                                                                   dpsgd_gradient_scale_vector,
                                                                   hessian_prior_mean,
                                                                   hessian_prior_std,
                                                                   optimal_phi_prior_mean,
                                                                   optimal_phi_prior_std,
                                                                   trace_burn_in_percentage,
                                                                   add_subsampling_noise=False,
                                                                   is_hessian_diagonal_matrix=True,
                                                                   initialize_params_with_priors=True,
                                                                   optimization_iterations=10000,
                                                                   optimization_learning_rate=1e-2,
                                                                   optimization_apply_trace_averaging=True,
                                                                   optimization_trace_averaging_burn_in=0.8,
                                                                   use_bfgs=False):
    phi_dimension = phi_trace.shape[-1]

    noise_variance = (dpsgd_noise_scale * dpsgd_clipping_threshold_vector / dpsgd_gradient_scale_vector) ** 2
    noise_variance = noise_variance * jnp.eye(phi_dimension, dtype=jnp.float32)
    trace_burn_in = int(phi_trace.shape[0] * trace_burn_in_percentage)

    if initialize_params_with_priors:
        if add_subsampling_noise:
            initial_params = jnp.concatenate([optimal_phi_prior_mean,
                                              hessian_prior_mean,
                                              jnp.zeros((phi_dimension,), jnp.float32)], axis=-1)
        else:
            initial_params = jnp.concatenate([optimal_phi_prior_mean,
                                              hessian_prior_mean], axis=-1)
        hessian_components = hessian_prior_mean.shape[-1]
    else:
        if is_hessian_diagonal_matrix:
            hessian_components = phi_dimension
        else:
            hessian_components = phi_dimension * (phi_dimension + 1) // 2

        if add_subsampling_noise:
            initial_params = jnp.concatenate([
                jnp.array(optimal_phi_prior_mean),  # this is for phi^*
                jnp.array(hessian_prior_mean),  # this is for the hessian A
                jnp.zeros((phi_dimension,), jnp.float32)  # this is for the subsampling noise
            ], axis=-1)
        else:
            initial_params = jnp.concatenate([
                jnp.array(optimal_phi_prior_mean),  # this is for phi^*
                jnp.array(hessian_prior_mean)  # this is for the hessian A
            ], axis=-1)

    def objective(params_):
        log_likelihood_sum = jax.vmap(gradient_based_inference_model_log_likelihood,
                                      in_axes=(0, 0, None, None, None, None, None),
                                      out_axes=0)(phi_trace[trace_burn_in:],
                                                  perturbed_gradient_trace[trace_burn_in:],
                                                  params_,
                                                  noise_variance,
                                                  sampling_rate,
                                                  add_subsampling_noise,
                                                  is_hessian_diagonal_matrix).sum()

        return -1.0 * (log_likelihood_sum + gradient_based_inference_model_log_prior(params_,
                                                                                     phi_dimension,
                                                                                     hessian_components,
                                                                                     hessian_prior_mean,
                                                                                     hessian_prior_std,
                                                                                     optimal_phi_prior_mean,
                                                                                     optimal_phi_prior_std,
                                                                                     add_subsampling_noise))

    if not use_bfgs:
        optimizer = optax.adam(optimization_learning_rate)
        optimizer_state = optimizer.init(initial_params)

        param_trace = jnp.zeros((optimization_iterations, initial_params.shape[0]))

        def update_params(t, args):
            params_, trace_, optimizer_state_ = args
            grads_ = jax.grad(objective)(params_)
            updates_, optimizer_state_ = optimizer.update(grads_, optimizer_state_)
            params_updated = optax.apply_updates(params_, updates_)
            # Polyak–Ruppert averaging to reduce noise:
            # params_ = (t * params_ + params_updated) / (t + 1)
            params_ = params_updated
            trace_ = trace_.at[t].set(params_)
            return params_, trace_, optimizer_state_

        (mode, param_trace, _) = jax.lax.fori_loop(0,
                                                   optimization_iterations,
                                                   update_params,
                                                   (initial_params,
                                                    param_trace,
                                                    optimizer_state))

        if optimization_apply_trace_averaging:
            burn_in = int(optimization_trace_averaging_burn_in * optimization_iterations)
            mode = jnp.mean(param_trace[burn_in:], axis=0)
    else:
        mode, *details = minimize(objective, initial_params, method="BFGS", tol=1e-12,
                                  options={'maxiter': 100000, 'gtol': 1e-12})
        param_trace = None

    objective_hessian = jax.hessian(objective)(mode)
    cov_matrix1 = jnp.linalg.inv(objective_hessian)
    identity_matrix = jnp.eye(objective_hessian.shape[0])
    cov_matrix_cho = jax.scipy.linalg.cho_factor(objective_hessian + 1e-7 * identity_matrix)
    cov_matrix2 = jax.scipy.linalg.cho_solve(cov_matrix_cho, identity_matrix)
    cov_matrix3 = svd_inverse(objective_hessian)

    return {
        'optimal_phi': mode[:phi_dimension],
        'hessian': vector_to_symmetric_matrix(mode[phi_dimension:], phi_dimension, is_hessian_diagonal_matrix)
    }, cov_matrix1, cov_matrix2, cov_matrix3, param_trace
