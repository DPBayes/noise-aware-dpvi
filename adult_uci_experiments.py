import math
import math
import os
import os.path
import pickle
import random
import time
from argparse import ArgumentParser
from collections import OrderedDict

import jax
import jax.numpy as jnp
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions
import pandas as pd
import tueplots
from scipy import stats, interpolate
from scipy.stats import multivariate_normal
from sklearn.calibration import calibration_curve
from tqdm import tqdm
from tueplots import bundles

from model import get_positive_definite_matrix, per_example_kl_divergence, logistic_regression_theta_transform
from noise_aware_dpsgd import calculate_estimates_from_mcmc_samples, perform_mcmc_sampling, \
    perform_dpsgd, calculate_hessian_and_optimal_phi_priors, perform_laplace_approximation, \
    calculate_estimates_from_laplace_approximation
from utils import DynamicAttributes, plot_trace

from sklearn import metrics

orig_data_description = OrderedDict({
    "age": "continuous",
    "workclass": "Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked",
    "fnlwgt": "continuous",
    "education": "Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool",
    "education-num": "continuous",
    "marital-status": "Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse",
    "occupation": "Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces",
    "relationship": "Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried",
    "race": "White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black",
    "sex": "Female, Male",
    "capital-gain": "continuous",
    "capital-loss": "continuous",
    "hours-per-week": "continuous",
    "native-country": "United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands",
    "income": "<=50K, >50K"
})


def load_orig_data(data_path):
    adult_train_path = os.path.join(data_path, "adult_train.csv")
    adult_test_path = os.path.join(data_path, "adult_test.csv")
    if os.path.isfile(adult_train_path) and os.path.isfile(adult_test_path):
        train_data = pd.read_csv(adult_train_path, index_col=0)
        test_data = pd.read_csv(adult_test_path, index_col=0)
    else:
        base_url = lambda s: f"https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.{s}"

        train_data = pd.read_csv(base_url("data"), header=None, names=orig_data_description.keys(), na_values="?",
                                 skipinitialspace=True)
        test_data = pd.read_csv(base_url("test"), header=None, names=orig_data_description.keys(), na_values="?",
                                skipinitialspace=True)

        train_data = train_data.dropna()
        test_data = test_data.dropna()

        ## for some reason, there is a "." in the end of every line in the test data, so remove that
        test_data["income"] = test_data["income"].map({"<=50K.": "<=50K", ">50K.": ">50K"})
        train_data.to_csv(adult_train_path)
        test_data.to_csv(adult_test_path)

    return train_data.copy(), test_data.copy()


def preprocess_adult(data_path):
    train_data, test_data = load_orig_data(data_path)

    data_description = orig_data_description.copy()
    ## remove the "education-num" feature as it is one-to-one with the "education"
    del train_data["education-num"]
    del train_data["native-country"]
    del train_data["relationship"]
    del test_data["education-num"]
    del test_data["native-country"]
    del test_data["relationship"]
    del data_description["education-num"]
    del data_description["native-country"]
    del data_description["relationship"]

    def onehot_encode(X, encoding=None):
        if encoding is None:
            encoded_X = X.astype("category").cat.codes
            encoding = {cat: code for cat, code in zip(X, encoded_X)}
        else:
            encoded_X = X.map(encoding)
        num_cat = len(encoding)
        diag_matrix = np.eye(num_cat)
        onehotted_X = diag_matrix[encoded_X]
        return onehotted_X, encoding

    ## Encode categorical variables
    preprocessed_train_data = pd.DataFrame()
    preprocessed_test_data = pd.DataFrame()

    encodings = {}

    for key, value in data_description.items():
        if value != "continuous" and key != "income":
            onehotted_feature, encoding = onehot_encode(train_data[key])
            sorted_cats = sorted(encoding, key=encoding.get)
            # convert onehotted numpy frame to pandas with category names appended to the feature
            df_names = [f"{key}:{name}" for name in sorted_cats]
            df = pd.DataFrame(onehotted_feature, columns=df_names)
            # add to preprocessed train data
            preprocessed_train_data = pd.concat((preprocessed_train_data, df), axis=1)
            encodings[key] = encoding
            ## for test
            onehotted_feature_test, _ = onehot_encode(test_data[key], encoding=encoding)
            df_test = pd.DataFrame(onehotted_feature_test, columns=df_names)
            preprocessed_test_data = pd.concat((preprocessed_test_data, df_test), axis=1)
        else:
            preprocessed_train_data[key] = train_data[key].values
            preprocessed_test_data[key] = test_data[key].values

    ## z-normalize the continuous features
    from sklearn.preprocessing import scale

    for key, value in data_description.items():
        if value == "continuous":
            preprocessed_train_data[key] = scale(preprocessed_train_data[key])
            preprocessed_test_data[key] = scale(preprocessed_test_data[key])

    ## label targets
    target_map = {"<=50K": 0, ">50K": 1}
    preprocessed_train_data["income"] = preprocessed_train_data["income"].map(target_map)
    preprocessed_test_data["income"] = preprocessed_test_data["income"].map(target_map)

    encodings["income"] = target_map

    train_data_array = preprocessed_train_data.to_numpy()
    test_data_array = preprocessed_test_data.to_numpy()
    train_xs = train_data_array[:, :-1]
    train_ys = train_data_array[:, -1]
    train_xs = np.concatenate([train_xs, np.ones((train_xs.shape[0], 1), np.float32)], axis=-1)

    test_xs = test_data_array[:, :-1]
    test_ys = test_data_array[:, -1]
    test_xs = np.concatenate([test_xs, np.ones((test_xs.shape[0], 1), np.float32)], axis=-1)

    return preprocessed_train_data, preprocessed_test_data, encodings, data_description, train_xs, train_ys, test_xs, test_ys


def set_default_args(experiment_args):
    experiment_args.random_seed = 984
    experiment_args.training_iterations = 10000
    experiment_args.target_epsilon = 1.0
    experiment_args.target_delta = 10 ** -5
    experiment_args.clipping_threshold = 11.0
    experiment_args.dpsgd_initial_mu_value = 0.0
    experiment_args.dpsgd_initial_pre_transformed_covariance_value = -6.0
    experiment_args.sampling_rate = 0.1
    experiment_args.mu_gradient_scale = 1.0
    experiment_args.pre_transformed_covariance_gradient_scale = experiment_args.clipping_threshold * 50.0
    experiment_args.dpsgd_mu_learning_rate = None
    experiment_args.dpsgd_pre_transformed_covariance_learning_rate = None
    experiment_args.mu_learning_rate_v = math.sqrt(2) * math.sqrt(57)
    experiment_args.pre_cov_learning_rate_v = math.sqrt(2) * math.sqrt(57)
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


def parse_program_args():
    parser = ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--target_epsilon', type=float, default=1.0, help='DP Epsilon')
    parser.add_argument('--method', type=str, default='mcmc', help='can be either "mcmc" or "laplace"')
    parser.add_argument('--task_id', type=int, default=0, help='Task ID')
    parser.add_argument('--output_directory', type=str, help='Output Directory')
    parser.add_argument('--debug_output_directory', type=str, help='Debug Output Directory')
    program_args, _ = parser.parse_known_args()
    return program_args


def main():
    program_args = parse_program_args()
    # program_args = DynamicAttributes()
    # program_args.output_directory = './'
    # program_args.debug_output_directory = './'
    # program_args.task_id = 1
    # program_args.method = 'mcmc'
    # program_args.target_epsilon = 0.1

    preprocessed_data_path = os.path.join(program_args.debug_output_directory, 'preprocessed_data.pickle')
    if not os.path.isfile(preprocessed_data_path):
        (train_data,
         test_data,
         encodings,
         data_description,
         train_xs,
         train_ys,
         test_xs,
         test_ys) = preprocess_adult(
            program_args.debug_output_directory)
        with open(preprocessed_data_path, 'wb') as f:
            pickle.dump((train_data,
                         test_data,
                         encodings,
                         data_description,
                         train_xs,
                         train_ys,
                         test_xs,
                         test_ys), f)
    else:
        with open(preprocessed_data_path, 'rb') as f:
            (_,
             _,
             _,
             _,
             train_xs,
             train_ys,
             test_xs,
             test_ys) = pickle.load(f)

    experiment_args = DynamicAttributes()
    experiment_args = set_default_args(experiment_args)

    if program_args.method == 'mcmc':
        experiment_args.use_mcmc_sampling = True
    else:
        experiment_args.use_mcmc_sampling = False

    # train_indices = np.arange(train_xs.shape[0])
    # np.random.shuffle(train_indices)
    # train_indices = train_indices[:5000]
    # experiment_args.train_xs = jnp.array(train_xs[train_indices], dtype=jnp.float32)
    # experiment_args.train_ys = jnp.array(train_ys[train_indices], dtype=jnp.float32)

    experiment_args.train_xs = jnp.array(train_xs, dtype=jnp.float32)
    experiment_args.train_ys = jnp.array(train_ys, dtype=jnp.float32)
    experiment_args.test_xs = jnp.array(test_xs, dtype=jnp.float32)
    experiment_args.test_ys = jnp.array(test_ys, dtype=jnp.float32)

    experiment_args.data_size = train_xs.shape[0] + test_xs.shape[0]
    experiment_args.training_data_size = train_xs.shape[0]
    experiment_args.test_data_size = test_xs.shape[0]

    task_id = program_args.task_id
    experiment_args.target_epsilon = program_args.target_epsilon
    experiment_args.task_id = task_id
    experiment_args.random_seed += task_id
    experiment_args.algorithm_prng_key = jax.random.PRNGKey(experiment_args.random_seed)
    experiment_args.theta_dimension = train_xs.shape[-1]

    experiment_tracker = DynamicAttributes()
    experiment_tracker_without_debug_info = DynamicAttributes()

    experiment_tracker.experiment_args = experiment_args
    experiment_tracker_without_debug_info.experiment_args = experiment_args
    experiment_tracker = perform_dpsgd(experiment_args, experiment_tracker)
    experiment_tracker_without_debug_info.mu_trace = experiment_tracker.mu_trace
    experiment_tracker_without_debug_info.phi_trace = experiment_tracker.phi_trace
    experiment_tracker_without_debug_info.dpsgd_noise_scale = experiment_tracker.dpsgd_noise_scale

    # joblib.dump(experiment_tracker,
    #             os.path.join(program_args.debug_output_directory,
    #                          f'experiment_uci_adult_{program_args.target_epsilon}_{program_args.task_id}.pkl'))

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

    joblib.dump(experiment_tracker,
                os.path.join(program_args.debug_output_directory,
                             f'experiment_uci_adult_{program_args.method}_{program_args.target_epsilon}_{program_args.task_id}_inference.pkl'))
    joblib.dump(experiment_tracker_without_debug_info,
                os.path.join(program_args.output_directory,
                             f'experiment_uci_adult_{program_args.method}_{program_args.target_epsilon}_{program_args.task_id}_inference.pkl'))


def calculate_accuracy(experiment_file_path):
    experiment = joblib.load(experiment_file_path)

    experiment_args = experiment.experiment_args

    xs = experiment.experiment_args.test_xs
    mean_theta = experiment.estimated_mu
    logits = jax.nn.sigmoid(jnp.sum(xs * mean_theta.reshape(1, -1), axis=-1)).ravel()
    accuracy = jnp.mean((logits >= 0.5) == (experiment_args.test_ys >= 0.5))
    print(accuracy)


def get_repeats_statistics_ci(repeated_coverages, alpha=0.05):
    mean = np.mean(repeated_coverages, axis=0)
    std_coverages = np.maximum(np.std(repeated_coverages, axis=0), 1e-9)
    lower = stats.norm(mean, std_coverages).ppf(alpha / 2)
    upper = stats.norm(mean, std_coverages).ppf(1 - alpha / 2)
    return lower, mean, upper


def get_repeats_statistics_std(repeated_coverages):
    mean = np.array([np.mean(coverages, axis=0) for coverages in repeated_coverages])
    std_coverages = np.array([np.maximum(np.std(coverages, axis=0), 1e-9) for coverages in repeated_coverages])
    lower = mean - std_coverages
    upper = mean + std_coverages
    return lower, mean, upper


def get_repeats_statistics_standard_error(repeated_coverages):
    mean = np.mean(repeated_coverages, axis=0)
    std_coverages = np.maximum(np.std(repeated_coverages, axis=0), 1e-9) / np.sqrt(len(repeated_coverages))
    lower = mean - std_coverages
    upper = mean + std_coverages
    return lower, mean, upper


def custom_calibration_curve(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    indices = np.argsort(y_prob)
    y_true_sorted = y_true[indices]
    y_prob_sorted = y_prob[indices]

    bin_size = len(y_true) // n_bins
    fraction_of_positives = []
    mean_predicted_value = []

    for i in range(n_bins):
        start = i * bin_size
        if i == n_bins - 1:
            end = len(y_true)
        else:
            end = (i + 1) * bin_size

        y_true_bin = y_true_sorted[start:end]
        y_prob_bin = y_prob_sorted[start:end]

        fraction_of_positives.append(np.mean(y_true_bin))
        mean_predicted_value.append(np.mean(y_prob_bin))

    return np.array(fraction_of_positives), np.array(mean_predicted_value)


def generate_predictions(base_dir, output_path):
    files = os.listdir(base_dir)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    for f in files:
        if not f.endswith('.pickle') and not f.endswith('.pkl'):
            continue
        file_path = os.path.join(base_dir, f)
        experiment = joblib.load(file_path)

        experiment_args = experiment.experiment_args
        experiment_args.pretty_print()

        theta_dimension = experiment_args.theta_dimension

        samples_count = 20000
        standard_dist = multivariate_normal(np.zeros(theta_dimension),
                                            np.eye(theta_dimension))
        standard_dist_samples = standard_dist.rvs(size=samples_count)

        if 'optimal_phi_samples' in experiment.to_dict():
            means = np.array(experiment.optimal_phi_samples[:, :experiment_args.theta_dimension])
            stds = np.array(experiment.covariance_optimal_phi_samples)
            distributions = np.concatenate([np.expand_dims(means, axis=1), np.expand_dims(stds, axis=1)],
                                           axis=1)

            dist_samples = distributions[np.random.randint(0, distributions.shape[0], size=samples_count)]
            means = dist_samples[:, 0, :]
            stds = dist_samples[:, 1, :]

            if theta_dimension == 1:
                standard_dist_samples = np.expand_dims(standard_dist_samples, -1)
        else:
            means = np.array([experiment.theta_means])
            stds = np.array([experiment.theta_stds])
        samples = means + stds * standard_dist_samples

        predicted_scores = []
        naive_predicted_scores = []
        test_xs = np.array(experiment_args.test_xs)
        test_ys = np.array(experiment_args.test_ys)

        test_points = test_xs.shape[0]
        progress_bar = tqdm(total=test_points)
        for i in range(test_points):
            logits = np.sum(np.array([test_xs[i]], np.float64) * samples, axis=-1)
            naive_logit = np.sum(experiment.mu_trace[-1] * test_xs[i])
            naive_predicted_scores.append(jnp.exp(numpyro.distributions.Bernoulli(logits=naive_logit).log_prob(1.0)))
            predicted_score = np.mean(jnp.exp(numpyro.distributions.Bernoulli(logits=logits).log_prob(1.0)))
            predicted_scores.append(predicted_score)
            progress_bar.update()
        progress_bar.close()

        predicted_scores = np.array(predicted_scores, np.float64)
        naive_predicted_scores = np.array(naive_predicted_scores, np.float64)
        joblib.dump((test_ys, predicted_scores, naive_predicted_scores), os.path.join(output_path, f))


def compute_calibrations(base_path, method, use_scikit_learn=False):
    prob_pred_list = []
    naive_prob_pred_list = []
    xs = np.linspace(0.01, 0.99, 100)
    files = os.listdir(base_path)
    for f in files:
        if not f.endswith('.pkl') and not f.endswith('.pickle'):
            continue
        (test_ys, predicted_scores, naive_predicted_scores) = joblib.load(os.path.join(base_path, f))
        try:
            if use_scikit_learn:
                prob_true, prob_pred = calibration_curve(test_ys, predicted_scores, n_bins=10)
                naive_prob_true, naive_prob_pred = calibration_curve(test_ys, naive_predicted_scores, n_bins=10)
            else:
                prob_true, prob_pred = custom_calibration_curve(test_ys, predicted_scores, n_bins=10)
                naive_prob_true, naive_prob_pred = custom_calibration_curve(test_ys, naive_predicted_scores, n_bins=10)
            if prob_true.shape[0] == 1:
                continue
            pred_interpolator = interpolate.interp1d(prob_true, prob_pred, fill_value='extrapolate')
            naive_pred_interpolator = interpolate.interp1d(naive_prob_true, naive_prob_pred, fill_value='extrapolate')

            prob_pred_list.append(pred_interpolator(xs))
            naive_prob_pred_list.append(naive_pred_interpolator(xs))
        except:
            print('yes')
            continue
    if method == 'ci':
        prob_pred_lower, mean_prob_pred, prob_pred_upper = get_repeats_statistics_ci(prob_pred_list, alpha=0.05)
        naive_prob_pred_lower, mean_naive_prob_pred, naive_prob_pred_upper = get_repeats_statistics_ci(
            naive_prob_pred_list,
            alpha=0.05)
    elif method == 'std':
        prob_pred_lower, mean_prob_pred, prob_pred_upper = get_repeats_statistics_std(prob_pred_list)
        naive_prob_pred_lower, mean_naive_prob_pred, naive_prob_pred_upper = get_repeats_statistics_std(
            naive_prob_pred_list)
    else:
        prob_pred_lower, mean_prob_pred, prob_pred_upper = get_repeats_statistics_standard_error(prob_pred_list)
        naive_prob_pred_lower, mean_naive_prob_pred, naive_prob_pred_upper = get_repeats_statistics_standard_error(
            naive_prob_pred_list)
    prob_pred = np.array(prob_pred_list)
    naive_pred = np.array(naive_prob_pred_list)
    return xs, prob_pred, naive_pred, mean_prob_pred, prob_pred_lower, prob_pred_upper, mean_naive_prob_pred, naive_prob_pred_lower, naive_prob_pred_upper


def plot_calibration_repeats(output_file_name,
                             base_path,
                             method='ci'):
    matplotlib.use('TKAgg')

    (xs, prob_pred, naive_pred, mean_prob_pred, prob_pred_lower, prob_pred_upper, mean_naive_prob_pred,
     naive_prob_pred_lower, naive_prob_pred_upper) = compute_calibrations(base_path, method)

    plt.rcParams.update(bundles.aistats2023())
    plt.rcParams.update(tueplots.figsizes.aistats2023_full())
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')

    x_ticks = np.linspace(0, 1, 6)
    y_ticks = np.linspace(0, 1, 6)

    fig, axs = plt.subplots(1, 1, figsize=(4, 2))
    axs.grid(True, linestyle='--', alpha=0.7)
    axs.set_xlabel('True Probabilities')
    axs.set_ylabel('Predicted Probabilities')
    axs.set_xticks(x_ticks)
    axs.set_yticks(y_ticks)
    axs.set_xlim(0, 1)
    axs.set_ylim(0, 1)

    axs.plot(xs, xs, linestyle='--', color='gray', label='Perfect calibration', linewidth=1.5)
    axs.plot(xs, mean_prob_pred, color='blue', label='Noise-aware', linewidth=1.0, alpha=0.8)
    axs.fill_between(xs,
                     prob_pred_lower,
                     prob_pred_upper,
                     color='blue', linewidth=1.0, alpha=0.25)

    axs.plot(xs, mean_naive_prob_pred, color='red', label='Naive', linewidth=1.0, alpha=0.8)
    axs.fill_between(xs,
                     naive_prob_pred_lower,
                     naive_prob_pred_upper,
                     color='red', linewidth=1.0, alpha=0.25)

    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', title='Legend')
    plt.tight_layout(rect=[0, 0, 1.0 / (1.75), 1])
    plt.savefig(f'{output_file_name}.pdf', dpi=300)
    plt.close('all')


if __name__ == '__main__':
    main()
    # use the following to generate calibration plots:
    # generate_predictions('path-to-pickled-experiment-files', './results')
    # plot_calibration_repeats('./adult-calibration-curve', './results')
