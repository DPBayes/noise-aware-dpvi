import math
import os
from argparse import ArgumentParser

import joblib
import matplotlib
import numpy as np
import tueplots
import tueplots.figsizes
from matplotlib import pyplot as plt, gridspec
from scipy import stats
from scipy.stats import multivariate_normal
from tueplots import bundles

matplotlib.use('TKAgg')

ERROR_BAND_ALPHA = 0.25
COVERAGE_LINE_ALPHA = 0.7


def compute_tarp_frequency(posterior_samples, theta_true, theta_ref):
    prior_ref_distance = np.sqrt(np.sum(np.square(theta_true - theta_ref)))
    sample_ref_distances = np.sqrt(np.sum(np.square(posterior_samples - theta_ref.reshape(1, -1)), axis=-1))
    return np.mean(sample_ref_distances < prior_ref_distance)


def compute_tarp_marginal_frequency(posterior_samples, theta_true, theta_ref):
    prior_ref_distances = np.abs(theta_true - theta_ref)
    sample_ref_distances = np.abs(posterior_samples - theta_ref.reshape(1, -1))
    return np.mean(sample_ref_distances < prior_ref_distances.reshape(1, -1), axis=0)


def default_theta_ref_generator(posterior_samples, uniform_distribution_scale=4.0):
    uniform_half_range = np.std(posterior_samples, axis=0) * uniform_distribution_scale

    def generator(transformed_data):
        return transformed_data + np.random.uniform(-uniform_half_range, uniform_half_range)

    return generator


class TARPExperiment:
    def __init__(self, posterior_samples, theta_true, data,
                 data_function=lambda x: x,
                 theta_ref_generator=None):
        self.posterior_samples = posterior_samples
        self.theta_true = theta_true
        self.data = data
        self.data_function = data_function

        if theta_ref_generator is None:
            self.theta_ref_generator = default_theta_ref_generator(posterior_samples)

        self.frequency = None
        self.marginal_frequency = None

    def get_tarp_frequency(self):
        theta_ref = self.theta_ref_generator(self.data_function(self.data))
        self.frequency = compute_tarp_frequency(self.posterior_samples, self.theta_true, theta_ref)
        return self.frequency

    def get_tarp_marginal_frequency(self):
        theta_ref = self.theta_ref_generator(self.data_function(self.data))
        self.marginal_frequency = compute_tarp_marginal_frequency(self.posterior_samples, self.theta_true, theta_ref)
        return self.marginal_frequency


class TARPEmpiricalCoverageTest:
    def __init__(self, tarp_experiments: list[TARPExperiment], credible_alpha_step_size=0.05):
        self.tarp_experiments = tarp_experiments

        self.credible_alphas = np.concatenate([np.ones((1,), np.float64) * 0.001,
                                               np.arange(credible_alpha_step_size, 1.0,
                                                         credible_alpha_step_size).astype(np.float64),
                                               np.ones((1,), np.float64) * 0.999])

        self.frequencies = None
        self.coverages = None
        self.marginal_frequencies = None
        self.marginal_coverages = None

        self.__theta_dimension = self.tarp_experiments[0].theta_true.shape[0]

    def get_coverages(self):
        self.frequencies = np.zeros(len(self.tarp_experiments), np.float64)
        for i, experiment in enumerate(self.tarp_experiments):
            self.frequencies[i] = experiment.get_tarp_frequency()
        self.coverages = np.zeros(self.credible_alphas.shape[0], np.float64)
        for i in range(self.credible_alphas.shape[0]):
            self.coverages[i] = np.mean(self.frequencies < (1 - self.credible_alphas[i]))
        return self.coverages

    def get_marginal_coverages(self):
        self.marginal_frequencies = np.zeros((self.__theta_dimension, len(self.tarp_experiments)),
                                             np.float64)
        for i, experiment in enumerate(self.tarp_experiments):
            self.marginal_frequencies[:, i] = experiment.get_tarp_marginal_frequency()
        self.marginal_coverages = np.zeros((self.__theta_dimension, self.credible_alphas.shape[0]),
                                           np.float64)
        for i in range(self.credible_alphas.shape[0]):
            credible_level_per_dimension = np.repeat(1 - self.credible_alphas[i],
                                                     self.__theta_dimension).reshape(-1, 1)
            self.marginal_coverages[:, i] = np.mean(self.marginal_frequencies < credible_level_per_dimension,
                                                    axis=-1)
        return self.marginal_coverages

    def get_coverages_with_repeats(self, repeats=20):
        repeated_coverages = np.zeros((repeats, self.credible_alphas.shape[0]), np.float64)
        for i in range(repeats):
            repeated_coverages[i] = self.get_coverages()
        return repeated_coverages

    def get_marginal_coverages_with_repeats(self, repeats=20):
        repeated_marginal_coverages = np.zeros((repeats, self.__theta_dimension, self.credible_alphas.shape[0]),
                                               np.float64)
        for i in range(repeats):
            repeated_marginal_coverages[i] = self.get_marginal_coverages()
        return repeated_marginal_coverages


def get_repeated_coverages_statistics_ci(repeated_coverages, alpha=0.05):
    mean_coverages = np.mean(repeated_coverages, axis=0)
    std_coverages = np.maximum(np.std(repeated_coverages, axis=0), 1e-9)
    lower_coverages = stats.norm(mean_coverages, std_coverages).ppf(alpha / 2)
    upper_coverages = stats.norm(mean_coverages, std_coverages).ppf(1 - alpha / 2)
    return lower_coverages, mean_coverages, upper_coverages


def get_repeated_coverages_statistics_std(repeated_coverages):
    mean_coverages = np.mean(repeated_coverages, axis=0)
    std_coverages = np.maximum(np.std(repeated_coverages, axis=0), 1e-9)
    lower_coverages = mean_coverages - std_coverages
    upper_coverages = mean_coverages + std_coverages
    return lower_coverages, mean_coverages, upper_coverages


def get_repeated_coverages_statistics_standard_error(repeated_coverages):
    mean_coverages = np.mean(repeated_coverages, axis=0)
    std_coverages = np.maximum(np.std(repeated_coverages, axis=0), 1e-9)
    standard_error = std_coverages / math.sqrt(repeated_coverages.shape[0])
    lower_coverages = mean_coverages - standard_error
    upper_coverages = mean_coverages + standard_error
    return lower_coverages, mean_coverages, upper_coverages


def plot_per_dimension_coverages(per_dimension_coverages,
                                 credible_levels,
                                 output_file_name,
                                 plot_title,
                                 dimension_titles,
                                 lower_per_dimension_coverages=None,
                                 upper_per_dimension_coverages=None,
                                 experiments_per_row=3):
    plt.rcParams.update(bundles.aistats2023())
    plt.rcParams.update(tueplots.figsizes.aistats2023_full())
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')

    x_ticks = np.linspace(0, 1, 6)
    y_ticks = np.linspace(0, 1, 6)

    theta_dimension = per_dimension_coverages.shape[0]

    rows = theta_dimension // experiments_per_row
    if theta_dimension % experiments_per_row != 0:
        rows += 1
    fig, axs = plt.subplots(rows, experiments_per_row, figsize=(2 * (experiments_per_row + 1), 2 * rows))
    for i in range(theta_dimension):
        if rows <= 1:
            axs_i = axs[i]
        else:
            axs_i = axs[i // experiments_per_row, i % experiments_per_row]
        axs_i.grid(True, linestyle='--', alpha=COVERAGE_LINE_ALPHA)
        axs_i.set_xlabel('1 - $\\alpha$')
        axs_i.set_ylabel('Coverage')
        axs_i.set_xticks(x_ticks)
        axs_i.set_yticks(y_ticks)
        axs_i.set_xlim(0, 1)
        axs_i.set_ylim(0, 1)
        if dimension_titles is None:
            axs_i.set_title(f'Dimension {i + 1}')
        else:
            axs_i.set_title(dimension_titles[i])
        axs_i.plot(credible_levels, credible_levels, linestyle='--', color='gray', label='Perfect Coverage',
                   linewidth=1.5)

    for i in range(theta_dimension):
        if rows <= 1:
            axs_i = axs[i]
        else:
            axs_i = axs[i // experiments_per_row, i % experiments_per_row]
        axs_i.plot(credible_levels, per_dimension_coverages[i], color='blue', label='TARP', linewidth=1.0,
                   alpha=COVERAGE_LINE_ALPHA)
        if lower_per_dimension_coverages is not None and upper_per_dimension_coverages is not None:
            axs_i.fill_between(credible_levels,
                               lower_per_dimension_coverages[i],
                               upper_per_dimension_coverages[i],
                               color='blue', linewidth=1.0, alpha=ERROR_BAND_ALPHA)

    unused = theta_dimension % experiments_per_row
    while unused != 0:
        if rows <= 1:
            axs_i = axs[unused]
        else:
            axs_i = axs[rows - 1, unused]
        fig.delaxes(axs_i)
        unused = (unused + 1) % experiments_per_row

    # fig.suptitle(plot_title)
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', title='Legend')
    plt.tight_layout(rect=[0, 0, experiments_per_row / (experiments_per_row + 0.75), 1])
    plt.savefig(f'{output_file_name}.pdf', dpi=300)
    plt.close('all')


def plot_marginal_coverages(tarp_empirical_coverage_test: TARPEmpiricalCoverageTest,
                            output_file_name,
                            plot_title,
                            dimension_titles=None,
                            experiments_per_row=3):
    plot_per_dimension_coverages(tarp_empirical_coverage_test.get_marginal_coverages(),
                                 1 - tarp_empirical_coverage_test.credible_alphas,
                                 output_file_name,
                                 plot_title, dimension_titles,
                                 experiments_per_row=experiments_per_row)


def plot_marginal_coverages_smoothed(tarp_empirical_coverage_test: TARPEmpiricalCoverageTest,
                                     output_file_name,
                                     plot_title,
                                     dimension_titles=None,
                                     repeats=20,
                                     experiments_per_row=3,
                                     method='ci'):
    repeated_marginal_coverages = tarp_empirical_coverage_test.get_marginal_coverages_with_repeats(repeats)

    if method == 'ci':
        lower_marginal_coverages, mean_marginal_coverages, upper_marginal_coverages = get_repeated_coverages_statistics_ci(
            repeated_marginal_coverages)
    elif method == 'std':
        lower_marginal_coverages, mean_marginal_coverages, upper_marginal_coverages = get_repeated_coverages_statistics_std(
            repeated_marginal_coverages)
    else:
        lower_marginal_coverages, mean_marginal_coverages, upper_marginal_coverages = get_repeated_coverages_statistics_standard_error(
            repeated_marginal_coverages)

    plot_per_dimension_coverages(mean_marginal_coverages,
                                 1 - tarp_empirical_coverage_test.credible_alphas,
                                 output_file_name,
                                 plot_title,
                                 dimension_titles,
                                 lower_marginal_coverages,
                                 upper_marginal_coverages,
                                 experiments_per_row)


def plot_full_coverages(coverages,
                        credible_levels,
                        output_file_name,
                        plot_title,
                        subtitle,
                        lower_coverages=None,
                        upper_coverages=None):
    plt.rcParams.update(bundles.aistats2023())
    plt.rcParams.update(tueplots.figsizes.aistats2023_full())
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')

    x_ticks = np.linspace(0, 1, 6)
    y_ticks = np.linspace(0, 1, 6)

    fig, axs = plt.subplots(1, 1)
    axs.grid(True, linestyle='--', alpha=COVERAGE_LINE_ALPHA)
    axs.set_xlabel('1 - $\\alpha$')
    axs.set_ylabel('Coverage')
    axs.set_xticks(x_ticks)
    axs.set_yticks(y_ticks)
    axs.set_xlim(0, 1)
    axs.set_ylim(0, 1)
    axs.set_title(subtitle)

    axs.plot(credible_levels, credible_levels, linestyle='--', color='gray', label='Perfect Coverage', linewidth=1.5)
    axs.plot(credible_levels, coverages, color='blue', label='TARP', linewidth=1.0, alpha=COVERAGE_LINE_ALPHA)
    if lower_coverages is not None and upper_coverages is not None:
        axs.fill_between(credible_levels,
                         lower_coverages,
                         upper_coverages,
                         color='blue', linewidth=1.0, alpha=ERROR_BAND_ALPHA)

    # fig.suptitle(plot_title)
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', title='Legend')
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.savefig(f'{output_file_name}.pdf', dpi=300)
    plt.close('all')


def plot_coverages(tarp_empirical_coverage_test: TARPEmpiricalCoverageTest,
                   output_file_name,
                   plot_title,
                   subtitle):
    plot_full_coverages(tarp_empirical_coverage_test.get_coverages(),
                        1 - tarp_empirical_coverage_test.credible_alphas,
                        output_file_name,
                        plot_title,
                        subtitle)


def plot_coverages_smoothed(tarp_empirical_coverage_test: TARPEmpiricalCoverageTest,
                            output_file_name,
                            plot_title,
                            subtitle,
                            repeats=20,
                            method='ci'):
    repeated_coverages = tarp_empirical_coverage_test.get_coverages_with_repeats(repeats)

    if method == 'ci':
        lower_coverages, mean_coverages, upper_coverages = get_repeated_coverages_statistics_ci(repeated_coverages)
    elif method == 'std':
        lower_coverages, mean_coverages, upper_coverages = get_repeated_coverages_statistics_std(repeated_coverages)
    else:
        lower_coverages, mean_coverages, upper_coverages = get_repeated_coverages_statistics_standard_error(
            repeated_coverages)

    plot_full_coverages(mean_coverages,
                        1 - tarp_empirical_coverage_test.credible_alphas,
                        output_file_name,
                        plot_title,
                        subtitle,
                        lower_coverages,
                        upper_coverages)


def plot_compare_per_dimension_coverages(per_dimension_coverages_list1,
                                         per_dimension_coverages_list2,
                                         credible_levels,
                                         output_file_name,
                                         plot_title,
                                         tests_dimension_titles,
                                         coverages1_label,
                                         coverages2_label,
                                         lower_per_dimension_coverages_list1=None,
                                         upper_per_dimension_coverages_list1=None,
                                         lower_per_dimension_coverages_list2=None,
                                         upper_per_dimension_coverages_list2=None,
                                         experiments_per_row=3):
    plt.rcParams.update(bundles.aistats2023())
    plt.rcParams.update(tueplots.figsizes.aistats2023_full())
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')

    x_ticks = np.linspace(0, 1, 6)
    y_ticks = np.linspace(0, 1, 6)

    if coverages1_label is None:
        coverages1_label = 'TARP 1'
    if coverages2_label is None:
        coverages2_label = 'TARP 2'

    dimensions = sum([coverages.shape[0] for coverages in per_dimension_coverages_list1])

    rows = dimensions // experiments_per_row
    if dimensions % experiments_per_row != 0:
        rows += 1
    fig, axs = plt.subplots(rows, experiments_per_row, figsize=(2 * (experiments_per_row + 1), 2 * rows))
    for i in range(dimensions):
        if rows <= 1:
            axs_i = axs[i]
        else:
            axs_i = axs[i // experiments_per_row, i % experiments_per_row]
        axs_i.grid(True, linestyle='--', alpha=COVERAGE_LINE_ALPHA)
        axs_i.set_xlabel('1 - $\\alpha$')
        axs_i.set_ylabel('Coverage')
        axs_i.set_xticks(x_ticks)
        axs_i.set_yticks(y_ticks)
        axs_i.set_xlim(0, 1)
        axs_i.set_ylim(0, 1)
        if tests_dimension_titles is None:
            axs_i.set_title(f'Dimension {i + 1}')
        else:
            axs_i.set_title(tests_dimension_titles[i])
        axs_i.plot(credible_levels, credible_levels, linestyle='--', color='gray', label='Perfect Coverage',
                   linewidth=1.5)

    coverages_index = 0
    coverages_dimension = 0
    for i in range(dimensions):
        if rows <= 1:
            axs_i = axs[i]
        else:
            axs_i = axs[i // experiments_per_row, i % experiments_per_row]

        coverages1 = per_dimension_coverages_list1[coverages_index][coverages_dimension]
        axs_i.plot(credible_levels, coverages1, color='blue', label=coverages1_label, linewidth=1.0,
                   alpha=COVERAGE_LINE_ALPHA)
        if lower_per_dimension_coverages_list1 is not None and upper_per_dimension_coverages_list1 is not None:
            axs_i.fill_between(credible_levels,
                               lower_per_dimension_coverages_list1[coverages_index][coverages_dimension],
                               upper_per_dimension_coverages_list1[coverages_index][coverages_dimension],
                               color='blue', linewidth=1.0, alpha=ERROR_BAND_ALPHA)

        coverages2 = per_dimension_coverages_list2[coverages_index][coverages_dimension]
        axs_i.plot(credible_levels, coverages2, color='red', label=coverages2_label, linewidth=1.0,
                   alpha=COVERAGE_LINE_ALPHA)
        if lower_per_dimension_coverages_list2 is not None and upper_per_dimension_coverages_list2 is not None:
            axs_i.fill_between(credible_levels,
                               lower_per_dimension_coverages_list2[coverages_index][coverages_dimension],
                               upper_per_dimension_coverages_list2[coverages_index][coverages_dimension],
                               color='red', linewidth=1.0, alpha=ERROR_BAND_ALPHA)

        coverages_dimension += 1
        if coverages_dimension >= per_dimension_coverages_list1[coverages_index].shape[0]:
            coverages_dimension = 0
            coverages_index += 1

    unused = dimensions % experiments_per_row
    while unused != 0:
        if rows <= 1:
            axs_i = axs[unused]
        else:
            axs_i = axs[rows - 1, unused]
        fig.delaxes(axs_i)
        unused = (unused + 1) % experiments_per_row

    # fig.suptitle(plot_title)
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', title='Legend')
    plt.tight_layout(rect=[0, 0, experiments_per_row / (experiments_per_row + 0.75), 1])
    plt.savefig(f'{output_file_name}.pdf', dpi=300)
    plt.close('all')


def plot_compare_coverages(coverages_list1,
                           coverages_list2,
                           credible_levels,
                           output_file_name,
                           plot_title,
                           tests_dimension_titles,
                           coverages1_label,
                           coverages2_label,
                           lower_per_dimension_coverages_list1=None,
                           upper_per_dimension_coverages_list1=None,
                           lower_per_dimension_coverages_list2=None,
                           upper_per_dimension_coverages_list2=None,
                           experiments_per_row=3):
    plt.rcParams.update(bundles.aistats2023())
    plt.rcParams.update(tueplots.figsizes.aistats2023_full())
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')

    x_ticks = np.linspace(0, 1, 6)
    y_ticks = np.linspace(0, 1, 6)

    if coverages1_label is None:
        coverages1_label = 'TARP 1'
    if coverages2_label is None:
        coverages2_label = 'TARP 2'

    dimensions = len(coverages_list1)

    rows = dimensions // experiments_per_row
    if dimensions % experiments_per_row != 0:
        rows += 1
    fig, axs = plt.subplots(rows, experiments_per_row, figsize=(2 * (experiments_per_row + 1), 2 * rows))
    for i in range(dimensions):
        if rows <= 1:
            if dimensions == 1:
                axs_i = axs
            else:
                axs_i = axs[i]
        else:
            axs_i = axs[i // experiments_per_row, i % experiments_per_row]
        axs_i.grid(True, linestyle='--', alpha=COVERAGE_LINE_ALPHA)
        axs_i.set_xlabel('1 - $\\alpha$')
        axs_i.set_ylabel('Coverage')
        axs_i.set_xticks(x_ticks)
        axs_i.set_yticks(y_ticks)
        axs_i.set_xlim(0, 1)
        axs_i.set_ylim(0, 1)
        if tests_dimension_titles is None:
            axs_i.set_title(f'Dimension {i + 1}')
        else:
            axs_i.set_title(tests_dimension_titles[i])
        axs_i.plot(credible_levels, credible_levels, linestyle='--', color='gray', label='Perfect Coverage',
                   linewidth=1.5)

    coverages_index = 0
    for i in range(dimensions):
        if rows <= 1:
            if dimensions == 1:
                axs_i = axs
            else:
                axs_i = axs[i]
        else:
            axs_i = axs[i // experiments_per_row, i % experiments_per_row]

        coverages1 = coverages_list1[i]
        axs_i.plot(credible_levels, coverages1, color='blue', label=coverages1_label, linewidth=1.0,
                   alpha=COVERAGE_LINE_ALPHA)
        if lower_per_dimension_coverages_list1 is not None and upper_per_dimension_coverages_list1 is not None:
            axs_i.fill_between(credible_levels,
                               lower_per_dimension_coverages_list1[i],
                               upper_per_dimension_coverages_list1[i],
                               color='blue', linewidth=1.0, alpha=ERROR_BAND_ALPHA)

        coverages2 = coverages_list2[i]
        axs_i.plot(credible_levels, coverages2, color='red', label=coverages2_label, linewidth=1.0,
                   alpha=COVERAGE_LINE_ALPHA)
        if lower_per_dimension_coverages_list2 is not None and upper_per_dimension_coverages_list2 is not None:
            axs_i.fill_between(credible_levels,
                               lower_per_dimension_coverages_list2[i],
                               upper_per_dimension_coverages_list2[i],
                               color='red', linewidth=1.0, alpha=ERROR_BAND_ALPHA)
        coverages_index += 1

    unused = dimensions % experiments_per_row
    while unused != 0:
        if rows <= 1:
            axs_i = axs[unused]
        else:
            axs_i = axs[rows - 1, unused]
        fig.delaxes(axs_i)
        unused = (unused + 1) % experiments_per_row

    # fig.suptitle(plot_title)
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', title='Legend')
    plt.tight_layout(rect=[0, 0, experiments_per_row / (experiments_per_row + 0.75), 1])
    plt.savefig(f'{output_file_name}.pdf', dpi=300)
    plt.close('all')


def plot_tarp_compare_per_dimension_coverages(tarp_empirical_coverage_tests1: list[TARPEmpiricalCoverageTest],
                                              tarp_empirical_coverage_tests2: list[TARPEmpiricalCoverageTest],
                                              output_file_name,
                                              plot_title,
                                              tests_dimension_titles=None,
                                              coverages_label1=None,
                                              coverages_label2=None,
                                              experiments_per_row=3):
    plot_compare_per_dimension_coverages([tarp_empirical_coverage_test.get_marginal_coverages()
                                          for tarp_empirical_coverage_test in tarp_empirical_coverage_tests1],
                                         [tarp_empirical_coverage_test.get_marginal_coverages()
                                          for tarp_empirical_coverage_test in tarp_empirical_coverage_tests2],
                                         1 - tarp_empirical_coverage_tests1[0].credible_alphas,
                                         output_file_name,
                                         plot_title,
                                         tests_dimension_titles,
                                         coverages_label1,
                                         coverages_label2,
                                         experiments_per_row=experiments_per_row)


def lower_mean_and_upper_coverages_smoothed(repeated_coverages_list, method='ci'):
    lower_per_dimension_coverages_list = []
    mean_per_dimension_coverages_list = []
    upper_per_dimension_coverages_list = []
    for repeated_coverages in repeated_coverages_list:
        if method == 'ci':
            lower_coverages, mean_coverages, upper_coverages = get_repeated_coverages_statistics_ci(repeated_coverages)
        elif method == 'std':
            lower_coverages, mean_coverages, upper_coverages = get_repeated_coverages_statistics_std(repeated_coverages)
        else:
            lower_coverages, mean_coverages, upper_coverages = get_repeated_coverages_statistics_standard_error(
                repeated_coverages)
        lower_per_dimension_coverages_list.append(lower_coverages)
        mean_per_dimension_coverages_list.append(mean_coverages)
        upper_per_dimension_coverages_list.append(upper_coverages)
    return lower_per_dimension_coverages_list, mean_per_dimension_coverages_list, upper_per_dimension_coverages_list


def plot_tarp_compare_per_dimension_coverages_smoothed(tarp_empirical_coverage_tests1: list[TARPEmpiricalCoverageTest],
                                                       tarp_empirical_coverage_tests2: list[TARPEmpiricalCoverageTest],
                                                       output_file_name,
                                                       plot_title,
                                                       tests_dimension_titles=None,
                                                       coverages_label1=None,
                                                       coverages_label2=None,
                                                       experiments_per_row=3,
                                                       repeats=20,
                                                       method='ci'):
    (lower_per_dimension_coverages_list1,
     mean_per_dimension_coverages_list1,
     upper_per_dimension_coverages_list1) = lower_mean_and_upper_coverages_smoothed(
        [tarp_empirical_coverage_test.get_marginal_coverages_with_repeats(repeats)
         for tarp_empirical_coverage_test in tarp_empirical_coverage_tests1],
        method)

    (lower_per_dimension_coverages_list2,
     mean_per_dimension_coverages_list2,
     upper_per_dimension_coverages_list2) = lower_mean_and_upper_coverages_smoothed(
        [tarp_empirical_coverage_test.get_marginal_coverages_with_repeats(repeats)
         for tarp_empirical_coverage_test in tarp_empirical_coverage_tests2],
        method)

    plot_compare_per_dimension_coverages(mean_per_dimension_coverages_list1,
                                         mean_per_dimension_coverages_list2,
                                         1 - tarp_empirical_coverage_tests1[0].credible_alphas,
                                         output_file_name,
                                         plot_title,
                                         tests_dimension_titles,
                                         coverages_label1,
                                         coverages_label2,
                                         lower_per_dimension_coverages_list1,
                                         upper_per_dimension_coverages_list1,
                                         lower_per_dimension_coverages_list2,
                                         upper_per_dimension_coverages_list2,
                                         experiments_per_row=experiments_per_row)


def plot_tarp_compare_coverages(tarp_empirical_coverage_tests1: list[TARPEmpiricalCoverageTest],
                                tarp_empirical_coverage_tests2: list[TARPEmpiricalCoverageTest],
                                output_file_name,
                                plot_title,
                                tests_dimension_titles=None,
                                coverages_label1=None,
                                coverages_label2=None,
                                experiments_per_row=3):
    plot_compare_coverages([tarp_empirical_coverage_test.get_coverages()
                            for tarp_empirical_coverage_test in tarp_empirical_coverage_tests1],
                           [tarp_empirical_coverage_test.get_coverages()
                            for tarp_empirical_coverage_test in tarp_empirical_coverage_tests2],
                           1 - tarp_empirical_coverage_tests1[0].credible_alphas,
                           output_file_name,
                           plot_title,
                           tests_dimension_titles,
                           coverages_label1,
                           coverages_label2,
                           experiments_per_row=experiments_per_row)


def plot_tarp_compare_coverages_smoothed(tarp_empirical_coverage_tests1: list[TARPEmpiricalCoverageTest],
                                         tarp_empirical_coverage_tests2: list[TARPEmpiricalCoverageTest],
                                         output_file_name,
                                         plot_title,
                                         tests_dimension_titles=None,
                                         coverages_label1=None,
                                         coverages_label2=None,
                                         experiments_per_row=3,
                                         repeats=20,
                                         method='ci'):
    (lower_coverages_list1,
     mean_coverages_list1,
     upper_coverages_list1) = lower_mean_and_upper_coverages_smoothed(
        [tarp_empirical_coverage_test.get_coverages_with_repeats(repeats)
         for tarp_empirical_coverage_test in tarp_empirical_coverage_tests1],
        method)
    (lower_coverages_list2,
     mean_coverages_list2,
     upper_coverages_list2) = lower_mean_and_upper_coverages_smoothed(
        [tarp_empirical_coverage_test.get_coverages_with_repeats(repeats)
         for tarp_empirical_coverage_test in tarp_empirical_coverage_tests2],
        method)

    plot_compare_coverages(mean_coverages_list1,
                           mean_coverages_list2,
                           1 - tarp_empirical_coverage_tests1[0].credible_alphas,
                           output_file_name,
                           plot_title,
                           tests_dimension_titles,
                           coverages_label1,
                           coverages_label2,
                           lower_coverages_list1,
                           upper_coverages_list1,
                           lower_coverages_list2,
                           upper_coverages_list2,
                           experiments_per_row=experiments_per_row)


def plot_tarp_coverages(experiments,
                        coverages_output_file_name,
                        plot_title,
                        multidimensional_title='',
                        marginal_titles=None):
    tarp_empirical_coverage_test = TARPEmpiricalCoverageTest(experiments['tarp_experiments'])
    filled_plot_title = plot_title.replace('#{epsilon}', str(experiments['epsilon']))
    dpsgd_noise_scale = experiments['dpsgd_noise_scale']
    if dpsgd_noise_scale is not None:
        filled_plot_title = filled_plot_title.replace('#{dpsgd_noise_scale}', str(dpsgd_noise_scale))
    sampling_rate = experiments['sampling_rate']
    if sampling_rate is not None:
        filled_plot_title = filled_plot_title.replace('#{sampling_rate}', str(sampling_rate))

    if experiments['theta_dimension'] > 1:
        plot_marginal_coverages(tarp_empirical_coverage_test,
                                coverages_output_file_name + '_marginal',
                                filled_plot_title,
                                marginal_titles,
                                experiments_per_row=min(experiments['theta_dimension'], 3))
        plot_marginal_coverages_smoothed(tarp_empirical_coverage_test,
                                         coverages_output_file_name + '_marginal_smoothed',
                                         filled_plot_title,
                                         marginal_titles,
                                         repeats=20,
                                         experiments_per_row=min(experiments['theta_dimension'], 3))

    plot_coverages(tarp_empirical_coverage_test,
                   coverages_output_file_name,
                   filled_plot_title,
                   multidimensional_title)
    plot_coverages_smoothed(tarp_empirical_coverage_test,
                            coverages_output_file_name + '_smoothed',
                            filled_plot_title,
                            multidimensional_title,
                            repeats=20)


def load_tarp_experiments(experiments_directory,
                          additional_filter='',
                          samples_count=20000):
    files = os.listdir(experiments_directory)
    experiments = []

    for f in files:
        if not f.endswith('.pkl') and not f.endswith('.pickle'):
            continue
        if additional_filter.strip() != '' and f.find(additional_filter) == -1:
            continue
        experiment_tracker = joblib.load(os.path.join(experiments_directory, f))
        experiment_tracker.file_name = os.path.join(experiments_directory, f)
        experiments.append((experiment_tracker.experiment_args, experiment_tracker))

    distributions_dict = {}
    for experiment_args, experiment_tracker in experiments:
        if 'optimal_phi_samples' in experiment_tracker.to_dict():
            means = np.array(experiment_tracker.optimal_phi_samples[:, :experiment_args.theta_dimension])
            stds = np.array(experiment_tracker.covariance_optimal_phi_samples)
            distributions_dict[experiment_args.task_id] = np.concatenate([np.expand_dims(means, axis=1),
                                                                          np.expand_dims(stds, axis=1)],
                                                                         axis=1)

    theta_dimension = experiments[0][0].theta_dimension

    tarp_experiments = []
    for k, (experiment_args, experiment_tracker) in enumerate(experiments):
        if 'optimal_phi_samples' not in experiment_tracker.to_dict():
            means = np.reshape(experiment_tracker.theta_means, (1, -1))
            stds = np.reshape(experiment_tracker.theta_stds, (1, -1))
        else:
            distributions = distributions_dict[experiment_args.task_id]
            dist_samples = distributions[np.random.randint(0, distributions.shape[0], size=samples_count)]
            means = dist_samples[:, 0, :]
            stds = dist_samples[:, 1, :]
        if np.isnan(means).any() or np.isnan(stds).any():
            continue

        standard_dist = multivariate_normal(np.zeros(theta_dimension), np.eye(theta_dimension))
        standard_dist_samples = standard_dist.rvs(size=samples_count)
        if theta_dimension == 1:
            standard_dist_samples = np.expand_dims(standard_dist_samples, -1)
        samples = means + stds * standard_dist_samples

        if 'theta_transform' not in experiment_args.to_dict():
            theta_true = np.array(experiment_args.theta_true)
        else:
            theta_true = experiment_args.theta_transform(experiment_args.theta_true)
        if 'trace_burn_in_percentage' in experiment_args.to_dict():
            noisy_data = np.mean(experiment_tracker.mu_trace[
                                 int(experiment_args.trace_burn_in_percentage * experiment_tracker.phi_trace.shape[
                                     0]):, :], axis=0)
        else:
            noisy_data = np.mean(experiment_tracker.mu_trace[int(0.2 * experiment_tracker.phi_trace.shape[0]):, :],
                                 axis=0)

        tarp_experiments.append(TARPExperiment(samples, theta_true, noisy_data))

    epsilon = float(experiments[0][0].target_epsilon)
    dpsgd_noise_scale = float(experiments[0][1].dpsgd_noise_scale)
    if 'sampling_rate' not in experiments[0][0].to_dict():
        sampling_rate = 1.0
    else:
        sampling_rate = float(experiments[0][0].sampling_rate)

    del experiments

    return {
        'dpsgd_noise_scale': dpsgd_noise_scale,
        'epsilon': epsilon,
        'sampling_rate': sampling_rate,
        'theta_dimension': theta_dimension,
        'tarp_experiments': tarp_experiments
    }


def parse_program_args():
    parser = ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--input_path', type=str, help='Coverage experiments path')
    parser.add_argument('--experiment_name', type=str, help='Experiment directory name')
    parser.add_argument('--output_path', type=str, help='Generated coverage plots directory path')
    program_args, _ = parser.parse_known_args()
    return program_args


def generate_plots():
    program_args = parse_program_args()

    for epsilon in [0.1, 0.3, 1.0]:
        for method in ['laplace', 'mcmc']:
            plot_dir = os.path.join(program_args.output_path, f'{program_args.experiment_name}/{epsilon}/{method}')
            if not os.path.isdir(plot_dir):
                os.makedirs(plot_dir)
            plot_file_name = os.path.join(plot_dir, 'coverages')
            plot_tarp_coverages(
                load_tarp_experiments(
                    os.path.join(program_args.input_path, f'{program_args.experiment_name}/{epsilon}/{method}')),
                plot_file_name,
                f'{program_args.experiment_name} ('
                '$\\epsilon = #{epsilon}$, '
                '$q_{\\text{dp}}'
                ' = #{sampling_rate}$, '
                '$\\sigma_{\\text{dp}}'
                ' = #{dpsgd_noise_scale}$)')


if __name__ == '__main__':
    generate_plots()
