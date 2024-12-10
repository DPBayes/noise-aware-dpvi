import os.path
from argparse import ArgumentParser

EXPERIMENT_NAMES = ['logistic_regression', 'gamma_exponential', 'beta_bernoulli', 'beta_binomial',
                    'dirichlet_categorical', 'linear_regression', 'linear_regression10d']


def parse_program_args():
    parser = ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--experiment_name', type=str, default='linear_regression10d',
                        help='One of the following: ' + ','.join(EXPERIMENT_NAMES))
    program_args, _ = parser.parse_known_args()
    return program_args


def main():
    program_args = parse_program_args()

    array_max_size = 40
    target_count = 500
    number_of_scripts = target_count // array_max_size
    scripts_directory = './scripts'

    if not os.path.isdir(scripts_directory):
        os.mkdir(scripts_directory)

    with open('slurm_coverage_tests.sh', 'r') as f:
        template_script = f.read()

    lines = []
    for epsilon in [0.1, 0.3, 1.0]:
        for use_mcmc in [True, False]:
            if use_mcmc:
                method = 'mcmc'
            else:
                method = 'laplace'
            for i in range(number_of_scripts):
                new_script = template_script.replace(
                    '{ARRAY_FROM}',
                    str(i * array_max_size)
                ).replace(
                    '{ARRAY_TO}',
                    str((i + 1) * array_max_size - 1)
                )
                new_script = new_script.replace('{METHOD}', method)
                new_script = new_script.replace('{EPSILON}', str(epsilon))
                if use_mcmc:
                    new_script = new_script.replace('{USE_MCMC}', '--use_mcmc_sampling True')
                else:
                    new_script = new_script.replace('{USE_MCMC}', '')
                new_script = new_script.replace('{EXPERIMENT_NAME}', program_args.experiment_name)
                script_name = f'puhti_coverage_tests_{program_args.experiment_name}_{epsilon}_{method}_{i}.sh'
                with open(os.path.join(scripts_directory, script_name), 'w') as f:
                    f.write(new_script)
                lines.append(f'sbatch {os.path.join(scripts_directory, script_name)}')

            remainder = number_of_scripts % array_max_size
            if remainder != 0:
                new_script = template_script.replace(
                    '{ARRAY_FROM}',
                    str(number_of_scripts * array_max_size)
                ).replace(
                    '{ARRAY_TO}',
                    str(target_count)
                )
                new_script = new_script.replace('{METHOD}', method)
                new_script = new_script.replace('{EPSILON}', str(epsilon))
                if use_mcmc:
                    new_script = new_script.replace('{USE_MCMC}', '--use_mcmc_sampling True')
                else:
                    new_script = new_script.replace('{USE_MCMC}', '')
                new_script = new_script.replace('{EXPERIMENT_NAME}', program_args.experiment_name)
                script_name = f'puhti_coverage_tests_{program_args.experiment_name}_{epsilon}_{method}_{number_of_scripts}.sh'
                with open(os.path.join(scripts_directory, script_name), 'w') as f:
                    f.write(new_script)
                lines.append(f'sbatch {os.path.join(scripts_directory, script_name)}')

    with open('full_puhti_coverage_tests.sh', 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
