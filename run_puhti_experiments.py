import os
import subprocess
import time

from tqdm import tqdm


def get_running_jobs_count():
    result = subprocess.run(['squeue', '-u', 'talalalr', '-h', '-r'], stdout=subprocess.PIPE, text=True)
    return len(result.stdout.splitlines())


def submit_job(task_id):
    env = os.environ.copy()
    env['TASK_ID'] = str(task_id)
    with open(os.devnull, 'w') as devnull:
        subprocess.run(['sbatch', 'puhti_coverage_test.sh'], env=env, stdout=devnull, stderr=devnull)


def main():
    total_jobs = 500
    max_running_jobs = 400
    task_id = 1

    progress_bar = tqdm(total=total_jobs)
    while task_id <= total_jobs:
        if get_running_jobs_count() < max_running_jobs:
            submit_job(task_id)
            task_id += 1
            progress_bar.update()
        else:
            time.sleep(10)
    progress_bar.close()


if __name__ == "__main__":
    main()
