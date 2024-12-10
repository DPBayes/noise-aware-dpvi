import os
import subprocess
import time
import traceback

MAX_JOBS = 400
SLEEP_TIME = 60
BASE_DIR = '/users/talalalr/noise-aware-dpsgd/scripts'


def get_running_jobs_count():
    result = subprocess.run(['squeue', '-u', 'talalalr', '-h', '-r'], stdout=subprocess.PIPE, text=True)
    return len(result.stdout.splitlines())


def submit_job(job_path):
    try:
        subprocess.run(['sbatch', job_path], check=True)
        print(f'Submitted job: {job_path}')
    except subprocess.CalledProcessError as e:
        traceback.print_exc()
        print(f'Error submitting job: {job_path}, error: {e}')


def main():
    jobs = os.listdir(BASE_DIR)

    total_jobs = len(jobs)
    i = 0
    for job in jobs:
        while get_running_jobs_count() >= MAX_JOBS:
            time.sleep(SLEEP_TIME)
        job_path = os.path.join(BASE_DIR, job)
        submit_job(job_path)
        os.remove(job_path)
        i += 1
        print(str(round((i / total_jobs) * 100, 2)) + '%')
    print('All jobs are submitted.')


if __name__ == '__main__':
    main()
