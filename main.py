# model configs
from fastai_configs import *
from wavelet_configs import *


def main():
    data_folder = '/data/ptbxl/'
    output_folder = '/output/'

    models = [conf_fastai_inception1d]

    # STANDARD SCP EXPERIMENTS ON PTB-XL

    experiments = [
        ('exp0', 'all'),
        ('exp1', 'diagnostic'),
        ('exp1.1', 'subdiagnostic'),
        ('exp1.1.1', 'superdiagnostic'),
        ('exp2', 'form'),
        ('exp3', 'rhythm')
    ]

    for name, task in experiments:
        e = SCP_Experiment(name, task, datafolder, output_folder, models)
        e.prepare()
        e.perform()
        e.evaluate()

    # generate summary table
    generate_ptbxl_summary_table()


if __name__ == "__main__":
    main()
