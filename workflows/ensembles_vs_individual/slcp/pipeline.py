r"""CASBI: Conservative Amortized Simulation-Based Inference

SLCP problem with 2 parameters. Although now it isn't really
tractable anymore! In this particular analysis we will
focus on the behaviour of the ratio estimators for
a given simulation budget.
"""

import argparse
import glob
import hypothesis as h
import hypothesis.workflow as w
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import papermill as pm
import shutil

from hypothesis.workflow import shell
from ratio_estimation import *
from tqdm import tqdm
from util import coverage_of_estimator, compute_sbc
from util import measure_diagnostic
from util import simulate

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--redo", action="store_true", help="Executes the workflow from scratch by removing all postconditions (default: false).")
parser.add_argument("--simulate-test-n", type=int, default=100000, help="Number of testing simulations (default: 100 000).")
parser.add_argument("--simulate-train-n", type=int, default=1000000, help="Number of training simulations (default: 10 000 000).")
parser.add_argument("--slurm", action="store_true", help="Execute the workflow on a Slurm-enabled HPC system (default: false).")
parser.add_argument("--test", action="store_true", help="Execute the workflow with fast hyper parameters for testing (default: false).")
parser.add_argument("--only_flows", action="store_true", help="Execute only the flow part of the workflow (default: false).")
arguments, _ = parser.parse_known_args()

### BEGIN Pre-workflow #########################################################

# Pipeline constants
root = os.path.dirname(os.path.abspath(__file__))
datadir = root + "/data"
outputdir = root + "/output"

# Hyperparameters
batch_size = 128
learning_rate = 0.001

if arguments.test:
    num_ensembles = 2
    epochs = 2
    simulations = [2 ** n for n in range(10, 11)]
    credible_interval_levels = [0.9, 0.95]
    simulation_block_size = 10
    arguments.simulate_train_n = 3000
    arguments.simulate_test_n = 20
    sbc_nb_rank_samples = 10
    sbc_nb_posterior_samples = 100
    diagnostic_n = 10
else:
    num_ensembles = 50
    epochs = 100
    simulations = [2 ** n for n in range(10, 18)]
    credible_interval_levels = [x/20 for x in range(1, 20)]
    simulation_block_size = 10000
    sbc_nb_rank_samples = 10000
    sbc_nb_posterior_samples = 1000
    diagnostic_n = 100000


assert arguments.simulate_train_n % simulation_block_size == 0
num_train_blocks = int(arguments.simulate_train_n / simulation_block_size)

# Check if everything needs to be cleaned.
if arguments.redo:
    shutil.rmtree(datadir, ignore_errors=True)
    shutil.rmtree(outputdir, ignore_errors=True)

# Simulation arguments
datadir_simulate_test = root + "/data/test"
datadir_simulate_train = root + "/data/train"

### END Pre-workflow ###########################################################

### BEGIN Workflow definition ##################################################

@w.root
def main():
    # Prepare simulation environment
    if not os.path.exists(datadir_simulate_train):
        logging.info("Creating training data directory.")
        os.makedirs(datadir_simulate_train)
    if not os.path.exists(datadir_simulate_test):
        logging.info("Creating testing data directory.")
        os.makedirs(datadir_simulate_test)
    # Prepare the output directory
    if not os.path.exists(outputdir):
        logging.info("Creating the output directory.")
        os.makedirs(outputdir)


@w.dependency(main)
@w.postcondition(w.num_files(datadir_simulate_train + "/block-*/inputs.npy", num_train_blocks))
@w.postcondition(w.num_files(datadir_simulate_train + "/block-*/outputs.npy", num_train_blocks))
@w.slurm.cpu_and_memory(1, "8g")
@w.slurm.timelimit("01:00:00")
@w.tasks(num_train_blocks)
def simulate_train(task_index):
    output_directory = datadir_simulate_train + "/block-" + str(task_index).zfill(5)
    # Check if the data has already been simulated
    dont_simulate = True
    dont_simulate &= os.path.exists(output_directory + "/inputs.npy")
    dont_simulate &= os.path.exists(output_directory + "/outputs.npy")
    if not dont_simulate:
        logging.info("Simulating training data block (" + str(task_index + 1) + " / " + str(num_train_blocks) + ")")
        simulate(n=simulation_block_size, directory=output_directory)


@w.dependency(simulate_train)
@w.postcondition(w.exists(datadir_simulate_train + "/inputs.npy"))
@w.postcondition(w.exists(datadir_simulate_train + "/outputs.npy"))
@w.slurm.cpu_and_memory(1, "16g")
@w.slurm.timelimit("01:00:00")
def merge_train():
    logging.info("Merging training data.")
    cwd = os.getcwd()
    os.chdir(root)
    shell("hypothesis merge --extension numpy --dimension 0 --in-memory --files 'data/train/block-*/inputs.npy' --sort --out data/train/inputs.npy")
    shell("hypothesis merge --extension numpy --dimension 0 --in-memory --files 'data/train/block-*/outputs.npy' --sort --out data/train/outputs.npy")
    shell("rm -rf data/train/block-*")
    os.chdir(cwd)


@w.dependency(main)
@w.postcondition(w.exists(datadir_simulate_test + "/inputs.npy"))
@w.postcondition(w.exists(datadir_simulate_test + "/outputs.npy"))
@w.slurm.cpu_and_memory(1, "4g")
@w.slurm.timelimit("01:00:00")
def simulate_test():
    logging.info("Simulating testing dataset.")
    simulate(n=arguments.simulate_test_n, directory=datadir_simulate_test)


dependencies = []
r"""
"""


def evaluate_point_classifier(simulation_budget, regularize, storagedir=None, cl_list=[0.95]):
    if storagedir is None:
        storagedir = outputdir + "/" + str(simulation_budget)
        if regularize:
            storagedir += "/with-regularization"
        else:
            storagedir += "/without-regularization"

    @w.dependency(simulate_test)
    @w.dependency(merge_train)
    @w.postcondition(w.num_files(storagedir + "/mlp-0*/weights.th", num_ensembles))
    @w.slurm.cpu_and_memory(4, "8g")
    @w.slurm.timelimit("48:00:00")
    @w.tasks(num_ensembles)
    def train_ratio_estimator(task_index):
        resultdir = storagedir + "/mlp-" + str(task_index).zfill(5)
        os.makedirs(resultdir, exist_ok=True)
        if not os.path.exists(resultdir + "/weights.th"):
            logging.info("Training classifier ratio estimator ({index} / {n}) for the small SLCP problem.".format(index=task_index + 1, n=num_ensembles))
            logging.info("Using the following hyper parameters:")
            logging.info(" - batch-size     : " + str(batch_size))
            logging.info(" - epochs         : " + str(epochs))
            logging.info(" - learning-rate  : " + str(learning_rate))
            logging.info(" - regularize     : " + str(regularize))
            logging.info(" - simulations    : " + str(simulation_budget))
            command = r"""python -m hypothesis.bin.ratio_estimation.train --batch-size {batch_size} \
                              --data-test ratio_estimation.DatasetJointTest \
                              --data-train ratio_estimation.DatasetJointTrain{simulations} \
                              --epochs {epochs} \
                              --estimator ratio_estimation.ClassifierRatioEstimator \
                              --hooks hooks.add \
                              --lr {lr} \
                              --lrsched-on-plateau \
                              --out {out} \
                              --show""".format(
                batch_size=batch_size,
                epochs=epochs,
                simulations=simulation_budget,
                lr=learning_rate,
                out=resultdir)
            if not regularize:
                command += " --criterion hypothesis.nn.ratio_estimation.BaseCriterion"
            # Execute the training procedure
            shell(command)


    @w.dependency(train_ratio_estimator)
    @w.postcondition(w.num_files(storagedir + "/mlp-0*/coverage.npy", num_ensembles))
    @w.slurm.cpu_and_memory(4, "8g")
    @w.slurm.timelimit("48:00:00")
    @w.tasks(num_ensembles)
    def coverage_individual(task_index):
        resultdir = storagedir + "/mlp-" + str(task_index).zfill(5)
        if not os.path.exists(resultdir + "/coverage.npy"):
            query = resultdir + "/weights.th"
            coverage = coverage_of_estimator(query, cl_list=cl_list)
            np.save(resultdir + "/coverage.npy", coverage)


    @w.dependency(train_ratio_estimator)
    @w.postcondition(w.exists(storagedir + "/coverage-classifier.npy"))
    @w.slurm.cpu_and_memory(4, "8g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("48:00:00")
    def coverage_ensemble():
        if not os.path.exists(storagedir + "/coverage-classifier.npy"):
            query = storagedir + "/mlp-0*/weights.th"
            coverage = coverage_of_estimator(query, cl_list=cl_list, max_samples=20000)
            np.save(storagedir + "/coverage-classifier.npy", coverage)

    # Add the dependencies for the summary notebook.
    dependencies.append(coverage_individual)
    dependencies.append(coverage_ensemble)



for simulation_budget in simulations:
    evaluate_point_classifier(simulation_budget, regularize=False, cl_list=credible_interval_levels)

### END Workflow definition ####################################################

# Execute the workflow
if __name__ == "__main__":
    if arguments.slurm:
        w.slurm.execute(directory=root)
    else:
        w.local.execute()
