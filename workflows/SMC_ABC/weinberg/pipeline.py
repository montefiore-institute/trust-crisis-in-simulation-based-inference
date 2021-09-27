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
from tqdm import tqdm
from util import simulate, coverage_of_estimator, mutual_information_of_estimator

from smc_abc import build_posterior

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--redo", action="store_true", help="Executes the workflow from scratch by removing all postconditions (default: false).")
parser.add_argument("--slurm", action="store_true", help="Executes the workflow on a Slurm-enabled HPC system (default: false).")
parser.add_argument("--test", action="store_true", help="Execute the workflow with fast hyper parameters for testing (default: false).")
arguments, _ = parser.parse_known_args()

### BEGIN Pre-workflow #########################################################

# Pipeline constants
root = os.path.dirname(os.path.abspath(__file__))
logdir = root + "/sbi-logs"
outputdir = root + "/output"

if arguments.test:
    num_ensembles = 2
    num_ensembles_gen = 3
    simulations = [2 ** n for n in range(10, 11)]
    credible_interval_levels = [0.9, 0.95]
else:
    num_ensembles = 250
    num_ensembles_gen = 300
    simulations = [2 ** n for n in range(10, 18)]
    credible_interval_levels = [x/20 for x in range(1, 20)]

# Check if everything needs to be cleaned.
if arguments.redo:
    shutil.rmtree(logdir, ignore_errors=True)
    shutil.rmtree(outputdir, ignore_errors=True)

### END Pre-workflow ###########################################################

### BEGIN Workflow definition ##################################################

@w.root
def main():
    # Prepare the output directory
    if not os.path.exists(outputdir):
        logging.info("Creating the output directory.")
        os.makedirs(outputdir)


def evaluate_rej_abc(simulation_budget):
    storagedir = outputdir + "/" + str(simulation_budget)

    @w.dependency(main)
    @w.postcondition(w.at_least_num_files(storagedir + "/run-*/posterior.pkl", num_ensembles))
    @w.slurm.cpu_and_memory(4, "4g")
    @w.slurm.timelimit("48:00:00")
    @w.tasks(num_ensembles_gen)
    def train_rej_abc(task_index):
        resultdir = storagedir + "/run-" + str(task_index).zfill(5)
        os.makedirs(resultdir, exist_ok=True)
        if not os.path.exists(os.path.join(resultdir, "posterior.pkl")):
            logging.info("Training posterior estimator ({index} / {n}) for the weinberg problem.".format(index=task_index + 1, n=num_ensembles))
            logging.info("Using the following hyper parameters:")
            logging.info(" - simulations    : " + str(simulation_budget))
            build_posterior(simulation_budget, resultdir, task_index, num_workers=4)

    @w.dependency(train_rej_abc)
    @w.postcondition(w.exists(storagedir + "/coverage.npy"))
    @w.slurm.cpu_and_memory(1, "4g")
    @w.slurm.timelimit("24:00:00")
    def coverage():
        if not os.path.exists(storagedir + "/coverage.npy"):
            query = storagedir + "/run-*/"
            coverage = coverage_of_estimator(query, num_ensembles, cl_list=credible_interval_levels)
            np.save(storagedir + "/coverage.npy", coverage)

    @w.dependency(train_rej_abc)
    @w.postcondition(w.exists(storagedir + "/mutual_information.npy"))
    @w.slurm.cpu_and_memory(1, "4g")
    @w.slurm.timelimit("24:00:00")
    def mutual_information():
        if not os.path.exists(storagedir + "/mutual_information.npy"):
            query = storagedir + "/run-*/"
            mutual_information = mutual_information_of_estimator(query, num_ensembles)
            np.save(storagedir + "/mutual_information.npy", mutual_information)

for simulation_budget in simulations:
    evaluate_rej_abc(simulation_budget)

### END Workflow definition ####################################################

# Execute the workflow
if __name__ == "__main__":
    if arguments.slurm:
        w.slurm.execute(directory=root)
    else:
        w.local.execute()
