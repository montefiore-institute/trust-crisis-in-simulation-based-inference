r"""Stellar Streams problem.
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
    sbc_nb_rank_samples = 10
    sbc_nb_posterior_samples = 100
    diagnostic_n = 10

else:
    num_ensembles = 5
    epochs = 100
    simulations = [2 ** n for n in range(10, 18)]
    credible_interval_levels = [x/20 for x in range(1, 20)]
    sbc_nb_rank_samples = 10000
    sbc_nb_posterior_samples = 1000
    diagnostic_n = 100000

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
@w.postcondition(w.exists(datadir_simulate_train + "/inputs.npy"))
@w.postcondition(w.exists(datadir_simulate_train + "/outputs.npy"))
@w.slurm.cpu_and_memory(1, "4g")
@w.slurm.timelimit("48:00:00")
def download_train():
    logging.info("Downloading train data.")
    shell("cd data && wget https://joerihermans.com/streams-train.zip")
    logging.info("Unpacking train data.")
    shell("cd data && unzip streams-train.zip && rm streams-train.zip")


@w.dependency(main)
@w.postcondition(w.exists(datadir_simulate_test + "/inputs.npy"))
@w.postcondition(w.exists(datadir_simulate_test + "/outputs.npy"))
@w.slurm.cpu_and_memory(1, "4g")
@w.slurm.timelimit("48:00:00")
def download_test():
    logging.info("Downloading test data.")
    shell("cd data && wget https://joerihermans.com/streams-test.zip")
    logging.info("Unpacking test data.")
    shell("cd data && unzip streams-test.zip && rm streams-test.zip")


dependencies = []
r"""
"""


def evaluate_point_classifier(simulation_budget, storagedir=None, cl_list=[0.95]):
    if storagedir is None:
        storagedir = outputdir + "/" + str(simulation_budget)
        storagedir += "/without-regularization"

    @w.dependency(download_test)
    @w.dependency(download_train)
    @w.postcondition(w.num_files(storagedir + "/mlp-0*/weights.th", num_ensembles))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("48:00:00")
    @w.tasks(num_ensembles)
    def train_ratio_estimator(task_index):
        resultdir = storagedir + "/mlp-" + str(task_index).zfill(5)
        os.makedirs(resultdir, exist_ok=True)
        if not os.path.exists(resultdir + "/weights.th"):
            logging.info("Training classifier ratio estimator ({index} / {n}) for the stellar streams problem.".format(index=task_index + 1, n=num_ensembles))
            logging.info("Using the following hyper parameters:")
            logging.info(" - batch-size     : " + str(batch_size))
            logging.info(" - epochs         : " + str(epochs))
            logging.info(" - learning-rate  : " + str(learning_rate))
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
            command += " --criterion hypothesis.nn.ratio_estimation.BaseCriterion"
            # Execute the training procedure
            shell(command)


    @w.dependency(train_ratio_estimator)
    @w.postcondition(w.num_files(storagedir + "/mlp-0*/coverage.npy", num_ensembles))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.timelimit("48:00:00")
    @w.slurm.gpu(1)
    @w.tasks(num_ensembles)
    def coverage_individual(task_index):
        resultdir = storagedir + "/mlp-" + str(task_index).zfill(5)
        if not os.path.exists(resultdir + "/coverage.npy"):
            query = resultdir + "/weights.th"
            coverage = coverage_of_estimator(query, cl_list=cl_list, test=arguments.test)
            np.save(resultdir + "/coverage.npy", coverage)


    @w.dependency(train_ratio_estimator)
    @w.postcondition(w.exists(storagedir + "/coverage-classifier.npy"))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.timelimit("48:00:00")
    @w.slurm.gpu(1)
    def coverage_ensemble():
        if not os.path.exists(storagedir + "/coverage-classifier.npy"):
            query = storagedir + "/mlp-0*/weights.th"
            coverage = coverage_of_estimator(query, cl_list=cl_list, test=arguments.test, max_samples=20000)
            np.save(storagedir + "/coverage-classifier.npy", coverage)


    @w.dependency(train_ratio_estimator)
    @w.postcondition(w.num_files(storagedir + "/mlp-0*/diagnostic.npy", num_ensembles))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.timelimit("00:15:00")
    @w.tasks(num_ensembles)
    def diagnostic_individual(task_index):
        resultdir = storagedir + "/mlp-" + str(task_index).zfill(5)
        outputfile = resultdir + "/diagnostic.npy"
        if not os.path.exists(outputfile):
            query = resultdir + "/weights.th"
            r = load_estimator(query)
            result = measure_diagnostic(r, n=diagnostic_n)
            np.save(outputfile, result)


    @w.dependency(train_ratio_estimator)
    @w.postcondition(w.exists(storagedir + "/diagnostic-classifier.npy"))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.timelimit("00:15:00")
    def diagnostic_ensemble():
        resultdir = storagedir
        outputfile = resultdir + "/diagnostic-classifier.npy"
        if not os.path.exists(outputfile):
            query = resultdir + "/mlp-0*/weights.th"
            r = load_estimator(query)
            result = measure_diagnostic(r, n=diagnostic_n)
            np.save(outputfile, result)


    @w.dependency(train_ratio_estimator)
    @w.postcondition(w.num_files(storagedir + "/mlp-0*/sbc.npy", num_ensembles))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("48:00:00")
    @w.tasks(num_ensembles)
    def sbc_individual(task_index):
        resultdir = storagedir + "/mlp-" + str(task_index).zfill(5)
        if not os.path.exists(resultdir + "/sbc.npy"):
            query = resultdir + "/weights.th"
            compute_sbc(query, sbc_nb_rank_samples, sbc_nb_posterior_samples, resultdir + "/sbc.npy")

    @w.dependency(train_ratio_estimator)
    @w.postcondition(w.exists(storagedir + "/sbc-classifier.npy"))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("48:00:00")
    def sbc_ensemble():
        if not os.path.exists(storagedir + "/sbc-classifier.npy"):
            query = storagedir + "/mlp-0*/weights.th"
            compute_sbc(query, sbc_nb_rank_samples, sbc_nb_posterior_samples, storagedir + "/sbc-classifier.npy")


    # Add the dependencies for the summary notebook.
    dependencies.append(diagnostic_individual)
    dependencies.append(diagnostic_ensemble)
    dependencies.append(coverage_individual)
    dependencies.append(coverage_ensemble)
    dependencies.append(sbc_individual)
    dependencies.append(sbc_ensemble)

    @w.dependency(download_test)
    @w.dependency(download_train)
    @w.postcondition(w.num_files(storagedir + "/mlp-bagging-0*/weights.th", num_ensembles))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("48:00:00")
    @w.tasks(num_ensembles)
    def train_ratio_estimator_bagging(task_index):
        resultdir = storagedir + "/mlp-bagging-" + str(task_index).zfill(5)
        os.makedirs(resultdir, exist_ok=True)
        if not os.path.exists(resultdir + "/weights.th"):
            logging.info("Training classifier ratio estimator ({index} / {n}) for the stellar streams problem.".format(index=task_index + 1, n=num_ensembles))
            logging.info("Using the following hyper parameters:")
            logging.info(" - batch-size     : " + str(batch_size))
            logging.info(" - epochs         : " + str(epochs))
            logging.info(" - learning-rate  : " + str(learning_rate))
            logging.info(" - simulations    : " + str(simulation_budget))
            command = r"""python -m hypothesis.bin.ratio_estimation.train --batch-size {batch_size} \
                              --data-test ratio_estimation.DatasetJointTest \
                              --data-train ratio_estimation.DatasetJointTrainBagging{simulations} \
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
            command += " --criterion hypothesis.nn.ratio_estimation.BaseCriterion"
            # Execute the training procedure
            shell(command)

    @w.dependency(train_ratio_estimator_bagging)
    @w.postcondition(w.exists(storagedir + "/coverage-classifier-bagging.npy"))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.timelimit("48:00:00")
    @w.slurm.gpu(1)
    def coverage_ensemble_bagging():
        if not os.path.exists(storagedir + "/coverage-classifier-bagging.npy"):
            query = storagedir + "/mlp-bagging-0*/weights.th"
            coverage = coverage_of_estimator(query, cl_list=cl_list, test=arguments.test, max_samples=20000)
            np.save(storagedir + "/coverage-classifier-bagging.npy", coverage)


    @w.dependency(train_ratio_estimator_bagging)
    @w.postcondition(w.exists(storagedir + "/diagnostic-classifier-bagging.npy"))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.timelimit("00:15:00")
    def diagnostic_ensemble_bagging():
        resultdir = storagedir
        outputfile = resultdir + "/diagnostic-classifier-bagging.npy"
        if not os.path.exists(outputfile):
            query = resultdir + "/mlp-bagging-0*/weights.th"
            r = load_estimator(query)
            result = measure_diagnostic(r, n=diagnostic_n)
            np.save(outputfile, result)

    @w.dependency(train_ratio_estimator_bagging)
    @w.postcondition(w.exists(storagedir + "/sbc-classifier-bagging.npy"))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("48:00:00")
    def sbc_ensemble_bagging():
        if not os.path.exists(storagedir + "/sbc-classifier-bagging.npy"):
            query = storagedir + "/mlp-bagging-0*/weights.th"
            compute_sbc(query, sbc_nb_rank_samples, sbc_nb_posterior_samples, storagedir + "/sbc-classifier-bagging.npy")

    # Add the dependencies for the summary notebook.
    dependencies.append(diagnostic_ensemble_bagging)
    dependencies.append(coverage_ensemble_bagging)
    dependencies.append(sbc_ensemble_bagging)

    @w.dependency(download_test)
    @w.dependency(download_train)
    @w.postcondition(w.num_files(storagedir + "/mlp-static-0*/weights.th", num_ensembles))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("48:00:00")
    @w.tasks(num_ensembles)
    def train_ratio_estimator_static(task_index):
        resultdir = storagedir + "/mlp-static-" + str(task_index).zfill(5)
        os.makedirs(resultdir, exist_ok=True)
        if not os.path.exists(resultdir + "/weights.th"):
            logging.info("Training classifier ratio estimator ({index} / {n}) for the stellar streams problem.".format(index=task_index + 1, n=num_ensembles))
            logging.info("Using the following hyper parameters:")
            logging.info(" - batch-size     : " + str(batch_size))
            logging.info(" - epochs         : " + str(epochs))
            logging.info(" - learning-rate  : " + str(learning_rate))
            logging.info(" - simulations    : " + str(simulation_budget))
            command = r"""python -m hypothesis.bin.ratio_estimation.train --batch-size {batch_size} \
                              --data-test ratio_estimation.DatasetJointTest \
                              --data-train ratio_estimation.DatasetJointTrainStatic{simulations} \
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
            command += " --criterion hypothesis.nn.ratio_estimation.BaseCriterion"
            # Execute the training procedure
            shell(command)

    @w.dependency(train_ratio_estimator_static)
    @w.postcondition(w.exists(storagedir + "/coverage-classifier-static.npy"))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.timelimit("48:00:00")
    @w.slurm.gpu(1)
    def coverage_ensemble_static():
        if not os.path.exists(storagedir + "/coverage-classifier-static.npy"):
            query = storagedir + "/mlp-static-0*/weights.th"
            coverage = coverage_of_estimator(query, cl_list=cl_list, test=arguments.test, max_samples=20000)
            np.save(storagedir + "/coverage-classifier-static.npy", coverage)


    @w.dependency(train_ratio_estimator_static)
    @w.postcondition(w.exists(storagedir + "/diagnostic-classifier-static.npy"))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.timelimit("00:15:00")
    def diagnostic_ensemble_static():
        resultdir = storagedir
        outputfile = resultdir + "/diagnostic-classifier-static.npy"
        if not os.path.exists(outputfile):
            query = resultdir + "/mlp-static-0*/weights.th"
            r = load_estimator(query)
            result = measure_diagnostic(r, n=diagnostic_n)
            np.save(outputfile, result)

    @w.dependency(train_ratio_estimator_static)
    @w.postcondition(w.exists(storagedir + "/sbc-classifier-static.npy"))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("48:00:00")
    def sbc_ensemble_static():
        if not os.path.exists(storagedir + "/sbc-classifier-static.npy"):
            query = storagedir + "/mlp-static-0*/weights.th"
            compute_sbc(query, sbc_nb_rank_samples, sbc_nb_posterior_samples, storagedir + "/sbc-classifier-static.npy")

    # Add the dependencies for the summary notebook.
    dependencies.append(diagnostic_ensemble_static)
    dependencies.append(coverage_ensemble_static)
    dependencies.append(sbc_ensemble_static)


def evaluate_point_flow_sbi(simulation_budget, storagedir=None, cl_list=[0.95]):
    if storagedir is None:
        storagedir = outputdir + "/" + str(simulation_budget) + "/without-regularization"


    @w.dependency(download_test)
    @w.dependency(download_train)
    @w.postcondition(w.num_files(storagedir + "/flow-sbi-0*/posterior.pkl", num_ensembles))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("48:00:00")
    @w.tasks(num_ensembles)
    def train_flow(task_index):
        resultdir = storagedir + "/flow-sbi-" + str(task_index).zfill(5)
        os.makedirs(resultdir, exist_ok=True)
        train_flow_sbi(simulation_budget, epochs, learning_rate, batch_size, resultdir, task_index)


    @w.dependency(train_flow)
    @w.postcondition(w.num_files(storagedir + "/flow-sbi-0*/coverage.npy", num_ensembles))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("48:00:00")
    @w.tasks(num_ensembles)
    def coverage_individual(task_index):
        resultdir = storagedir + "/flow-sbi-" + str(task_index).zfill(5)
        if not os.path.exists(resultdir + "/coverage.npy"):
            query = resultdir + "/posterior.pkl"
            coverage = coverage_of_estimator(query, cl_list=cl_list, flow_sbi=True, test=arguments.test)
            np.save(resultdir + "/coverage.npy", coverage)


    @w.dependency(train_flow)
    @w.postcondition(w.exists(storagedir + "/coverage-flow-sbi.npy"))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("48:00:00")
    def coverage_ensemble():
        if not os.path.exists(storagedir + "/coverage-flow-sbi.npy"):
            query = storagedir + "/flow-sbi-0*/posterior.pkl"
            coverage = coverage_of_estimator(query, cl_list=cl_list, flow_sbi=True, test=arguments.test, max_samples=20000)
            np.save(storagedir + "/coverage-flow-sbi.npy", coverage)


    @w.dependency(train_flow)
    @w.postcondition(w.num_files(storagedir + "/flow-sbi-0*/sbc.npy", num_ensembles))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("48:00:00")
    @w.tasks(num_ensembles)
    def sbc_individual(task_index):
        resultdir = storagedir + "/flow-sbi-" + str(task_index).zfill(5)
        if not os.path.exists(resultdir + "/sbc.npy"):
            query = resultdir + "/posterior.pkl"
            compute_sbc(query, sbc_nb_rank_samples, sbc_nb_posterior_samples, resultdir + "/sbc.npy", flow_sbi=True)

    @w.dependency(train_flow)
    @w.postcondition(w.exists(storagedir + "/sbc-flow-sbi.npy"))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("48:00:00")
    def sbc_ensemble():
        if not os.path.exists(storagedir + "/sbc-flow-sbi.npy"):
            query = storagedir + "/flow-sbi-0*/posterior.pkl"
            compute_sbc(query, sbc_nb_rank_samples, sbc_nb_posterior_samples, storagedir + "/sbc-flow-sbi.npy", flow_sbi=True)



    # Add the dependencies for the summary notebook.
    dependencies.append(coverage_individual)
    dependencies.append(coverage_ensemble)
    dependencies.append(sbc_individual)
    dependencies.append(sbc_ensemble)

    @w.dependency(download_test)
    @w.dependency(download_train)
    @w.postcondition(w.num_files(storagedir + "/flow-sbi-bagging-0*/posterior.pkl", num_ensembles))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("48:00:00")
    @w.tasks(num_ensembles)
    def train_flow_bagging(task_index):
        resultdir = storagedir + "/flow-sbi-bagging-" + str(task_index).zfill(5)
        os.makedirs(resultdir, exist_ok=True)
        train_flow_sbi(simulation_budget, epochs, learning_rate, batch_size, resultdir, task_index, bagging=True)

    @w.dependency(train_flow_bagging)
    @w.postcondition(w.exists(storagedir + "/coverage-flow-sbi-bagging.npy"))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("48:00:00")
    def coverage_ensemble_bagging():
        if not os.path.exists(storagedir + "/coverage-flow-sbi-bagging.npy"):
            query = storagedir + "/flow-sbi-bagging-0*/posterior.pkl"
            coverage = coverage_of_estimator(query, cl_list=cl_list, flow_sbi=True, test=arguments.test, max_samples=20000)
            np.save(storagedir + "/coverage-flow-sbi-bagging.npy", coverage)

    @w.dependency(train_flow_bagging)
    @w.postcondition(w.exists(storagedir + "/sbc-flow-sbi-bagging.npy"))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("48:00:00")
    def sbc_ensemble_bagging():
        if not os.path.exists(storagedir + "/sbc-flow-sbi-bagging.npy"):
            query = storagedir + "/flow-sbi-bagging-0*/posterior.pkl"
            compute_sbc(query, sbc_nb_rank_samples, sbc_nb_posterior_samples, storagedir + "/sbc-flow-sbi-bagging.npy", flow_sbi=True)

    # Add the dependencies for the summary notebook.
    dependencies.append(coverage_ensemble_bagging)
    dependencies.append(sbc_ensemble_bagging)

    @w.dependency(download_test)
    @w.dependency(download_train)
    @w.postcondition(w.num_files(storagedir + "/flow-sbi-static-0*/posterior.pkl", num_ensembles))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("48:00:00")
    @w.tasks(num_ensembles)
    def train_flow_static(task_index):
        resultdir = storagedir + "/flow-sbi-static-" + str(task_index).zfill(5)
        os.makedirs(resultdir, exist_ok=True)
        train_flow_sbi(simulation_budget, epochs, learning_rate, batch_size, resultdir, task_index, static=True)

    @w.dependency(train_flow_static)
    @w.postcondition(w.exists(storagedir + "/coverage-flow-sbi-static.npy"))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("48:00:00")
    def coverage_ensemble_static():
        if not os.path.exists(storagedir + "/coverage-flow-sbi-static.npy"):
            query = storagedir + "/flow-sbi-static-0*/posterior.pkl"
            coverage = coverage_of_estimator(query, cl_list=cl_list, flow_sbi=True, test=arguments.test, max_samples=20000)
            np.save(storagedir + "/coverage-flow-sbi-static.npy", coverage)

    @w.dependency(train_flow_static)
    @w.postcondition(w.exists(storagedir + "/sbc-flow-sbi-static.npy"))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("48:00:00")
    def sbc_ensemble_static():
        if not os.path.exists(storagedir + "/sbc-flow-sbi-static.npy"):
            query = storagedir + "/flow-sbi-static-0*/posterior.pkl"
            compute_sbc(query, sbc_nb_rank_samples, sbc_nb_posterior_samples, storagedir + "/sbc-flow-sbi-static.npy", flow_sbi=True)

    # Add the dependencies for the summary notebook.
    dependencies.append(coverage_ensemble_static)
    dependencies.append(sbc_ensemble_static)


for simulation_budget in simulations:
    if arguments.only_flows:
        evaluate_point_flow_sbi(simulation_budget, cl_list=credible_interval_levels)
    else:
        evaluate_point_classifier(simulation_budget, cl_list=credible_interval_levels)
        evaluate_point_flow_sbi(simulation_budget, cl_list=credible_interval_levels)



### END Workflow definition ####################################################

# Execute the workflow
if __name__ == "__main__":
    if arguments.slurm:
        w.slurm.execute(directory=root)
    else:
        w.local.execute()
