r"""CASBI: Conservative Amortized Simulation-Based Inference

Demonstration that the AUC metric might be deceitful for
establishing the statistical performance.
"""

import argparse
import hypothesis as h
import hypothesis.workflow as w
import logging
import os
import papermill as pm
import shutil


# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--redo", action="store_true", help="Executes the workflow from scratch by removing all postconditions (default: false).")
parser.add_argument("--slurm", action="store_true", help="Execute the workflow on a Slurm-enabled HPC system (default: false).")
arguments, _ = parser.parse_known_args()

### BEGIN Pre-workflow #########################################################


# Pipeline constants
root = os.path.dirname(os.path.abspath(__file__))
outputdir = root + "/output"

# Check if everything needs to be cleaned.
if arguments.redo:
    shutil.rmtree(outputdir, ignore_errors=True)


### END Pre-workflow ###########################################################

### BEGIN Workflow definition ##################################################

@w.root
def main():
    # Prepare the output directory
    if not os.path.exists(outputdir):
        logging.info("Creating the output directory.")
        os.makedirs(outputdir)


@w.dependency(main)
@w.postcondition(w.exists(outputdir + "/demonstration.ipynb"))
@w.slurm.cpu_and_memory(4, "4g")
@w.slurm.timelimit("01:00:00")
def summary():
    pm.execute_notebook(
        root + "/demonstration.ipynb",
        outputdir + "/demonstration.ipynb",
        parameters={
            "root": root,
            "outputdir": outputdir})


### END Workflow definition ####################################################

# Execute the workflow
if __name__ == "__main__":
    if arguments.slurm:
        w.slurm.execute(directory=root)
    else:
        w.local.execute()
