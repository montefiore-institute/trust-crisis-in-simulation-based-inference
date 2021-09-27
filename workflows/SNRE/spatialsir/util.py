r"""CASBI: Conservative Amortized Simulation-Based Inference

Simplified SLCP problem.
"""

import numpy as np
import os
import torch

from hypothesis.benchmark.spatialsir import Prior
from hypothesis.benchmark.spatialsir import Simulator

from snre import load_estimators_parameters_observables, coverage, mutual_information


@torch.no_grad()
def simulate(n=10000, directory="."):
    simulator = Simulator()
    prior = Prior()
    inputs = prior.sample((n,))
    outputs = simulator(inputs)
    if directory is not None:
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(directory + "/inputs.npy", inputs.numpy())
        np.save(directory + "/outputs.npy", outputs.numpy())

    return inputs, outputs


@torch.no_grad()
def coverage_of_estimator(query, nb_posteriors, cl_list=[0.95]):
    posteriors, parameters, observables = load_estimators_parameters_observables(query)
    alphas = [1 - cl for cl in cl_list]
    empirical_coverage = coverage(posteriors[:nb_posteriors], parameters[:nb_posteriors], observables[:nb_posteriors], alphas)

    return empirical_coverage

@torch.no_grad()
def mutual_information_of_estimator(query, nb_posteriors):
    posteriors, parameters, observables = load_estimators_parameters_observables(query)
    empirical_mutual_information = mutual_information(posteriors[:nb_posteriors], parameters[:nb_posteriors], observables[:nb_posteriors])

    return empirical_mutual_information
