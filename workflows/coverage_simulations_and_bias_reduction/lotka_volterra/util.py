r"""CASBI: Conservative Amortized Simulation-Based Inference

Spatial SIR problem.
"""

import numpy as np
import os
import torch
import glob
import pickle

from hypothesis.benchmark.lotka_volterra_small import Prior
from hypothesis.benchmark.lotka_volterra_small import Simulator
from hypothesis.nn.ratio_estimation import expectation_marginals_ratio

from ratio_estimation import DatasetJointTest as Dataset
from ratio_estimation import coverage
from ratio_estimation import load_estimator

from sbc import sbc_run

@torch.no_grad()
def simulate(n=10000, directory="."):
    simulator = Simulator()
    prior = Prior()
    inputs = prior.sample((n,)).cpu()
    outputs = simulator(inputs).view(len(inputs), -1) / 100
    if directory is not None:
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(directory + "/inputs.npy", inputs.numpy())
        np.save(directory + "/outputs.npy", outputs.numpy())

    return inputs, outputs


class FlowEnsemble():
    def __init__(self, flows):
        self.flows = flows

    def log_prob(self, *args, **kwargs):
        posteriors = [flow.log_prob(*args, **kwargs) for flow in self.flows]
        return torch.stack(posteriors, axis=0).exp().mean(axis=0).log()

class CpuPrior():
    def __init__(self, prior):
        self.prior = prior

    def log_prob(self, samples):
        return self.prior.log_prob(samples).cpu()


@torch.no_grad()
def coverage_of_estimator(path_to_weights, cl_list=[0.95], reduce="ratio_mean", flow_sbi=False, max_samples=None):
    if flow_sbi:
        paths = glob.glob(path_to_weights)
        flows = [pickle.load(open(path, "rb")) for path in paths]
        for flow in flows:
            flow._prior = CpuPrior(flow._prior)

        if len(flows) > 1:
            r = FlowEnsemble(flows)
        else:
            r = flows[0]

    else:
        r = load_estimator(path_to_weights, reduce=reduce)

    d = Dataset()

    if max_samples is None:
        inputs = d[:]["inputs"]
        outputs = d[:]["outputs"]
    else:
        inputs = d[:max_samples]["inputs"]
        outputs = d[:max_samples]["outputs"]
        
    alphas = [1 - cl for cl in cl_list]
    emperical_coverage = coverage(r, inputs, outputs, alphas,flow_sbi=flow_sbi)

    return emperical_coverage

#100000
@torch.no_grad()
def measure_diagnostic(r, n=10):
    d = Dataset()

    return expectation_marginals_ratio(d, r, n=n)

@torch.no_grad()
def importance_posterior_sampling(prior, posterior, nb_samples, nb_gen_samples):
    init_samples = prior.sample((nb_gen_samples,))
    weights = (posterior.log_prob(init_samples).squeeze() - prior.log_prob(init_samples).squeeze()).exp().numpy().squeeze()
    weights = weights/weights.sum()
    indices = np.arange(len(init_samples))
    samples_indices = np.random.choice(indices, size=nb_samples, replace=False, p=weights)
    samples = init_samples[samples_indices, :]

    return samples

@torch.no_grad()
def compute_sbc(path_to_weights, nb_rank_samples, nb_posterior_samples, save_name, reduce="ratio_mean", flow_sbi=False):
    prior = Prior()
    simulator = Simulator()

    if flow_sbi:
        paths = glob.glob(path_to_weights)
        flows = [pickle.load(open(path, "rb")) for path in paths]
        for flow in flows:
            flow._prior = CpuPrior(flow._prior)

        if len(flows) > 1:
            flow_posterior = FlowEnsemble(flows)
        else:
            flow_posterior = flows[0]

        class FlowPosterior():
            def __init__(self, x, flow):
                self.x = x
                self.flow = flow

            def log_prob(self, theta):
                tmp = self.flow.log_prob(theta, x=self.x, norm_posterior=False)[0]
                assert(tmp.shape == (len(theta),))
                return tmp


        def sample_posterior(x, nb_samples):
            posterior = FlowPosterior(x, flow_posterior)
            return importance_posterior_sampling(prior, posterior, nb_samples, nb_samples*100)

    else:
        r = load_estimator(path_to_weights, reduce=reduce)

        class RatioPosterior():
            def __init__(self, x, prior, ratio):
                self.x = x
                self.prior = prior
                self.ratio = ratio

            def log_prob(self, theta):
                outputs = self.x.repeat(len(theta), 1, 1, 1).float()
                return self.prior.log_prob(theta) + self.ratio.log_ratio(inputs=theta, outputs=outputs)

        def sample_posterior(x, nb_samples):
            posterior = RatioPosterior(x, prior, r)
            return importance_posterior_sampling(prior, posterior, nb_samples, nb_samples*100)

    sbc_run(prior, simulator, sample_posterior, nb_rank_samples, nb_posterior_samples, save_name)
