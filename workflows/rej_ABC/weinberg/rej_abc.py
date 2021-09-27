import pickle
import os
import glob
import hypothesis as h
import numpy as np
import torch
import time

from sbi.inference.base import infer
from hypothesis.benchmark.weinberg import Prior
from hypothesis.benchmark.weinberg import Simulator
from hypothesis.stat import highest_density_level
from sbi.inference import MCABC
from tqdm import tqdm
from scipy.stats import gaussian_kde

prior = Prior()
extent = [ # I know, this isn't very nice :(
    prior.low.item(), prior.high.item()]


def build_posterior(simulation_budget, out, task_index, num_workers):
    prior = Prior()
    prior = prior.expand((1,))
    simulator = Simulator(default_beam_energy=40.0, num_samples=4)
    min_samples_to_keep = 100.
    min_quantile = min_samples_to_keep/simulation_budget
    quantile = max([min_quantile, 0.01])

    theta_0 = prior.sample((1,))
    x_0 = simulator(theta_0)

    inference = MCABC(simulator, prior, num_workers=num_workers)
    posterior = inference(x_0, simulation_budget, quantile=quantile)

    # Better way?
    theta_accepted = posterior._samples
    posterior = gaussian_kde(np.swapaxes(theta_accepted.numpy(), 0, 1))

    with open(os.path.join(out, "posterior.pkl"), "wb") as handle:
        pickle.dump(posterior, handle)

    with open(os.path.join(out, "x_0.pkl"), "wb") as handle:
        pickle.dump(x_0, handle)

    with open(os.path.join(out, "theta_0.pkl"), "wb") as handle:
        pickle.dump(theta_0, handle)


@torch.no_grad()
def load_estimators_parameters_observables(query):
    paths = glob.glob(query)
    posteriors = [pickle.load(open(os.path.join(path, "posterior.pkl"), "rb")) for path in paths]
    parameters = [pickle.load(open(os.path.join(path, "theta_0.pkl"), "rb")).to(h.accelerator) for path in paths]
    observables = [pickle.load(open(os.path.join(path, "x_0.pkl"), "rb")).to(h.accelerator) for path in paths]

    return posteriors, parameters, observables

@torch.no_grad()
def compute_log_posterior(posterior, observable, resolution=100):
    # Prepare grid
    epsilon = 0.00001
    inputs = torch.linspace(extent[0], extent[1] - epsilon, resolution)  # Account for half-open interval of uniform prior
    inputs = inputs.view(-1, 1)
    inputs = inputs.to(h.accelerator)

    observable = observable.to(h.accelerator)

    inputs = np.swapaxes(inputs.numpy(), 0, 1)
    log_posterior = posterior.logpdf(inputs)
    #log_posterior = torch.stack([posterior.log_prob(inputs[i, :]) for i in range(len(inputs))], axis=0)
    assert (log_posterior.shape == (resolution,))

    return log_posterior

@torch.no_grad()
def coverage(posteriors, nominals, observables, alphas=[0.05]):
    n = len(nominals)
    covered = [0 for _ in alphas]

    for posterior, nominal, observable in tqdm(zip(posteriors, nominals, observables), "Coverages evaluated"):
        pdf = np.exp(compute_log_posterior(posterior, observable))
        #pdf = compute_log_posterior(posterior, observable).exp()
        nominal_pdf = np.exp(posterior.logpdf(np.swapaxes(nominal.numpy(), 0, 1)))
        #nominal_pdf = posterior.log_prob(nominal.squeeze()).exp()
        for i, alpha in enumerate(alphas):
            level = highest_density_level(pdf, alpha)
            if nominal_pdf >= level:
                covered[i] += 1

    return [x / n for x in covered]

@torch.no_grad()
def mutual_information(posteriors, nominals, observables):
    prior = Prior()
    n = len(nominals)
    mi = 0

    for posterior, nominal, observable in tqdm(zip(posteriors, nominals, observables), "Mutual information evaluated"):
        #log_posterior = posterior.log_prob(nominal.squeeze())
        log_posterior = torch.Tensor(posterior.logpdf(np.swapaxes(nominal.numpy(), 0, 1)))
        log_prior = prior.log_prob(nominal)
        log_r = log_posterior - log_prior
        mi += log_r

    return mi/n
