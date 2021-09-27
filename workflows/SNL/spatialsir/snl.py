import pickle
import os
import glob
import hypothesis as h
import numpy as np
import torch
import time

from sbi.inference.base import infer
from hypothesis.benchmark.spatialsir import Prior
from hypothesis.benchmark.spatialsir import Simulator
from hypothesis.stat import highest_density_level
from sbi.inference import SNLE, prepare_for_sbi, simulate_for_sbi
from sbi.inference.posteriors.likelihood_based_posterior import LikelihoodBasedPosterior
from sbi.utils.get_nn_models import likelihood_nn
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter

prior = Prior()
extent = [ # I know, this isn't very nice :(
    prior.low[0].item(), prior.high[0].item(),
    prior.low[1].item(), prior.high[1].item()]

def build_embedding():
    hidden = 64
    latent = 10
    return torch.nn.Sequential(
        torch.nn.Linear(8, hidden),
        torch.nn.SELU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.SELU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.SELU(),
        torch.nn.Linear(hidden, latent))

def build_posterior(simulation_budget, out, lr, epochs, batch_size, task_index):
    prior = Prior()
    s = Simulator()
    @torch.no_grad()
    def simulator(thetas):
        n = len(thetas)
        xs = s(thetas).float().view(n, -1)
        return xs
    num_rounds = 10
    #num_rounds = 2

    executed_without_errors = False
    current_try = 0

    while not executed_without_errors and current_try < 10:
        try:
            theta_0 = prior.sample((1,))
            x_0 = simulator(theta_0)

            os.makedirs("sbi-logs",  exist_ok=True)
            log_writer = SummaryWriter(log_dir=os.path.join("sbi-logs", "{}_{}_{}".format(simulation_budget, task_index, current_try)))
            density_estimator_build_fun = likelihood_nn(model='nsf', hidden_features=64, num_transforms=3)
            inference = SNLE(prior=prior, density_estimator=density_estimator_build_fun, summary_writer=log_writer)

            proposal = prior

            for _ in range(num_rounds):
                theta, x = simulate_for_sbi(simulator, proposal, num_simulations=simulation_budget//num_rounds)
                density_estimator = inference.append_simulations(theta, x)
                density_estimator.train(learning_rate=lr, max_num_epochs=epochs, training_batch_size=batch_size, show_train_summary=True)
                posterior = density_estimator.build_posterior(mcmc_method="slice")
                proposal = posterior.set_default_x(x_0)

            executed_without_errors = True

        #rerun if AssertionError: NaN/Inf present in prior eval. is raised
        except AssertionError as e:
            current_try += 1
            print("Error occured during training: {}".format(e))


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
    p1 = torch.linspace(extent[0], extent[1], resolution)
    p2 = torch.linspace(extent[2], extent[3], resolution)
    p1 = p1.to(h.accelerator)
    p2 = p2.to(h.accelerator)
    g1, g2 = torch.meshgrid(p1.view(-1), p2.view(-1))
    # Vectorize
    inputs = torch.cat([g1.reshape(-1, 1), g2.reshape(-1, 1)], dim=1)

    log_prior_probabilities = prior.log_prob(inputs).view(-1, 1)
    observable = observable.to(h.accelerator)
    log_posterior = posterior.log_prob(inputs, x=observable)[0]
    assert (log_posterior.shape == (resolution**2,))

    return log_posterior

@torch.no_grad()
def coverage(posteriors, nominals, observables, alphas=[0.05]):
    n = len(nominals)
    covered = [0 for _ in alphas]

    for posterior, nominal, observable in tqdm(zip(posteriors, nominals, observables), "Coverages evaluated"):
        pdf = compute_log_posterior(posterior, observable).exp()
        nominal_pdf = posterior.log_prob(nominal, x=observable).exp()
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
        log_posterior = posterior.log_prob(nominal, x=observable)
        log_prior = prior.log_prob(nominal)
        log_r = log_posterior - log_prior
        mi += log_r

    return mi/n
