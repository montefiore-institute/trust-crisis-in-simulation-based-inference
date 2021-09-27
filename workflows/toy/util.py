r"""CASBI: Conservative Amortized Simulation-Based Inference

Toy problem highlighting the connection between conditional flows
and ratio estimators.
"""

import hypothesis as h
import numpy as np
import torch

from hypothesis.util.data import NamedDataset
from torch.utils.data import TensorDataset


delta = + 0.001


def Prior(low=-10.0, high=10.0, accelerator=h.accelerator):
    lower = torch.tensor(low).float()
    lower = lower.to(accelerator)
    upper = torch.tensor(high).float()
    upper = upper.to(accelerator)
    return torch.distributions.uniform.Uniform(lower, upper + delta)


@torch.no_grad()
def compute_log_pdf(r, observable, resolution=250):
    prior = Prior()
    extent = torch.linspace(prior.low.item(), prior.high.item() - delta, resolution)
    extent = extent.to(h.accelerator)
    x = observable.view(1, -1).repeat(resolution, 1)
    log_ratios = r.log_ratio(inputs=inputs, outputs=outputs)
    log_prior = prior.log_prob(extent).view(-1, 1)
    log_pdf = log_prior + log_ratios

    return log_pdf


@torch.no_grad()
def simulate(n=1):
    prior = Prior(accelerator="cpu")
    inputs = prior.sample((n,)).view(-1, 1).numpy()
    outputs = np.random.normal(size=n).reshape(-1, 1) + inputs
    inputs = torch.from_numpy(inputs).float()
    outputs = torch.from_numpy(outputs).float()

    return inputs, outputs


class DatasetJointTrain(NamedDataset):

    def __init__(self):
        inputs, outputs = simulate(n=1000000)
        dataset_inputs = TensorDataset(inputs)
        dataset_outputs = TensorDataset(outputs)
        super(DatasetJointTrain, self).__init__(
            inputs=dataset_inputs,
            outputs=dataset_outputs)


class DatasetJointTest(NamedDataset):

    def __init__(self):
        inputs, outputs = simulate(n=100000)
        dataset_inputs = TensorDataset(inputs)
        dataset_outputs = TensorDataset(outputs)
        super(DatasetJointTest, self).__init__(
            inputs=dataset_inputs,
            outputs=dataset_outputs)
