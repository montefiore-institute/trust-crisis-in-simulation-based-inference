import NF
import glob
import hypothesis as h
import numpy as np
import torch
import pickle
import os

from hypothesis.benchmark.tractable_small import Prior
from hypothesis.nn import build_ratio_estimator
from hypothesis.nn.ratio_estimation import BaseRatioEstimator
from hypothesis.nn.ratio_estimation import RatioEstimatorEnsemble
from hypothesis.stat import highest_density_level
from hypothesis.util.data import NamedDataset
from hypothesis.util.data import NumpyDataset
from torch.utils.data import TensorDataset
from tqdm import tqdm

from sbi.inference.base import infer
from sbi.inference import SNPE
from sbi.utils.get_nn_models import posterior_nn
from torch.utils.tensorboard.writer import SummaryWriter

### Utilities ##################################################################

prior = Prior()

extent = [ # I know, this isn't very nice :(
    prior.low[0].item(), prior.high[0].item(),
    prior.low[1].item(), prior.high[1].item()]


@torch.no_grad()
def load_estimator(query, reduce="ratio_mean"):
    if isinstance(query, str):
        paths = glob.glob(query)
    else:
        assert h.util.is_iterable(query)
        paths = query  # A list of paths has been specified.
    if len(paths) > 1:
        estimators = [load_estimator(p) for p in paths]
        r = RatioEstimatorEnsemble(estimators, reduce=reduce)
    else:
        path = paths[0]
        if "/flow" in path:
            r = FlowRatioEstimator()
        else:
            r = ClassifierRatioEstimator()
        r.load_state_dict(torch.load(path))
    r = r.to(h.accelerator)
    r.eval()

    return r


@torch.no_grad()
def compute_log_posterior(r, observable, resolution=100, flow_sbi=False):
    # Prepare grid
    epsilon = 0.00001
    p1 = torch.linspace(extent[0], extent[1] - epsilon, resolution)  # Account for half-open interval of uniform prior
    p2 = torch.linspace(extent[2], extent[3] - epsilon, resolution)  # Account for half-open interval of uniform prior
    p1 = p1.to(h.accelerator)
    p2 = p2.to(h.accelerator)
    g1, g2 = torch.meshgrid(p1.view(-1), p2.view(-1))
    # Vectorize
    inputs = torch.cat([g1.reshape(-1, 1), g2.reshape(-1, 1)], dim=1)

    if flow_sbi:
        observable = observable.to(h.accelerator)
        log_posterior = r.log_prob(inputs, x=observable, norm_posterior=False)[0].view(resolution, resolution).cpu()
    else:
        log_prior_probabilities = prior.log_prob(inputs).view(-1, 1)
        observables = observable.repeat(resolution ** 2, 1, 1, 1).float()
        observables = observables.to(h.accelerator)
        log_ratios = r.log_ratio(inputs=inputs, outputs=observables)
        log_posterior = (log_prior_probabilities + log_ratios).view(resolution, resolution).cpu()

    return log_posterior


@torch.no_grad()
def compute_log_pdf(r, inputs, outputs, flow_sbi=False):
    inputs = inputs.to(h.accelerator)
    outputs = outputs.to(h.accelerator)

    if flow_sbi:
        log_posterior = r.log_prob(inputs, x=outputs, norm_posterior=False)
        return log_posterior.squeeze()
    else:
        log_ratios = r.log_ratio(inputs=inputs, outputs=outputs)
        log_prior = prior.log_prob(inputs)

        return (log_prior + log_ratios).squeeze()


@torch.no_grad()
def coverage(r, inputs, outputs, alphas=[0.05], flow_sbi=False):
    n = len(inputs)
    covered = [0 for _ in alphas]
    for index in tqdm(range(n), "Coverages evaluated"):
        # Prepare setup
        nominal = inputs[index].squeeze().unsqueeze(0)
        observable = outputs[index].squeeze().unsqueeze(0)
        nominal = nominal.to(h.accelerator)
        observable = observable.to(h.accelerator)
        pdf = compute_log_posterior(r, observable, flow_sbi=flow_sbi).exp()
        nominal_pdf = compute_log_pdf(r, nominal, observable, flow_sbi=flow_sbi).exp()

        for i, alpha in enumerate(alphas):
            level = highest_density_level(pdf, alpha)
            if nominal_pdf >= level:
                covered[i] += 1

    return [x / n for x in covered]

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

def train_flow_sbi(simulation_budget, epochs, lr, batch_size, out, task_index, bagging=False, static=False):
    inputs = np.load("data/train/inputs.npy")
    outputs = np.load("data/train/outputs.npy")

    if bagging:
        # Bagging
        indices = np.random.choice(np.arange(simulation_budget), simulation_budget, replace=True)

    elif static:
        # Always select the same samples
        indices = np.arange(simulation_budget)

    else:
        # Sample from whole dataset
        indices = np.random.choice(np.arange(len(inputs)), simulation_budget, replace=False)

    inputs = inputs[indices, :]
    outputs = outputs[indices, :]
    inputs = torch.from_numpy(inputs)
    outputs = torch.from_numpy(outputs)

    os.makedirs("sbi-logs",  exist_ok=True)
    log_writer = SummaryWriter(log_dir=os.path.join("sbi-logs", "{}_{}".format(simulation_budget, task_index)))
    embedding = build_embedding()
    density_estimator_build_fun = posterior_nn(model='nsf', hidden_features=64, num_transforms=3, embedding_net=embedding)
    inference = SNPE(prior=prior, density_estimator=density_estimator_build_fun, summary_writer=log_writer)
    density_estimator = inference.append_simulations(inputs, outputs).train(learning_rate=lr, max_num_epochs=epochs, training_batch_size=batch_size)
    posterior = inference.build_posterior(density_estimator)

    with open(os.path.join(out, "posterior.pkl"), "wb") as handle:
        pickle.dump(posterior, handle)



### Ratio estimators ###########################################################

class ClassifierRatioEstimator(BaseRatioEstimator):

    def __init__(self):
        random_variables = {"inputs": (2,), "outputs": (8,)}
        Class = build_ratio_estimator("mlp", random_variables)
        activation = torch.nn.SELU
        trunk = [128] * 3
        r = Class(activation=activation, trunk=trunk)
        super(ClassifierRatioEstimator, self).__init__(r=r)
        self._r = r

    def log_ratio(self, **kwargs):
        return self._r.log_ratio(**kwargs)

class FlowRatioEstimator(BaseRatioEstimator):

    def __init__(self):
        denominator = "inputs|outputs"
        random_variables = {"inputs": (2,), "outputs": (8,)}
        super(FlowRatioEstimator, self).__init__(
            denominator=denominator,
            random_variables=random_variables)
        # Flow definition for now a simple conditionnal autoregressive affine
        conditioner_type = NF.AutoregressiveConditioner
        conditioner_args = {"in_size": np.prod(random_variables['inputs']),
                            "hidden": [128, 128, 128], "out_size": 2,
                            "cond_in": np.prod(random_variables['outputs'])}
        normalizer_type = NF.AffineNormalizer
        normalizer_args = {}
        nb_flow = 5
        self.flow = NF.buildFCNormalizingFlow(nb_flow, conditioner_type, conditioner_args, normalizer_type, normalizer_args)
        self._prior = Prior()

    def log_ratio(self, inputs, outputs, **kwargs):
        b_size = inputs.shape[0]
        log_posterior, _ = self.flow.compute_ll(inputs.view(b_size, -1), outputs.view(b_size, -1))
        log_prior = self._prior.log_prob(inputs)

        return log_posterior.view(-1, 1) - log_prior.view(-1, 1)

    def log_posterior(self, inputs, outputs, **kwargs):
        b_size = inputs.shape[0]
        outputs = outputs.float()
        log_posterior, _ = self.flow.compute_ll(inputs.view(b_size, -1), outputs.view(b_size, -1))
        return log_posterior.view(-1, 1)

### Datasets ###################################################################

class DatasetJointTrain(NamedDataset):

    def __init__(self, n=None, bagging=False, static=False):
        inputs = np.load("data/train/inputs.npy")
        outputs = np.load("data/train/outputs.npy")
        if n is not None:
            if bagging:
                # Bagging
                indices = np.random.choice(np.arange(n), n, replace=True)

            elif static:
                # Always select the same samples
                indices = np.arange(n)

            else:
                # Sample from whole dataset
                indices = np.random.choice(np.arange(len(inputs)), n, replace=False)

            inputs = inputs[indices, :]
            outputs = outputs[indices, :]
        inputs = TensorDataset(torch.from_numpy(inputs))
        outputs = TensorDataset(torch.from_numpy(outputs))
        super(DatasetJointTrain, self).__init__(
            inputs=inputs,
            outputs=outputs)


class DatasetJointTest(NamedDataset):

    def __init__(self):
        inputs = NumpyDataset("data/test/inputs.npy")
        outputs = NumpyDataset("data/test/outputs.npy")
        super(DatasetJointTest, self).__init__(
            inputs=inputs,
            outputs=outputs)


class DatasetJointTrain1024(DatasetJointTrain):

    def __init__(self, n=1024):
        super(DatasetJointTrain1024, self).__init__(n=n)


class DatasetJointTrain2048(DatasetJointTrain):

    def __init__(self, n=2048):
        super(DatasetJointTrain2048, self).__init__(n=n)


class DatasetJointTrain4096(DatasetJointTrain):

    def __init__(self, n=4096):
        super(DatasetJointTrain4096, self).__init__(n=n)


class DatasetJointTrain8192(DatasetJointTrain):

    def __init__(self, n=8192):
        super(DatasetJointTrain8192, self).__init__(n=n)


class DatasetJointTrain16384(DatasetJointTrain):

    def __init__(self, n=16384):
        super(DatasetJointTrain16384, self).__init__(n=n)


class DatasetJointTrain32768(DatasetJointTrain):

    def __init__(self, n=32768):
        super(DatasetJointTrain32768, self).__init__(n=n)


class DatasetJointTrain65536(DatasetJointTrain):

    def __init__(self, n=65536):
        super(DatasetJointTrain65536, self).__init__(n=n)


class DatasetJointTrain131072(DatasetJointTrain):

    def __init__(self, n=131072):
        super(DatasetJointTrain131072, self).__init__(n=n)

#Bagging datasets
class DatasetJointTrainBagging1024(DatasetJointTrain):

    def __init__(self, n=1024, bagging=True):
        super(DatasetJointTrainBagging1024, self).__init__(n=n, bagging=bagging)


class DatasetJointTrainBagging2048(DatasetJointTrain):

    def __init__(self, n=2048, bagging=True):
        super(DatasetJointTrainBagging2048, self).__init__(n=n, bagging=bagging)


class DatasetJointTrainBagging4096(DatasetJointTrain):

    def __init__(self, n=4096, bagging=True):
        super(DatasetJointTrainBagging4096, self).__init__(n=n, bagging=bagging)


class DatasetJointTrainBagging8192(DatasetJointTrain):

    def __init__(self, n=8192, bagging=True):
        super(DatasetJointTrainBagging8192, self).__init__(n=n, bagging=bagging)


class DatasetJointTrainBagging16384(DatasetJointTrain):

    def __init__(self, n=16384, bagging=True):
        super(DatasetJointTrainBagging16384, self).__init__(n=n, bagging=bagging)


class DatasetJointTrainBagging32768(DatasetJointTrain):

    def __init__(self, n=32768, bagging=True):
        super(DatasetJointTrainBagging32768, self).__init__(n=n, bagging=bagging)


class DatasetJointTrainBagging65536(DatasetJointTrain):

    def __init__(self, n=65536, bagging=True):
        super(DatasetJointTrainBagging65536, self).__init__(n=n, bagging=bagging)


class DatasetJointTrainBagging131072(DatasetJointTrain):

    def __init__(self, n=131072, bagging=True):
        super(DatasetJointTrainBagging131072, self).__init__(n=n, bagging=bagging)

#Static datasets
class DatasetJointTrainStatic1024(DatasetJointTrain):

    def __init__(self, n=1024, static=True):
        super(DatasetJointTrainStatic1024, self).__init__(n=n, static=static)


class DatasetJointTrainStatic2048(DatasetJointTrain):

    def __init__(self, n=2048, static=True):
        super(DatasetJointTrainStatic2048, self).__init__(n=n, static=static)


class DatasetJointTrainStatic4096(DatasetJointTrain):

    def __init__(self, n=4096, static=True):
        super(DatasetJointTrainStatic4096, self).__init__(n=n, static=static)


class DatasetJointTrainStatic8192(DatasetJointTrain):

    def __init__(self, n=8192, static=True):
        super(DatasetJointTrainStatic8192, self).__init__(n=n, static=static)


class DatasetJointTrainStatic16384(DatasetJointTrain):

    def __init__(self, n=16384, static=True):
        super(DatasetJointTrainStatic16384, self).__init__(n=n, static=static)


class DatasetJointTrainStatic32768(DatasetJointTrain):

    def __init__(self, n=32768, static=True):
        super(DatasetJointTrainStatic32768, self).__init__(n=n, static=static)


class DatasetJointTrainStatic65536(DatasetJointTrain):

    def __init__(self, n=65536, static=True):
        super(DatasetJointTrainStatic65536, self).__init__(n=n, static=static)


class DatasetJointTrainStatic131072(DatasetJointTrain):

    def __init__(self, n=131072, static=True):
        super(DatasetJointTrainStatic131072, self).__init__(n=n, static=static)
