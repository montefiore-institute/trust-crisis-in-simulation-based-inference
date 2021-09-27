r"""CASBI: Conservative Amortized Simulation-Based Inference

Contains the definition of the ratio estimators
and the corresponding utilities.
"""

import NF
import glob
import hypothesis as h
import numpy as np
import torch
import torch.nn as nn
import pickle
import os

from hypothesis.benchmark.gravitational_waves import Prior
from hypothesis.nn import build_ratio_estimator
from hypothesis.nn.ratio_estimation import BaseRatioEstimator
from hypothesis.nn.ratio_estimation import RatioEstimatorEnsemble
from hypothesis.stat import highest_density_level
from hypothesis.util.data import NamedDataset
from hypothesis.util.data import NumpyDataset
from torch.utils.data import Dataset, TensorDataset
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
    paths = glob.glob(query)
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
def compute_log_posterior(r, observable, resolution=100, batch_size=64, flow_sbi=False):
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
        log_posterior = torch.empty(resolution**2)

        for b in range(0, inputs.shape[0], batch_size):
            cur_inputs = inputs[b:b+batch_size]
            log_posterior[b:b+batch_size] = r.log_prob(cur_inputs, x=observable, norm_posterior=False)[0]

        log_posterior = log_posterior.view(resolution, resolution).cpu()

    else:
        log_prior_probabilities = prior.log_prob(inputs).flatten()

        log_ratios = torch.empty(resolution**2)
        
        for b in range(0, inputs.shape[0], batch_size):
            cur_inputs = inputs[b:b+batch_size]
            observables = observable.repeat(cur_inputs.shape[0], 1, 1).float()
            observables = observables.to(h.accelerator)
            log_ratios[b:b+batch_size] = r.log_ratio(inputs=cur_inputs, outputs=observables).squeeze(1)

        log_prior_probabilities = log_prior_probabilities.cpu()
        log_ratios = log_ratios.cpu()

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
    nb_channels = 16

    cnn = [nn.Conv1d(in_channels=2, out_channels=nb_channels, kernel_size=1)]

    for i in range(13):
        cnn.append(nn.Conv1d(in_channels=nb_channels, out_channels=nb_channels, kernel_size=2, dilation=2**i))
        cnn.append(nn.SELU())

    cnn.append(nn.Flatten())

    return nn.Sequential(*cnn)

def train_flow_sbi(simulation_budget, epochs, lr, batch_size, out, task_index, bagging=False, static=False):
    inputs = np.load("data/train/inputs.npy")
    outputs = np.load("data/train/outputs.npy", mmap_mode='r')

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
    outputs = outputs.float()

    os.makedirs("sbi-logs",  exist_ok=True)
    log_writer = SummaryWriter(log_dir=os.path.join("sbi-logs", "{}_{}".format(simulation_budget, task_index)))
    embedding = build_embedding()
    density_estimator_build_fun = posterior_nn(model='nsf', hidden_features=64, num_transforms=3, embedding_net=embedding)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inference = SNPE(prior=prior, density_estimator=density_estimator_build_fun, summary_writer=log_writer, device=device)
    density_estimator = inference.append_simulations(inputs, outputs).train(learning_rate=lr, max_num_epochs=epochs, training_batch_size=batch_size)
    posterior = inference.build_posterior(density_estimator)

    with open(os.path.join(out, "posterior.pkl"), "wb") as handle:
        pickle.dump(posterior, handle)


### Ratio estimators ###########################################################

class ClassifierRatioEstimator(BaseRatioEstimator):

    def __init__(self):
        super(ClassifierRatioEstimator, self).__init__(denominator = "inputs|outputs", 
                                                       random_variables={"inputs": (2,), "outputs": (2, 8192)})

        nb_channels = 16
        fc_layers = [nb_channels + 2, 128, 128, 128, 1]

        cnn = [nn.Conv1d(in_channels=2, out_channels=nb_channels, kernel_size=1)]

        for i in range(13):
            cnn.append(nn.Conv1d(in_channels=nb_channels, out_channels=nb_channels, kernel_size=2, dilation=2**i))
            cnn.append(nn.SELU())


        self.features = nn.Sequential(*cnn)
        fc = []
        for i in range(len(fc_layers) - 1):
            fc.append(nn.Linear(fc_layers[i], fc_layers[i+1]))
            fc.append(nn.SELU())

        fc.pop()
        self.fc = nn.Sequential(*fc)

        self.features.type(torch.float32)
        self.fc.type(torch.float32)
        

    def log_ratio(self, inputs, outputs, **kwargs):
        inputs = inputs.type(torch.float32)
        outputs = outputs.type(torch.float32)
        features = self.features(outputs).view(outputs.shape[0], -1)
        concat = torch.cat((features, inputs), 1)
        return self.fc(concat)

class FlowRatioEstimator(BaseRatioEstimator):
    # Need change
    def __init__(self):
        denominator = "inputs|outputs"
        random_variables = {"inputs": (1,), "outputs": (4,)}
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

### Datasets ###################################################################

class NumpyDiskDataset(Dataset):
    def __init__(self, array, indices=None):
        self.array = array
        self.indices = indices

    def __len__(self):
        if self.indices is None:
            return len(self.array)
        else:
            return len(self.indices)

    def __getitem__(self, idx):
        if self.indices is None:
            return torch.from_numpy(np.copy(self.array[idx]))
        else:
            return torch.from_numpy(np.copy(self.array[self.indices[idx]]))

class DatasetJointTrain(NamedDataset):

    def __init__(self, n=None, bagging=False, static=False):
        inputs = np.load("data/train/inputs.npy")
        outputs = np.load("data/train/outputs.npy", mmap_mode='r')
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

            indices = np.sort(indices)  # Decrease probability of page-fault

        inputs = NumpyDiskDataset(inputs, indices=indices)
        outputs = NumpyDiskDataset(outputs, indices=indices)
        super(DatasetJointTrain, self).__init__(
            inputs=inputs,
            outputs=outputs)


class DatasetJointTest(NamedDataset):

    def __init__(self):
        inputs = np.load("data/test/inputs.npy")
        outputs = np.load("data/test/outputs.npy", mmap_mode='r')
        inputs = NumpyDiskDataset(inputs)
        outputs = NumpyDiskDataset(outputs)
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