r"""CASBI: Conservative Amortized Simulation-Based Inference

Toy problem highlighting the connection between conditional flows
and ratio estimators.
"""

import hypothesis as h
import numpy as np
import torch

from hypothesis.nn import build_ratio_estimator
from hypothesis.nn.ratio_estimation import BaseRatioEstimator
from util import Prior


### Ratio estimators ###########################################################


class RatioEstimator(BaseRatioEstimator):

    def __init__(self):
        random_variables = {"inputs": (1,), "outputs": (1,)}
        Class = build_ratio_estimator("mlp", random_variables)
        activation = torch.nn.SELU
        trunk = [128] * 3
        r = Class(activation=activation, trunk=trunk)
        super(RatioEstimator, self).__init__(r=r)
        self._r = r

    def log_ratio(self, **kwargs):
        return self._r.log_ratio(**kwargs)


class FlowRatioEstimator(BaseRatioEstimator):

    def __init__(self):
        # Ratio estimator setup
        denominator = "inputs|outputs"
        random_variables = {"inputs": (1,), "outputs": (1,)}
        super(FlowRatioEstimator, self).__init__(
            denominator=denominator,
            random_variables=random_variables)
        # Flow definition
        self._prior = Prior()
        # TODO Flow definition
        raise NotImplementedError

    def log_ratio(self, inputs, outputs, **kwargs):
        # TODO Implement computation of the ratio.
        raise NotImplementedError
