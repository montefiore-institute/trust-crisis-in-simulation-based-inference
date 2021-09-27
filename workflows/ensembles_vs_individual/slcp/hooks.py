r"""CASBI: Conservative Amortized Simulation-Based Inference

Custom hooks for the trainer. The following events are available:
 - batch_test_complete
 - batch_test_start
 - batch_train_complete
 - batch_train_start
 - batch_validate_complete
 - batch_validate_start
 - epoch_complete
 - epoch_start
 - fit_complete
 - fit_start
 - new_best_test
 - new_best_train
 - new_best_validate
 - test_complete
 - test_start
 - train_complete
 - train_start
 - validate_complete
 - validate_start

"""

import hypothesis as h
import logging
import numpy as np
import torch

from ratio_estimation import DatasetJointTest


def add(arguments, trainer):
    add_loss_monitor(arguments, trainer)


def add_loss_monitor(arguments, trainer):
    losses = []

    @torch.no_grad()
    def record_loss_functional(trainer, **kwargs):
        size = 10000
        estimator = trainer.estimator
        dataset = DatasetJointTest()
        n = len(dataset)
        L = torch.nn.BCELoss()
        ones = torch.ones(n, 1)
        ones = ones.to(h.accelerator)
        zeros = torch.zeros(n, 1)
        zeros = zeros.to(h.accelerator)
        # Joint
        inputs = dataset[:]["inputs"]
        outputs = dataset[:]["outputs"]
        inputs = inputs.to(h.accelerator)
        outputs = outputs.to(h.accelerator)
        d, _ = estimator(inputs=inputs, outputs=outputs)
        loss_joint = L(d, ones)
        # Marginals
        inputs = inputs[torch.randperm(n)]
        d, _ = estimator(inputs=inputs, outputs=outputs)
        loss_marginals = L(d, zeros)
        # Store result
        losses.append(loss_joint.item() + loss_marginals.item())


    def save_loss(trainer, **kwargs):
        np.save(arguments.out + "/test-loss-functionals.npy", np.array(losses))

    # Register the hooks.
    trainer.add_event_handler(trainer.events.fit_start, record_loss_functional)
    trainer.add_event_handler(trainer.events.epoch_complete, record_loss_functional)
    trainer.add_event_handler(trainer.events.fit_complete, save_loss)
