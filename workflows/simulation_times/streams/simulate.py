import hypothesis
import numpy as np
import torch
import time

from simulators import GD1StreamSimulator
from simulators import WDMSubhaloSimulator
from util import allocate_prior_stream_age as PriorAge
from util import allocate_prior_wdm_mass as PriorMass


prior_age = PriorAge()
prior_mass = PriorMass()
n = 1000
times = []
for _ in range(5):
    time_start = time.time()
    ages = prior_age.sample().view(-1, 1)
    masses = prior_mass.sample((n,))
    simulator = GD1StreamSimulator()
    stream = simulator(ages)[0]
    simulator = WDMSubHaloSimulator(stream, record_impacts=True)
    times.append(time.time() - time_start)
    print("Done!")

print(np.mean(times))
print(np.std(times))
