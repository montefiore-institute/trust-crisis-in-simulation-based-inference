import time
import numpy as np

from hypothesis.benchmark.spatialsir import Prior
from hypothesis.benchmark.spatialsir import Simulator


times = []
n = 1000
for _ in range(5):
    time_start = time.time()
    prior = Prior()
    simulator = Simulator()
    inputs = prior.sample((n,))
    outputs = simulator(inputs)
    times.append(time.time() - time_start)

print(np.mean(times))
print(np.std(times))
