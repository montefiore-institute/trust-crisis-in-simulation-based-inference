from ratio_estimation import *
from util import *
from plotting import *
import torch
import hypothesis as h
import matplotlib.pyplot as plt

h.plot.activate()


r = load_estimator("output/65536/with-regularization/mlp-*/weights.th")
print(measure_diagnostic(r))

d = DatasetJointTest()
indices = np.random.choice(np.arange(len(d)), size=10, replace=False)
for index in indices:
    inputs = d[index]["inputs"]
    outputs = d[index]["outputs"]
    plot_posterior(plt.gca(), r, outputs, nominal=inputs)
    plot_contours(plt.gca(), r, outputs)
    plt.show()
