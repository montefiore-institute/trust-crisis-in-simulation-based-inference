import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

def sbc_run(prior, dataset, posterior_sample_function, nb_rank_samples, nb_posterior_samples, save_name):
    ranks = []
    for i in tqdm(range(nb_rank_samples)):
        #theta = prior.sample()
        #x = simulator(theta)
        theta = dataset[i]["inputs"]
        x = dataset[i]["outputs"]
        posterior_samples = posterior_sample_function(x, nb_posterior_samples)

        f_theta = theta.mean()
        f_posterior_samples = posterior_samples.mean(axis=1)
        ranks.append((f_posterior_samples < f_theta).sum())

    ranks = np.array(ranks)
    np.save(save_name, ranks)


def plot_sbc(file_name, save_name, bins=50):
    ranks = np.load(file_name)
    plt.hist(ranks, bins=bins)
    plt.savefig(save_name)
