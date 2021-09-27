## Abstract


We present extensive empirical evidence showing that current Bayesian simulation-based inference algorithms are inadequate for the falsificationist methodology of scientific inquiry. Our results collected through massive experimental computations show that all benchmarked algorithms -- (S)NPE, (S)NRE, SNL and variants of ABC -- may produce overconfident posterior approximations, which makes them demonstrably unreliable and dangerous if one's scientific goal is to constrain parameters of interest. We believe that failing to address this issue will lead to a well-founded trust crisis in simulation-based inference. For this reason, we argue that research efforts should now focus on theoretical and methodological developments of conservative approximate inference algorithms and present research directions towards this objective. In this regard, we show empirical evidence that ensembles are consistently more reliable.


## Using the code

> **Recommended**. **This installs a Python 3 environment by default.**

```console
you@computer:~ $ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
you@computer:~ $ sh Miniconda3-latest-Linux-x86_64.sh
```

Next, install the necessary dependencies.

```console
you@computer:~ conda env create -f environment.yml
you@computer:~ conda activate crisissbi
```

After the environment has been activated, there are 2 ways to execute the pipelines depending on your setup.
The first only requires your laptop. In that regard simply execute a pipeline as follows:
```console
you@computer:~ cd workflows/auc_demonstration
you@computer:~ python pipeline.py
```
The other approach is on a Slurm enabled HPC cluster. To exploit the parallelism, execute the script as
```console
you@computer:~ cd workflows/auc_demonstration
you@computer:~ python pipeline.py --slurm
```
The jobs will be automatically submitted to the default Slurm queue.


## Citation

See `CITATION.cff`

## License

Described the `LICENSE` file.
