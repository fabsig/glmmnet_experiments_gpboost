# Code and results for running the experiments of Avanzi et al. (2023) for GPBoost

This repository contains code to run the experiments of [Avanzi et al. (2023, "Machine Learning with High-Cardinality Categorical Features in Actuarial Applications")](https://arxiv.org/abs/2301.12710)  when choosing tuning parameters using cross-validation for [GPBoost](https://github.com/fabsig/GPBoost) instead of using an arbitrary choice.

Avanzi et al. (2023, *version [v1] Mon, 30 Jan 2023 07:35:18 UTC on arXiv*) use unfavorable tuning parameters for GPBoost when running their experiments, in particular, they do not select tuning parameters in an objective and systematic way. In this repository, we reproduce the simulated experiments of Avanzi et al. (2023) for GPBoost when choosing tuning parameters using cross-validation (on the training data in  every simulation run). 

The results are reported in the [results directory](https://github.com/fabsig/glmmnet_experiments_gpboost/tree/main/results). We obtain better results for GPBoost compared to almost all results reported in Avanzi et al. (2023) including those for GLMMNet. The file [run_experiments_gpboost.py](https://github.com/fabsig/glmmnet_experiments_gpboost/blob/main/run_experiments_gpboost.py) contains the code to run these experiments.

GPBoost version 1.0.1 was used for obtaining the results reported here. *Note that a relatively simple approach is used for choosing tuning parameters: 4-fold CV with a deterministic search on a coarse grid. It is likely that better results could be obtained by considering (i) additional tuning parameters (l1 / l2 penalties on leaves, num_leaves with max_depth=-1, etc.), (ii) a finer grid, and (iii) more folds (e.g., 10-fold CV).*
