"""
Run the simulated experiments of Avanzi et al. (2023, "Machine Learning with High-Cardinality Categorical Features in Actuarial Applications", 
                                                https://arxiv.org/abs/2301.12710) 
for GPBoost when choosing tuning parameters using cross-validation on the training data. 

Code for running and evaluating the experiments is the same as in Avanzi et al. (2023)
    taken from https://github.com/agi-lab/glmmnet (after this commit: https://github.com/agi-lab/glmmnet/commit/366f41fee475a3cd74a70afc3744ed36f2dcc3e9)
"""

path = "SET_PATH_FOR_RESULTS"

import numpy as np
import pandas as pd
from sklearn.datasets import make_friedman1, make_friedman3
import gpboost as gpb
import time
from matplotlib import pyplot as plt
from sklearn.utils import check_random_state
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from scipy.stats import norm, gamma, lognorm
from scipy.special import beta
plt.style.use('ggplot')

"""
Define functions used for experiments
"""

def make_sim(
        X, f_X, n_categories, signal_to_noise=np.array([4, 2, 1]),
        y_dist="gaussian", inverse_link=None, cat_dist="balanced", random_state=None):
    """
    Generate synthetic regression data from mixed effects models:
        `g(mu) = f(X) + Z*u`,
    where `f(X)` is a nonlinear function of `X` (feature matrix), `Z` is 
    a matrix of random effects variables, `u` is a vector of random effects,
    and `g(mu)` is a nonlinear function of `mu` (mean response).

    Parameters
    ----------
    X : feature matrix of fixed effects, pd.DataFrame of shape (n_samples, n_features).
    f_X : nonlinear deterministic function of `X`, ndarray of shape (n_samples, ).
    n_categories : int, number of categories/groups.
    signal_to_noise : ndarray of shape (3, ), default=np.array([4, 2, 1])
        The relative ratio of signal from fixed effects, signal from random effects, 
        and noise in the response. It will be normalized to sum to 1.
    y_dist : str, default="gaussian"
        The distribution of the response variable, "gaussian", "gamma", or "lognormal".
    inverse_link : callable, inverse of link function.
        If None, the exp function is used when y_dist="gamma", and the identity function
        is used otherwise.
    cat_dist : str, default="balanced"
        "balanced" or "skewed" distributions for the allocation of categories.
        "balanced" allocates approx equal number of observations to each category.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset noise. Pass an int
        for reproducible output across multiple function calls.
    
    Returns
    -------
    X : pd.DataFrame of shape (n_samples, n_features + 1).
        The input samples, including a column of category labels from 0 to n_categories-1.
    y : ndarray of shape (n_samples,)
        The output values.
    truth : ndarray of shape (n_samples,)
        The true mean response values (unobservable in practice).
    Zu : ndarray of shape (n_samples,)
        The raw random effects.
    """
    rng = check_random_state(random_state)
    n_samples = X.shape[0]
    n_features = X.shape[1]
    X = pd.DataFrame(X, columns=["X" + str(i) for i in range(1, n_features + 1)])
    if (cat_dist == "balanced"):
        X = X.assign(
            # Generate a random number from 0 to (n_categories - 1)
            category = rng.randint(low=0, high=n_categories, size=n_samples).astype(object)
        )
    elif (cat_dist == "skewed"):
        X = X.assign(
            category = np.floor(rng.beta(a=2, b=3, size=n_samples) * n_categories).astype(int).astype(object)
        )
    
    # Generate random effects
    signal_to_noise = signal_to_noise / sum(signal_to_noise)
    signal_FE, signal_RE, noise = tuple(signal_to_noise)
    u = rng.standard_normal(size=n_categories) * signal_RE
    Zu = u[X.category.astype(int)]

    # Scale the fixed effects so that the mean of f(X) = signal_FE
    f_X = f_X / f_X.mean() * signal_FE

    # Generate response variable
    if inverse_link is None:
        # By default, use log link for gamma and identity link for lognormal or gaussian
        inverse_link = np.exp if y_dist == "gamma" else (lambda x: x)
    truth = inverse_link(f_X + Zu)
    if (y_dist == "gaussian"):
        y = rng.normal(loc=truth, scale=noise, size=n_samples)
    elif (y_dist == "gamma"):
        # We need:
        # 1. gamma_mean = truth, i.e. gamma_shape * gamma_scale = truth
        # 2. gamma_variance = (noise ** 2) * (truth ** 2), where gamma_variance = gamma_mean ** 2 / gamma_shape
        gamma_shape = 1 / (noise ** 2)
        gamma_scale = (noise ** 2) * truth
        y = rng.gamma(shape=gamma_shape, scale=gamma_scale, size=n_samples)
    elif (y_dist == "lognormal"):
        # 1. ln_mean = truth
        # 2. ln_variance = noise ** 2
        ln_sigma = np.sqrt(np.log(1 + (noise ** 2) / (truth ** 2)))
        ln_mean = np.log(truth) - ln_sigma ** 2 / 2
        y = rng.lognormal(mean=ln_mean, sigma=ln_sigma, size=n_samples)
    
    return X, y, truth, Zu


def split(X, y, n_train):
    """
    Deterministic split of the data into training and test sets at n_train.
    """
    X_train = X.iloc[:n_train, :]
    X_test = X.iloc[n_train:, :]
    y_train = y[:n_train]
    y_test = y[n_train:]
    
    return X_train, X_test, y_train, y_test


def crps_norm(y, loc, scale):
    """
    Compute CRPS of a location-scale transformed normal distribution.
    Translated from the R package `scoringRules`.
    Source code: https://github.com/FK83/scoringRules/blob/master/R/scores_norm.R
    """
    y = np.array(y, dtype=float)
    y = y - loc
    z = np.divide(y, scale, out=np.zeros_like(y), where=(~ np.isclose(y, 0) | ~ np.isclose(scale, 0)))
    crps = scale * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    return crps

def crps_gamma(y, shape, scale):
    """
    Compute CRPS of a gamma distribution.
    Translated from the R package `scoringRules`.
    Source code: https://github.com/FK83/scoringRules/blob/master/R/scores_gamma.R
    """
    y = np.array(y, dtype=float)
    p1 = gamma.cdf(y, a=shape, scale=scale)
    p2 = gamma.cdf(y, a=np.add(shape, 1), scale=scale)
    crps = y * (2*p1 - 1) - scale * (shape * (2*p2 - 1) + 1 / beta(0.5, shape))
    return crps

def crps_lognorm(y, meanlog, sdlog):
    """
    Compute CRPS of a lognormal distribution.
    Translated from the R package `scoringRules`.
    Source code: https://github.com/FK83/scoringRules/blob/master/R/scores_lnorm.R
    """
    y = np.array(y, dtype=float)
    c1 = y * (2 * lognorm.cdf(y, s=sdlog, scale=np.exp(meanlog)) - 1)
    c2 = 2 * np.exp(np.add(meanlog, np.power(sdlog, 2) / 2))
    c3 = lognorm.cdf(y, s=sdlog, scale=np.exp(np.add(meanlog, np.power(sdlog, 2)))) + norm.cdf(sdlog / np.sqrt(2)) - 1
    crps = c1 - c2 * c3
    return crps


def evaluate_predictions(y, y_pred, categories, likelihood="gaussian", **kwargs):
    """
    Evaluate the performance of a model based on the predictions.
    Specify the likelihood of the predictions with the keyword argument `likelihood`.

    Parameters
    ----------
    y: true values
    y_pred: predicted values
    categories: categories of the observations
    likelihood: form of the response distribution, one of the following:
        - "gaussian": normal distribution;
        - "gamma": gamma distribution;
        - "lognorm": lognormal distribution;
        - "loggamma": loggamma distribution.
    **kwargs: additional keyword arguments to specify the estimated parameters of the likelihood (to compute CRPS):
        - "gaussian" requires `loc` and `scale`;
        - "gamma" requires `shape` (or synonymously `gamma_shape`) and `gamma_scale`;
        - "lognorm" requires `meanlog` and `sdlog`.
        - "loggamma" requires `shape` (or synonymously `gamma_shape`) and `gamma_scale`.
    """
    if likelihood == "gaussian":
        loc = kwargs.get("loc", y_pred)
        scale = kwargs.get("scale", np.sqrt(mean_squared_error(y, y_pred)))

    # Take y and y_pred back to the original scale
    if likelihood == "lognorm":
        # Transform the observations back to the original scale
        y = np.exp(y)
        # Transform the predictions back to the original scale
        meanlog = kwargs.get("meanlog", y_pred)                                      # Infer the predictive parameters from predictions
        sdlog = kwargs.get("sdlog", np.sqrt(mean_squared_error(np.log(y), meanlog))) # Infer the predictive parameters from predictions
        y_pred = np.exp(np.add(meanlog, np.power(sdlog, 2) / 2))

    if likelihood in ["gamma", "loggamma"]:
        gamma_shape = kwargs.get("gamma_shape", kwargs.get("shape", None))
        gamma_scale = kwargs.get("gamma_scale", None)
        if gamma_shape is None:
            # Estimate shape parameter of gamma distribution by Pearson's method
            # Related discussion: https://stats.stackexchange.com/questions/367560/
            # `statsmodels` ref: https://www.statsmodels.org/dev/_modules/statsmodels/genmod/generalized_linear_model.html#GLM.estimate_scale
            resid = np.power(y - y_pred, 2)
            var = np.power(y_pred, 2)
            gamma_dispersion = np.sum(resid / var) / len(y)
            gamma_shape = 1 / gamma_dispersion
        if gamma_scale is None:
            gamma_scale = y_pred / gamma_shape

        if likelihood == "loggamma":
            if any(gamma_scale >= 1):
                raise ValueError("The scale parameter of the loggamma distribution must be < 1 otherwise expectation is infinite.")
            # Transform the observations back to the original scale
            y = np.exp(y)
            # Transform the predictions back to the original scale
            y_pred = np.exp(y_pred)

    scores = dict()

    scores["MAE"] = mean_absolute_error(y, y_pred)
    scores["MedAE"] = median_absolute_error(y, y_pred) # for a more robust estimate of the error
    scores["MedPE"] = np.median(np.divide(np.abs(np.subtract(y, y_pred)), y)) # median percentage error
    scores["RMSE"] = np.sqrt(mean_squared_error(y, y_pred))
    scores["R2"] = r2_score(y, y_pred)

    # RMSE of average prediction for each category
    data = pd.concat([
        pd.Series(categories, name="category").reset_index(drop=True), 
        pd.Series(y, name="y").reset_index(drop=True), 
        pd.Series(y_pred, name="y_pred").reset_index(drop=True)
        ], axis=1)
    gb = data.groupby("category", as_index=False)
    counts = gb.size()
    avg_by_cat = gb[["y", "y_pred"]].mean()
    scores["RMSE_avg"] = np.sqrt(mean_squared_error(avg_by_cat["y"], avg_by_cat["y_pred"]))

    # Volume weighted RMSE of average prediction for each category
    scores["RMSE_avg_weighted"] = np.sqrt(
        mean_squared_error(avg_by_cat["y"], avg_by_cat["y_pred"], sample_weight=counts["size"]))

    # CRPS to quantify accuracy of probabilistic predictions
    if likelihood == "gaussian":
        scores["CRPS"] = crps_norm(y, loc, scale).mean()
    elif likelihood == "gamma":
        scores["CRPS"] = crps_gamma(y, gamma_shape, gamma_scale).mean()
    elif likelihood == "lognorm":
        scores["CRPS"] = crps_norm(np.log(y), meanlog, sdlog).mean()
    elif likelihood == "loggamma":
        scores["CRPS"] = crps_gamma(np.log(y), gamma_shape, gamma_scale).mean()
    
    # Negative log-likelihood of probabilistic predictions
    if likelihood == "gaussian":
        scores["NLL"] = -norm.logpdf(y, loc=loc, scale=scale).mean()
    elif likelihood == "gamma":
        scores["NLL"] = -gamma.logpdf(y, a=gamma_shape, scale=gamma_scale).mean()
    elif likelihood == "lognorm":
        scores["NLL"] = -np.log(norm.pdf(np.log(y), loc=meanlog, scale=sdlog) / y).mean()
    elif likelihood == "loggamma":
        scores["NLL"] = -np.log(gamma.pdf(np.log(y), a=gamma_shape, scale=gamma_scale) / y).mean()

    return scores




"""
Run simulation experiments
"""
# Configure simulation parameters
n_train = 5000                     # number of training observations
n_test = 2500                      # number of test observations
n = n_train + n_test               # total number of observations
n_categories = 100                 # number of categories for the categorical variable
f_structure = "friedman1"          # structure of the fixed effects f(X)
nsim = 50 # number of simulation repetitions

for exp_id in np.arange(1,7):
    print("******** Starting experiment " + str(exp_id))

    if exp_id == 1:
        sig2noise = np.array([4, 1, 1])    # relative signal-to-noise of the fixed effects, random effects and irreducible error
        y_dist = "gaussian"                # distribution of the response variable y
        inverse_link = lambda x: x         # inverse of the identity link
        cat_dist = "balanced"              # distribution of the categorical variable
    elif exp_id == 2:
        sig2noise = np.array([4, 1, 1])    # relative signal-to-noise of the fixed effects, random effects and irreducible error
        y_dist = "gamma"                   # distribution of the response variable y
        inverse_link = np.exp              # inverse of the log link
        cat_dist = "balanced"              # distribution of the categorical variable
    elif exp_id == 3:
        sig2noise = np.array([4, 1, 1])    # relative signal-to-noise of the fixed effects, random effects and irreducible error
        y_dist = "gaussian"                # distribution of the response variable y
        inverse_link = lambda x: x         # inverse of the identity link
        cat_dist = "skewed"                # distribution of the categorical variable
    elif exp_id == 4:
        exp_id = 4                         # experiment id
        sig2noise = np.array([4, 1, 2])    # relative signal-to-noise of the fixed effects, random effects and irreducible error
        y_dist = "gaussian"                # distribution of the response variable y
        inverse_link = lambda x: x         # inverse of the identity link
        cat_dist = "balanced"              # distribution of the categorical variable
    elif exp_id == 5:
        sig2noise = np.array([8, 1, 4])    # relative signal-to-noise of the fixed effects, random effects and irreducible error
        y_dist = "gaussian"                # distribution of the response variable y
        inverse_link = lambda x: x         # inverse of the log link
        cat_dist = "balanced"              # distribution of the categorical variable
    elif exp_id == 6:
        sig2noise = np.array([8, 1, 4])    # relative signal-to-noise of the fixed effects, random effects and irreducible error
        y_dist = "gamma"                   # distribution of the response variable y
        inverse_link = np.exp              # inverse of the log link
        cat_dist = "skewed"                # distribution of the categorical variable

    results = np.zeros(shape=(nsim, 4))
    results_avanzi = np.zeros(shape=(nsim, 4))
    start_sim = time.time()
    for i in range(nsim):
        if i % 10 == 0:
            print("         Simulation number " + str(i))
        
        # Generate data
        if (f_structure == "friedman1"):
            # Out of the n_features features, only 5 are actually used to compute y.
            # The remaining features are independent of y.
            X, f_X = make_friedman1(n_samples=n, n_features=10, noise=0.0, random_state=i)
        elif (f_structure == "friedman3"):
            X, f_X = make_friedman3(n_samples=n, noise=0.0, random_state=i)
        
        # Simulate random effects and therefore the response variable y
        X, y, truth, Zu = make_sim(X, f_X, n_categories, sig2noise, y_dist, inverse_link, cat_dist, i)
        
        # Split data into training and testing sets
        hicard_var = "category"
        X["category"] = X["category"].astype(int)
        X_train, X_test, y_train, y_test = split(X, y, n_train)
        # X_train.shape, X_test.shape
        
        objective = "regression"
        if y_dist == "gamma":
            objective = "gamma"
        elif y_dist == "gaussian":
            objective = "regression"
        
        #--------------------Choosing tuning parameters----------------
        param_grid = {'learning_rate': [1,0.1,0.01], 
                      'min_data_in_leaf': [10,100,1000],
                      'max_depth': [1,2,3,5,10]}
        
        gp_model = gpb.GPModel(group_data=X_train[hicard_var], likelihood=y_dist)
        data_train = gpb.Dataset(X_train.drop([hicard_var], axis=1), y_train)
        params = {'objective': objective, 'verbose': -1, 'num_leaves': 2**10}
        opt_params = gpb.grid_search_tune_parameters(param_grid=param_grid, params=params, nfold=4,
                                                      gp_model=gp_model, train_set=data_train,
                                                      verbose_eval=-1, seed=i,
                                                      num_boost_round=1000, early_stopping_rounds=10)
        
        #--------------------Train model and make predictions----------------
        gp_model = gpb.GPModel(group_data=X_train[hicard_var], likelihood=y_dist)
        params = {'objective': objective, 
                  'learning_rate': opt_params['best_params']['learning_rate'], 
                  'max_depth': opt_params['best_params']['max_depth'],
                  'min_data_in_leaf': opt_params['best_params']['min_data_in_leaf'], 
                  'num_leaves': 2**10,
                  'verbose': 0}
        num_boost_round = opt_params['best_iter']
               
        bst = gpb.train(params=params, train_set=data_train, gp_model=gp_model,
                        num_boost_round=num_boost_round)
        pred = bst.predict(data=X_test.drop([hicard_var], axis=1), group_data_pred=X_test[hicard_var],
                            pred_latent=False, predict_var=False)
            
        scale = None
        gamma_shape = None
        if y_dist == "gaussian":
            scale = np.sqrt(gp_model.get_cov_pars().iloc[0,0])
        elif y_dist == "gamma":
            gamma_shape = gp_model.get_aux_pars().iloc[0,0]
        eval_pred = evaluate_predictions(y=y_test, y_pred=pred['response_mean'], 
                                         categories=X_test[hicard_var], likelihood=y_dist,
                                         scale=scale, gamma_shape=gamma_shape)         

        results[i] = [eval_pred['MAE'], eval_pred['RMSE'], eval_pred['CRPS'], eval_pred['RMSE_avg']]
        
        # Choice of parameters being used in Avanzi et al. (2023)
        params = {'objective': objective, 'learning_rate': 0.01, 'max_depth': 1, 
                  'verbose': 0, 'use_nesterov_acc': True}
        num_boost_round = 100
        gp_model = gpb.GPModel(group_data=X_train[hicard_var], likelihood=y_dist)
        bst = gpb.train(params=params, train_set=data_train, gp_model=gp_model,
                        num_boost_round=num_boost_round)
        pred = bst.predict(data=X_test.drop([hicard_var], axis=1), group_data_pred=X_test[hicard_var],
                            pred_latent=False, predict_var=False)
        scale = None
        gamma_shape = None
        if y_dist == "gaussian":
            scale = np.sqrt(gp_model.get_cov_pars().iloc[0,0])
        elif y_dist == "gamma":
            gamma_shape = gp_model.get_aux_pars().iloc[0,0]
        eval_pred = evaluate_predictions(y=y_test, y_pred=pred['response_mean'], 
                                          categories=X_test[hicard_var], likelihood=y_dist,
                                          scale=scale, gamma_shape=gamma_shape)         
        results_avanzi[i] = [eval_pred['MAE'], eval_pred['RMSE'], eval_pred['CRPS'], eval_pred['RMSE_avg']]
    
    end_sim = time.time()
    print("  Total time for experiment " + str(exp_id) + ": " + str(end_sim - start_sim))
    ## Approx. 10 mins for experiments with a "gaussian" likelihood on a laptop
    ## Approx. 20 mins for experiments with a "gamma" likelihood on a laptop
    
    results_pd = pd.DataFrame(results)
    results_pd.columns = ['MAE', 'RMSE', 'CRPS', 'RMSE_avg']
    results_pd.mean().to_csv(path+'results/results_gpboost_experiment='+str(exp_id)+'.csv', header=['GPBoost'])
    fig, ax = plt.subplots()
    results_pd.boxplot()
    plt.title("Results GPBoost - Experiment " + str(exp_id))
    ax.grid(axis='y', which = "minor", linestyle="dotted")
    ax.minorticks_on()
    plt.savefig(path+'results/results_gpboost_experiment='+str(exp_id)+'.jpeg', dpi=100)
    plt.show()
    
    results_pd = pd.DataFrame(results_avanzi)
    results_pd.columns = ['MAE', 'RMSE', 'CRPS', 'RMSE_avg']
    results_pd.mean().to_csv(path+'results_avanzi/results_gpboost_experiment='+str(exp_id)+'.csv', header=['GPBoost'])
    fig, ax = plt.subplots()
    results_pd.boxplot()
    plt.title("Results GPBoost - Experiment " + str(exp_id))
    ax.grid(axis='y', which = "minor", linestyle="dotted")
    ax.minorticks_on()
    plt.savefig(path+'results_avanzi/results_gpboost_experiment='+str(exp_id)+'.jpeg', dpi=100)
    plt.show()

