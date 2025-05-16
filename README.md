# SentimentAnanlysis
Proper Bayes Model Analysis

## Introduction

This project explores Bayesian modeling using two key approaches:

Metropolis-Hastings MCMC

Hamiltonian Monte Carlo (HMC)

We utilize Amazon review data to predict customer ratings using binary word features as predictors.

## Model Overview

### Model Type

Bayesian Regression

### Sampling Methods:

Metropolis Hastings MCMC

Hamiltonian Monte Carlo

### Priors

Intercept (α) ∼ N (0, 2)

Slope of each word (βᵢ) ∼ N (0, 2)

### Likelihood

Customer ratings, y ∼ Binomial(n = 5, p = θ)

θ = logit⁻¹(α + ∑ βᵢ · wordᵢ)

## Model Architecture and Implementation

Utilizes Amazon review data processed via the DataGenerator class.

Implements two models:

ProperBayes (MCMC)

ProperBayesHMC (HMC)

Supports GPU acceleration, model saving/loading using pickle.

## Experimental Results of MCMC

Without Feature Selection

Step size = 0.01, n_samp = 5000

Posterior mode ≈ 3.0 (neutral ratings on average).

With Feature Selection

Step size = 0.03, n_samp = 2000

Posterior mode ≈ 3.0, better convergence.

## Experimental Results of HMC

With Feature Selection

Step size = 0.01, n_samp = 500

Posterior mode ≈ 2.8, faster convergence.

Step size = 0.01, n_samp = 1000

Posterior mode ≈ 3.1238 (Neutral Ratings).

Step size = 0.01, n_samp = 400

Posterior mode ≈ 2.7270, quick trace stabilization.

## Insights

HMC is computationally expensive but converges faster due to gradient information.

Feature Selection significantly improves convergence.

Both methods achieve an accuracy score of ≈ 65-70%.

