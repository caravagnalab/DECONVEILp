// -----------------------------------------------------------------------------
// Gene-by-gene single-group CN-expression model (log link)
//
// Tumour mean:
//   tumour_mu = sf * purity *
//                exp(
//                  b0
//                  + dose_log * b_scaling
//                  + dev * b_deviation
//                )
//
// where:
//   dose_log = log(CN / 2)   (0 at CN = 2)
//   dev      = (CN - 2) / 2  (0 at CN = 2)
//
// Full mean:
//   mu = tumour_mu
//        + sf * (1 - purity) * exp(b_noncancer_log)
//
// Interpretation:
//   b0          = log tumour baseline at diploid CN
//   b_scaling   = proportional dosage sensitivity
//   b_deviation = linear deviation from proportional scaling
// -----------------------------------------------------------------------------

data {
  int<lower=1> N;
  array[N] int<lower=0> y;

  // Size factors must be strictly positive because log(sf) is used.
  vector<lower=1e-12>[N] sf;

  vector<lower=0, upper=1>[N] purity;

  vector[N] dose_log;  // log(CN / 2)
  vector[N] dev;       // (CN - 2) / 2
}

parameters {
  real b0;
  real b_scaling;
  real b_deviation;

  real b_noncancer_log;
  real<lower=1e-6, upper=1e3> phi;
}

transformed parameters {
  vector[N] log_mu;

  for (n in 1:N) {
    real tumour_linpred =
      b0
      + dose_log[n] * b_scaling
      + dev[n] * b_deviation;

    // log(mu) for the sum of tumour and non-cancer expression.
    log_mu[n] =
      log(sf[n])
      + log_sum_exp(
          log(purity[n]) + tumour_linpred,
          log1m(purity[n]) + b_noncancer_log
        );
  }
}

model {
  // Empirical biology-informed priors.
  b0 ~ normal(5.0, 2.0);
  b_scaling ~ normal(0.4, 0.5);
  b_deviation ~ normal(0.0, 0.3);

  b_noncancer_log ~ normal(log(5.0), 0.5);
  phi ~ lognormal(log(20.0), 0.5);

  y ~ neg_binomial_2_log(log_mu, phi);
}

generated quantities {
  // CN 2 -> 1
  real lp_scaling_2to1;
  real lp_dev_2to1;
  real lp_2to1;
  real tumor_fc_2to1;
  real fracCN_2to1;
  real cancel_index_2to1;

  // CN 2 -> 3
  real lp_scaling_2to3;
  real lp_dev_2to3;
  real lp_2to3;
  real tumor_fc_2to3;
  real fracCN_2to3;
  real cancel_index_2to3;

  // CN 2 -> 4
  real lp_scaling_2to4;
  real lp_dev_2to4;
  real lp_2to4;
  real tumor_fc_2to4;
  real fracCN_2to4;
  real cancel_index_2to4;

  vector[N] log_lik;
  vector[N] mu_rep;
  array[N] int y_rep;

  // CN 2 -> 1: dose_log = log(1 / 2), dev = -0.5
  lp_scaling_2to1 = log(1.0 / 2.0) * b_scaling;
  lp_dev_2to1 = -0.5 * b_deviation;
  lp_2to1 = lp_scaling_2to1 + lp_dev_2to1;

  tumor_fc_2to1 = exp(lp_2to1);
  fracCN_2to1 = tumor_fc_2to1 - 1.0;

  cancel_index_2to1 =
    lp_dev_2to1
    / fmax(abs(lp_scaling_2to1), 1e-12);

  // CN 2 -> 3: dose_log = log(3 / 2), dev = +0.5
  lp_scaling_2to3 = log(3.0 / 2.0) * b_scaling;
  lp_dev_2to3 = 0.5 * b_deviation;
  lp_2to3 = lp_scaling_2to3 + lp_dev_2to3;

  tumor_fc_2to3 = exp(lp_2to3);
  fracCN_2to3 = tumor_fc_2to3 - 1.0;

  cancel_index_2to3 =
    lp_dev_2to3
    / fmax(abs(lp_scaling_2to3), 1e-12);

  // CN 2 -> 4: dose_log = log(4 / 2), dev = +1
  lp_scaling_2to4 = log(4.0 / 2.0) * b_scaling;
  lp_dev_2to4 = b_deviation;
  lp_2to4 = lp_scaling_2to4 + lp_dev_2to4;

  tumor_fc_2to4 = exp(lp_2to4);
  fracCN_2to4 = tumor_fc_2to4 - 1.0;

  cancel_index_2to4 =
    lp_dev_2to4
    / fmax(abs(lp_scaling_2to4), 1e-12);

  for (n in 1:N) {
    real log_mu_safe = fmin(log_mu[n], 20.0);

    log_lik[n] =
      neg_binomial_2_log_lpmf(
        y[n] | log_mu[n], phi
      );

    mu_rep[n] = exp(log_mu[n]);

    // The cap prevents RNG overflow for extreme fitted means.
    y_rep[n] =
      neg_binomial_2_log_rng(
        log_mu_safe,
        phi
      );
  }
}
