/* header file for updating variances using classical Gibbs sampler */

#ifndef SIMPLE_GIBBS_H
#define SIMPLE_GIBBS_H

#include <cmath>
#include <RcppArmadillo.h>


double sampleV(
    const double& vA,
    const double& vB,
    const arma::vec& xi
);

double sampleV0(
    const double& v0A,
    const double& v0B,
    const double& xi0
);

double sampleW(
    const double& wA,
    const double& wB,
    const arma::vec& zetas
);

double sampleW0(
    const double& wA,
    const double& wB,
    const double& zeta0//arma::rowvec& zeta0
);

double sampleTau(
    const double& tauA,
    const double& tauB,
    // const arma::vec& betas
    const arma::mat& betas
);

#endif
