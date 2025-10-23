/* header file for updating variances using classical Gibbs sampler */

#ifndef SIMPLE_GIBBS_H
#define SIMPLE_GIBBS_H

#include <cmath>
#include <RcppArmadillo.h>


double sampleV(
    const double a,
    const double b,
    const arma::vec& xi
);

double sampleV0(
    const double a,
    const double b,
    const double xi0
);

// double sampleW(
//     const double a,
//     const double b,
//     const arma::vec& zetas
// );

// double sampleW0(
//     const double a,
//     const double b,
//     const double zeta0//arma::rowvec& zeta0
// );

// double sampleTau(
//     const double a,
//     const double b,
//     // const arma::vec& betas
//     const arma::mat& betas
// );

#endif
