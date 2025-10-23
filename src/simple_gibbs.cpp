// Gibbs sampling for variance parameters

#include "simple_gibbs.h"
#include <stdio.h>


// update \xi's variance vSq
double sampleV(
    const double a,
    const double b,
    const arma::vec& xi
)
{
    // xi.shed_row(0);
    double a_post = a + 0.5 * arma::accu(xi != 0.);
    // double b_post = b + 0.5 * arma::as_scalar(xi.t() * xi);
    double b_post = b + 0.5 * arma::accu(xi % xi);

    double vSq = 1. / R::rgamma(a_post, 1. / b_post);

    return vSq;
}

// update \xi0 variance v0Sq
double sampleV0(
    const double a,
    const double b,
    const double xi0
)
{
    double a_post = a + 0.5;
    double b_post = b + 0.5 * xi0 * xi0;

    double v0Sq = 1. / R::rgamma(a_post, 1. / b_post);

    return v0Sq;
}

// // update \zetas' variance wSq
// double sampleW(
//     const double a,
//     const double b,
//     const arma::vec& zetas
// )
// {
//     double a_post = a + 0.5 * arma::accu(zetas != 0.);
//     double b_post = b + 0.5 * arma::accu(zetas % zetas);

//     double wSq = 1. / R::rgamma(a_post, 1. / b_post);

//     return wSq;
// }

// // update \zeta0's variance w0Sq
// double sampleW0(
//     const double a,
//     const double b,
//     const double zeta0//arma::rowvec& zeta0
// )
// {
//     double a_post = a + 0.5;// * arma::accu(zeta0 != 0.);
//     double b_post = b + 0.5 * zeta0 * zeta0;//arma::as_scalar(zeta0 * zeta0.t());

//     double wSq = 1. / R::rgamma(a_post, 1. / b_post);

//     return wSq;

// }

// // update \betas' variance tauSq
// double sampleTau(
//     const double a,
//     const double b,
//     // const arma::vec& betas
//     const arma::mat& betas
// )
// {
//     double a_post = a + 0.5 * arma::accu(betas != 0.);
//     double b_post = b + 0.5 * arma::accu(betas % betas);

//     double tauSq = 1. / R::rgamma(a_post, 1. / b_post);

//     return tauSq;
// }
