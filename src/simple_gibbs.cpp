// Gibbs sampling for variance parameters

#include "simple_gibbs.h"
#include <stdio.h>


// update \xi's variance vSq
double sampleV(
    const double& vA,
    const double& vB,
    const arma::vec& xi
)
{
    // xi.shed_row(0);
    double vA_post = vA + 0.5 * arma::accu(xi != 0.);
    double vB_post = vB + 0.5 * arma::as_scalar(xi.t() * xi);

    double vSq = 1. / R::rgamma(vA_post, 1. / vB_post);

    return vSq;
}

// update \xi0 variance v0Sq
double sampleV0(
    const double& v0A,
    const double& v0B,
    const double& xi0
)
{
    double v0A_post = v0A + 0.5;
    double v0B_post = v0B + 0.5 * xi0 * xi0;

    double v0Sq = 1. / R::rgamma(v0A_post, 1. / v0B_post);

    return v0Sq;
}

// update \zetas' variance wSq
double sampleW(
    const double& wA,
    const double& wB,
    const arma::vec& zetas
)
{
    double wA_post = wA + 0.5 * arma::accu(zetas != 0.);
    double wB_post = wB + 0.5 * arma::accu(zetas % zetas);

    double wSq = 1. / R::rgamma(wA_post, 1. / wB_post);

    return wSq;
}

// update \zeta0's variance w0Sq
double sampleW0(
    const double& wA,
    const double& wB,
    const double& zeta0//arma::rowvec& zeta0
)
{
    double wA_post = wA + 0.5;// * arma::accu(zeta0 != 0.);
    double wB_post = wB + 0.5 * zeta0 * zeta0;//arma::as_scalar(zeta0 * zeta0.t());

    double wSq = 1. / R::rgamma(wA_post, 1. / wB_post);

    return wSq;

}

// update \betas' variance tauSq
double sampleTau(
    const double& tauA,
    const double& tauB,
    // const arma::vec& betas
    const arma::mat& betas
)
{
    double tauA_post = tauA + 0.5 * arma::accu(betas != 0.);
    double tauB_post = tauB + 0.5 * arma::accu(betas % betas);

    double tauSq = 1. / R::rgamma(tauA_post, 1. / tauB_post);

    return tauSq;
}
