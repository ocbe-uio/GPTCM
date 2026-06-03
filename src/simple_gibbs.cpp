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
