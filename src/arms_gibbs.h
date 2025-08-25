/* header file for univariate and multivariate arms for all parameters */

#ifndef ARMS_GIBBS_H
#define ARMS_GIBBS_H

#include <cmath>

#include "arms.h"
#include "eval_func.h"
#include "global.h"


class ARMS_Gibbs
{
public:

    static void arms_gibbs_xi(
        // int n,
        // int nsamp,
        // int ninit,
        // int metropolis,
        // bool simple,
        // double convex,
        // int npoint,
        const armsParmClass armsPar,
        arma::vec& currentPars,
        double v0Sq,
        double vSq,
        // double vA,
        // double vB,
        arma::mat datProportion,
        arma::mat weibullS,
        const DataClass& dataclass
    );

    static void arms_gibbs_beta(
        const armsParmClass armsPar,
        arma::mat& currentPars,
        arma::vec& tauSq,
        double& tau0Sq,
        double tauA,
        double tauB,
        double tau0A,
        double tau0B,

        // const arma::umat& gammas,
        arma::umat gammas,

        double kappa,
        arma::vec& datTheta,
        arma::mat datMu,
        arma::mat& datProportion,
        arma::mat weibullS,
        const DataClass& dataclass,
        double& logPosteriorBeta
    );

    static void arms_gibbs_betaK(
        const unsigned int k,
        const armsParmClass armsPar,
        arma::mat& currentPars,
        double tau0Sq,
        double tauSqK,
        double tauA,
        double tauB,

        // const arma::umat& gammas,
        arma::umat gammas,

        double kappa,
        arma::vec& datTheta,
        arma::mat datMu,
        arma::mat& datProportion,
        arma::mat weibullS,
        const DataClass& dataclass,
        double& logPosteriorBeta
    );

    static void arms_gibbs_zeta(
        const armsParmClass armsPar,
        arma::mat& currentPars,
        double& w0Sq,
        arma::vec& wSq,
        // bool w0IGamma,
        double w0A,
        double w0B,
        double wA,
        double wB,

        arma::umat etas,

        double kappa,
        bool dirichlet,
        arma::vec& datTheta,
        arma::mat weibullS,
        arma::mat weibullLambda,
        const DataClass& dataclass,
        double& logPosteriorZeta
    );

    static void arms_gibbs_zetaK(
        const unsigned int k,
        const armsParmClass armsPar,
        arma::mat& currentPars,
        double w0Sq,
        double wSq,
        // bool w0IGamma,
        // double w0A,
        // double w0B,
        double wA,
        double wB,

        arma::umat etas,

        double kappa,
        bool dirichlet,
        arma::vec& datTheta,
        arma::mat weibullS,
        arma::mat weibullLambda,
        const DataClass& dataclass,
        double& logPosteriorZeta
    );

    static void arms_kappa(
        const armsParmClass armsPar,
        double& currentPars,
        double kappaA,
        double kappaB,
        bool invGamma,
        arma::vec datTheta,
        arma::mat datMu,
        arma::mat datProportion,
        const DataClass& dataclass
    );

    static void slice_sample(
        double (*logfn)(double par, void *mydata),
        void *mydata,
        double& x,
        const unsigned int steps,
        const double w,
        const double lower,
        const double upper
    );

};


#endif
