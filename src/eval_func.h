/* header file for evaluation functions (i.e. log densities) */

#ifndef EVAL_FUNC_H
#define EVAL_FUNC_H

#include <stdio.h>
#include <RcppArmadillo.h>


// typedef std::vector<double> stdvec;


typedef struct common_data
{
    // members
    double *currentPars;

    unsigned int jj;
    unsigned int l;
    unsigned int p;
    unsigned int L;
    unsigned int N;

    double v0Sq;
    double vSq;
    double vA;
    double vB;
    double tau0Sq;
    double tauSq;
    double w0Sq;
    double wSq;
    double phi;
    double Delta;
    bool dirichlet;

    // double mrfA;
    // double mrfB;
    // double mrfG;
    // double piA;
    // double piB;

    double kappa;
    double kappaA;
    double kappaB;
    bool invGamma;
    double *datTheta;
    double *datMu;
    const double *datX;
    double *datProportion;
    const double *datProportionConst;
    const unsigned int *datEvent;
    const double *datTime;
    double *weibullS;
    double *weibullLambda;
} dataS;

class EvalFunction
{
public:

    static double log_dens_xis
    (
        double par,
        void *abc_data
    );

    static double log_dens_betas
    (
        double par,
        void *abc_data
    );

    static double log_dens_zetas
    (
        double par,
        void *abc_data
    );

    static double log_dens_phi
    (
        double par,
        void *abc_data
    );

    static double log_dens_kappa
    (
        double par,
        void *abc_data
    );

    static double pdfTruncNorm
    (
        double x,
        double m,
        double sd,
        double lower,
        double upper
    );
};

#endif
