/* header file for global variables*/

#ifndef GLOBAL_H
#define GLOBAL_H

#include <RcppArmadillo.h>

// Define constants for bounds
constexpr double UPPER_BOUND = 700.0;
constexpr double UPPER_BOUND_3 = 170.0; // 270 here and 1.0e-20 below resulted in slightly worse \zetas. Check to use Log-Sum-Exp trick
constexpr double LOWER_BOUND = 1.0e-10;

// Using the constants inline where necessary
inline double upperbound = UPPER_BOUND;
inline double upperbound3 = UPPER_BOUND_3;
// inline double upperbound3 = 700.; // not improve results when n=2000
inline double lowerbound = LOWER_BOUND;
// inline double lowerbound = 1.0e-100; // not improve results when n=2000


class DataClass
{
public:
    // Use const for immutable members
    const arma::uvec datEvent;
    const arma::vec datTime;
    const arma::cube datX;
    const arma::mat datX0;
    const arma::mat datProportionConst;

    // // Use shared_ptr for mutable members
    // std::shared_ptr<arma::mat> datProportion;
    // std::shared_ptr<arma::vec> datTheta;
    // std::shared_ptr<arma::vec> datMu;
    // std::shared_ptr<arma::mat> weibullLambda;
    // std::shared_ptr<arma::mat> weibullS;

    // Constructor to initialize the constants
    DataClass(
        const arma::uvec& datEvent_,
        const arma::vec& datTime_,
        const arma::cube& datX_,
        const arma::mat& datX0_,
        const arma::mat& datProportionConst_//,
        // std::shared_ptr<arma::mat> datProportion_,
        // std::shared_ptr<arma::vec> datTheta_,
        // std::shared_ptr<arma::vec> datMu_,
        // std::shared_ptr<arma::mat> weibullLambda_,
        // std::shared_ptr<arma::mat> weibullS_
    ) :
        datEvent(datEvent_),
        datTime(datTime_),
        datX(datX_),
        datX0(datX0_),
        datProportionConst(datProportionConst_)//,
        // datProportion(datProportion_),
        // datTheta(datTheta_),
        // datMu(datMu_),
        // weibullLambda(weibullLambda_),
        // weibullS(weibullS_)
    {}
} ;

/*
typedef struct arms_parameters
{
    // members
    int n;
    int nsamp;
    int ninit;
    int metropolis;
    bool simple;
    double convex;
    int npoint;
} armsParS;
*/

class armsParmClass
{
public:
    // members
    const unsigned int n;
    const int nsamp;
    const int ninit;
    const int metropolis;
    const bool simple;
    const double convex;
    const int npoint;

    // ranges of key parameters
    const double xiMin;
    const double xiMax;
    const double zetaMin;
    const double zetaMax;
    const double kappaMin;
    const double kappaMax;
    const double betaMin;
    const double betaMax;

    // Constructor to initialize the constants
    armsParmClass(
        unsigned int n_,
        int nsamp_,
        int ninit_,
        int metropolis_,
        bool simple_,
        double convex_,
        int npoint_,

        double xiMin_,
        double xiMax_,
        double zetaMin_,
        double zetaMax_,
        double kappaMin_,
        double kappaMax_,
        double betaMin_,
        double betaMax_
    ) :
        n(n_),
        nsamp(nsamp_),
        ninit(ninit_),
        metropolis(metropolis_),
        simple(simple_),
        convex(convex_),
        npoint(npoint_),

        xiMin(xiMin_),
        xiMax(xiMax_),
        zetaMin(zetaMin_),
        zetaMax(zetaMax_),
        kappaMin(kappaMin_),
        kappaMax(kappaMax_),
        betaMin(betaMin_),
        betaMax(betaMax_)
    {}

} ;

#endif
