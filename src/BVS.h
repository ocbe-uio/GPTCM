/* header file for Bayesian variable selection via Metropolis-Hastings sampler*/

#ifndef BVS_H
#define BVS_H

#include "global.h"

#include <RcppArmadillo.h>

enum class Gamma_Sampler_Type
{
    bandit = 1, mc3
}; // scoped enum

enum class Gamma_Prior_Type
{
    bernoulli = 1, mrf
}; // scoped enum

enum class Eta_Sampler_Type
{
    bandit = 1, mc3
}; // scoped enum

enum class Eta_Prior_Type
{
    bernoulli = 1, mrf
}; // scoped enum

typedef struct HyperparData
{
    // members
    double mrfA;
    double mrfB;
    const unsigned int *mrfG;
    const double *mrfG_weights;
    unsigned int mrfG_edge_n;
    double piA;
    double piB;

    double mrfA_prop;
    double mrfB_prop;
    const unsigned int *mrfG_prop;
    const double *mrfG_prop_weights;
    unsigned int mrfG_prop_edge_n;
    double rhoA;
    double rhoB;

    // double vSq;
    double vA;
    double vB;
    // double v0Sq;
    double v0A;
    double v0B;
    // double tau0Sq;
    double tau0A;
    double tau0B;
    // double tauSq;
    double tauA;
    double tauB;
    // double wSq;
    double wA;
    double wB;
    // double w0Sq;
    double w0A;
    double w0B;
    bool w0IGamma;

    bool kappaIGamma;
    double kappaA;
    double kappaB;
} hyperparS;

class BVS_Sampler
{
public:
    // the following class constructor is not yet used in current version
    BVS_Sampler(
        // const HyperparData& hyperpar,
        const DataClass& dataclass
    ) :
        // hyperpar_(hyperpar),
        dataclass_(dataclass) {}

    // log-density of survival and measurement error data
    static void loglikelihood(
        const arma::vec& xi,
        const arma::mat& zetas,
        const arma::mat& betas,
        double kappa,

        bool proportion_model,
        const DataClass &dataclass,
        arma::vec& loglik//,
        //arma::vec& loglik0 // log-density of survival data
    );

    // log-density of survival data
    static void loglikelihood0(
        const arma::vec& xi,
        const arma::mat& zetas,
        const arma::mat& betas,
        double kappa,

        bool proportion_model,
        const DataClass &dataclass,
        arma::vec& loglik
    );

    static void sampleGamma(
        arma::umat& gammas_,
        Gamma_Prior_Type gamma_prior,
        Gamma_Sampler_Type gamma_sampler,
        arma::mat& logP_gamma_,
        unsigned int& gamma_acc_count_,
        arma::vec& log_likelihood_,

        // int n,
        // int nsamp,
        // int ninit,
        // int metropolis,
        // bool simple,
        // double convex,
        // int npoint,
        const armsParmClass armsPar,
        void *hyperpar_,

        const arma::vec& xi_,
        const arma::mat& zetas_,
        arma::mat& betas_,
        double kappa_,
        double tau0Sq_,
        arma::vec tauSq_,
        double pi0,

        bool proportion_model,

        double& logPosteriorBeta,
        arma::mat& datProportion,
        arma::vec& datTheta,
        arma::mat datMu,
        arma::mat weibullS,
        const DataClass &dataclass
    );

    static void sampleEta(
        arma::umat& etas_,
        Eta_Prior_Type eta_prior,
        Eta_Sampler_Type eta_sampler,
        arma::mat& logP_eta_,
        unsigned int& eta_acc_count_,
        arma::vec& log_likelihood_,

        const armsParmClass armsPar,
        void *hyperpar_,
        arma::mat& zetas_,
        const arma::mat& betas_,
        const arma::vec& xi_,
        double kappa_,
        double w0Sq_,
        arma::vec wSq_,
        double rho0,

        bool dirichlet,

        double& logPosteriorZeta,
        arma::vec& datTheta,
        arma::mat weibullS,
        arma::mat weibullLambda,
        const DataClass &dataclass
    );

    static double logPDFBernoulli(unsigned int x, double pi);
    static double lBeta(double a,double b);
    static double logPDFBeta(double x, double a, double b);

    // static void banditInit(unsigned int p, unsigned int L, unsigned int N);

    // static void banditInitEta(unsigned int p, unsigned int L, unsigned int N);

private:
    // HyperparData hyperpar_;
    DataClass dataclass_;

    static double gammaMC3Proposal(
        arma::umat& mutantGammas,
        const arma::umat gammas_,
        arma::uvec& updateIdx,
        unsigned int componentUpdateIdx_
    );

    static double gammaBanditProposal(
        arma::umat& mutantGammas,
        const arma::umat gammas_,
        arma::uvec& updateIdx,
        unsigned int componentUpdateIdx_,
        arma::mat& banditAlpha
    );

    static double etaBanditProposal(
        arma::umat& mutantEtas,
        const arma::umat etas_,
        arma::uvec& updateIdx,
        unsigned int componentUpdateIdx_,
        arma::mat& banditAlpha2
    );

    static arma::uvec randWeightedIndexSampleWithoutReplacement(
        unsigned int populationSize,
        const arma::vec& weights,
        unsigned int sampleSize
    );

    static arma::uvec randWeightedIndexSampleWithoutReplacement(
        unsigned int populationSize,
        unsigned int sampleSize
    );

    static unsigned int randWeightedIndexSampleWithoutReplacement(
        const arma::vec& weights
    );

    static double logPDFWeightedIndexSampleWithoutReplacement(
        const arma::vec& weights,
        const arma::uvec& indexes
    );

    static double logspace_add(
        double a,
        double b
    );

    static double logPDFNormal(
        const arma::vec& x,
        const double& sigmaSq
    );
};

// function logPDFMRF() is not yet used
// double logPDFMRF(const arma::umat& externalGamma, const arma::mat& mrfG, double a, double b );

/*
// Bandit-sampling related quantities for gammas
static unsigned int n_updates_bandit;
static arma::mat banditZeta;
static arma::mat banditAlpha;
static arma::mat banditBeta;
static arma::vec mismatch;
static arma::vec normalised_mismatch;
static arma::vec normalised_mismatch_backwards;

static double banditLimit;
static double banditIncrement;


// Bandit-sampling related quantities for etas
static unsigned int n_updates_bandit2;
static arma::mat banditZeta2;
static arma::mat banditAlpha2;
static arma::mat banditBeta2;
static arma::vec mismatch2;
static arma::vec normalised_mismatch2;
static arma::vec normalised_mismatch_backwards2;

static double banditLimit2;
static double banditIncrement2;
*/

#endif
