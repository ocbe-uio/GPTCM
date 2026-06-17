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

    double augBetaVar;
    double augZetaVar;

    double vA;
    double vB;
    double v0A;
    double v0B;
    double tau0A;
    double tau0B;
    double tauA;
    double tauB;
    double wA;
    double wB;
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
        const arma::umat& etas,
        const arma::umat& gammas,
        double kappa,

        bool proportion_model,
        const DataClass &dataclass,
        arma::vec& loglik
    );

    static void loglikelihood_noBVS(
        double kappa,
        bool proportion_model,
        arma::mat& alphas,
        arma::mat& updateProportions,
        arma::mat& weibullS,
        arma::mat& weibullLambda,
        arma::mat& logTheta,
        arma::mat& datTheta,
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
        bool CMH,
        const armsParmClass& armsPar,
        void *hyperpar_,
        const arma::vec& xi_,
        const arma::mat& zetas_,
        const arma::umat& etas_,
        arma::mat& betas_,
        double kappa_,
        double tau0Sq_,
        const arma::vec& tauSq_,
        const arma::vec& pi,
        arma::vec& logZ_gamma,
        bool proportion_model,
        arma::mat& datProportion,
        arma::vec& datTheta,
        const arma::mat& datMu,
        const arma::mat& weibullS,
        const DataClass &dataclass
    );

    static double logsumexp(const arma::vec& x);

    static void sampleEta(
        arma::umat& etas_,
        Eta_Prior_Type eta_prior,
        Eta_Sampler_Type eta_sampler,
        arma::mat& logP_eta_,
        unsigned int& eta_acc_count_,
        arma::vec& log_likelihood_,
        bool CMH,
        const armsParmClass& armsPar,
        void *hyperpar_,
        arma::mat& zetas_,
        const arma::mat& betas_,
        const arma::umat& gammas_,
        const arma::vec& xi_,
        double kappa_,
        double w0Sq_,
        arma::vec wSq_,
        const arma::vec& rho,
        arma::vec& logZ_eta,
        bool dirichlet,
        arma::vec& datTheta,
        const arma::mat& weibullS,
        arma::mat& weibullLambda,
        const DataClass &dataclass
    );

    static double logPDFBernoulli(unsigned int x, double pi);

private:
    // HyperparData hyperpar_;
    DataClass dataclass_;

    static double logPbetaK(
        const unsigned int k,
        const arma::mat& betas,
        const arma::umat& gammas,

        const double tauSq,
        const double kappa,
        const arma::vec& datTheta,
        const arma::mat& datProportion,
        const DataClass& dataclass
    );

    static double logPzetaK(
        const unsigned int k,
        const arma::mat& zetas,
        const arma::umat& etas,
        const double wSq,
        const double kappa,

        const arma::vec& datTheta,
        const arma::mat& weibullS,
        const arma::mat& weibullLambda,
        const DataClass& dataclass
    );

    static double logPDFNormal(
        const arma::vec& x, 
        double sigmaSq
    );

    static double logPDFNormal(
        double x, 
        double mean, 
        double var
    );

    static double logSlabPriorNormal(
        double x, 
        double var
    );

    static double logPseudoPriorNormal(
        double x, 
        double var
    );

    static double logAugBetaPriorColumn(
        const arma::vec& beta_col_nonintercept,
        const arma::uvec& gamma_col,
        double slab_var
    );

    static double logAugZetaPriorColumn(
        const arma::vec& zeta_col_nonintercept,
        const arma::uvec& eta_col,
        double slab_var
    );

    static double gammaMC3Proposal(
        unsigned int p,
        arma::umat& mutantGammas,
        const arma::umat gammas_,
        arma::uvec& updateIdx,
        unsigned int componentUpdateIdx_
    );

    static double gammaBanditProposal(
        unsigned int p,
        arma::umat& mutantGammas,
        const arma::umat gammas_,
        arma::uvec& updateIdx,
        unsigned int componentUpdateIdx_,
        arma::mat& banditAlpha
    );

    static double etaBanditProposal(
        unsigned int p,
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

    static arma::vec randMvNormal(
        const arma::vec &m,
        const arma::mat &Sigma
    );

    static arma::vec randVecNormal(const unsigned int n);

    static double logPDFNormal(
        const arma::vec& x,
        const arma::vec& m,
        const arma::mat& Sigma
    );
    
    static arma::uvec setdiff_preserve_order(
        const arma::uvec& A, 
        const arma::uvec& B
    );
};


#endif
