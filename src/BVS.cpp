/* Core updates for Bayesian variable selection */

#include <memory> // Include for smart pointers
#include <algorithm>
#include <unordered_set>
#include <limits>
#include <cmath>

#include "BVS.h"
#include "arms_gibbs.h"

// -----------------------------------------------------------------------------
// Carlin--Chib helpers
// -----------------------------------------------------------------------------
//
// Important modelling note:
//   - The likelihood is evaluated with masked coefficients gamma * beta and
//     eta * zeta.
//   - Inactive coefficients remain in the state. Under full Carlin--Chib they
//     should be drawn from pseudo-priors in the beta/zeta update step.
//   - Because no pseudo-prior hyperparameters are currently passed into BVS.cpp,
//     this file uses the slab prior N(0, tauSq_l) / N(0, wSq_l) also as the
//     default pseudo-prior. With that default the augmented-prior correction
//     cancels, and the gamma/eta update is the masked-likelihood conditional MH
//     special case of Carlin--Chib.
//   - To obtain the main mixing benefit of Carlin--Chib, add pseudo-prior means
//     and variances to hyperparS / BVS.h / drive.cpp, then modify
//     logPseudoPriorNormal() below accordingly.
// -----------------------------------------------------------------------------

namespace {
/*
arma::mat maskBetasByGamma(const arma::mat& betas, const arma::umat& gammas)
{
    arma::mat out = betas;
    const unsigned int L = gammas.n_cols;

    for(unsigned int l = 0; l < L; ++l)
    {
        arma::uvec off_l = arma::find(gammas.col(l) == 0);
        if(!off_l.is_empty())
        {
            for(arma::uword ii = 0; ii < off_l.n_elem; ++ii)
            {
                out(1 + off_l[ii], l) = 0.0;
            }
        }
    }

    return out;
}

arma::mat maskZetasByEta(const arma::mat& zetas, const arma::umat& etas)
{
    arma::mat out = zetas;
    const unsigned int L = etas.n_cols;

    for(unsigned int l = 0; l < L; ++l)
    {
        arma::uvec off_l = arma::find(etas.col(l) == 0);
        if(!off_l.is_empty())
        {
            for(arma::uword ii = 0; ii < off_l.n_elem; ++ii)
            {
                out(1 + off_l[ii], l) = 0.0;
            }
        }
    }

    return out;
}
*/

inline double logNormalScalar(double x, double mean, double var)
{
    const double eps = 1e-12;
    var = std::max(var, eps);
    double z = x - mean;
    return -0.5 * std::log(2.0 * M_PI) - 0.5 * std::log(var) - 0.5 * z * z / var;
}

inline double logSlabPriorNormal(double x, double var)
{
    return logNormalScalar(x, 0.0, var);
}

inline double logPseudoPriorNormal(double x, double var)
{
    // Default pseudo-prior: same as slab prior.
    // Replace this by N(pseudo_mean[j,l], pseudo_var[j,l]) if pseudo-prior
    // hyperparameters are added to hyperparS / BVS.h / drive.cpp.
    return logNormalScalar(x, 0.0, var);
}

double logAugBetaPriorColumn(
    const arma::vec& beta_col_nonintercept,
    const arma::uvec& gamma_col,
    double slab_var)
{
    double out = 0.0;
    const unsigned int p = gamma_col.n_elem;

    for(unsigned int j = 0; j < p; ++j)
    {
        double b = beta_col_nonintercept[j];
        if(gamma_col[j] == 1)
            out += logSlabPriorNormal(b, slab_var);
        else
            out += logPseudoPriorNormal(b, slab_var);
    }

    return out;
}

double logAugZetaPriorColumn(
    const arma::vec& zeta_col_nonintercept,
    const arma::uvec& eta_col,
    double slab_var)
{
    double out = 0.0;
    const unsigned int p = eta_col.n_elem;

    for(unsigned int j = 0; j < p; ++j)
    {
        double z = zeta_col_nonintercept[j];
        if(eta_col[j] == 1)
            out += logSlabPriorNormal(z, slab_var);
        else
            out += logPseudoPriorNormal(z, slab_var);
    }

    return out;
}

} // anonymous namespace

// TODO: loglikelihood can be updated in 'ARMS_Gibbs::logPbetas()' and 'ARMS_Gibbs::logPzetas()',
//          so that it does not need to updated twice in 'BVS_Sampler::sampleGamma()' and 'BVS_Sampler::sampleEta()'.
// log-density for coefficient xis
void BVS_Sampler::loglikelihood(
    const arma::vec& xi,
    const arma::mat& zetas,
    const arma::mat& betas,
    const arma::umat& etas,
    const arma::umat& gammas,
    double kappa,

    bool proportion_model,
    const DataClass &dataclass,
    arma::vec& loglik)
{
    // dimensions
    unsigned int N = dataclass.datX.n_rows;
    unsigned int p = dataclass.datX.n_cols;
    unsigned int L = dataclass.datX.n_slices;

    arma::mat updateProportions = dataclass.datProportionConst;
    arma::mat alphas = arma::zeros<arma::mat>(N, L);
    arma::vec alphas_Rowsum;
    if(proportion_model)
    {
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif

        for(unsigned int l=0; l<L; ++l)
        {
            arma::vec zetaMask_l = zetas.submat(1, l, p, l);
            zetaMask_l.elem(arma::find(etas.col(l) == 0)).fill(0.0);
            alphas.col(l) = arma::exp( zetas(0, l) + dataclass.datX.slice(l) * zetaMask_l );
        }
        alphas.elem(arma::find(alphas > upperbound3)).fill(upperbound3);
        alphas.elem(arma::find(alphas < lowerbound)).fill(lowerbound);
        alphas_Rowsum = arma::sum(alphas, 1);
        updateProportions = alphas / arma::repmat(alphas_Rowsum, 1, L);
    }

    arma::vec logTheta = dataclass.datX0 * xi;
    logTheta.elem(arma::find(logTheta > upperbound)).fill(upperbound);
    arma::vec thetas = arma::exp( logTheta );

    arma::vec f = arma::zeros<arma::vec>(N);
    arma::vec survival_pop = arma::zeros<arma::vec>(N);

    for(unsigned int l=0; l<L; ++l)
    {
        arma::vec betaMask_l = betas.submat(1, l, p, l);
        betaMask_l.elem(arma::find(gammas.col(l) == 0)).fill(0.0);
        arma::vec logMu_l = betas(0, l) + dataclass.datX.slice(l) * betaMask_l;
        logMu_l.elem(arma::find(logMu_l > upperbound)).fill(upperbound);
        arma::vec mu_l = arma::exp(logMu_l);

        arma::vec weibull_lambdas_l = mu_l / std::tgamma(1. + 1./kappa);
        arma::vec weibullS_l = arma::exp( - arma::pow( dataclass.datTime / weibull_lambdas_l, kappa) );
        arma::vec weibull_pdf = arma::exp(-kappa * arma::log(weibull_lambdas_l) - arma::pow(dataclass.datTime/weibull_lambdas_l, kappa));

        survival_pop += updateProportions.col(l) % weibullS_l;

        f += kappa * arma::pow(dataclass.datTime, kappa - 1.0) % updateProportions.col(l) % weibull_pdf;
    }

    // summarize density of the Weibull's survival part
    arma::vec log_survival_pop = - thetas % (1. - survival_pop);
    f.elem(arma::find(f < lowerbound)).fill(lowerbound);
    arma::vec log_f_pop = logTheta + arma::log(f) + log_survival_pop;

    // summarize density of the Dirichlet part
    arma::vec log_dirichlet = arma::zeros<arma::vec>(N);
    if (proportion_model)
    {
        log_dirichlet =
            arma::lgamma(alphas_Rowsum) - arma::sum(arma::lgamma(alphas), 1) +
            arma::sum( (alphas - 1.0) % arma::log(dataclass.datProportionConst), 1 );
    }

    log_f_pop.elem(dataclass.eventIndex).fill(0.);
    log_survival_pop.elem(dataclass.eventIndex).fill(0.);
    loglik = log_f_pop + log_survival_pop + log_dirichlet;
}

// loglikelihood for 'BVS = FALSE'
void BVS_Sampler::loglikelihood_noBVS(
    const arma::vec& xi,
    const arma::mat& zetas,
    const arma::mat& betas,
    double kappa,

    bool proportion_model,
    arma::mat& alphas,
    arma::mat& updateProportions,
    arma::mat& weibullS,
    arma::mat& weibullLambda,
    const DataClass &dataclass,
    arma::vec& loglik)
{
    // dimensions
    unsigned int N = dataclass.datX.n_rows;
    // unsigned int p = dataclass.datX.n_cols;
    unsigned int L = dataclass.datX.n_slices;

    // arma::mat updateProportions = dataclass.datProportionConst;
    // arma::mat alphas = arma::zeros<arma::mat>(N, L);
    // arma::vec alphas_Rowsum;
    // if(proportion_model)
    // {
    //     #ifdef _OPENMP
    //     #pragma omp parallel for
    //     #endif

    //     for(unsigned int l=0; l<L; ++l)
    //     {
    //         arma::vec zetaMask_l = zetas.submat(1, l, p, l);
    //         alphas.col(l) = arma::exp( zetas(0, l) + dataclass.datX.slice(l) * zetaMask_l );
    //     }
    //     alphas.elem(arma::find(alphas > upperbound3)).fill(upperbound3);
    //     alphas.elem(arma::find(alphas < lowerbound)).fill(lowerbound);
    //     alphas_Rowsum = arma::sum(alphas, 1);
    //     updateProportions = alphas / arma::repmat(alphas_Rowsum, 1, L);
    // }

    arma::vec logTheta = dataclass.datX0 * xi;
    logTheta.elem(arma::find(logTheta > upperbound)).fill(upperbound);
    arma::vec thetas = arma::exp( logTheta );

    arma::vec f = arma::zeros<arma::vec>(N);
    arma::vec survival_pop = arma::zeros<arma::vec>(N);

    for(unsigned int l=0; l<L; ++l)
    {
        // arma::vec betaMask_l = betas.submat(1, l, p, l);
        // arma::vec logMu_l = betas(0, l) + dataclass.datX.slice(l) * betaMask_l;
        // logMu_l.elem(arma::find(logMu_l > upperbound)).fill(upperbound);
        // arma::vec mu_l = arma::exp(logMu_l);

        // arma::vec weibull_lambdas_l = mu_l / std::tgamma(1. + 1./kappa);
        // arma::vec weibullS_l = arma::exp( - arma::pow( dataclass.datTime / weibull_lambdas_l, kappa) );
        // arma::vec weibull_pdf = arma::exp(-kappa * arma::log(weibull_lambdas_l) - arma::pow(dataclass.datTime/weibull_lambdas_l, kappa));
        arma::vec weibull_pdf = arma::exp(-kappa * arma::log(weibullLambda.col(l)) - arma::pow(dataclass.datTime/weibullLambda.col(l), kappa));

        survival_pop += updateProportions.col(l) % weibullS.col(l);

        f += kappa * arma::pow(dataclass.datTime, kappa - 1.0) % updateProportions.col(l) % weibull_pdf;
    }

    // summarize density of the Weibull's survival part
    arma::vec log_survival_pop = - thetas % (1. - survival_pop);
    f.elem(arma::find(f < lowerbound)).fill(lowerbound);
    arma::vec log_f_pop = logTheta + arma::log(f) + log_survival_pop;

    // summarize density of the Dirichlet part
    arma::vec log_dirichlet = arma::zeros<arma::vec>(N);
    if (proportion_model)
    {
        arma::vec alphas_Rowsum = arma::sum(alphas, 1);
        log_dirichlet =
            arma::lgamma(alphas_Rowsum) - arma::sum(arma::lgamma(alphas), 1) +
            arma::sum( (alphas - 1.0) % arma::log(dataclass.datProportionConst), 1 );
    }

    log_f_pop.elem(dataclass.eventIndex).fill(0.);
    log_survival_pop.elem(dataclass.eventIndex).fill(0.);
    loglik = log_f_pop + log_survival_pop + log_dirichlet;
}


void BVS_Sampler::sampleGamma(
    arma::umat& gammas_,
    Gamma_Prior_Type gamma_prior,
    Gamma_Sampler_Type gamma_sampler,
    arma::mat& logP_gamma_,
    unsigned int& gamma_acc_count_,
    arma::vec& log_likelihood_,

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
    arma::vec& logZ_gamma_,

    bool proportion_model,

    // double& logPosteriorBeta,
    arma::mat& datProportion,
    arma::vec& datTheta,
    const arma::mat& datMu,
    const arma::mat& weibullS,
    const DataClass &dataclass)
{
    (void)armsPar;
    (void)tau0Sq_;
    (void)datProportion;
    (void)datTheta;
    (void)datMu;
    (void)weibullS;
    (void)logZ_gamma_; // kept in signature for compatibility with the previous AIS version

    std::unique_ptr<hyperparS> hyperpar = std::make_unique<hyperparS>();
    *hyperpar = *(hyperparS *)hyperpar_;

    arma::umat proposedGamma = gammas_;
    arma::mat proposedGammaPrior;
    arma::uvec updateIdx;

    double logProposalRatio = 0;

    unsigned int p = gammas_.n_rows;
    unsigned int L = gammas_.n_cols;

    static arma::mat banditAlpha = arma::mat(p, L, arma::fill::value(0.5));
    static arma::mat banditBeta = arma::mat(p, L, arma::fill::value(0.5));

    unsigned int componentUpdateIdx = static_cast<unsigned int>( R::runif( 0, L ) );
    arma::uvec singleIdx_k = { componentUpdateIdx };

    switch( gamma_sampler )
    {
    case Gamma_Sampler_Type::bandit:
        logProposalRatio += gammaBanditProposal( p, proposedGamma, gammas_, updateIdx, componentUpdateIdx, banditAlpha );
        break;

    case Gamma_Sampler_Type::mc3:
        logProposalRatio += gammaMC3Proposal( p, proposedGamma, gammas_, updateIdx, componentUpdateIdx );
        break;
    }

    double logPriorGammaRatio = 0.;

    switch(gamma_prior)
    {
    case Gamma_Prior_Type::bernoulli:
    {
        proposedGammaPrior = logP_gamma_;

        // pi is a latent MCMC state updated in drive.cpp.
        // Conditional prior ratio: log p(gamma_new | pi_l) - log p(gamma_old | pi_l)
        double pi_l = pi[componentUpdateIdx];

        for(auto i: updateIdx)
        {
            double logp_new = logPDFBernoulli(proposedGamma(i, componentUpdateIdx), pi_l);
            double logp_old = logPDFBernoulli(gammas_(i, componentUpdateIdx), pi_l);

            proposedGammaPrior(i, componentUpdateIdx) = logp_new;
            logPriorGammaRatio += logp_new - logp_old;
        }

        break;
    }

    case Gamma_Prior_Type::mrf:
    {
        arma::umat mrfG(const_cast<unsigned int*>(hyperpar->mrfG), hyperpar->mrfG_edge_n, 2, false);
        arma::vec mrfG_weights(const_cast<double*>(hyperpar->mrfG_weights), hyperpar->mrfG_edge_n, false);

        logPriorGammaRatio += hyperpar->mrfA * (
            (double)(arma::accu(proposedGamma.submat(updateIdx, singleIdx_k))) -
            (double)(arma::accu(gammas_.submat(updateIdx, singleIdx_k)))
        );

        arma::uvec updateIdxGlobal = updateIdx + p * componentUpdateIdx;
        arma::uvec updateIdxMRF_common = arma::intersect(updateIdxGlobal, mrfG);

        if((updateIdxMRF_common.n_elem > 0) && (hyperpar->mrfB > 0))
        {
            #ifdef _OPENMP
            #pragma omp parallel for default(shared) reduction(+:logPriorGammaRatio)
            #endif

            for(unsigned int i=0; i<hyperpar->mrfG_edge_n; ++i)
            {
                if( mrfG(i, 0) != mrfG(i, 1))
                {
                    logPriorGammaRatio += hyperpar->mrfB * 2.0 * mrfG_weights(i) *
                                             ((double)(proposedGamma(mrfG(i, 0)) * proposedGamma(mrfG(i, 1))) -
                                              (double)(gammas_(mrfG(i, 0)) * gammas_(mrfG(i, 1))));
                }
                else
                {
                    logPriorGammaRatio += hyperpar->mrfB * mrfG_weights(i) *
                                             ((double)(proposedGamma(mrfG(i, 0))) -
                                              (double)(gammas_(mrfG(i, 0))));
                }
            }
        }
        break;
    }
    }

    // -------------------------------------------------------------------------
    // Carlin--Chib / masked-likelihood step for gamma.
    // The full beta vector is unchanged, but the effective likelihood uses
    // gamma * beta. The augmented-prior ratio includes slab prior for active
    // coefficients and pseudo-prior for inactive coefficients.
    // -------------------------------------------------------------------------
    // arma::mat beta_current_masked  = maskBetasByGamma(betas_, gammas_);
    // arma::mat beta_proposed_masked = maskBetasByGamma(betas_, proposedGamma);

    arma::vec currentLikelihood;
    arma::vec proposedLikelihood;

    loglikelihood(
        xi_,
        zetas_,
        betas_,
        etas_,
        gammas_,
        kappa_,
        proportion_model,
        dataclass,
        currentLikelihood
    );

    loglikelihood(
        xi_,
        zetas_,
        betas_,
        etas_,
        proposedGamma,
        kappa_,
        proportion_model,
        dataclass,
        proposedLikelihood
    );

    double logLikelihoodRatio = arma::accu(proposedLikelihood) - arma::accu(currentLikelihood);

    double logAugBetaCurrent = logAugBetaPriorColumn(
        betas_.submat(1, componentUpdateIdx, p, componentUpdateIdx),
        gammas_.col(componentUpdateIdx),
        tauSq_[componentUpdateIdx]
    );

    double logAugBetaProposed = logAugBetaPriorColumn(
        betas_.submat(1, componentUpdateIdx, p, componentUpdateIdx),
        proposedGamma.col(componentUpdateIdx),
        tauSq_[componentUpdateIdx]
    );

    double logAugBetaPriorRatio = logAugBetaProposed - logAugBetaCurrent;

    double logAccProb = logLikelihoodRatio +
                        logPriorGammaRatio +
                        logAugBetaPriorRatio +
                        logProposalRatio;

    if( std::log(R::runif(0,1)) < logAccProb )
    {
        gammas_ = proposedGamma;

        if( gamma_prior == Gamma_Prior_Type::bernoulli )
        {
            logP_gamma_ = proposedGammaPrior;
        }

        log_likelihood_ = proposedLikelihood;
        ++gamma_acc_count_;
    }
    else
    {
        log_likelihood_ = currentLikelihood;
    }

    // Do NOT zero or overwrite betas_ here. Inactive coefficients are auxiliary
    // Carlin--Chib variables. For a full CC implementation, inactive betas should
    // be sampled from pseudo-priors in the beta update step.

    if( gamma_sampler == Gamma_Sampler_Type::bandit )
    {
        double banditLimit = (double)(dataclass.datTime.n_elem);
        double banditIncrement = 1.;

        for(auto iter: updateIdx)
        {
            if( banditAlpha(iter,componentUpdateIdx) + banditBeta(iter,componentUpdateIdx) <= banditLimit )
            {
                banditAlpha(iter,componentUpdateIdx) += banditIncrement * gammas_(iter,componentUpdateIdx);
                banditBeta(iter,componentUpdateIdx) += banditIncrement * (1-gammas_(iter,componentUpdateIdx));
            }
        }
    }
}


// Returns log(sum_i exp(x[i])) in a numerically stable way
double BVS_Sampler::logsumexp(const arma::vec& x) {
    if (x.is_empty()) return -std::numeric_limits<double>::infinity();
    double m = x.max();
    if (!std::isfinite(m)) return m;
    double s = 0.0;
    for (double xi : x) s += std::exp(xi - m);
    return m + std::log(s);
}


void BVS_Sampler::sampleEta(
    arma::umat& etas_,
    Eta_Prior_Type eta_prior,
    Eta_Sampler_Type eta_sampler,
    arma::mat& logP_eta_,
    unsigned int& eta_acc_count_,
    arma::vec& log_likelihood_,

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
    arma::vec& logZ_eta_,

    bool dirichlet,

    // double& logPosteriorZeta,
    arma::vec& datTheta,
    const arma::mat& weibullS,
    arma::mat& weibullLambda,
    const DataClass &dataclass)
{
    (void)armsPar;
    (void)w0Sq_;
    (void)datTheta;
    (void)weibullS;
    (void)weibullLambda;
    (void)logZ_eta_; // kept in signature for compatibility with the previous AIS version

    std::unique_ptr<hyperparS> hyperpar = std::make_unique<hyperparS>();
    *hyperpar = *(hyperparS *)hyperpar_;

    arma::umat proposedEta = etas_;
    arma::mat proposedEtaPrior;
    arma::uvec updateIdx;

    double logProposalRatio = 0;

    unsigned int p = etas_.n_rows;
    unsigned int L = etas_.n_cols;

    static arma::mat banditAlpha2 = arma::mat(p, L, arma::fill::value(0.5));
    static arma::mat banditBeta2 = arma::mat(p, L, arma::fill::value(0.5));

    unsigned int componentUpdateIdx = static_cast<unsigned int>( R::runif( 0, L ) );
    arma::uvec singleIdx_k = { componentUpdateIdx };

    switch( eta_sampler )
    {
    case Eta_Sampler_Type::bandit:
        logProposalRatio += etaBanditProposal( p, proposedEta, etas_, updateIdx, componentUpdateIdx, banditAlpha2 );
        break;

    case Eta_Sampler_Type::mc3:
        logProposalRatio += gammaMC3Proposal( p, proposedEta, etas_, updateIdx, componentUpdateIdx );
        break;
    }

    double logPriorEtaRatio = 0.;

    switch(eta_prior)
    {
    case Eta_Prior_Type::bernoulli:
    {
        proposedEtaPrior = logP_eta_;

        // rho is a latent MCMC state updated in drive.cpp.
        // Conditional prior ratio: log p(eta_new | rho_l) - log p(eta_old | rho_l)
        double rho_l = rho[componentUpdateIdx];

        for(auto i: updateIdx)
        {
            double logp_new = logPDFBernoulli(proposedEta(i, componentUpdateIdx), rho_l);
            double logp_old = logPDFBernoulli(etas_(i, componentUpdateIdx), rho_l);

            proposedEtaPrior(i, componentUpdateIdx) = logp_new;
            logPriorEtaRatio += logp_new - logp_old;
        }

        break;
    }

    case Eta_Prior_Type::mrf:
    {
        arma::umat mrfG(const_cast<unsigned int*>(hyperpar->mrfG_prop), hyperpar->mrfG_prop_edge_n, 2, false);
        arma::vec mrfG_weights(const_cast<double*>(hyperpar->mrfG_prop_weights), hyperpar->mrfG_prop_edge_n, false);

        logPriorEtaRatio = hyperpar->mrfA_prop * (
            (double)(arma::accu(proposedEta.submat(updateIdx, singleIdx_k))) -
            (double)(arma::accu(etas_.submat(updateIdx, singleIdx_k)))
        );

        arma::uvec updateIdxGlobal = updateIdx + p * componentUpdateIdx;
        arma::uvec updateIdxMRF_common = arma::intersect(updateIdxGlobal, mrfG);

        if((updateIdxMRF_common.n_elem > 0) && (hyperpar->mrfB_prop > 0))
        {
            #ifdef _OPENMP
            #pragma omp parallel for default(shared) reduction(+:logPriorEtaRatio)
            #endif

            for(unsigned int i=0; i<hyperpar->mrfG_prop_edge_n; ++i)
            {
                if( mrfG(i, 0) != mrfG(i, 1))
                {
                    logPriorEtaRatio += hyperpar->mrfB_prop * 2.0 * mrfG_weights(i) *
                                           ((double)(proposedEta(mrfG(i, 0)) * proposedEta(mrfG(i, 1))) -
                                            (double)(etas_(mrfG(i, 0)) * etas_(mrfG(i, 1))));
                }
                else
                {
                    logPriorEtaRatio += hyperpar->mrfB_prop * mrfG_weights(i) *
                                           ((double)(proposedEta(mrfG(i, 0))) -
                                            (double)(etas_(mrfG(i, 0))));
                }
            }
        }
        break;
    }
    }

    // -------------------------------------------------------------------------
    // Carlin--Chib / masked-likelihood step for eta.
    // The full zeta vector is unchanged, but the effective likelihood uses
    // eta * zeta. The augmented-prior ratio includes slab prior for active
    // coefficients and pseudo-prior for inactive coefficients.
    // -------------------------------------------------------------------------
    // arma::mat zeta_current_masked  = maskZetasByEta(zetas_, etas_);
    // arma::mat zeta_proposed_masked = maskZetasByEta(zetas_, proposedEta);

    arma::vec currentLikelihood;
    arma::vec proposedLikelihood;

    loglikelihood(
        xi_,
        zetas_,
        betas_,
        etas_,
        gammas_,
        kappa_,
        dirichlet,
        dataclass,
        currentLikelihood
    );

    loglikelihood(
        xi_,
        zetas_,
        betas_,
        proposedEta,
        gammas_,
        kappa_,
        dirichlet,
        dataclass,
        proposedLikelihood
    );

    double logLikelihoodRatio = arma::accu(proposedLikelihood) - arma::accu(currentLikelihood);

    double logAugZetaCurrent = logAugZetaPriorColumn(
        zetas_.submat(1, componentUpdateIdx, p, componentUpdateIdx),
        etas_.col(componentUpdateIdx),
        wSq_[componentUpdateIdx]
    );

    double logAugZetaProposed = logAugZetaPriorColumn(
        zetas_.submat(1, componentUpdateIdx, p, componentUpdateIdx),
        proposedEta.col(componentUpdateIdx),
        wSq_[componentUpdateIdx]
    );

    double logAugZetaPriorRatio = logAugZetaProposed - logAugZetaCurrent;

    double logAccProb = logLikelihoodRatio +
                        logPriorEtaRatio +
                        logAugZetaPriorRatio +
                        logProposalRatio;

    if( std::log(R::runif(0,1)) < logAccProb )
    {
        etas_ = proposedEta;

        if( eta_prior == Eta_Prior_Type::bernoulli )
        {
            logP_eta_ = proposedEtaPrior;
        }

        log_likelihood_ = proposedLikelihood;
        ++eta_acc_count_;
    }
    else
    {
        log_likelihood_ = currentLikelihood;
    }

    // Do NOT zero or overwrite zetas_ here. Inactive coefficients are auxiliary
    // Carlin--Chib variables. For a full CC implementation, inactive zetas should
    // be sampled from pseudo-priors in the zeta update step.

    if( eta_sampler == Eta_Sampler_Type::bandit )
    {
        double banditLimit = (double)(dataclass.datTime.n_elem);
        double banditIncrement = 1.;

        for(auto iter: updateIdx)
        {
            if( banditAlpha2(iter,componentUpdateIdx) + banditBeta2(iter,componentUpdateIdx) <= banditLimit )
            {
                banditAlpha2(iter,componentUpdateIdx) += banditIncrement * etas_(iter,componentUpdateIdx);
                banditBeta2(iter,componentUpdateIdx) += banditIncrement * (1-etas_(iter,componentUpdateIdx));
            }
        }
    }
}


double BVS_Sampler::gammaMC3Proposal(
    unsigned int p,
    arma::umat& mutantGamma,
    const arma::umat gammas_,
    arma::uvec& updateIdx,
    unsigned int componentUpdateIdx_ )
{
    unsigned int n_updates_MC3 = std::max(5., std::ceil( (double)(p) / 5. ));
    Rcpp::IntegerVector entireIdx = Rcpp::seq( 0, p - 1);
    updateIdx = Rcpp::as<arma::uvec>(Rcpp::sample(entireIdx, n_updates_MC3, false));

    for( auto i : updateIdx)
    {
        mutantGamma(i,componentUpdateIdx_) =
            ( R::runif(0,1) < 0.5 ) ? gammas_(i,componentUpdateIdx_) : 1-gammas_(i,componentUpdateIdx_);
    }

    return 0.;
}


double BVS_Sampler::gammaBanditProposal(
    unsigned int p,
    arma::umat& mutantGamma,
    const arma::umat gammas_,
    arma::uvec& updateIdx,
    unsigned int componentUpdateIdx_,
    arma::mat& banditAlpha )
{
    static arma::vec banditZeta = arma::vec(p);
    static arma::vec mismatch = arma::vec(p);
    static arma::vec normalised_mismatch = arma::vec(p);
    static arma::vec normalised_mismatch_backwards = arma::vec(p);

    unsigned int n_updates_bandit = 4;
    double logProposalRatio = 0.;

    for(unsigned int j=0; j<p; ++j)
    {
        banditZeta(j) = R::rbeta(banditAlpha(j,componentUpdateIdx_), banditAlpha(j,componentUpdateIdx_));
        mismatch(j) = (mutantGamma(j,componentUpdateIdx_)==0) ? banditZeta(j) : (1.-banditZeta(j));
    }

    normalised_mismatch = mismatch / arma::sum(mismatch);

    if( R::runif(0,1) < 0.5 )
    {
        updateIdx = arma::zeros<arma::uvec>(1);
        updateIdx(0) = randWeightedIndexSampleWithoutReplacement(normalised_mismatch);

        mutantGamma(updateIdx(0),componentUpdateIdx_) = 1 - gammas_(updateIdx(0),componentUpdateIdx_);

        normalised_mismatch_backwards = mismatch;
        normalised_mismatch_backwards(updateIdx(0)) = 1. - normalised_mismatch_backwards(updateIdx(0));
        normalised_mismatch_backwards = normalised_mismatch_backwards / arma::sum(normalised_mismatch_backwards);

        logProposalRatio =
            std::log( normalised_mismatch_backwards(updateIdx(0)) ) -
            std::log( normalised_mismatch(updateIdx(0)) );
    }
    else
    {
        updateIdx = arma::zeros<arma::uvec>(n_updates_bandit);
        updateIdx = randWeightedIndexSampleWithoutReplacement(p, normalised_mismatch, n_updates_bandit);

        normalised_mismatch_backwards = mismatch;

        for(unsigned int i=0; i<n_updates_bandit; ++i)
        {
            unsigned int j = R::rbinom( 1, banditZeta(updateIdx(i)));
            mutantGamma(updateIdx(i),componentUpdateIdx_) = j;

            normalised_mismatch_backwards(updateIdx(i)) = 1.- normalised_mismatch_backwards(updateIdx(i));

            logProposalRatio +=
                logPDFBernoulli(gammas_(updateIdx(i),componentUpdateIdx_), banditZeta(updateIdx(i))) -
                logPDFBernoulli(mutantGamma(updateIdx(i),componentUpdateIdx_), banditZeta(updateIdx(i)));
        }

        normalised_mismatch_backwards = normalised_mismatch_backwards / arma::sum(normalised_mismatch_backwards);

        logProposalRatio +=
            logPDFWeightedIndexSampleWithoutReplacement(normalised_mismatch_backwards, updateIdx) -
            logPDFWeightedIndexSampleWithoutReplacement(normalised_mismatch, updateIdx);
    }

    return logProposalRatio;
}


double BVS_Sampler::etaBanditProposal(
    unsigned int p,
    arma::umat& mutantEta,
    const arma::umat etas_,
    arma::uvec& updateIdx,
    unsigned int componentUpdateIdx_,
    arma::mat& banditAlpha2)
{
    static arma::vec banditZeta2 = arma::vec(p);
    static arma::vec mismatch2 = arma::vec(p);
    static arma::vec normalised_mismatch2 = arma::vec(p);
    static arma::vec normalised_mismatch_backwards2 = arma::vec(p);

    unsigned int n_updates_bandit = 4;
    double logProposalRatio = 0.;

    for(unsigned int j=0; j<p; ++j)
    {
        banditZeta2(j) = R::rbeta(banditAlpha2(j,componentUpdateIdx_), banditAlpha2(j,componentUpdateIdx_));
        mismatch2(j) = (mutantEta(j,componentUpdateIdx_)==0) ? banditZeta2(j) : (1.-banditZeta2(j));
    }

    normalised_mismatch2 = mismatch2 / arma::sum(mismatch2);

    if( R::runif(0,1) < 0.5 )
    {
        updateIdx = arma::zeros<arma::uvec>(1);
        updateIdx(0) = randWeightedIndexSampleWithoutReplacement(normalised_mismatch2);

        mutantEta(updateIdx(0),componentUpdateIdx_) = 1 - etas_(updateIdx(0),componentUpdateIdx_);

        normalised_mismatch_backwards2 = mismatch2;
        normalised_mismatch_backwards2(updateIdx(0)) = 1. - normalised_mismatch_backwards2(updateIdx(0));
        normalised_mismatch_backwards2 = normalised_mismatch_backwards2 / arma::sum(normalised_mismatch_backwards2);

        logProposalRatio =
            std::log( normalised_mismatch_backwards2(updateIdx(0)) ) -
            std::log( normalised_mismatch2(updateIdx(0)) );
    }
    else
    {
        updateIdx = arma::zeros<arma::uvec>(n_updates_bandit);
        updateIdx = randWeightedIndexSampleWithoutReplacement(p, normalised_mismatch2, n_updates_bandit);

        normalised_mismatch_backwards2 = mismatch2;

        for(unsigned int i=0; i<n_updates_bandit; ++i)
        {
            unsigned int j = R::rbinom( 1, banditZeta2(updateIdx(i)));
            mutantEta(updateIdx(i),componentUpdateIdx_) = j;

            normalised_mismatch_backwards2(updateIdx(i)) = 1.- normalised_mismatch_backwards2(updateIdx(i));

            logProposalRatio +=
                logPDFBernoulli(etas_(updateIdx(i),componentUpdateIdx_), banditZeta2(updateIdx(i))) -
                logPDFBernoulli(mutantEta(updateIdx(i),componentUpdateIdx_), banditZeta2(updateIdx(i)));
        }

        normalised_mismatch_backwards2 = normalised_mismatch_backwards2 / arma::sum(normalised_mismatch_backwards2);

        logProposalRatio +=
            logPDFWeightedIndexSampleWithoutReplacement(normalised_mismatch_backwards2, updateIdx) -
            logPDFWeightedIndexSampleWithoutReplacement(normalised_mismatch2, updateIdx);
    }

    return logProposalRatio;
}


double BVS_Sampler::logPbetaK(
    const unsigned int k,
    const arma::mat& betas,
    const arma::umat& gammas,

    const double tauSq,
    const double kappa,
    const arma::vec& datTheta,
    const arma::mat& datProportion,
    const DataClass& dataclass)
{
    double logP = 0.;

    unsigned int N = dataclass.datX.n_rows;
    unsigned int p = dataclass.datX.n_cols;
    unsigned int L = dataclass.datX.n_slices;

    double logprior = - arma::accu(betas.submat(1, k, p, k) % betas.submat(1, k, p, k)) / tauSq / 2.;

    arma::vec logpost_first = arma::zeros<arma::vec>(N);
    double logpost_second_sum = 0;
    for(unsigned int l=0; l<L; ++l)
    {
        arma::vec betaMask_l = betas.submat(1, l, p, l);
        betaMask_l.elem(arma::find(gammas.col(l) == 0)).fill(0.0);
        arma::vec logMu_l = betas(0,l) + dataclass.datX.slice(l) * betaMask_l;
        logMu_l.elem(arma::find(logMu_l > upperbound)).fill(upperbound);
        arma::vec weibull_lambdas_tmp = arma::exp(logMu_l) / std::tgamma(1. + 1./kappa);
        arma::vec tmp = datProportion.col(l) %
                        arma::exp( - arma::pow( dataclass.datTime / weibull_lambdas_tmp, kappa) );
        logpost_first += tmp % (kappa / weibull_lambdas_tmp) %
                         arma::pow(dataclass.datTime/weibull_lambdas_tmp, kappa - 1.0);

        if(l == k)
            logpost_second_sum = arma::sum(datTheta % tmp);
    }

    double logpost_first_sum = arma::sum( arma::log( logpost_first.elem(dataclass.eventIndex) ) );

    logP = logpost_first_sum + logpost_second_sum + logprior;

    return logP;
}


double BVS_Sampler::logPzetaK(
    const unsigned int k,
    const arma::mat& zetas,
    const arma::umat& etas,
    const double wSq,
    const double kappa,

    const arma::vec& datTheta,
    const arma::mat& weibullS,
    const arma::mat& weibullLambda,
    const DataClass &dataclass)
{
    double logP = 0.;

    unsigned int N = dataclass.datX.n_rows;
    unsigned int p = dataclass.datX.n_cols;
    unsigned int L = dataclass.datX.n_slices;

    arma::mat alphas = arma::zeros<arma::mat>(N, L);

    for(unsigned int l=0; l<L; ++l) 
    {
        arma::vec zetaMask_l = zetas.submat(1, l, p, l);
        zetaMask_l.elem(arma::find(etas.col(l) == 0)).fill(0.0);
        alphas.col(l) = arma::exp( zetas(0, l) + dataclass.datX.slice(l) * zetaMask_l );
    }
    alphas.elem(arma::find(alphas > upperbound3)).fill(upperbound3);
    alphas.elem(arma::find(alphas < lowerbound)).fill(lowerbound);
    arma::vec alphas_Rowsum = arma::sum(alphas, 1);

    double logprior = - (arma::accu(zetas.submat(1, k, p, k) % zetas.submat(1, k, p, k))) / wSq / 2.;

    arma::vec logpost_first = arma::zeros<arma::vec>(N);
    arma::vec logpost_second= arma::zeros<arma::vec>(N);

    for(unsigned int l=0; l<L; ++l)
    {
        arma::vec tmp = alphas.col(l) / alphas_Rowsum %  weibullS.col(l);
        logpost_first += arma::pow(weibullLambda.col(l), - kappa) % tmp;
        logpost_second += tmp;
    }

    double logpost_first_sum = arma::sum( arma::log( logpost_first.elem(dataclass.eventIndex) ) );

    double logpost_second_sum = arma::sum(datTheta % logpost_second);

    double log_dirichlet_sum = arma::sum(
        arma::lgamma(alphas_Rowsum) - arma::sum(arma::lgamma(alphas), 1) +
        arma::sum( (alphas - 1.0) % arma::log(dataclass.datProportionConst), 1 )
    );

    logP = logprior + logpost_first_sum + logpost_second_sum + log_dirichlet_sum;

    return logP;
}


// subfunctions used for bandit proposal

arma::uvec BVS_Sampler::randWeightedIndexSampleWithoutReplacement(
    unsigned int populationSize,
    const arma::vec& weights,
    unsigned int sampleSize
)
{
    arma::vec tmp = Rcpp::rexp( populationSize, 1. );
    arma::vec score = tmp - weights;
    arma::uvec result = arma::sort_index( score,"ascend" );

    return result.subvec(0,sampleSize-1);
}


unsigned int BVS_Sampler::randWeightedIndexSampleWithoutReplacement(
    const arma::vec& weights
)
{
    double u = R::runif(0,1);
    double tmp = weights(0);
    unsigned int t = 0;

    while(u > tmp)
    {
        tmp += weights(++t);
    }

    return t;
}


double BVS_Sampler::logPDFWeightedIndexSampleWithoutReplacement(
    const arma::vec& weights,
    const arma::uvec& indexes
)
{
    double logP_permutation = -std::numeric_limits<double>::infinity();
    double tmp;

    std::vector<unsigned int> v = arma::conv_to<std::vector<unsigned int>>::from(arma::sort(indexes));

    arma::uvec current_permutation;
    arma::vec current_weights;

    do
    {
        current_permutation = arma::conv_to<arma::uvec>::from(v);
        current_weights = weights;
        tmp = 0.;

        while( current_permutation.n_elem > 0 )
        {
            tmp += log(current_weights(current_permutation(0)));
            current_permutation.shed_row(0);
            if(current_permutation.n_elem > 0)
                current_weights = current_weights/arma::sum(current_weights(current_permutation));
        }

        logP_permutation = logspace_add( logP_permutation,tmp );

    }
    while (std::next_permutation(v.begin(), v.end()));

    return logP_permutation;
}


double BVS_Sampler::logspace_add(
    double a,
    double b)
{
    if(a <= std::numeric_limits<double>::lowest())
        return b;
    if(b <= std::numeric_limits<double>::lowest())
        return a;

    return std::max(a, b) + std::log( (double)(1. + std::exp( (double)-std::abs((double)(a - b)) )));
}


double BVS_Sampler::logPDFBernoulli(unsigned int x, double pi)
{
    if( x > 1 )
        return -std::numeric_limits<double>::infinity();

    const double eps = 1e-12;
    pi = std::min(std::max(pi, eps), 1.0 - eps);

    return static_cast<double>(x) * std::log(pi) +
           (1.0 - static_cast<double>(x)) * std::log(1.0 - pi);
}


double BVS_Sampler::logPDFNormal(
    const arma::vec& x,
    double sigmaSq)
{
    unsigned int k = x.n_elem;
    double tmp = (double)k * std::log(sigmaSq);

    return -0.5*(double)k*log(2.*M_PI) -0.5*tmp - 0.5 * arma::as_scalar( x.t() * x ) / sigmaSq;
}


arma::vec BVS_Sampler::randMvNormal(
    const arma::vec &m,
    const arma::mat &Sigma)
{
    unsigned int d = m.n_elem;

    if(Sigma.n_rows != d || Sigma.n_cols != d )
    {
        throw std::runtime_error("Dimension not matching in the multivariate normal sampler");
    }

    arma::mat A;
    arma::vec eigval;
    arma::mat eigvec;
    arma::rowvec res;

    if( arma::chol(A,Sigma) )
    {
        res = randVecNormal(d).t() * A ;
    }
    else
    {
        if( eig_sym(eigval, eigvec, Sigma) )
        {
            res = (eigvec * arma::diagmat(arma::sqrt(eigval)) * randVecNormal(d)).t();
        }
        else
        {
            throw std::runtime_error("randMvNorm failing because of singular Sigma matrix");
        }
    }

    return res.t() + m;
}


arma::vec BVS_Sampler::randVecNormal(const unsigned int n)
{
    arma::vec res = Rcpp::as<arma::vec>(Rcpp::rnorm(n));
    return res;
}


double BVS_Sampler::logPDFNormal(
    const arma::vec& x,
    const arma::vec& m,
    const arma::mat& Sigma)
{
    unsigned int k = Sigma.n_cols;

    double sign, tmp;
    arma::log_det(tmp, sign, Sigma );

    return -0.5*(double)k*log(2.*M_PI) -0.5*tmp -0.5* arma::as_scalar( (x-m).t() * arma::inv_sympd(Sigma) * (x-m) );
}


// Compute set difference: elements of A that are not in B, preserving A's order
arma::uvec BVS_Sampler::setdiff_preserve_order(
    const arma::uvec& A,
    const arma::uvec& B)
{
    std::unordered_set<arma::uword> bset;
    bset.reserve(B.n_elem);
    for (arma::uword x : B) bset.insert(x);

    std::vector<arma::uword> out;
    out.reserve(A.n_elem);
    for (arma::uword x : A) {
        if (bset.find(x) == bset.end()) {
            out.push_back(x);
        }
    }

    return arma::uvec(out);
}