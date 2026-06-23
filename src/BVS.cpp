/* Core updates for Bayesian variable selection */

#include <memory> // Include for smart pointers
#include <cstdlib>
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
//   - By default using the slab prior N(0, tauSq_l) / N(0, wSq_l)  as the
//     default pseudo-prior. With that default the augmented-prior correction
//     cancels, and the gamma/eta update is the masked-likelihood conditional MH
//     special case of Carlin--Chib.
//   - To obtain the main mixing benefit of Carlin--Chib, specify pseudo-prior 
//     (means and) variances to hyperpar->augBetaVar, hyperpar->augZetaVar.
// -----------------------------------------------------------------------------

/*
namespace {

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

} // anonymous namespace
*/

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

    arma::mat datProportion;
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
            // zetaMask_l.elem(arma::find(etas.col(l) == 0)).fill(0.0);
            for (arma::uword j = 0; j < p; ++j) { if (etas(j,l) == 0) zetaMask_l[j] = 0.0; }
            alphas.col(l) = arma::exp( zetas(0, l) + dataclass.datX.slice(l) * zetaMask_l );
        }
        // alphas.elem(arma::find(alphas > upperbound3)).fill(upperbound3);
        // alphas.elem(arma::find(alphas < lowerbound)).fill(lowerbound);
        alphas = arma::min(alphas, arma::mat(N,L).fill(upperbound)); // faster alternative
        alphas = arma::max(alphas, arma::mat(N,L).fill(lowerbound)); 
        alphas_Rowsum = arma::sum(alphas, 1);
        // datProportion = alphas / arma::repmat(alphas_Rowsum, 1, L);
        datProportion = alphas.each_col() / alphas_Rowsum; // faster alternative to above
    }
    else
    {
        datProportion = dataclass.datProportionConst;
    }

    arma::vec logTheta = dataclass.datX0 * xi;
    // logTheta.elem(arma::find(logTheta > upperbound)).fill(upperbound);
    logTheta = arma::min(logTheta, arma::vec(N).fill(upperbound)); 
    arma::vec thetas = arma::exp( logTheta );

    arma::vec f = arma::zeros<arma::vec>(N);
    arma::vec survival_pop = arma::zeros<arma::vec>(N);
    
    arma::vec betaMask_l(p);
    arma::vec mu_l(N);
    arma::vec weibull_lambdas_l(N);
    arma::vec weibullS_l(N);
    arma::vec weibull_pdf(N);
    double GammaFuncKappa = std::tgamma(1. + 1./kappa);

    for(unsigned int l=0; l<L; ++l)
    {
        betaMask_l = betas.submat(1, l, p, l);
        // betaMask_l.elem(arma::find(gammas.col(l) == 0)).fill(0.0);
        // // faster alternative to above
        // betaMask_l %= arma::conv_to<arma::vec>::from(gammas.col(l)); 
        // betaMask_l %= arma::vec(gammas.col(l));
        for (arma::uword j = 0; j < p; ++j) { if (gammas(j,l) == 0) betaMask_l[j] = 0.0; }
        mu_l = betas(0, l) + dataclass.datX.slice(l) * betaMask_l;
        // mu_l.elem(arma::find(mu_l > upperbound)).fill(upperbound);
        mu_l = arma::exp(mu_l);
        mu_l = arma::min(mu_l, arma::vec(N).fill(upperbound)); 
        mu_l = arma::max(mu_l, arma::vec(N).fill(lowerbound)); 

        weibull_lambdas_l = mu_l / GammaFuncKappa;
        weibullS_l = arma::exp( - arma::pow( dataclass.datTime / weibull_lambdas_l, kappa) );
        weibull_pdf = arma::exp(-kappa * arma::log(weibull_lambdas_l) - arma::pow(dataclass.datTime/weibull_lambdas_l, kappa));

        survival_pop += datProportion.col(l) % weibullS_l;

        f += kappa * arma::pow(dataclass.datTime, kappa - 1.0) % datProportion.col(l) % weibull_pdf;
    }

    // summarize density of the Weibull's survival part
    arma::vec log_survival_pop = - thetas % (1. - survival_pop);
    // f.elem(arma::find(f < lowerbound)).fill(lowerbound);
    f = arma::max(f, arma::vec(N).fill(lowerbound)); // faster alternative
    arma::vec log_f_pop = logTheta + arma::log(f) + log_survival_pop;

    // summarize density of the Dirichlet part
    arma::vec log_dirichlet = arma::zeros<arma::vec>(N);
    if (proportion_model)
    {
        log_dirichlet =
            arma::lgamma(alphas_Rowsum) - arma::sum(arma::lgamma(alphas), 1) +
            arma::sum( (alphas - 1.0) % arma::log(dataclass.datProportionConst), 1 );
    }

    // log_f_pop.elem(arma::find(dataclass.datEvent == 0)).fill(0.);
    // log_survival_pop.elem(arma::find(dataclass.datEvent)).fill(0.);
    loglik = log_dirichlet;
    for (arma::uword i = 0; i < N; ++i) 
    { 
        if (dataclass.datEvent[i])
        {
            loglik[i] += log_f_pop[i];
            loglik[i] += log_survival_pop[i];
        } 
    }
    // loglik = log_f_pop + log_survival_pop + log_dirichlet;
}

// loglikelihood given all relevant quantities
void BVS_Sampler::loglikelihood_noBVS(
    // const arma::vec& xi,
    // const arma::mat& zetas,
    // const arma::mat& betas,
    double kappa,

    bool proportion_model,
    const arma::mat& alphas,
    const arma::mat& datProportion,
    const arma::mat& weibullS,
    const arma::mat& weibullLambda,
    const arma::vec& logTheta,
    const arma::vec& datTheta,
    const DataClass& dataclass,
    arma::vec& loglik)
{
    // dimensions
    unsigned int N = dataclass.datX.n_rows;
    // unsigned int p = dataclass.datX.n_cols;
    unsigned int L = dataclass.datX.n_slices;

    arma::vec f = arma::zeros<arma::vec>(N);
    arma::vec survival_pop = arma::zeros<arma::vec>(N);
    arma::vec weibull_pdf(N);

    for(unsigned int l=0; l<L; ++l)
    {
        weibull_pdf = arma::exp(-kappa * arma::log(weibullLambda.col(l)) - arma::pow(dataclass.datTime/weibullLambda.col(l), kappa));
        survival_pop += datProportion.col(l) % weibullS.col(l);
        f += kappa * arma::pow(dataclass.datTime, kappa - 1.0) % datProportion.col(l) % weibull_pdf;
    }

    // summarize density of the Weibull's survival part
    arma::vec log_survival_pop = - datTheta % (1. - survival_pop);
    // f.elem(arma::find(f < lowerbound)).fill(lowerbound);
    f = arma::max(f, arma::vec(N).fill(lowerbound)); 
    arma::vec log_f_pop = logTheta + arma::log(f) + log_survival_pop;

    // summarize density of the Dirichlet part
    arma::vec log_dirichlet = arma::zeros<arma::vec>(N);
    if (proportion_model)
    {
        // arma::vec alphas_Rowsum = arma::sum(alphas, 1);
        log_dirichlet =
            arma::lgamma(arma::sum(alphas, 1)) - 
            arma::sum(arma::lgamma(alphas), 1) +
            arma::sum( (alphas - 1.0) % arma::log(dataclass.datProportionConst), 1 );
    }

    // log_f_pop.elem(arma::find(dataclass.datEvent == 0)).fill(0.);
    // log_survival_pop.elem(arma::find(dataclass.datEvent)).fill(0.);
    loglik = log_dirichlet;
    for (arma::uword i = 0; i < N; ++i) 
    { 
        if (dataclass.datEvent[i])
        {
            loglik[i] += log_f_pop[i];
            loglik[i] += log_survival_pop[i];
        } 
    }
    // loglik = log_f_pop + log_survival_pop + log_dirichlet;
}


void BVS_Sampler::sampleGamma(
    arma::umat& gammas_,
    Gamma_Prior_Type gamma_prior,
    Gamma_Sampler_Type gamma_sampler,
    arma::mat& logP_gamma_,
    unsigned int& gamma_acc_count_,
    arma::vec& log_likelihood_,
    bool CMH,
    arma::mat& gammaBanditAlpha,
    arma::mat& gammaBanditBeta,
    const armsParmClass& armsPar,
    void *hyperpar_,
    const arma::mat& pseudoMean,
    const arma::mat& pseudoVar,
    const arma::vec& xi_,
    const arma::mat& zetas_,
    const arma::umat& etas_,
    arma::mat& betas_,
    double kappa,
    double tau0Sq_,
    const arma::vec& tauSq_,
    const arma::vec& pi,
    arma::vec& logZ_gamma_,
    bool proportion_model,
    // arma::mat& datProportion,
    // arma::vec& datTheta,
    arma::mat& datMu,
    arma::mat& weibullS,
    arma::mat& weibullLambda,
    const DataClass &dataclass)
{
    (void)armsPar;
    (void)tau0Sq_;
    (void)logZ_gamma_; // kept in signature for compatibility with the previous AIS version

    // std::unique_ptr<hyperparS> hyperpar = std::make_unique<hyperparS>();
    // *hyperpar = *(hyperparS *)hyperpar_;
    auto* hyperpar = static_cast<hyperparS*>(hyperpar_);

    arma::umat proposedGamma = gammas_;
    arma::mat proposedGammaPrior;
    arma::uvec updateIdx;

    double logProposalRatio = 0;

    unsigned int N = log_likelihood_.n_elem;
    unsigned int p = gammas_.n_rows;
    unsigned int L = gammas_.n_cols;

    // static arma::mat gammaBanditAlpha = arma::mat(p, L, arma::fill::value(0.5));
    // static arma::mat gammaBanditBeta = arma::mat(p, L, arma::fill::value(0.5));

    unsigned int componentUpdateIdx = static_cast<unsigned int>( R::runif( 0, L ) );
    arma::uvec singleIdx_k = { componentUpdateIdx };

    switch( gamma_sampler )
    {
    case Gamma_Sampler_Type::bandit:
        logProposalRatio += gammaBanditProposal( p, proposedGamma, gammas_, updateIdx, componentUpdateIdx, gammaBanditAlpha, gammaBanditBeta );
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

        logPriorGammaRatio += BVS_Sampler::mrfEdgeRatio(
            proposedGamma,
            gammas_,
            mrfG,
            mrfG_weights,
            updateIdxGlobal,
            hyperpar->mrfB
        );

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
        kappa,
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
        kappa,
        proportion_model,
        dataclass,
        proposedLikelihood
    );

    double logLikelihoodRatio = arma::sum(proposedLikelihood) - arma::sum(currentLikelihood);

    double logAugBetaCurrent = 0.0;
    double logAugBetaProposed = 0.0;
    double logAugBetaPriorRatio = 0.0;

    // choose to use conditional MH or Carlin-Chib augmented MH
    if( !CMH )
    {
        // arma::vec pseudoMean = augBetaMean.col(componentUpdateIdx);
        // arma::vec pseudoVar = augBetaVar.col(componentUpdateIdx);
        // if (pseudoVar.is_zero()) pseudoVar.fill( tauSq_[componentUpdateIdx] );

        logAugBetaCurrent = logAugBetaPriorColumn(
            betas_.submat(1, componentUpdateIdx, p, componentUpdateIdx),
            gammas_.col(componentUpdateIdx),
            tauSq_[componentUpdateIdx],
            pseudoMean.col(componentUpdateIdx),
            pseudoVar.col(componentUpdateIdx)
        );

        logAugBetaProposed = logAugBetaPriorColumn(
            betas_.submat(1, componentUpdateIdx, p, componentUpdateIdx),
            proposedGamma.col(componentUpdateIdx),
            tauSq_[componentUpdateIdx],
            pseudoMean.col(componentUpdateIdx),
            pseudoVar.col(componentUpdateIdx)
        );

        logAugBetaPriorRatio = logAugBetaProposed - logAugBetaCurrent;
    }

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

        // update quantities needed for ARMS_Gibbs::arms_gibbs_beta() in the main MCMC-loop in drive.cpp
        arma::vec logMu_k = betas_(0, componentUpdateIdx) + dataclass.datX.slice(componentUpdateIdx) * 
            (betas_.submat(1, componentUpdateIdx, p, componentUpdateIdx) % gammas_.col(componentUpdateIdx)) ;
        // logMu_k.elem(arma::find(logMu_k > upperbound)).fill(upperbound);
        logMu_k = arma::min(logMu_k, arma::vec(N).fill(upperbound)); 
        datMu.col(componentUpdateIdx) = arma::exp( logMu_k );
        weibullLambda.col(componentUpdateIdx) = datMu.col(componentUpdateIdx) / std::tgamma(1. + 1./kappa);
        // weibullLambda.elem(arma::find(lambdas > upperbound)).fill(upperbound);
        weibullS.col(componentUpdateIdx) = arma::exp(        
            -arma::pow(            
                dataclass.datTime / weibullLambda.col(componentUpdateIdx),           
                kappa        
            )    
        );
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
        double banditLimit = (double)(N);
        double banditIncrement = 1.;

        for(auto iter: updateIdx)
        {
            if( gammaBanditAlpha(iter,componentUpdateIdx) + gammaBanditBeta(iter,componentUpdateIdx) < banditLimit )
            {
                gammaBanditAlpha(iter,componentUpdateIdx) += banditIncrement * gammas_(iter,componentUpdateIdx);
                gammaBanditBeta(iter,componentUpdateIdx) += banditIncrement * (1-gammas_(iter,componentUpdateIdx));
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
    bool CMH,
    arma::mat& etaBanditAlpha,
    arma::mat& etaBanditBeta,

    const armsParmClass& armsPar,
    void *hyperpar_,
    const arma::mat& pseudoMean,
    const arma::mat& pseudoVar,
    arma::mat& zetas_,
    const arma::mat& betas_,
    const arma::umat& gammas_,
    const arma::vec& xi_,
    double kappa,
    double w0Sq_,
    const arma::vec& wSq_,
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

    // std::unique_ptr<hyperparS> hyperpar = std::make_unique<hyperparS>();
    // *hyperpar = *(hyperparS *)hyperpar_;
    auto* hyperpar = static_cast<hyperparS*>(hyperpar_);

    arma::umat proposedEta = etas_;
    arma::mat proposedEtaPrior;
    arma::uvec updateIdx;

    double logProposalRatio = 0;

    unsigned int N = log_likelihood_.n_elem;
    unsigned int p = etas_.n_rows;
    unsigned int L = etas_.n_cols;

    // static arma::mat etaBanditAlpha = arma::mat(p, L, arma::fill::value(0.5));
    // static arma::mat etaBanditBeta = arma::mat(p, L, arma::fill::value(0.5));

    unsigned int componentUpdateIdx = static_cast<unsigned int>( R::runif( 0, L ) );
    arma::uvec singleIdx_k = { componentUpdateIdx };

    switch( eta_sampler )
    {
    case Eta_Sampler_Type::bandit:
        logProposalRatio += etaBanditProposal( p, proposedEta, etas_, updateIdx, componentUpdateIdx, etaBanditAlpha, etaBanditBeta );
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

        logPriorEtaRatio += BVS_Sampler::mrfEdgeRatio(
            proposedEta,
            etas_,
            mrfG,
            mrfG_weights,
            updateIdxGlobal,
            hyperpar->mrfB_prop
        );

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
        kappa,
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
        kappa,
        dirichlet,
        dataclass,
        proposedLikelihood
    );

    double logLikelihoodRatio = arma::sum(proposedLikelihood) - arma::sum(currentLikelihood);

    double logAugZetaCurrent = 0.0;
    double logAugZetaProposed = 0.0;
    double logAugZetaPriorRatio = 0.0;

    // use conditional MH or Carlin-Chib augmented MH
    if( !CMH )
    {
        // arma::mat augZetaMean(const_cast<double*>(hyperpar->augZetaMean), p, L, false);
        // arma::mat augZetaVar(const_cast<double*>(hyperpar->augZetaVar), p, L, false);

        // arma::vec pseudoMean = augZetaMean.col(componentUpdateIdx);
        // arma::vec pseudoVar = augZetaVar.col(componentUpdateIdx);
        // if (pseudoVar.is_zero()) pseudoVar.fill( wSq_[componentUpdateIdx] );

        logAugZetaCurrent = logAugZetaPriorColumn(
            zetas_.submat(1, componentUpdateIdx, p, componentUpdateIdx),
            etas_.col(componentUpdateIdx),
            wSq_[componentUpdateIdx],
            pseudoMean.col(componentUpdateIdx),
            pseudoVar.col(componentUpdateIdx)
        );

        logAugZetaProposed = logAugZetaPriorColumn(
            zetas_.submat(1, componentUpdateIdx, p, componentUpdateIdx),
            proposedEta.col(componentUpdateIdx),
            wSq_[componentUpdateIdx],
            pseudoMean.col(componentUpdateIdx),
            pseudoVar.col(componentUpdateIdx)
        );

        logAugZetaPriorRatio = logAugZetaProposed - logAugZetaCurrent;
    }

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
        double banditLimit = (double)(N);
        double banditIncrement = 1.;

        for(auto iter: updateIdx)
        {
            if( etaBanditAlpha(iter,componentUpdateIdx) + etaBanditBeta(iter,componentUpdateIdx) < banditLimit )
            {
                etaBanditAlpha(iter,componentUpdateIdx) += banditIncrement * etas_(iter,componentUpdateIdx);
                etaBanditBeta(iter,componentUpdateIdx) += banditIncrement * (1-etas_(iter,componentUpdateIdx));
            }
        }
    }
}


double BVS_Sampler::gammaMC3Proposal(
    unsigned int p,
    arma::umat& mutantGamma,
    const arma::umat& gammas_,
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
    const arma::umat& gammas_,
    arma::uvec& updateIdx,
    unsigned int componentUpdateIdx_,
    arma::mat& gammaBanditAlpha,
    arma::mat& gammaBanditBeta)
{
    arma::vec banditPi(p, arma::fill::none);
    arma::vec mismatch(p, arma::fill::none);
    arma::vec normalised_mismatch;
    arma::vec normalised_mismatch_backwards;

    double logProposalRatio = 0.0;

    if (p == 0) {
        Rcpp::stop("'p' must be positive in gammaBanditProposal().");
    }

    /*
     * Thompson-style draw of latent inclusion probabilities.
     *
     * gammaBanditAlpha tracks inclusion counts;
     * gammaBanditBeta  tracks exclusion counts.
     */
    for(unsigned int j = 0; j < p; ++j)
    {
        double a = gammaBanditAlpha(j, componentUpdateIdx_);
        double b = gammaBanditBeta(j, componentUpdateIdx_);

        const double eps = 1e-12;
        a = std::max(a, eps);
        b = std::max(b, eps);

        banditPi(j) = R::rbeta(a, b);

        /*
         * Mismatch weight:
         *   if gamma_j = 0, weight = pi_j;
         *   if gamma_j = 1, weight = 1 - pi_j.
         */
        mismatch(j) =
            (gammas_(j, componentUpdateIdx_) == 0)
                ? banditPi(j)
                : (1.0 - banditPi(j));
    }

    double eps = 1e-12;
    mismatch.elem(arma::find_nonfinite(mismatch)).fill(0.0);
    // mismatch.elem(arma::find(mismatch < eps)).fill(eps);
    mismatch = arma::max(mismatch, arma::vec(p).fill(eps)); 

    normalised_mismatch = mismatch / arma::sum(mismatch);

    /*
     * Option A:
     *   - with probability 0.5, single deterministic flip;
     *   - with probability 0.5, multi deterministic flip with J > 1.
     *
     * For p == 1, only the single-flip move is possible.
     */
    bool singleMove = true;

    if (p >= 2) {
        singleMove = (R::runif(0.0, 1.0) < 0.5);
    }

    if (singleMove)
    {
        updateIdx = arma::zeros<arma::uvec>(1);
        updateIdx(0) = randWeightedIndexSampleWithoutReplacement(normalised_mismatch);

        unsigned int idx = updateIdx(0);

        mutantGamma(idx, componentUpdateIdx_) =
            1 - gammas_(idx, componentUpdateIdx_);

        /*
         * Backward mismatch under the proposed state.
         * Only flipped coordinates change their mismatch weight.
         */
        normalised_mismatch_backwards = mismatch;
        normalised_mismatch_backwards(idx) =
            1.0 - normalised_mismatch_backwards(idx);

        // normalised_mismatch_backwards.elem(
        //     arma::find(normalised_mismatch_backwards < eps)
        // ).fill(eps);
        normalised_mismatch_backwards = arma::max(normalised_mismatch_backwards, arma::vec(p).fill(eps)); 

        normalised_mismatch_backwards =
            normalised_mismatch_backwards / arma::sum(normalised_mismatch_backwards);

        logProposalRatio =
            std::log(normalised_mismatch_backwards(idx)) -
            std::log(normalised_mismatch(idx));
    }
    else
    {
        /*
         * Multi-flip branch.
         * Use J >= 2 so that this branch is disjoint from the single-flip branch.
         */
        unsigned int n_updates_bandit = std::min<unsigned int>(4, p);
        n_updates_bandit = std::max<unsigned int>(2, n_updates_bandit);

        updateIdx = randWeightedIndexSampleWithoutReplacement(
            p,
            normalised_mismatch,
            n_updates_bandit
        );

        /*
         * Deterministically flip all selected variables.
         */
        for(unsigned int i = 0; i < n_updates_bandit; ++i)
        {
            unsigned int idx = updateIdx(i);

            mutantGamma(idx, componentUpdateIdx_) =
                1 - gammas_(idx, componentUpdateIdx_);
        }

        /*
         * Backward mismatch under proposed state.
         * For every flipped coordinate, mismatch becomes 1 - old mismatch.
         */
        normalised_mismatch_backwards = mismatch;

        for(unsigned int i = 0; i < n_updates_bandit; ++i)
        {
            unsigned int idx = updateIdx(i);
            normalised_mismatch_backwards(idx) =
                1.0 - normalised_mismatch_backwards(idx);
        }

        // normalised_mismatch_backwards.elem(
        //     arma::find(normalised_mismatch_backwards < eps)
        // ).fill(eps);
        normalised_mismatch_backwards = arma::max(normalised_mismatch_backwards, arma::vec(p).fill(eps)); 

        normalised_mismatch_backwards =
            normalised_mismatch_backwards / arma::sum(normalised_mismatch_backwards);

        /*
         * No Bernoulli-resampling density terms are needed now.
         * The proposal ratio is only the weighted-without-replacement
         * set-selection probability backward divided by forward.
         */
        logProposalRatio =
            logPDFWeightedIndexSampleWithoutReplacement(
                normalised_mismatch_backwards,
                updateIdx
            ) -
            logPDFWeightedIndexSampleWithoutReplacement(
                normalised_mismatch,
                updateIdx
            );
    }

    return logProposalRatio;
}


double BVS_Sampler::etaBanditProposal(
    unsigned int p,
    arma::umat& mutantEta,
    const arma::umat& etas_,
    arma::uvec& updateIdx,
    unsigned int componentUpdateIdx_,
    arma::mat& etaBanditAlpha,
    arma::mat& etaBanditBeta)
{
    arma::vec banditPi(p, arma::fill::none);
    arma::vec mismatch(p, arma::fill::none);
    arma::vec normalised_mismatch;
    arma::vec normalised_mismatch_backwards;

    double logProposalRatio = 0.0;

    if (p == 0) {
        Rcpp::stop("'p' must be positive in etaBanditProposal().");
    }

    /*
     * Thompson-style draw of latent inclusion probabilities.
     *
     * etaBanditAlpha tracks inclusion counts;
     * etaBanditBeta  tracks exclusion counts.
     */
    for(unsigned int j = 0; j < p; ++j)
    {
        double a = etaBanditAlpha(j, componentUpdateIdx_);
        double b = etaBanditBeta(j, componentUpdateIdx_);

        const double eps = 1e-12;
        a = std::max(a, eps);
        b = std::max(b, eps);

        banditPi(j) = R::rbeta(a, b);

        /*
         * Mismatch weight:
         *   if eta_j = 0, weight = pi_j;
         *   if eta_j = 1, weight = 1 - pi_j.
         */
        mismatch(j) =
            (etas_(j, componentUpdateIdx_) == 0)
                ? banditPi(j)
                : (1.0 - banditPi(j));
    }

    double eps = 1e-12;
    mismatch.elem(arma::find_nonfinite(mismatch)).fill(0.0);
    // mismatch.elem(arma::find(mismatch < eps)).fill(eps);
    mismatch = arma::max(mismatch, arma::vec(p).fill(eps)); 

    normalised_mismatch = mismatch / arma::sum(mismatch);

    /*
     * Option A:
     *   - with probability 0.5, single deterministic flip;
     *   - with probability 0.5, multi deterministic flip with J > 1.
     *
     * For p == 1, only the single-flip move is possible.
     */
    bool singleMove = true;

    if (p >= 2) {
        singleMove = (R::runif(0.0, 1.0) < 0.5);
    }

    if (singleMove)
    {
        updateIdx = arma::zeros<arma::uvec>(1);
        updateIdx(0) = randWeightedIndexSampleWithoutReplacement(normalised_mismatch);

        unsigned int idx = updateIdx(0);

        mutantEta(idx, componentUpdateIdx_) =
            1 - etas_(idx, componentUpdateIdx_);

        /*
         * Backward mismatch under proposed state.
         * Only flipped coordinates change their mismatch weight.
         */
        normalised_mismatch_backwards = mismatch;
        normalised_mismatch_backwards(idx) =
            1.0 - normalised_mismatch_backwards(idx);

        // normalised_mismatch_backwards.elem(
        //     arma::find(normalised_mismatch_backwards < eps)
        // ).fill(eps);
        normalised_mismatch_backwards = arma::max(normalised_mismatch_backwards, arma::vec(p).fill(eps)); 

        normalised_mismatch_backwards =
            normalised_mismatch_backwards / arma::sum(normalised_mismatch_backwards);

        logProposalRatio =
            std::log(normalised_mismatch_backwards(idx)) -
            std::log(normalised_mismatch(idx));
    }
    else
    {
        /*
         * Multi-flip branch.
         * Use J >= 2 so that this branch is disjoint from the single-flip branch.
         */
        unsigned int n_updates_bandit = std::min<unsigned int>(4, p);
        n_updates_bandit = std::max<unsigned int>(2, n_updates_bandit);

        updateIdx = randWeightedIndexSampleWithoutReplacement(
            p,
            normalised_mismatch,
            n_updates_bandit
        );

        /*
         * Deterministically flip all selected variables.
         */
        for(unsigned int i = 0; i < n_updates_bandit; ++i)
        {
            unsigned int idx = updateIdx(i);

            mutantEta(idx, componentUpdateIdx_) =
                1 - etas_(idx, componentUpdateIdx_);
        }

        /*
         * Backward mismatch under proposed state.
         * For every flipped coordinate, mismatch becomes 1 - old mismatch.
         */
        normalised_mismatch_backwards = mismatch;

        for(unsigned int i = 0; i < n_updates_bandit; ++i)
        {
            unsigned int idx = updateIdx(i);
            normalised_mismatch_backwards(idx) =
                1.0 - normalised_mismatch_backwards(idx);
        }

        // normalised_mismatch_backwards.elem(
        //     arma::find(normalised_mismatch_backwards < eps)
        // ).fill(eps);
        normalised_mismatch_backwards = arma::max(normalised_mismatch_backwards, arma::vec(p).fill(eps)); 

        normalised_mismatch_backwards =
            normalised_mismatch_backwards / arma::sum(normalised_mismatch_backwards);

        /*
         * No Bernoulli-resampling density terms are needed now.
         * The proposal ratio is only the weighted-without-replacement
         * set-selection probability backward divided by forward.
         */
        logProposalRatio =
            logPDFWeightedIndexSampleWithoutReplacement(
                normalised_mismatch_backwards,
                updateIdx
            ) -
            logPDFWeightedIndexSampleWithoutReplacement(
                normalised_mismatch,
                updateIdx
            );
    }

    return logProposalRatio;
}

// helper function for cleaner MRF code
double BVS_Sampler::mrfEdgeRatio(
    const arma::umat& proposed,
    const arma::umat& current,
    const arma::umat& edges,
    const arma::vec& weights,
    const arma::uvec& updated_global_idx,
    double mrfB)
{
    if (mrfB <= 0.0 || updated_global_idx.is_empty()) {
        return 0.0;
    }

    std::unordered_set<arma::uword> updated_nodes;
    updated_nodes.reserve(updated_global_idx.n_elem);

    for (arma::uword idx : updated_global_idx) {
        updated_nodes.insert(idx);
    }

    double out = 0.0;

    for (arma::uword e = 0; e < edges.n_rows; ++e)
    {
        arma::uword a = edges(e, 0);
        arma::uword b = edges(e, 1);

        bool touches_updated =
            updated_nodes.find(a) != updated_nodes.end() ||
            updated_nodes.find(b) != updated_nodes.end();

        if (!touches_updated) {
            continue;
        }

        if (a != b)
        {
            out +=
                mrfB * 2.0 * weights(e) *
                (
                    static_cast<double>(proposed(a) * proposed(b)) -
                    static_cast<double>(current(a) * current(b))
                );
        }
        else
        {
            out +=
                mrfB * weights(e) *
                (
                    static_cast<double>(proposed(a)) -
                    static_cast<double>(current(a))
                );
        }
    }

    return out;
}


/*
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

    double logpost_first_sum = arma::sum( arma::log( logpost_first.elem(arma::find(dataclass.datEvent)) ) );

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

    double logpost_first_sum = arma::sum( arma::log( logpost_first.elem(arma::find(dataclass.datEvent)) ) );

    double logpost_second_sum = arma::sum(datTheta % logpost_second);

    double log_dirichlet_sum = arma::sum(
        arma::lgamma(alphas_Rowsum) - arma::sum(arma::lgamma(alphas), 1) +
        arma::sum( (alphas - 1.0) % arma::log(dataclass.datProportionConst), 1 )
    );

    logP = logprior + logpost_first_sum + logpost_second_sum + log_dirichlet_sum;

    return logP;
}
*/

// subfunctions used for bandit proposal

arma::uvec BVS_Sampler::randWeightedIndexSampleWithoutReplacement(
    unsigned int populationSize,
    const arma::vec& weights,
    unsigned int sampleSize
)
{
    if (sampleSize > populationSize) {
        Rcpp::stop("'sampleSize' cannot exceed 'populationSize'.");
    }

    arma::vec safe_weights = weights;

    double eps = 1e-12;
    safe_weights.elem(arma::find_nonfinite(safe_weights)).fill(0.0);
    // safe_weights.elem(arma::find(safe_weights < eps)).fill(eps);
    safe_weights = arma::max(safe_weights, arma::vec(weights.n_elem).fill(eps)); 

    arma::vec u = Rcpp::runif(populationSize);

    // Exponential race: smaller score means earlier selection.
    arma::vec score = -arma::log(u) / safe_weights;

    arma::uvec result = arma::sort_index(score, "ascend");

    return result.subvec(0, sampleSize - 1);
}


unsigned int BVS_Sampler::randWeightedIndexSampleWithoutReplacement(
    const arma::vec& weights
)
{
    if (weights.is_empty()) {
        Rcpp::stop("'weights' is empty.");
    }

    arma::vec safe_weights = weights;
    safe_weights.elem(arma::find_nonfinite(safe_weights)).fill(0.0);
    // safe_weights.elem(arma::find(safe_weights < 0.0)).fill(0.0);
    safe_weights = arma::max(safe_weights, arma::vec(weights.n_elem).fill(0.0)); 

    double total = arma::accu(safe_weights);

    if (!std::isfinite(total) || total <= 0.0) {
        Rcpp::stop("Invalid weights in randWeightedIndexSampleWithoutReplacement().");
    }

    safe_weights /= total;

    double u = R::runif(0.0, 1.0);
    double cdf = 0.0;

    for (arma::uword i = 0; i < safe_weights.n_elem; ++i)
    {
        cdf += safe_weights[i];
        if (u <= cdf) {
            return static_cast<unsigned int>(i);
        }
    }

    return static_cast<unsigned int>(safe_weights.n_elem - 1);
}


double BVS_Sampler::logPDFWeightedIndexSampleWithoutReplacement(
    const arma::vec& weights,
    const arma::uvec& indexes
)
{
    if (indexes.is_empty()) return 0.0;

    const unsigned int populationSize = weights.n_elem;
    const unsigned int sampleSize = indexes.n_elem;

    if (sampleSize > populationSize) 
        return -std::numeric_limits<double>::infinity();

    arma::vec safe_weights = weights;

    // Numerical protection.
    double eps = 1e-12;
    safe_weights.elem(arma::find_nonfinite(safe_weights)).fill(0.0);
    // safe_weights.elem(arma::find(safe_weights < eps)).fill(eps);
    safe_weights = arma::max(safe_weights, arma::vec(weights.n_elem).fill(eps)); 

    double total_weight = arma::accu(safe_weights);

    if (!std::isfinite(total_weight) || total_weight <= 0.0)
        return -std::numeric_limits<double>::infinity();

    // Check indexes are valid and unique.
    std::vector<unsigned int> idx = arma::conv_to<std::vector<unsigned int>>::from(indexes);

    std::sort(idx.begin(), idx.end());

    if (std::unique(idx.begin(), idx.end()) != idx.end())
        return -std::numeric_limits<double>::infinity();

    for (unsigned int id : idx)
        if (id >= populationSize)
            return -std::numeric_limits<double>::infinity();

    double logP_set = -std::numeric_limits<double>::infinity();

    do {
        double remaining_weight = total_weight;
        double logP_perm = 0.0;
        bool valid = true;

        for (unsigned int id : idx) {
            double w = safe_weights[id];

            if (w <= 0.0 || remaining_weight <= 0.0) {
                valid = false;
                break;
            }

            logP_perm += std::log(w) - std::log(remaining_weight);
            remaining_weight -= w;
        }

        if (valid) logP_set = logspace_add(logP_set, logP_perm);

    } while (std::next_permutation(idx.begin(), idx.end()));

    return logP_set;
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

double BVS_Sampler::logPDFNormal(double x, double mean, double var)
{
    const double eps = 1e-12;
    var = std::max(var, eps);
    double z = x - mean;
    
    return -0.5 * std::log(2.0 * M_PI) - 0.5 * std::log(var) - 0.5 * z * z / var;
}

double BVS_Sampler::logSlabPriorNormal(double x, double var)
{
    return logPDFNormal(x, 0.0, var);
}

double BVS_Sampler::logPseudoPriorNormal(double x, double var)
{
    // Default pseudo-prior: same as slab prior.
    // Replace this by N(pseudo_mean[j,l], pseudo_var[j,l]) if pseudo-prior with different mean and var
    return logPDFNormal(x, 0.0, var);
}

double BVS_Sampler::logAugBetaPriorColumn(
    const arma::vec& beta_col_nonintercept,
    const arma::uvec& gamma_col,
    double slab_var,
    const arma::vec& pseudo_mean,
    const arma::vec& pseudo_var)
{
    double out = 0.0;
    for (unsigned int j = 0; j < gamma_col.n_elem; ++j)
    {
        double b = beta_col_nonintercept[j];
        if (gamma_col[j] == 1)
            out += logSlabPriorNormal(b, slab_var);
        else
            out += logPDFNormal(b, pseudo_mean[j], pseudo_var[j]);
    }
    return out;
}

double BVS_Sampler::logAugZetaPriorColumn(
    const arma::vec& zeta_col_nonintercept,
    const arma::uvec& eta_col,
    double slab_var,
    const arma::vec& pseudo_mean,
    const arma::vec& pseudo_var)
{
    double out = 0.0;
    for (unsigned int j = 0; j < eta_col.n_elem; ++j)
    {
        double b = zeta_col_nonintercept[j];
        if (eta_col[j] == 1)
            out += logSlabPriorNormal(b, slab_var);
        else
            out += logPDFNormal(b, pseudo_mean[j], pseudo_var[j]);
    }
    return out;
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
