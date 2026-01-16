/* Log-likelihood for the use in Metropolis-Hastings sampler*/

#include <memory> // Include for smart pointers

#include "BVS.h"
#include "arms_gibbs.h"

// TODO: loglikelihood can be updated in 'ARMS_Gibbs::logPbetas()' and 'ARMS_Gibbs::logPzetas()',
//          so that it does not need to updated twice in 'BVS_Sampler::sampleGamma()' and 'BVS_Sampler::sampleEta()'.
// log-density for coefficient xis
void BVS_Sampler::loglikelihood(
    const arma::vec& xi,
    const arma::mat& zetas,
    const arma::mat& betas,
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
            alphas.col(l) = arma::exp( zetas(0, l) + dataclass.datX.slice(l) * zetas.submat(1, l, p, l) );
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
        // arma::vec logMu_l = dataclass.datX.slice(l) * betas.col(l);
        arma::vec logMu_l = betas(0, l) + dataclass.datX.slice(l) * betas.submat(1, l, p, l);
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
    arma::vec log_f_pop = logTheta + arma::log(f) + log_survival_pop;

    // summarize density of the Dirichlet part
    // double log_dirichlet = 0.;
    arma::vec log_dirichlet = arma::zeros<arma::vec>(N);
    if (proportion_model)
    {
        log_dirichlet = //arma::accu(
            arma::lgamma(alphas_Rowsum) - arma::sum(arma::lgamma(alphas), 1) +
            arma::sum( (alphas - 1.0) % arma::log(dataclass.datProportionConst), 1 );
        //);
    }

    // double loglik =  arma::accu( log_f_pop.elem(arma::find(dataclass.datEvent)) ) +
    //                  arma::accu( log_survival_pop.elem(arma::find(dataclass.datEvent == 0)) ) +
    //                  log_dirichlet;
    log_f_pop.elem(arma::find(dataclass.datEvent == 0)).fill(0.);
    log_survival_pop.elem(arma::find(dataclass.datEvent)).fill(0.);
    // arma::vec loglik = log_f_pop + log_survival_pop + log_dirichlet;
    loglik = log_f_pop + log_survival_pop + log_dirichlet;

    // // log-likelihood of survival data without including the measurement modeling part
    // loglik0 = log_f_pop + log_survival_pop;

    //return loglik;
}

// NO NEED 'loglikelihood0()', SINCE ALL ARE BASED ON JOINT LIKELIHOOD 'loglikelihood()'
// log-density for survival data only
/*
void BVS_Sampler::loglikelihood0(
    const arma::vec& xi,
    const arma::mat& zetas,
    const arma::mat& betas,
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
        for(unsigned int l=0; l<L; ++l)
        {
            alphas.col(l) = arma::exp( zetas(0, l) + dataclass.datX.slice(l) * zetas.submat(1, l, p, l) );
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
        // arma::vec logMu_l = dataclass.datX.slice(l) * betas.col(l);
        arma::vec logMu_l = betas(0, l) + dataclass.datX.slice(l) * betas.submat(1, l, p, l);
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
    arma::vec log_f_pop = logTheta + arma::log(f) + log_survival_pop;

    // // summarize density of the Dirichlet part
    // arma::vec log_dirichlet = arma::zeros<arma::vec>(N);
    // if (proportion_model)
    // {
    //     log_dirichlet = //arma::accu(
    //         arma::lgamma(alphas_Rowsum) - arma::sum(arma::lgamma(alphas), 1) +
    //         arma::sum( (alphas - 1.0) % arma::log(dataclass.datProportionConst), 1 );
    //     //);
    // }

    log_f_pop.elem(arma::find(dataclass.datEvent == 0)).fill(0.);
    log_survival_pop.elem(arma::find(dataclass.datEvent)).fill(0.);
    // loglik = log_f_pop + log_survival_pop + log_dirichlet;

    // log-likelihood of survival data without including the measurement modeling part
    loglik = log_f_pop + log_survival_pop;

    //return loglik;
}
*/

void BVS_Sampler::sampleGamma(
    arma::umat& gammas_,
    Gamma_Prior_Type gamma_prior,
    Gamma_Sampler_Type gamma_sampler,
    arma::mat& logP_gamma_,
    unsigned int& gamma_acc_count_,
    arma::vec& log_likelihood_,
    const std::string& rw_mh,
    double sigmaMH_beta,

    const armsParmClass& armsPar,
    void *hyperpar_,

    const arma::vec& xi_,
    const arma::mat& zetas_,
    arma::mat& betas_,
    double kappa_,
    double tau0Sq_,
    const arma::vec& tauSq_,
    double pi0,

    bool proportion_model,

    // double& logPosteriorBeta,
    arma::mat& datProportion,
    arma::vec& datTheta,
    const arma::mat& datMu,
    const arma::mat& weibullLambda,
    const arma::mat& weibullS,
    const DataClass &dataclass)
{

    // reallocate struct variables
    // hyperparS *hyperpar = (hyperparS *)calloc(sizeof(hyperparS), sizeof(hyperparS));
    std::unique_ptr<hyperparS> hyperpar = std::make_unique<hyperparS>();
    *hyperpar = *(hyperparS *)hyperpar_;

    arma::umat proposedGamma = gammas_; // copy the original gammas and later change the address of the copied one
    arma::mat proposedGammaPrior;
    arma::uvec updateIdx;

    double logProposalRatio = 0;

    unsigned int p = gammas_.n_rows;
    unsigned int L = gammas_.n_cols;
    unsigned int n = datTheta.n_elem;

    // define static variables for global updates for the use of bandit algorithm
    // initial value 0.5 here forces shrinkage toward 0 or 1
    static arma::mat banditAlpha = arma::mat(p, L, arma::fill::value(0.5));
    static arma::mat banditBeta = arma::mat(p, L, arma::fill::value(0.5));

    // decide on one component
    unsigned int componentUpdateIdx = static_cast<unsigned int>( R::runif( 0, L ) );
    // Rcpp::IntegerVector entireIdx = Rcpp::seq( 0, L - 1);
    // unsigned int componentUpdateIdx = Rcpp::sample(entireIdx, 1, false)[0];
    arma::uvec singleIdx_k = { componentUpdateIdx };

    // Update the proposed Gamma with 'updateIdx' renewed via its address
    //if ( gamma_sampler_bandit )
    switch( gamma_sampler )
    {
    case Gamma_Sampler_Type::bandit:
        logProposalRatio += gammaBanditProposal( p, proposedGamma, gammas_, updateIdx, componentUpdateIdx, banditAlpha );
        break;

    case Gamma_Sampler_Type::mc3:
        logProposalRatio += gammaMC3Proposal( p, proposedGamma, gammas_, updateIdx, componentUpdateIdx );
        break;
    }

    // note only one outcome is updated
    // update log probabilities

    // compute logProposalGammaRatio, i.e. proposedGammaPrior - logP_gamma
    double logPriorGammaRatio = 0.;
    //if(gamma_prior_bernoulli)
    switch(gamma_prior)
    {
    case Gamma_Prior_Type::bernoulli:
    {
        proposedGammaPrior = logP_gamma_; // copy the original one and later change the address of the copied one
        // update corresponding Bernoulli probabilities (allow cluster-specific sparsity)
        // pi_proposed = R::rbeta(hyperpar->piA + (double)(arma::accu(proposedGamma.col(componentUpdateIdx))),
        //                      hyperpar->piB + (double)(p) - (double)(arma::accu(proposedGamma.col(componentUpdateIdx))));

        double pi = pi0;
        for(auto i: updateIdx)
        {   // This chunk is only valid for beta-Bernoulli prior
            if(pi0 == 0.) // this is to control if using pre-defined constant pi0 or hyperprior
            {
                //// feature-specific Bernoulli probability
                // pi = R::rbeta(hyperpar->piA + (double)(gammas_(i,componentUpdateIdx)),
                //                 hyperpar->piB + 1.0 - (double)(gammas_(i,componentUpdateIdx)));
                //// column-specific Bernoulli probability
                // pi = R::rbeta(hyperpar->piA + (double)(arma::sum(gammas_.col(componentUpdateIdx))),
                //                 hyperpar->piB + (double)(p) - (double)(arma::sum(gammas_.col(componentUpdateIdx))));
                //// row-specific Bernoulli probability
                pi = R::rbeta(hyperpar->piA + (double)(arma::accu(gammas_.row(i))),
                                    hyperpar->piB + (double)(L) - (double)(arma::accu(gammas_.row(i))));
            }
            // logProposalGammaRatio += logPDFBernoulli( proposedGamma(i,componentUpdateIdx), pi_proposed ) - logPDFBernoulli( gammas_(i,componentUpdateIdx), pi[componentUpdateIdx] );
            proposedGammaPrior(i,componentUpdateIdx) = logPDFBernoulli( proposedGamma(i,componentUpdateIdx), pi );
            logPriorGammaRatio +=  proposedGammaPrior(i, componentUpdateIdx) - logP_gamma_(i, componentUpdateIdx);
        }
        // // std::cout << "debug gamma - pi=" << pi << 
        // "; logProposalGammaRatio=" << logProposalGammaRatio <<
        // "; piA=" << hyperpar->piA << "; piA+=" << (double)(arma::accu(proposedGamma.col(componentUpdateIdx))) << 
        // "; piB=" << hyperpar->piB << "; piB+=" << (double)(p) - (double)(arma::accu(proposedGamma.col(componentUpdateIdx))) << 
        // "\n";
        break;
    }

    case Gamma_Prior_Type::mrf:
    {
        arma::umat mrfG(const_cast<unsigned int*>(hyperpar->mrfG), hyperpar->mrfG_edge_n, 2, false);
        arma::vec mrfG_weights(const_cast<double*>(hyperpar->mrfG_weights), hyperpar->mrfG_edge_n, false);
        // update corresponding to MRF prior
        // logProposalGammaRatio += logPDFMRF( proposedGamma, mrfG, hyperpar->mrfA, hyperpar->mrfB ) - logPDFMRF( gammas_, mrfG, hyperpar->mrfA, hyperpar->mrfB );

        // log-ratio/difference from the first-order term in MRF prior
        logPriorGammaRatio += hyperpar->mrfA * ( (double)(arma::accu(proposedGamma.submat(updateIdx, singleIdx_k))) -
                                (double)(arma::accu(gammas_.submat(updateIdx, singleIdx_k))) );

        // convert 'updateIdx' from ONE component to its index among multiple components
        arma::uvec updateIdxGlobal = updateIdx + p * componentUpdateIdx;
        // log-ratio/difference from the second-order term in MRF prior
        //arma::umat mrfG_idx = arma::conv_to<arma::umat>::from(mrfG);
        //mrfG_idx.shed_col(2);
        arma::uvec updateIdxMRF_common = arma::intersect(updateIdxGlobal, mrfG); //mrfG_idx);
        // // std::cout << "...debug4  updateIdxMRF_common=" << updateIdxMRF_common.t() << "\n";
        if((updateIdxMRF_common.n_elem > 0) && (hyperpar->mrfB > 0))
        {
            /*
            for(auto i: updateIdxMRF_common)
            {
                arma::uvec idxG = arma::find(mrfG.col(0) == i || mrfG.col(1) == i);
                for(auto ii: idxG)
                {
                    logProposalGammaRatio += hyperpar->mrfB * 2.0 * mrfG(ii, 2) * (proposedGamma(i) - gammas_(i));
                }
            }
            */
            #ifdef _OPENMP
            #pragma omp parallel for default(shared) reduction(+:logPriorGammaRatio)
            #endif

            for(unsigned int i=0; i<hyperpar->mrfG_edge_n; ++i)
            {
                if( mrfG(i, 0) != mrfG(i, 1))
                {

                    logPriorGammaRatio += hyperpar->mrfB * 2.0 * mrfG_weights(i) * //mrfG(i, 2) *
                                             ((double)(proposedGamma(mrfG(i, 0)) * proposedGamma(mrfG(i, 1))) -
                                              (double)(gammas_(mrfG(i, 0)) * gammas_(mrfG(i, 1))));
                }
                else
                {
                    logPriorGammaRatio += hyperpar->mrfB * mrfG_weights(i) * //mrfG(i, 2) *
                                             ((double)(proposedGamma(mrfG(i, 0))) - (double)(gammas_(mrfG(i, 0))));
                }

                /*
                    // std::cout << "; logProposalGammaRatio21=" << logProposalGammaRatio <<
                    " (eq=" << eq <<
                    "; tmp=" << tmp <<
                    "; tmp1=" << hyperpar->mrfB * mrfG_weights(i) <<
                    "; tmp2=" << (double)(proposedGamma(mrfG(i, 0))) - (double)(gammas_(mrfG(i, 0))) <<
                    "; mrfB=" << hyperpar->mrfB <<
                    "; mrfG_weights(i)=" << mrfG_weights(i) <<
                    "; proposedGamma(mrfG(i, 0))=" << (double)(proposedGamma(mrfG(i, 0))) <<
                    "; gammas_(mrfG(i, 0))=" << (double)(gammas_(mrfG(i, 0))) << "\n";
                */
            }
        }
        break;
    }
    }

    // compute logProposalBetaRatio given proposedGamma, i.e. proposedBetaPrior - logP_beta

    // update betas based on the proposal gammas
    arma::mat proposedBeta = betas_;
    
    // RW-MH
    if (arma::as_scalar(arma::any(proposedGamma(updateIdx, singleIdx_k)))) 
    {
        unsigned int J = updateIdx.n_elem;
        if (rw_mh != "symmetric")
        {
            // double c = std::exp(a);

            // Precompute a and powers
            arma::vec a = dataclass.datTime / weibullLambda.col(componentUpdateIdx);      // n-vector
            arma::vec a_power_kappa = arma::pow(a, kappa_);                               // z_il^kappa

            // s_l = κ / λ_il * (t_i/λ_il)^{κ-1} * p_il * S_l(t_i)
            arma::vec s_l = (kappa_ / weibullLambda.col(componentUpdateIdx))
                        % arma::pow(a, kappa_ - 1.0)
                        % datProportion.col(componentUpdateIdx)
                        % weibullS.col(componentUpdateIdx);

            // S = sum over components of κ / λ * (t/λ)^{κ-1} * p * S(t)
            arma::vec S(n, arma::fill::zeros);
            for (unsigned int ll = 0; ll < L; ++ll) {
                arma::vec a_tmp = dataclass.datTime / weibullLambda.col(ll);
                S += (kappa_ / weibullLambda.col(ll))
                % arma::pow(a_tmp, kappa_ - 1.0)
                % datProportion.col(ll)
                % weibullS.col(ll);
            }

            // Per-observation weights w_i (likelihood-only score part)
            arma::vec w_i(n);

            // Stabilize division s_l / S
            const double eps_div = 1e-12;
            arma::vec S_safe = S + eps_div;

            // First term: −δ_i * (s_il / S_i) * (1 − z_il^κ)
            w_i  = - dataclass.datEvent
                % (s_l / S_safe)
                % (1.0 - a_power_kappa);

            // Second term: + θ_i * p_il * S_l(t_i) * z_il^κ
            w_i += datTheta
                % datProportion.col(componentUpdateIdx)
                % weibullS.col(componentUpdateIdx)
                % a_power_kappa;

            // Design matrix and likelihood-only per-observation gradients
            arma::mat X_l = dataclass.datX.slice(componentUpdateIdx).cols(updateIdx);     // n x d
            const double G = std::tgamma(1.0 + 1.0 / kappa_);                              // Γ(1 + 1/κ)

            // gradient_betaK_i0 = (κ / G) * X * w_i (likelihood-only)
            arma::mat gradient_betaK_i0 = (kappa_ / G) * (X_l.each_col() % w_i);          // n x d

            // Empirical Fisher (OPG) for the sum log-likelihood: H = sum_i s_i s_i^T
            arma::mat H = gradient_betaK_i0.t() * gradient_betaK_i0;                       // d x d

            // Metric Δ = H + prior precision + damping
            arma::mat Delta_betaK = H;
            double nu = 0.5;  // damping parameter
            Delta_betaK.diag() += 1.0 / tauSq_[componentUpdateIdx] + nu;

            // Enforce symmetry (numerical robustness)
            Delta_betaK = 0.5 * (Delta_betaK + Delta_betaK.t());

            // Invert to get the metric M ≈ (H + D^{-1} + ν I)^{-1}
            arma::mat M;
            if (!arma::inv_sympd(M, Delta_betaK)) {
                arma::inv(M, Delta_betaK, arma::inv_opts::allow_approx);
            }

            // Full posterior gradient (sum-likelihood − prior): d-vector
            arma::vec gradient_betaK = arma::sum(gradient_betaK_i0, 0).t();               // sum over i
            gradient_betaK -= betas_(1 + updateIdx, singleIdx_k) / tauSq_[componentUpdateIdx];

            // MALA mean and covariance
            double eps = sigmaMH_beta;
            arma::vec  m     = 0.5 * eps * eps * (M * gradient_betaK);
            arma::mat  Sigma =        eps * eps * M;

            // Draw MALA proposal increment u ~ N(m, Sigma)
            arma::vec u = randMvNormal(m, Sigma);

            // Update proposed parameters on the selected coordinates
            proposedBeta(1 + updateIdx, singleIdx_k) += u;
            
        } else {//if( arma::any(gammas_(updateIdx,singleIdx_k)) ) {
            // (symmetric) random-walk Metropolis with optimal standard deviation O(d^{-1/2})
            arma::vec u = Rcpp::rnorm( J, 0., sigmaMH_beta * 1. / std::sqrt(J) ); 
            proposedBeta(1+updateIdx, singleIdx_k) += u;
        }
    }
    proposedBeta(1+arma::find(proposedGamma.col(componentUpdateIdx) == 0), singleIdx_k).fill(0.); // assure 0 for corresponding proposed betas with 0

    // prior ratio of beta
    double logPriorBetaRatio = 0.;
    logPriorBetaRatio += logPDFNormal(proposedBeta(1+updateIdx,singleIdx_k), tauSq_[componentUpdateIdx]);
    logPriorBetaRatio -= logPDFNormal(betas_(1+updateIdx,singleIdx_k), tauSq_[componentUpdateIdx]);

    // compute logLikelihoodRatio, i.e. proposedLikelihood - log_likelihood
    arma::vec proposedLikelihood = log_likelihood_;
    // loglikelihood( xi_, zetas_, betas_, kappa_, proportion_model, dataclass, log_likelihood_ );
    loglikelihood( xi_, zetas_, proposedBeta, kappa_, proportion_model, dataclass, proposedLikelihood );

    double logLikelihoodRatio = arma::sum(proposedLikelihood - log_likelihood_);

    // Here we need always compute the proposal and original ratios, in particular the likelihood, since betas are updated
    //logProposalGammaRatio = arma::accu(proposedGammaPrior - logP_gamma);
    double logAccProb = logLikelihoodRatio +
                        logPriorGammaRatio +
                        logPriorBetaRatio +
                        logProposalRatio;
    /*
    // std::cout << "...debug logAccProb=" <<  logAccProb <<
    "; logProposalGammaRatio=" << logProposalGammaRatio <<
    "; logPriorBetaRatio=" << logPriorBetaRatio <<
    "; logLikelihoodRatio=" << logLikelihoodRatio <<
    "; logProposalRatio=" << logProposalRatio <<
    "; logProposalBetaRatio=" << logProposalBetaRatio <<
    "; logPosteriorBeta=" << logPosteriorBeta <<
    "; logPosteriorBeta_proposal=" << logPosteriorBeta_proposal <<
    "\n";
    if(logAccProb == 0.)
        // std::cout << "........updateIdx=" << updateIdx.t() <<
    "; gammas(updateIdx, k)=" <<  gammas_.submat(updateIdx, singleIdx_k).t() <<
    "; proposedGamma(updateIdx, k)=" << proposedGamma.submat(updateIdx, singleIdx_k).t() <<
    "\n";
    */
    if( std::log(R::runif(0,1)) < logAccProb )
    {
        gammas_ = proposedGamma;
        if( gamma_prior == Gamma_Prior_Type::bernoulli )
        {
            logP_gamma_ = proposedGammaPrior;
            // pi[componentUpdateIdx] = pi_proposed;

        }
        log_likelihood_ = proposedLikelihood;
        betas_ = proposedBeta;
        // logPosteriorBeta = logPosteriorBeta_proposal;

        ++gamma_acc_count_;
    }

    // after A/R, update bandit Related variables
    if( gamma_sampler == Gamma_Sampler_Type::bandit )
    {
        // banditLimit to control the beta prior with relatively large variance
        double banditLimit = (double)(log_likelihood_.n_elem);
        double banditIncrement = 1.;

        for(auto iter: updateIdx)
            // for(arma::uvec::iterator iter = updateIdx.begin(); iter != updateIdx.end(); ++iter)
        {
            // FINITE UPDATE
            if( banditAlpha(iter,componentUpdateIdx) + banditBeta(iter,componentUpdateIdx) <= banditLimit )
                // if( banditAlpha(*iter,componentUpdateIdx) + banditBeta(*iter,componentUpdateIdx) <= banditLimit )
            {
                banditAlpha(iter,componentUpdateIdx) += banditIncrement * gammas_(iter,componentUpdateIdx);
                // banditAlpha(*iter,componentUpdateIdx) += banditIncrement * gammas_(*iter,componentUpdateIdx);
                banditBeta(iter,componentUpdateIdx) += banditIncrement * (1-gammas_(iter,componentUpdateIdx));
                // banditBeta(*iter,componentUpdateIdx) += banditIncrement * (1-gammas_(*iter,componentUpdateIdx));
            }

            // // CONTINUOUS UPDATE, alternative to the above, at most one has to be uncommented

            // banditAlpha(*iter,componentUpdateIdx) += banditIncrement * gamma(*iter,componentUpdateIdx);
            // banditBeta(*iter,componentUpdateIdx) += banditIncrement * (1-gamma(*iter,componentUpdateIdx));

            // // renormalise
            // if( banditAlpha(*iter,componentUpdateIdx) + banditBeta(*iter) > banditLimit )
            // {
            //     banditAlpha(*iter,componentUpdateIdx) = banditLimit * ( banditAlpha(*iter,componentUpdateIdx) / ( banditAlpha(*iter,componentUpdateIdx) + banditBeta(*iter,componentUpdateIdx) ));
            //     banditBeta(*iter,componentUpdateIdx) = banditLimit * (1. - ( banditAlpha(*iter,componentUpdateIdx) / ( banditAlpha(*iter,componentUpdateIdx) + banditBeta(*iter,componentUpdateIdx) )) );
            // }

        }
    }
    // free(hyperpar);

    // return gammas_;
}


void BVS_Sampler::sampleEta(
    arma::umat& etas_,
    Eta_Prior_Type eta_prior,
    Eta_Sampler_Type eta_sampler,
    arma::mat& logP_eta_,
    unsigned int& eta_acc_count_,
    arma::vec& log_likelihood_,
    const std::string& rw_mh,
    double sigmaMH_zeta,

    const armsParmClass& armsPar,
    void *hyperpar_,
    arma::mat& zetas_,
    const arma::mat& betas_,
    const arma::vec& xi_,
    double kappa_,
    double w0Sq_,
    arma::vec wSq_,
    double rho0,

    bool dirichlet,

    // double& logPosteriorZeta,
    arma::vec& datTheta,
    const arma::mat& weibullS,
    arma::mat& weibullLambda,
    const DataClass &dataclass)
{

    // reallocate struct variables
    // hyperparS *hyperpar = (hyperparS *)calloc(sizeof(hyperparS), sizeof(hyperparS));
    std::unique_ptr<hyperparS> hyperpar = std::make_unique<hyperparS>();
    *hyperpar = *(hyperparS *)hyperpar_;

    arma::umat proposedEta = etas_; // copy the original etas and later change the address of the copied one
    arma::mat proposedEtaPrior;
    arma::uvec updateIdx;

    double logProposalRatio = 0;

    unsigned int p = etas_.n_rows;
    unsigned int L = etas_.n_cols;
    unsigned int n = datTheta.n_elem;

    // define static variables for global updates for the use of bandit algorithm
    static arma::mat banditAlpha2 = arma::mat(p, L, arma::fill::value(0.5));
    static arma::mat banditBeta2 = arma::mat(p, L, arma::fill::value(0.5));

    // decide on one component
    unsigned int componentUpdateIdx = static_cast<unsigned int>( R::runif( 0, L ) );
    arma::uvec singleIdx_k = { componentUpdateIdx };

    // Update the proposed Eta with 'updateIdx' renewed via its address
    //if ( gamma_sampler_bandit )
    switch( eta_sampler )
    {
    case Eta_Sampler_Type::bandit:
        logProposalRatio += etaBanditProposal( p, proposedEta, etas_, updateIdx, componentUpdateIdx, banditAlpha2 );
        break;

    case Eta_Sampler_Type::mc3:
        logProposalRatio += gammaMC3Proposal( p, proposedEta, etas_, updateIdx, componentUpdateIdx );
        break;
    }

    //// use M-H sampler instead of Gibbs for variable selection indicator's Bernoulli probability
    // static arma::mat rho = arma::mat(p, L, arma::fill::value(0.01));

    // note only one outcome is updated
    // update log probabilities

    double logPriorEtaRatio = 0.;
    // // std::cout << "BVS.cpp ... eta_prior=" << static_cast<std::underlying_type<Eta_Prior_Type>::type>(eta_prior)  << "\n";
    switch(eta_prior)
    {
    case Eta_Prior_Type::bernoulli:
    {
        proposedEtaPrior = logP_eta_; // copy the original one and later change the address of the copied one
        // update corresponding Bernoulli probabilities (allow cluster-specific sparsity)
        // rho_proposed = R::rbeta(hyperpar->rhoA + (double)(arma::accu(proposedEta.col(componentUpdateIdx))),
        //                       hyperpar->rhoB + (double)(p) - (double)(arma::accu(proposedEta.col(componentUpdateIdx))));
        
        double rho = rho0;
        for(auto i: updateIdx)
        {   // This chunk is only valid for beta-Bernoulli prior
            if(rho0 == 0.)
            {
                //// feature-specific Bernoulli probability
                // rho = R::rbeta(hyperpar->rhoA + (double)(etas_(i,componentUpdateIdx)),
                //                 hyperpar->rhoB + 1.0 - (double)(etas_(i,componentUpdateIdx)));
                //// column-specific Bernoulli probability
                // rho = R::rbeta(hyperpar->rhoA + (double)(arma::sum(etas_.col(componentUpdateIdx))),
                //                 hyperpar->rhoB + (double)(p) - (double)(arma::sum(etas_.col(componentUpdateIdx))));
                //// row-specific Bernoulli probability
                rho = R::rbeta(hyperpar->rhoA + (double)(arma::accu(etas_.row(i))),
                                    hyperpar->rhoB + (double)(L) - (double)(arma::accu(etas_.row(i))));
            }
            //// the following random-walk MH sampler result in increasing rho 
            // double proposedRho = std::exp( std::log( rho(i, componentUpdateIdx) ) + R::rnorm(0.0, 1.0) );
            // if( proposedRho <= 1.0 )
            // {
            //     double proposedRhoPrior = logPDFBeta( proposedRho, hyperpar->rhoA, hyperpar->rhoB );
            //     double logP_rho = logPDFBeta( rho(i, componentUpdateIdx), hyperpar->rhoA, hyperpar->rhoB );
            //     double proposedEtaPrior0 = logPDFBernoulli( proposedEta(i,componentUpdateIdx), proposedRho);
                
            //     // A/R
            //     double logAccProb0 = (proposedRhoPrior + proposedEtaPrior0) - (logP_rho + logP_eta_(i, componentUpdateIdx));
                
            //     if( std::log(R::runif(0,1)) < logAccProb0 )
            //     {
            //         rho(i, componentUpdateIdx) = proposedRho;

            //         logProposalEtaRatio += proposedEtaPrior0 - logP_eta_(i, componentUpdateIdx);

            //         logP_eta_(i, componentUpdateIdx) = proposedEtaPrior0;
            //     }
            // }
            // logProposalEtaRatio += logPDFBernoulli( proposedEta(i,componentUpdateIdx), rho_proposed ) - logPDFBernoulli( etas_(i,componentUpdateIdx), rho[componentUpdateIdx] );
            proposedEtaPrior(i,componentUpdateIdx) = logPDFBernoulli( proposedEta(i,componentUpdateIdx), rho );
            logPriorEtaRatio +=  proposedEtaPrior(i, componentUpdateIdx) - logP_eta_(i, componentUpdateIdx);
        }
        // // std::cout << "debug eta - rho=" << rho << 
        // "; logProposalEtaRatio=" << logProposalEtaRatio <<
        // "; rhoA=" << hyperpar->rhoA << "; rhoA+=" << (double)(arma::accu(proposedEta.col(componentUpdateIdx))) << 
        // "; rhoB=" << hyperpar->rhoB << "; rhoB+=" << (double)(p) - (double)(arma::accu(proposedEta.col(componentUpdateIdx))) << 
        // "\n";
        break;
    }

    case Eta_Prior_Type::mrf:
    {
        arma::umat mrfG(const_cast<unsigned int*>(hyperpar->mrfG_prop), hyperpar->mrfG_prop_edge_n, 2, false);
        arma::vec mrfG_weights(const_cast<double*>(hyperpar->mrfG_prop_weights), hyperpar->mrfG_prop_edge_n, false);
        // update corresponding to MRF prior

        // log-ratio/difference from the first-order term in MRF prior
        logPriorEtaRatio += hyperpar->mrfA_prop * ( (double)(arma::accu(proposedEta.submat(updateIdx, singleIdx_k))) -
                              (double)(arma::accu(etas_.submat(updateIdx, singleIdx_k))) );

        // convert 'updateIdx' from ONE component to its index among multiple components
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
                                           ((double)(proposedEta(mrfG(i, 0))) - (double)(etas_(mrfG(i, 0))));
                }
            }
        }
        break;

    }
    }

    // update other quantities related to acceptance ratio
    arma::mat proposedZeta = zetas_;
    
    // RW-MH
    if (arma::as_scalar(arma::any(proposedEta(updateIdx, singleIdx_k))))
    {
        if (rw_mh != "symmetric")
        {
            // double c = std::exp(a);

            // Compute alphas: n x L
            arma::mat alphas = arma::zeros<arma::mat>(n, L);
            for (unsigned int j = 0; j < L; ++j) {
                // zetas_: (p+1) x L: intercept zeta(0,j); coefficients zeta(1..p, j)
                alphas.col(j) = arma::exp( zetas_(0, j) + dataclass.datX.slice(j) * zetas_.submat(1, j, p, j) );
            }
            arma::vec alpha0 = arma::sum(alphas, 1);  

            const unsigned int l = componentUpdateIdx;  // current component index

            // Weights and C for component l
            arma::vec omega_l = alphas.col(l) / alpha0;
            arma::vec C_l = arma::pow(weibullLambda.col(l), -kappa_) % weibullS.col(l);

            // M_i = sum_j ω_ij C_ij and N_i = sum_j ω_ij S_j(t_i)
            arma::vec M_i = arma::zeros<arma::vec>(n);
            arma::vec N_i = arma::zeros<arma::vec>(n);
            for (unsigned int j = 0; j < L; ++j) {
                arma::vec omega_j = alphas.col(j) / alpha0;
                arma::vec C_j = arma::pow(weibullLambda.col(j), -kappa_) % weibullS.col(j);
                M_i += omega_j % C_j;
                N_i += omega_j % weibullS.col(j);
            }

            // Per-observation scalar weights w_i (likelihood-only score part)
            arma::vec w_i(n);

            // Stabilize division C_l / M_i
            const double eps_div = 1.0e-12;
            arma::vec M_i_safe = M_i + eps_div;

            w_i  = dataclass.datEvent % omega_l % (C_l / M_i_safe - 1.0);                 // δ_i * ω_il * (C_il/M_i - 1)
            w_i += datTheta           % omega_l % (weibullS.col(l) - N_i);                 // θ_i * ω_il * (S_l - N_i)

            // Digamma and Dirichlet-like terms
            arma::vec digamma_alpha0  = Rcpp::digamma(Rcpp::wrap(alpha0));
            arma::vec alpha_l         = alphas.col(l);
            arma::vec digamma_alpha_l = Rcpp::digamma(Rcpp::wrap(alpha_l));

            w_i += alpha_l % (digamma_alpha0 - digamma_alpha_l);                           // α_il [ψ(α_i0) − ψ(α_il)]
            w_i += alpha_l % arma::log(dataclass.datProportionConst.col(l));               // α_il log p̃_il

            // Design matrix for component l, restricted to updated coordinates
            arma::mat X_l = dataclass.datX.slice(l).cols(updateIdx);                       // n x d (d = updateIdx.n_elem)

            // Likelihood-only per-observation gradients: n x d
            arma::mat gradient_zetaK_i0 = X_l.each_col() % w_i;

            // Empirical Fisher (OPG) for the sum log-likelihood: d x d
            arma::mat H = gradient_zetaK_i0.t() * gradient_zetaK_i0;

            // Add prior precision and damping to the diagonal: Δ = H + D_l^{−1} + ν I
            arma::mat Delta_zetaK = H;
            double nu = 0.1;  // damping parameter to ensure positive definiteness
            Delta_zetaK.diag() += 1.0 / wSq_[l] + nu;

            // Enforce symmetry (numerical robustness)
            Delta_zetaK = 0.5 * (Delta_zetaK + Delta_zetaK.t());

            // Invert to get the metric M ≈ (H + D^{-1} + ν I)^{-1}
            arma::mat M;
            if (!arma::inv_sympd(M, Delta_zetaK)) {
                arma::inv(M, Delta_zetaK, arma::inv_opts::allow_approx);
            }

            // Full posterior gradient (sum-likelihood − prior): d-vector
            arma::vec gradient_zetaK = arma::sum(gradient_zetaK_i0, 0).t();                // sum over i
            gradient_zetaK -= zetas_(1 + updateIdx, singleIdx_k) / wSq_[l];                // − ζ_l / w_l^2

            // MALA mean and covariance
            double eps = sigmaMH_zeta;                                                      // ε*
            arma::vec  m     = 0.5 * eps * eps * (M * gradient_zetaK);
            arma::mat  Sigma =        eps * eps * M;

            // Draw MALA proposal increment u ~ N(m, Sigma)
            arma::vec u = randMvNormal(m, Sigma);

            // Update proposed parameters on the selected coordinates
            proposedZeta(1 + updateIdx, singleIdx_k) += u;
            
        } else {//if( arma::any(etas_(updateIdx,singleIdx_k)) ) {
            // (symmetric) random-walk Metropolis with optimal standard deviation O(d^{-1/2})
            arma::vec u = Rcpp::rnorm( updateIdx.n_elem, 0., sigmaMH_zeta * 1. / std::sqrt(updateIdx.n_elem) ); 
            proposedZeta(1+updateIdx, singleIdx_k) += u;
        }
    }
    proposedZeta(1+arma::find(proposedEta.col(componentUpdateIdx) == 0), singleIdx_k).fill(0.); // assure 0 for corresponding proposed betas with 0

    // prior ratio of zeta
    double logPriorZetaRatio = 0.;
    logPriorZetaRatio += logPDFNormal(proposedZeta(1+updateIdx,singleIdx_k), wSq_[componentUpdateIdx]);
    logPriorZetaRatio -= logPDFNormal(zetas_(1+updateIdx,singleIdx_k), wSq_[componentUpdateIdx]);

    // compute logLikelihoodRatio, i.e. proposedLikelihood - log_likelihood
    arma::vec proposedLikelihood = log_likelihood_;
    // loglikelihood( xi_, zetas_, betas_, kappa_, true, dataclass, log_likelihood_ );
    loglikelihood( xi_, proposedZeta, betas_, kappa_, true, dataclass, proposedLikelihood );

    double logLikelihoodRatio = arma::sum(proposedLikelihood - log_likelihood_);

    // Here we need always compute the proposal and original ratios, in particular the likelihood, since betas are updated
    double logAccProb = logLikelihoodRatio +
                        logPriorEtaRatio +
                        logPriorZetaRatio +
                        logProposalRatio;

    if( std::log(R::runif(0,1)) < logAccProb )
    {
        etas_ = proposedEta;
        if( eta_prior == Eta_Prior_Type::bernoulli )
        {
            logP_eta_ = proposedEtaPrior;
            // rho[componentUpdateIdx] = rho_proposed;

        }
        log_likelihood_ = proposedLikelihood;
        zetas_ = proposedZeta;
        // logPosteriorZeta = logPosteriorZeta_proposal;

        ++eta_acc_count_;
    }

    // after A/R, update bandit Related variables
    if( eta_sampler == Eta_Sampler_Type::bandit )
    {
        double banditLimit = (double)(log_likelihood_.n_elem);
        double banditIncrement = 1.;

        for(auto iter: updateIdx)
            // for(arma::uvec::iterator iter = updateIdx.begin(); iter != updateIdx.end(); ++iter)
        {
            // FINITE UPDATE
            if( banditAlpha2(iter,componentUpdateIdx) + banditBeta2(iter,componentUpdateIdx) <= banditLimit )
                // if( banditAlpha2(*iter,componentUpdateIdx) + banditBeta2(*iter,componentUpdateIdx) <= banditLimit )
            {
                banditAlpha2(iter,componentUpdateIdx) += banditIncrement * etas_(iter,componentUpdateIdx);
                // banditAlpha2(*iter,componentUpdateIdx) += banditIncrement * etas_(*iter,componentUpdateIdx);
                banditBeta2(iter,componentUpdateIdx) += banditIncrement * (1-etas_(iter,componentUpdateIdx));
                // banditBeta2(*iter,componentUpdateIdx) += banditIncrement * (1-etas_(*iter,componentUpdateIdx));
            }

            // // CONTINUOUS UPDATE, alternative to the above, at most one has to be uncommented

            // banditAlpha(*iter,componentUpdateIdx) += banditIncrement * gamma(*iter,componentUpdateIdx);
            // banditBeta(*iter,componentUpdateIdx) += banditIncrement * (1-gamma(*iter,componentUpdateIdx));

            // // renormalise
            // if( banditAlpha(*iter,componentUpdateIdx) + banditBeta(*iter) > banditLimit )
            // {
            //     banditAlpha(*iter,componentUpdateIdx) = banditLimit * ( banditAlpha(*iter,componentUpdateIdx) / ( banditAlpha(*iter,componentUpdateIdx) + banditBeta(*iter,componentUpdateIdx) ));
            //     banditBeta(*iter,componentUpdateIdx) = banditLimit * (1. - ( banditAlpha(*iter,componentUpdateIdx) / ( banditAlpha(*iter,componentUpdateIdx) + banditBeta(*iter,componentUpdateIdx) )) );
            // }

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
    //arma::umat mutantGamma = gammas_;
    unsigned int n_updates_MC3 = std::max(5., std::ceil( (double)(p) / 5. )); //arbitrary number, should I use something different?
    // unsigned int n_updates_MC3 = (p>5)? 5 : (p-1);
    //TODO: re-run all high-dimensional cases to check if upto 20 covariates result in similar results as before
    // int n_updates_MC3 = (p>3)? 3 : (p-1);//std::ceil( p / 40 );
    // n_updates_MC3 = std::min( std::max(5, n_updates_MC3), 20); // For super large p, only update 20 variables each time
    // n_updates_MC3 = (n_updates_MC3 > p)? p : n_updates_MC3;
    /*
    updateIdx = arma::uvec( n_updates_MC3 );

    for( int i=0; i<n_updates_MC3; ++i)
    {
        updateIdx(i) = static_cast<unsigned int>( R::runif( 0, p ) );    // note that I might be updating multiple times the same coeff
    }
    */
    //arma::uvec entireIdx = arma::linspace<arma::uvec>( 0, p - 1, p );
    Rcpp::IntegerVector entireIdx = Rcpp::seq( 0, p - 1);
    updateIdx = Rcpp::as<arma::uvec>(Rcpp::sample(entireIdx, n_updates_MC3, false)); // here 'replace = false'

    for( auto i : updateIdx)
    {
        mutantGamma(i,componentUpdateIdx_) = ( R::runif(0,1) < 0.5 )? gammas_(i,componentUpdateIdx_) : 1-gammas_(i,componentUpdateIdx_); // could simply be ( 0.5 ? 1 : 0) ;
    }

    //return mutantGamma ;
    return 0. ; // pass this to the outside, it's the (symmetric) logProposalRatio
}

// sampler for proposed updates on gammas_
double BVS_Sampler::gammaBanditProposal(
    unsigned int p,
    arma::umat& mutantGamma,
    const arma::umat gammas_,
    arma::uvec& updateIdx,
    unsigned int componentUpdateIdx_,
    arma::mat& banditAlpha )
{
    // define static variables for global updates
    // 'banditZeta' corresponds to pi in GPTCM
    static arma::vec banditZeta = arma::vec(p);
    /*
    static arma::mat banditAlpha = arma::mat(gammas_.n_rows, gammas_.n_cols, arma::fill::value(0.5));
    // banditAlpha.fill( 0.5 );
    static arma::mat banditBeta = arma::mat(gammas_.n_rows, gammas_.n_cols, arma::fill::value(0.5));
    // banditBeta.fill( 0.5 );
    */
    static arma::vec mismatch = arma::vec(p);
    static arma::vec normalised_mismatch = arma::vec(p);
    static arma::vec normalised_mismatch_backwards = arma::vec(p);

    unsigned int n_updates_bandit = 4; // this needs to be low as its O(n_updates!)
    // banditLimit = (double)N;
    // banditIncrement = 1.;


    //int nOutcomes = gammas_.n_cols;
    //arma::umat mutantGamma = gammas_;
    double logProposalRatio = 0.;

    for(unsigned int j=0; j<p; ++j)
    {
        // Sample Zs (only for relevant component)
        banditZeta(j) = R::rbeta(banditAlpha(j,componentUpdateIdx_),banditAlpha(j,componentUpdateIdx_));

        // Create mismatch (only for relevant outcome)
        mismatch(j) = (mutantGamma(j,componentUpdateIdx_)==0)?(banditZeta(j)):(1.-banditZeta(j));   //mismatch
    }

    // Normalise
    // mismatch = arma::log(mismatch); //logscale ??? TODO
    // normalised_mismatch = mismatch - Utils::logspace_add(mismatch);

    // normalised_mismatch = mismatch / arma::as_scalar(arma::sum(mismatch));
    normalised_mismatch = mismatch / arma::sum(mismatch);

    if( R::runif(0,1) < 0.5 )   // one deterministic update
    {
        // Decide which to update
        updateIdx = arma::zeros<arma::uvec>(1);
        //updateIdx(0) = randWeightedIndexSampleWithoutReplacement(p,normalised_mismatch); // sample the one
        updateIdx(0) = randWeightedIndexSampleWithoutReplacement(normalised_mismatch); // sample the one

        // Update
        mutantGamma(updateIdx(0),componentUpdateIdx_) = 1 - gammas_(updateIdx(0),componentUpdateIdx_); // deterministic, just switch

        // Compute logProposalRatio probabilities
        normalised_mismatch_backwards = mismatch;
        normalised_mismatch_backwards(updateIdx(0)) = 1. - normalised_mismatch_backwards(updateIdx(0)) ;

        // normalised_mismatch_backwards = normalised_mismatch_backwards - Utils::logspace_add(normalised_mismatch_backwards);
        // normalised_mismatch_backwards = normalised_mismatch_backwards / arma::as_scalar(arma::sum(normalised_mismatch_backwards));
        normalised_mismatch_backwards = normalised_mismatch_backwards / arma::sum(normalised_mismatch_backwards);

        logProposalRatio = ( std::log( normalised_mismatch_backwards(updateIdx(0)) ) ) -
                           ( std::log( normalised_mismatch(updateIdx(0)) ) );

    }
    else
    {
        /*
        n_updates_bandit random (bern) updates
        Note that we make use of column indexing here for armadillo matrices
        */

        // logProposalRatio = 0.;
        // Decide which to update
        updateIdx = arma::zeros<arma::uvec>(n_updates_bandit);
        updateIdx = randWeightedIndexSampleWithoutReplacement(p,normalised_mismatch,n_updates_bandit); // sample n_updates_bandit indexes

        normalised_mismatch_backwards = mismatch; // copy for backward proposal

        // Update
        for(unsigned int i=0; i<n_updates_bandit; ++i)
        {
            // mutantGamma(updateIdx(i),componentUpdateIdx_) = static_cast<unsigned int>(R::rbinom( 1, banditZeta(updateIdx(i)))); // random update
            unsigned int j = R::rbinom( 1, banditZeta(updateIdx(i))); // random update
            mutantGamma(updateIdx(i),componentUpdateIdx_) = j;

            normalised_mismatch_backwards(updateIdx(i)) = 1.- normalised_mismatch_backwards(updateIdx(i));

            logProposalRatio += logPDFBernoulli(gammas_(updateIdx(i),componentUpdateIdx_),banditZeta(updateIdx(i))) -
                                logPDFBernoulli(mutantGamma(updateIdx(i),componentUpdateIdx_),banditZeta(updateIdx(i)));
        }

        // Compute logProposalRatio probabilities
        // normalised_mismatch_backwards = normalised_mismatch_backwards - Utils::logspace_add(normalised_mismatch_backwards);
        // normalised_mismatch_backwards = normalised_mismatch_backwards / arma::as_scalar(arma::sum(normalised_mismatch_backwards));
        normalised_mismatch_backwards = normalised_mismatch_backwards / arma::sum(normalised_mismatch_backwards);

        logProposalRatio += logPDFWeightedIndexSampleWithoutReplacement(normalised_mismatch_backwards,updateIdx) -
                            logPDFWeightedIndexSampleWithoutReplacement(normalised_mismatch,updateIdx);
    }

    return logProposalRatio; // pass this to the outside
    //return mutantGamma;

}

// sampler for proposed updates on etas_
double BVS_Sampler::etaBanditProposal(
    unsigned int p,
    arma::umat& mutantEta,
    const arma::umat etas_,
    arma::uvec& updateIdx,
    unsigned int componentUpdateIdx_,
    arma::mat& banditAlpha2)
{
    // define static variables for global updates
    // 'banditZeta2' corresponds to rho in GPTCM
    static arma::vec banditZeta2 = arma::vec(p);
    /*
    static arma::mat banditAlpha = arma::mat(gammas_.n_rows, gammas_.n_cols, arma::fill::value(0.5));
    // banditAlpha.fill( 0.5 );
    static arma::mat banditBeta = arma::mat(gammas_.n_rows, gammas_.n_cols, arma::fill::value(0.5));
    // banditBeta.fill( 0.5 );
    */
    static arma::vec mismatch2 = arma::vec(p);
    static arma::vec normalised_mismatch2 = arma::vec(p);
    static arma::vec normalised_mismatch_backwards2 = arma::vec(p);

    unsigned int n_updates_bandit = 4; // this needs to be low as its O(n_updates!)
    // banditLimit = (double)N;
    // banditIncrement = 1.;

    double logProposalRatio = 0.;

    for(unsigned int j=0; j<p; ++j)
    {
        // Sample Zs (only for relevant component)
        banditZeta2(j) = R::rbeta(banditAlpha2(j,componentUpdateIdx_),banditAlpha2(j,componentUpdateIdx_));

        // Create mismatch (only for relevant outcome)
        mismatch2(j) = (mutantEta(j,componentUpdateIdx_)==0)?(banditZeta2(j)):(1.-banditZeta2(j));   //mismatch
    }

    // Normalise
    // mismatch = arma::log(mismatch); //logscale ??? TODO
    // normalised_mismatch = mismatch - Utils::logspace_add(mismatch);

    // normalised_mismatch2 = mismatch2 / arma::as_scalar(arma::sum(mismatch2));
    normalised_mismatch2 = mismatch2 /arma::sum(mismatch2);

    // Use "ε-Greedy Strategy for Bernoulli Bandits". Choose ε=0.5 to balance exploration and exploitation
    if( R::runif(0,1) < 0.5 )   // one deterministic update
    {
        // Decide which to update
        updateIdx = arma::zeros<arma::uvec>(1);
        //updateIdx(0) = randWeightedIndexSampleWithoutReplacement(p,normalised_mismatch); // sample the one
        updateIdx(0) = randWeightedIndexSampleWithoutReplacement(normalised_mismatch2); // sample the one

        // Update
        mutantEta(updateIdx(0),componentUpdateIdx_) = 1 - etas_(updateIdx(0),componentUpdateIdx_); // deterministic, just switch

        // Compute logProposalRatio probabilities
        normalised_mismatch_backwards2 = mismatch2;
        normalised_mismatch_backwards2(updateIdx(0)) = 1. - normalised_mismatch_backwards2(updateIdx(0)) ;

        // normalised_mismatch_backwards = normalised_mismatch_backwards - Utils::logspace_add(normalised_mismatch_backwards);
        // normalised_mismatch_backwards2 = normalised_mismatch_backwards2 / arma::as_scalar(arma::sum(normalised_mismatch_backwards2));
        normalised_mismatch_backwards2 = normalised_mismatch_backwards2 / arma::sum(normalised_mismatch_backwards2);

        logProposalRatio = ( std::log( normalised_mismatch_backwards2(updateIdx(0)) ) ) -
                           ( std::log( normalised_mismatch2(updateIdx(0)) ) );

    }
    else
    {
        /*
        n_updates_bandit random (bern) updates
        Note that we make use of column indexing here for armadillo matrices
        */

        // logProposalRatio = 0.;
        // Decide which to update
        updateIdx = arma::zeros<arma::uvec>(n_updates_bandit);
        updateIdx = randWeightedIndexSampleWithoutReplacement(p,normalised_mismatch2,n_updates_bandit); // sample n_updates_bandit indexes

        normalised_mismatch_backwards2 = mismatch2; // copy for backward proposal

        // Update
        for(unsigned int i=0; i<n_updates_bandit; ++i)
        {
            // mutantEta(updateIdx(i),componentUpdateIdx_) = static_cast<unsigned int>(R::rbinom( 1, banditZeta2(updateIdx(i)))); // random update
            unsigned int j = R::rbinom( 1, banditZeta2(updateIdx(i))); // random update
            mutantEta(updateIdx(i),componentUpdateIdx_) = j;

            normalised_mismatch_backwards2(updateIdx(i)) = 1.- normalised_mismatch_backwards2(updateIdx(i));

            logProposalRatio += logPDFBernoulli(etas_(updateIdx(i),componentUpdateIdx_),banditZeta2(updateIdx(i))) -
                                logPDFBernoulli(mutantEta(updateIdx(i),componentUpdateIdx_),banditZeta2(updateIdx(i)));
        }

        // Compute logProposalRatio probabilities
        // normalised_mismatch_backwards = normalised_mismatch_backwards - Utils::logspace_add(normalised_mismatch_backwards);
        // normalised_mismatch_backwards2 = normalised_mismatch_backwards2 / arma::as_scalar(arma::sum(normalised_mismatch_backwards2));
        normalised_mismatch_backwards2 = normalised_mismatch_backwards2 / arma::sum(normalised_mismatch_backwards2);

        logProposalRatio += logPDFWeightedIndexSampleWithoutReplacement(normalised_mismatch_backwards2,updateIdx) -
                            logPDFWeightedIndexSampleWithoutReplacement(normalised_mismatch2,updateIdx);
    }

    return logProposalRatio; // pass this to the outside
    //return mutantGamma;

}

double BVS_Sampler::logPbetaK(
    const unsigned int k,
    const arma::mat& betas,

    const double tauSq,
    const double kappa,
    const arma::vec& datTheta,
    const arma::mat& datProportion,
    const DataClass& dataclass)
{
    double logP = 0.;

    // dimensions
    unsigned int N = dataclass.datX.n_rows;
    unsigned int p = dataclass.datX.n_cols;
    unsigned int L = dataclass.datX.n_slices;

    // compute log density
    double logprior = - arma::sum(betas.col(k) % betas.col(k)) / tauSq / 2.;

    arma::vec logpost_first = arma::zeros<arma::vec>(N);
    // arma::vec logpost_second= arma::zeros<arma::vec>(N);
    double logpost_second_sum = 0;
    for(unsigned int l=0; l<L; ++l)
    {
        arma::vec logMu_l = betas(0,l) + dataclass.datX.slice(l) * betas.submat(1, l, p, l);
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
    const double wSq,
    const double kappa,

    const arma::vec& datTheta,
    const arma::mat& weibullS,
    const arma::mat& weibullLambda,
    const DataClass &dataclass)
{
    double logP = 0.;

    // dimensions
    unsigned int N = dataclass.datX.n_rows;
    unsigned int p = dataclass.datX.n_cols;
    unsigned int L = dataclass.datX.n_slices;

    // update Dirichlet concentrations based on zetas
    arma::mat alphas = arma::zeros<arma::mat>(N, L);

    for(unsigned int l=0; l<L; ++l)
    {
        alphas.col(l) = arma::exp( zetas(0, l) + dataclass.datX.slice(l) * zetas.submat(1, l, p, l) );
    }
    alphas.elem(arma::find(alphas > upperbound3)).fill(upperbound3);
    alphas.elem(arma::find(alphas < lowerbound)).fill(lowerbound);
    arma::vec alphas_Rowsum = arma::sum(alphas, 1);

    // compute log density. NOTE: Subtract intercept
    double logprior = - (arma::accu(zetas.submat(1, k, p, k) % zetas.submat(1, k, p, k))) / wSq / 2.;

    arma::vec logpost_first = arma::zeros<arma::vec>(N);
    arma::vec logpost_second= arma::zeros<arma::vec>(N);
    // double logpost_second_sum = 0;
    for(unsigned int l=0; l<L; ++l)
    {
        arma::vec tmp = alphas.col(l) / alphas_Rowsum %  weibullS.col(l);
        logpost_first += arma::pow(weibullLambda.col(l), - kappa) % tmp;
        logpost_second += tmp;
    }

    double logpost_first_sum = 0.;
    logpost_first_sum = arma::sum( arma::log( logpost_first.elem(arma::find(dataclass.datEvent)) ) );

    double logpost_second_sum = 0.;
    logpost_second_sum = arma::sum(datTheta % logpost_second);

    double log_dirichlet_sum = 0.;
    log_dirichlet_sum = arma::sum(
                            arma::lgamma(alphas_Rowsum) - arma::sum(arma::lgamma(alphas), 1) +
                            arma::sum( (alphas - 1.0) % arma::log(dataclass.datProportionConst), 1 )
                        );

    logP = logprior + logpost_first_sum + logpost_second_sum + log_dirichlet_sum;

    return logP;
}

// subfunctions used for bandit proposal

arma::uvec BVS_Sampler::randWeightedIndexSampleWithoutReplacement(
    unsigned int populationSize,    // size of set sampling from
    const arma::vec& weights,       // (log) probability for each element
    unsigned int sampleSize         // size of each sample
) // sample is a zero-offset indices to selected items, output is the subsampled population.
{
    // note I can do everything in the log scale as the ordering won't change!
    arma::vec tmp = Rcpp::rexp( populationSize, 1. );
    arma::vec score = tmp - weights;
    arma::uvec result = arma::sort_index( score,"ascend" );

    return result.subvec(0,sampleSize-1);
}

// Overload with equal weights
/*
arma::uvec BVS_Sampler::randWeightedIndexSampleWithoutReplacement(
    unsigned int populationSize,    // size of set sampling from
    unsigned int sampleSize         // size of each sample
) // sample is a zero-offset indices to selected items, output is the subsampled population.
{
    // note I can do everything in the log scale as the ordering won't change!
    arma::vec score = Rcpp::rexp( populationSize, 1. );
    arma::uvec result = arma::sort_index( score,"ascend" );

    return result.subvec(0,sampleSize-1);
}
*/

// overload with sampleSize equal to one
unsigned int BVS_Sampler::randWeightedIndexSampleWithoutReplacement(
    const arma::vec& weights     // probability for each element
) // sample is a zero-offset indices to selected items, output is the subsampled population.
{
    // note I can do everything in the log scale as the ordering won't change!

    double u = R::runif(0,1);
    double tmp = weights(0);
    unsigned int t = 0;

    while(u > tmp)
    {
        // tmp = Utils::logspace_add(tmp,logWeights(++t));
        tmp += weights(++t);
    }

    return t;
}

// logPDF rand Weighted Indexes (need to implement the one for the original starting vector?)
double BVS_Sampler::logPDFWeightedIndexSampleWithoutReplacement(
    const arma::vec& weights,
    const arma::uvec& indexes
)
{
    // arma::vec logP_permutation = arma::zeros<arma::vec>((int)std::tgamma(indexes.n_elem+1));  //too big of a vector
    double logP_permutation = 0.;
    double tmp;

    std::vector<unsigned int> v = arma::conv_to<std::vector<unsigned int>>::from(arma::sort(indexes));
    // vector should be sorted at the beginning.

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
            current_weights = current_weights/arma::sum(current_weights(current_permutation));   // this will gets array weights that do not sum to 1 in total, but will only use relevant elements
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

    if(a <= std::numeric_limits<float>::lowest())
        return b;
    if(b <= std::numeric_limits<float>::lowest())
        return a;
    return std::max(a, b) + std::log( (double)(1. + std::exp( (double)-std::abs((double)(a - b)) )));
}

double BVS_Sampler::logPDFBernoulli(unsigned int x, double pi)
{
    if( x > 1 ||  x < 0 )
        return -std::numeric_limits<double>::infinity();
    else
        return (double)(x) * std::log(pi) + (1.-(double)(x)) * std::log(1. - pi);
}
/*
double BVS_Sampler::lBeta(double a,double b)
{    //log beta function
		return std::lgamma(a) + std::lgamma(b) - std::lgamma(a+b);
}

double BVS_Sampler::logPDFBeta(double x, double a, double b)
{
		if( x <= 0. || x >= 1. )
			return -std::numeric_limits<double>::infinity();
		else
			return -lBeta(a,b) + (a-1)*log(x) + (b-1)*log(1-x);
}
*/
double BVS_Sampler::logPDFNormal(const arma::vec& x, const double& sigmaSq)  // zeroMean and independentVar
{
    unsigned int k = x.n_elem;
    double tmp = (double)k * std::log(sigmaSq); // log-determinant(Sigma)

    return -0.5*(double)k*log(2.*M_PI) -0.5*tmp - 0.5 * arma::as_scalar( x.t() * x ) / sigmaSq;

}

arma::vec BVS_Sampler::randMvNormal(
    const arma::vec &m,
    const arma::mat &Sigma)
{
    unsigned int d = m.n_elem;
    //check
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
    // arma::vec res(n);
    // for(unsigned int i=0; i<n; ++i)
    // {
    //     res(i) = R::rnorm( 0., 1. );
    // }

    arma::vec res = Rcpp::rnorm(n);
    return res;
}


/*
double logPDFMRF(const arma::umat& externalGamma, const arma::mat& mrfG, double a, double b )
{
    double logP = 0.;

    // calculate the linear and quadratic parts in MRF by using all edges of G
    arma::vec gammaVec = arma::conv_to< arma::vec >::from(arma::vectorise(externalGamma));
    double quad_mrf = 0.;
    double linear_mrf = 0.;
    //int count_linear_mrf = 0; // If the MRF graph matrix has diagonals 0, count_linear_mrf is always 0.
    for( unsigned i=0; i < mrfG->n_rows; ++i )
    {
        if( (*mrfG)(i,0) != (*mrfG)(i,1) ){
            quad_mrf += 2.0 * gammaVec( (*mrfG)(i,0) ) * gammaVec( (*mrfG)(i,1) ) * (*mrfG)(i,2);
        }else{
                if( gammaVec( (*mrfG)(i,0) ) == 1 ){
                    linear_mrf += (*mrfG)(i,2); // should this be 'linear_mrf += e * (externalMRFG)(i,2)'?
                    //count_linear_mrf ++;
                }
        }
    }
    //logP = arma::as_scalar( linear_mrf + d * (arma::accu( externalGamma ) - count_linear_mrf) + e * 2.0 * quad_mrf );
    // Should logP be the following?
    logP = arma::as_scalar( d * arma::accu( externalGamma ) + e * (linear_mrf + quad_mrf) );

    return logP;
}
 */

// Bandit-sampling related methods
/*
void BVS_Sampler::banditInit(
    unsigned int p,
    unsigned int L,
    unsigned int N
)// initialise all the private memebers
{
    banditZeta = arma::vec(p);

    banditAlpha = arma::mat(p, L);
    banditAlpha.fill( 0.5 );

    banditBeta = arma::mat(p, L);
    banditBeta.fill( 0.5 );

    mismatch = arma::vec(p);
    normalised_mismatch = arma::vec(p);
    normalised_mismatch_backwards = arma::vec(p);

    n_updates_bandit = 4; // this needs to be low as its O(n_updates!)

    banditLimit = (double)N;
    banditIncrement = 1.;
}


void BVS_Sampler::banditInitEta(
    unsigned int p,
    unsigned int L,
    unsigned int N
)// initialise all the private memebers
{
    banditZeta2 = arma::vec(p);

    banditAlpha2 = arma::mat(p, L);
    banditAlpha2.fill( 0.5 );

    banditBeta2 = arma::mat(p, L);
    banditBeta2.fill( 0.5 );

    mismatch2 = arma::vec(p);
    normalised_mismatch2 = arma::vec(p);
    normalised_mismatch_backwards2 = arma::vec(p);

    n_updates_bandit2 = 4; // this needs to be low as its O(n_updates!)

    banditLimit2 = (double)N;
    banditIncrement2 = 1.;
}
*/