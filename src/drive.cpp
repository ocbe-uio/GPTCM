// Main function for the MCMC loop


#include "simple_gibbs.h"
#include "arms_gibbs.h"
#include "BVS.h"
#include "global.h"

#ifdef _OPENMP
 extern omp_lock_t RNGlock; /*defined in global.h*/
 #include <omp.h>
#endif

#include <Rcpp.h>
// [[Rcpp::plugins(openmp)]]

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]


//' Main function implemented in C++ for the MCMC loop
//'
//' @name run_mcmc
//'
//' @param nIter number of MCMC iterations
//' @param burnin length of MCMC burn-in period
//' @param thin number of thinning
//' @param tick an integer used for printing the iteration index
//' @param n number of samples to draw
//' @param nsamp how many samples to draw for generating each sample; only the last draw will be kept
//' @param ninit number of initials as meshgrid values for envelop search
//' @param convex adjustment for convexity (non-negative value, default 1.0)
//' @param npoint maximum number of envelope points
//' @param dirichlet not yet implemented
//' @param proportion_model logical value for modeling the proportions data
//' @param BVS logical value for implementing Bayesian variable selection
//' @param threads maximum threads used for parallelization. Default is 1
//' @param gamma_prior one of \code{c("bernoulli", "MRF")}
//' @param gamma_sampler one of \code{c("mc3", "bandit")}
//' @param eta_prior one of \code{c("bernoulli", "MRF")}
//' @param eta_sampler one of \code{c("mc3", "bandit")}
//' @param initList a list of initial values for parameters "kappa", "xi", "betas", and "zetas"
//' @param rangeList a list of ranges of initial values for parameters "kappa", "xi", "betas", and "zetas"
//' @param hyperparList a list of relevant hyperparameters
//' @param datEvent a vector of survival status
//' @param datTime a vector of survival times
//' @param datX an array of cluster-specific covariates
//' @param datX0 a matrix of mandatory variables
//' @param datProportionConst an array of cluster-specific proportions
//'
// [[Rcpp::export]]
Rcpp::List run_mcmc(
    unsigned int nIter,
    unsigned int burnin,
    unsigned int thin,
    unsigned int tick,
    unsigned int n,
    int nsamp,
    int ninit,
    double convex,
    int npoint,
    bool dirichlet,
    bool proportion_model,
    bool BVS,
    int threads,
    const std::string& gamma_prior,
    const std::string& gamma_sampler,
    const std::string& eta_prior,
    const std::string& eta_sampler,

    const Rcpp::List& initList,
    const Rcpp::List& rangeList,
    const Rcpp::List& hyperparList,

    const arma::uvec& datEvent,
    const arma::vec& datTime,
    const arma::cube& datX,
    const arma::mat& datX0,
    const arma::mat& datProportionConst)
{
    #ifdef _OPENMP
    if( threads == 1 ){
        omp_set_max_active_levels( 1 );
        omp_set_num_threads( 1 );
    } else {
        omp_init_lock(&RNGlock);
        omp_set_max_active_levels( 1 );
        omp_set_num_threads( threads );
    }
    #endif

    // dimensions
    unsigned int N = datX.n_rows;
    unsigned int p = datX.n_cols;
    unsigned int L = datX.n_slices;

    // input constant data sets in a class
    DataClass dataclass(datEvent, datTime, datX, datX0, datProportionConst);

    // arms parameters in a class
    int metropolis = 1;
    armsParmClass armsPar(n, nsamp, ninit, metropolis, convex, npoint,
                          Rcpp::as<double>(rangeList["xiMin"]),
                          Rcpp::as<double>(rangeList["xiMax"]),
                          Rcpp::as<double>(rangeList["zetaMin"]),
                          Rcpp::as<double>(rangeList["zetaMax"]),
                          Rcpp::as<double>(rangeList["kappaMin"]),
                          Rcpp::as<double>(rangeList["kappaMax"]),
                          Rcpp::as<double>(rangeList["betaMin"]),
                          Rcpp::as<double>(rangeList["betaMax"]));

    // hyperparameters
    hyperparS *hyperpar = (hyperparS *)malloc(sizeof (hyperparS));

    arma::umat mrfG;
    arma::vec mrfG_weights;
    if ( gamma_prior == "mrf" )
    {
        hyperpar->mrfA = Rcpp::as<double>(hyperparList["mrfA"]);
        hyperpar->mrfB = Rcpp::as<double>(hyperparList["mrfB"]);
        mrfG = Rcpp::as<arma::umat>(hyperparList["mrfG"]);
        mrfG_weights = Rcpp::as<arma::vec>(hyperparList["mrfG.weights"]);
        hyperpar->mrfG = mrfG.memptr();
        hyperpar->mrfG_weights = mrfG_weights.memptr();
        hyperpar->mrfG_edge_n = mrfG.n_rows;
    }
    else
    {
        hyperpar->piA = Rcpp::as<double>(hyperparList["piA"]);
        hyperpar->piB = Rcpp::as<double>(hyperparList["piB"]);
    }

    arma::umat mrfG_prop;
    arma::vec mrfG_prop_weights;
    if ( eta_prior == "mrf" )
    {
        hyperpar->mrfA_prop = Rcpp::as<double>(hyperparList["mrfA.prop"]);
        hyperpar->mrfB_prop = Rcpp::as<double>(hyperparList["mrfB.prop"]);
        mrfG_prop = Rcpp::as<arma::umat>(hyperparList["mrfG.prop"]);
        mrfG_prop_weights = Rcpp::as<arma::vec>(hyperparList["mrfG.prop.weights"]);
        hyperpar->mrfG_prop = mrfG_prop.memptr();
        hyperpar->mrfG_prop_weights = mrfG_prop_weights.memptr();
        hyperpar->mrfG_prop_edge_n = mrfG_prop.n_rows;
    }
    else
    {
        hyperpar->rhoA = Rcpp::as<double>(hyperparList["rhoA"]);
        hyperpar->rhoB = Rcpp::as<double>(hyperparList["rhoB"]);
    }

    // NOTE:
    // pi and rho are now latent variables.
    // The fixed hyperparameters hyperparList["pi"] and hyperparList["rho"]
    // are no longer used here.

    double vSq = Rcpp::as<double>(hyperparList["vSq"]);
    hyperpar->vA = Rcpp::as<double>(hyperparList["vA"]);
    hyperpar->vB = Rcpp::as<double>(hyperparList["vB"]);

    double v0Sq = Rcpp::as<double>(hyperparList["v0Sq"]);
    hyperpar->v0A = Rcpp::as<double>(hyperparList["v0A"]);
    hyperpar->v0B = Rcpp::as<double>(hyperparList["v0B"]);

    arma::vec tauSq = Rcpp::as<arma::vec>(hyperparList["tauSq"]);
    hyperpar->tauA = Rcpp::as<double>(hyperparList["tauA"]);
    hyperpar->tauB = Rcpp::as<double>(hyperparList["tauB"]);

    double tau0Sq = Rcpp::as<double>(hyperparList["tau0Sq"]);
    hyperpar->tau0A = Rcpp::as<double>(hyperparList["tau0A"]);
    hyperpar->tau0B = Rcpp::as<double>(hyperparList["tau0B"]);

    arma::vec wSq = Rcpp::as<arma::vec>(hyperparList["wSq"]);
    hyperpar->wA = Rcpp::as<double>(hyperparList["wA"]);
    hyperpar->wB = Rcpp::as<double>(hyperparList["wB"]);
    hyperpar->w0IGamma = Rcpp::as<bool>(hyperparList["w0IGamma"]);

    double w0Sq = Rcpp::as<double>(hyperparList["w0Sq"]);
    hyperpar->w0A = Rcpp::as<double>(hyperparList["w0A"]);
    hyperpar->w0B = Rcpp::as<double>(hyperparList["w0B"]);

    hyperpar->kappaA = Rcpp::as<double>(hyperparList["kappaA"]);
    hyperpar->kappaB = Rcpp::as<double>(hyperparList["kappaB"]);
    hyperpar->kappaIGamma = Rcpp::as<bool>(hyperparList["kappaIGamma"]);

    // Gamma sampler
    Gamma_Sampler_Type gammaSampler;
    if ( gamma_sampler == "bandit" )
        gammaSampler = Gamma_Sampler_Type::bandit;
    else if ( gamma_sampler == "mc3" )
        gammaSampler = Gamma_Sampler_Type::mc3;
    else
    {
        Rprintf("ERROR: Wrong type of Gamma Sampler given!");
        return 1;
    }

    Gamma_Prior_Type gammaPrior;
    if ( gamma_prior == "bernoulli" )
        gammaPrior = Gamma_Prior_Type::bernoulli;
    else if ( gamma_prior == "mrf" )
        gammaPrior = Gamma_Prior_Type::mrf;
    else
    {
        Rprintf("ERROR: Wrong type of Gamma Prior given!");
        return 1;
    }

    // Eta sampler
    Eta_Sampler_Type etaSampler;
    if ( eta_sampler == "bandit" )
        etaSampler = Eta_Sampler_Type::bandit;
    else if ( eta_sampler == "mc3" )
        etaSampler = Eta_Sampler_Type::mc3;
    else
    {
        Rprintf("ERROR: Wrong type of Eta Sampler given!");
        return 1;
    }

    Eta_Prior_Type etaPrior;
    if ( eta_prior == "bernoulli" )
        etaPrior = Eta_Prior_Type::bernoulli;
    else if ( eta_prior == "mrf" )
        etaPrior = Eta_Prior_Type::mrf;
    else
    {
        Rprintf("ERROR: Wrong type of Eta Prior given!");
        return 1;
    }

    // initial values of key parameters
    arma::vec xi = Rcpp::as<arma::vec>(initList["xi"]);
    arma::mat zetas = Rcpp::as<arma::mat>(initList["zetas"]);
    arma::mat betas = Rcpp::as<arma::mat>(initList["betas"]);
    double kappa = Rcpp::as<double>(initList["kappa"]);

    unsigned int nIter_thin = nIter / thin;

    // initializing mcmc results
    arma::vec vSq_mcmc = arma::zeros<arma::vec>(1+nIter_thin);
    vSq_mcmc[0] = vSq;

    arma::mat xi_mcmc = arma::zeros<arma::mat>(1+nIter_thin, xi.n_elem);
    xi_mcmc.row(0) = xi.t();

    arma::vec wSq_mcmc = arma::zeros<arma::vec>(1+nIter_thin);
    wSq_mcmc[0] = wSq[0];

    arma::mat zeta_mcmc = arma::zeros<arma::mat>(1+nIter_thin, (p+1)*L);
    zeta_mcmc.row(0) = arma::vectorise(zetas).t();

    arma::vec tauSq_mcmc = arma::zeros<arma::vec>(1+nIter_thin);
    tauSq_mcmc[0] = tauSq[0];

    arma::mat beta_mcmc = arma::zeros<arma::mat>(1+nIter_thin, (p+1)*L);
    beta_mcmc.row(0) = arma::vectorise(betas).t();

    arma::vec kappa_mcmc = arma::zeros<arma::vec>(1+nIter_thin);
    kappa_mcmc[0] = kappa;

    // ============================================================
    // Gamma variable selection quantities
    // ============================================================

    arma::umat gammas = arma::ones<arma::umat>(p, L);
    arma::umat gamma_mcmc;
    arma::mat logP_gamma;
    unsigned int gamma_acc_count = 0;

    // Latent component-specific Bernoulli probabilities for gammas
    arma::vec pi = arma::zeros<arma::vec>(L);
    arma::mat pi_mcmc;
    arma::vec pi_post = arma::zeros<arma::vec>(L);

    // Marginal likelihood for AIS
    arma::vec logZ_gamma = arma::vec(L, arma::fill::value(std::numeric_limits<double>::quiet_NaN()));
    arma::vec logZ_eta   = arma::vec(L, arma::fill::value(std::numeric_limits<double>::quiet_NaN()));

    if(BVS)
    {
        logP_gamma = arma::zeros<arma::mat>(p, L);
        gamma_acc_count = 0;

        for(unsigned int l=0; l<L; ++l)
        {
            double pi_init;

            if(gamma_prior == "mrf")
            {
                pi_init = 1. / (1. + std::exp(- hyperpar->mrfA));
            }
            else
            {
                pi[l] = R::rbeta(hyperpar->piA, hyperpar->piB);
                pi_init = pi[l];
            }

            for(unsigned int j=0; j<p; ++j)
            {
                gammas(j, l) = R::rbinom(1, pi_init);
                logP_gamma(j, l) = BVS_Sampler::logPDFBernoulli(gammas(j, l), pi_init);
            }
        }

        gamma_mcmc = arma::zeros<arma::umat>(1+nIter_thin, p*L);
        gamma_mcmc.row(0) = arma::vectorise(gammas).t();

        if(gammaPrior == Gamma_Prior_Type::bernoulli)
        {
            pi_mcmc = arma::zeros<arma::mat>(1+nIter_thin, L);
            pi_mcmc.row(0) = pi.t();
        }
    }

    // ============================================================
    // Eta variable selection quantities
    // ============================================================

    arma::umat etas;
    arma::umat eta_mcmc;
    arma::mat logP_eta;
    unsigned int eta_acc_count = 0;

    // Latent component-specific Bernoulli probabilities for etas
    arma::vec rho = arma::zeros<arma::vec>(L);
    arma::mat rho_mcmc;
    arma::vec rho_post = arma::zeros<arma::vec>(L);

    if(BVS)
    {
        logP_eta = arma::zeros<arma::mat>(p, L);
        eta_acc_count = 0;

        etas = arma::zeros<arma::umat>(p, L);

        if(proportion_model)
        {
            for(unsigned int l=0; l<L; ++l)
            {
                double rho_init;

                if(eta_prior == "mrf")
                {
                    rho_init = 1. / (1. + std::exp(- hyperpar->mrfA_prop));
                }
                else
                {
                    rho[l] = R::rbeta(hyperpar->rhoA, hyperpar->rhoB);
                    rho_init = rho[l];
                }

                for(unsigned int j=0; j<p; ++j)
                {
                    etas(j, l) = R::rbinom(1, rho_init);
                    logP_eta(j, l) = BVS_Sampler::logPDFBernoulli(etas(j, l), rho_init);
                }
            }

            eta_mcmc = arma::zeros<arma::umat>(1+nIter_thin, p*L);
            eta_mcmc.row(0) = arma::vectorise(etas).t();

            if(etaPrior == Eta_Prior_Type::bernoulli)
            {
                rho_mcmc = arma::zeros<arma::mat>(1+nIter_thin, L);
                rho_mcmc.row(0) = rho.t();
            }
        }
    }
    else
    {
        etas = arma::ones<arma::umat>(p, L);
    }

    // quantity 1
    arma::vec datTheta = arma::zeros<arma::vec>(N);
    arma::vec logTheta = dataclass.datX0 * xi;
    logTheta.elem(arma::find(logTheta > upperbound)).fill(upperbound);
    datTheta = arma::exp( logTheta );

    // quantity 2
    arma::mat datMu = arma::zeros<arma::mat>(N, L);
    for(unsigned int l=0; l<L; ++l)
    {
        arma::vec logMu_l = betas(0, l) + dataclass.datX.slice(l) * betas.submat(1, l, p, l);
        logMu_l.elem(arma::find(logMu_l > upperbound)).fill(upperbound);
        datMu.col(l) = arma::exp( logMu_l );
    }

    // quantity 3
    arma::mat datProportion = dataclass.datProportionConst;
    if(proportion_model)
    {
        arma::mat alphas = arma::zeros<arma::mat>(N, L);

        for(unsigned int l=0; l<L; ++l)
        {
            alphas.col(l) = arma::exp( zetas(0, l) + dataclass.datX.slice(l) * zetas.submat(1, l, p, l) );
        }

        alphas.elem(arma::find(alphas > upperbound3)).fill(upperbound3);
        alphas.elem(arma::find(alphas < lowerbound)).fill(lowerbound);
        datProportion = alphas / arma::repmat(arma::sum(alphas, 1), 1, L);
    }

    arma::mat weibullS = arma::zeros<arma::mat>(N, L);
    arma::mat weibullLambda = arma::zeros<arma::mat>(N, L);

    for(unsigned int l=0; l<L; ++l)
    {
        arma::vec lambdas = arma::pow( dataclass.datTime / (datMu.col(l) / std::tgamma(1. + 1./kappa)), kappa );
        lambdas.elem(arma::find(lambdas > upperbound)).fill(upperbound);
        weibullS.col(l) = arma::exp( -lambdas );
        weibullLambda.col(l) = datMu.col(l) / std::tgamma(1.0+1.0/kappa);
    }

    // initializing posterior mean
    double kappa_post = 0.;
    arma::vec xi_post = arma::zeros<arma::vec>(arma::size(xi));
    arma::mat zeta_post = arma::zeros<arma::mat>(arma::size(zetas));
    arma::mat beta_post = arma::zeros<arma::mat>(arma::size(betas));
    arma::umat gamma_post = arma::zeros<arma::umat>(arma::size(gammas));
    arma::umat eta_post = arma::zeros<arma::umat>(arma::size(etas));

    arma::mat loglikelihood_mcmc = arma::zeros<arma::mat>(1+nIter_thin, N);
    arma::vec log_likelihood;

    BVS_Sampler::loglikelihood(
        xi,
        zetas,
        betas,
        etas,
        gammas,
        kappa,

        proportion_model,
        dataclass,
        log_likelihood
    );

    loglikelihood_mcmc.row(0) = log_likelihood.t();

    // ###########################################################
    // ## MCMC loop
    // ###########################################################

    const unsigned int cTotalLength = tick;

    Rprintf("Running MCMC iterations ...\n");

    unsigned int nIter_thin_count = 0;

    if (BVS)
    {
        for (unsigned int m=0; m<nIter; ++m)
        {
            if ((m+1) % cTotalLength == 0) {
                double gamma_acc = (m == 0) ? 0.0 : (double)(gamma_acc_count)/(double)(m+1) * 1000.0;
                double eta_acc = (m == 0) ? 0.0 : (double)(eta_acc_count)/(double)(m+1) * 1000.0;

                Rcpp::Rcout << " Running iteration " << m+1 << " ... Acc Rate: ~ gamma: "
                            << std::round(gamma_acc)/1000.0
                            << " ... ~ eta: " << std::round(eta_acc)/1000.0 << "\n";
            }

            // update beta variances tau0Sq, tauSq
            tau0Sq = sampleV(hyperpar->tau0A, hyperpar->tau0B, betas.row(0).t());

            for (unsigned int l=0; l<L; ++l)
            {
                tauSq[l] = sampleV(hyperpar->tauA, hyperpar->tauB, betas.submat(1,l,p,l));
            }

            // update xi variance vSq
            v0Sq = sampleV0(hyperpar->v0A, hyperpar->v0B, xi[0]);
            vSq = sampleV(hyperpar->vA, hyperpar->vB, xi.subvec(1, xi.n_elem - 1));

            // update xi in cure fraction
            ARMS_Gibbs::arms_gibbs_xi
            (
                armsPar,
                xi,
                v0Sq,
                vSq,
                datProportion,
                weibullS,
                dataclass
            );

            // update cure rate based on new xi
            logTheta = dataclass.datX0 * xi;
            logTheta.elem(arma::find(logTheta > upperbound)).fill(upperbound);
            datTheta = arma::exp( logTheta );

            // update parameters in the proportion model
            if(proportion_model)
            {
                if(dirichlet)
                {
                    // update zeta variances w0Sq, wSq
                    w0Sq = sampleV(hyperpar->w0A, hyperpar->w0B, zetas.row(0).t());

                    for (unsigned int l=0; l<L; ++l)
                    {
                        wSq[l] = sampleV(hyperpar->wA, hyperpar->wB, zetas.submat(1,l,p,l));
                    }

                    if(BVS)
                    {
                        // ============================================================
                        // Update latent Bernoulli probabilities rho | eta
                        // ============================================================

                        if(etaPrior == Eta_Prior_Type::bernoulli)
                        {
                            for(unsigned int l=0; l<L; ++l)
                            {
                                double s_l = arma::accu(etas.col(l));

                                rho[l] = R::rbeta(hyperpar->rhoA + s_l,
                                                hyperpar->rhoB + static_cast<double>(p) - s_l);

                                // Recompute logP_eta so it is consistent with current rho.
                                for(unsigned int j=0; j<p; ++j)
                                {
                                    logP_eta(j, l) = BVS_Sampler::logPDFBernoulli(etas(j, l), rho[l]);
                                }
                            }
                        }

                        // logZ_eta.fill(std::numeric_limits<double>::quiet_NaN());
                        BVS_Sampler::sampleEta(
                            etas,
                            etaPrior,
                            etaSampler,
                            logP_eta,
                            eta_acc_count,
                            log_likelihood,

                            armsPar,
                            hyperpar,

                            zetas,
                            betas,
                            gammas,
                            xi,
                            kappa,
                            w0Sq,
                            wSq,
                            rho,
                            logZ_eta,

                            dirichlet,
                            datTheta,
                            weibullS,
                            weibullLambda,
                            dataclass
                        );
                    }

                    // One more round update besides sampleEta()
                    ARMS_Gibbs::arms_gibbs_zeta(
                        armsPar,
                        zetas,
                        w0Sq,
                        wSq,
                        etas,

                        kappa,
                        dirichlet,
                        datTheta,
                        weibullS,
                        weibullLambda,
                        dataclass
                    );

                    // update Dirichlet concentrations and proportions based on new zetas
                    arma::mat alphas = arma::zeros<arma::mat>(N, L);

                    for(unsigned int l=0; l<L; ++l)
                    {
                        arma::vec zetaMask_l = zetas.submat(1, l, p, l);
                        zetaMask_l.elem(arma::find(etas.col(l) == 0)).fill(0.0);
                        alphas.col(l) = arma::exp( zetas(0, l) + dataclass.datX.slice(l) * zetaMask_l );
                    }

                    alphas.elem(arma::find(alphas > upperbound3)).fill(upperbound3);
                    alphas.elem(arma::find(alphas < lowerbound)).fill(lowerbound);
                    datProportion = alphas / arma::repmat(arma::sum(alphas, 1), 1, L);
                }
                else
                {
                    Rprintf("Warning: In arms_gibbs_zeta(), Dirichlet modeling with logit/alr-link is not implement!\n");
                    break;
                }
            }

            // update Weibull shape parameter kappa
            ARMS_Gibbs::arms_kappa(
                armsPar,
                kappa,
                hyperpar->kappaA,
                hyperpar->kappaB,
                hyperpar->kappaIGamma,
                datTheta,
                datMu,
                datProportion,
                dataclass
            );

            // update Weibull quantities based on new kappa
            for(unsigned int l=0; l<L; ++l)
            {
                weibullLambda.col(l) = datMu.col(l) / std::tgamma(1.0+1.0/kappa);
                weibullS.col(l) = arma::exp(- arma::pow( dataclass.datTime/weibullLambda.col(l), kappa));
            }

            // ============================================================
            // Update latent Bernoulli probabilities pi | gamma
            // ============================================================

            if(BVS)
            {
                if(gammaPrior == Gamma_Prior_Type::bernoulli)
                {
                    for(unsigned int l=0; l<L; ++l)
                    {
                        double s_l = arma::accu(gammas.col(l));

                        pi[l] = R::rbeta(hyperpar->piA + s_l,
                                        hyperpar->piB + static_cast<double>(p) - s_l);

                        // Recompute logP_gamma so it is consistent with current pi.
                        for(unsigned int j=0; j<p; ++j)
                        {
                            logP_gamma(j, l) = BVS_Sampler::logPDFBernoulli(gammas(j, l), pi[l]);
                        }
                    }
                }
            //}

            // update gammas -- variable selection indicators
            //if(BVS)
            //{
                // logZ_gamma.fill(std::numeric_limits<double>::quiet_NaN());
                BVS_Sampler::sampleGamma(
                    gammas,
                    gammaPrior,
                    gammaSampler,
                    logP_gamma,
                    gamma_acc_count,
                    log_likelihood,

                    armsPar,
                    hyperpar,

                    xi,
                    zetas,
                    etas,
                    betas,
                    kappa,
                    tau0Sq,
                    tauSq,
                    pi,        
                    logZ_gamma,

                    proportion_model,

                    datProportion,
                    datTheta,
                    datMu,
                    weibullS,
                    dataclass
                );
            }

            // update betas in non-cure fraction
            ARMS_Gibbs::arms_gibbs_beta(
                armsPar,
                betas,
                tauSq,
                tau0Sq,

                gammas,

                kappa,
                datTheta,
                datMu,
                datProportion,
                weibullS,
                dataclass
            );

            #ifdef _OPENMP
            #pragma omp parallel for
            #endif

            // update Weibull quantities based on new betas
            for(unsigned int l=0; l<L; ++l)
            {
                arma::vec betaMask_l = betas.submat(1, l, p, l);
                betaMask_l.elem(arma::find(gammas.col(l) == 0)).fill(0.0);
                arma::vec logMu_l = betas(0, l) + dataclass.datX.slice(l) * betaMask_l;
                logMu_l.elem(arma::find(logMu_l > upperbound)).fill(upperbound);

                datMu.col(l) = arma::exp( logMu_l );
                weibullLambda.col(l) = datMu.col(l) / std::tgamma(1.0+1.0/kappa);
                weibullS.col(l) = arma::exp(- arma::pow( dataclass.datTime/weibullLambda.col(l), kappa));
            }

            // save results for un-thinned posterior mean
            if(m >= burnin)
            {
                xi_post += xi;
                zeta_post += zetas;
                kappa_post += kappa;
                beta_post += betas;

                if(BVS)
                {
                    gamma_post += gammas;

                    if(gammaPrior == Gamma_Prior_Type::bernoulli)
                    {
                        pi_post += pi;
                    }

                    if(proportion_model)
                    {
                        eta_post += etas;

                        if(etaPrior == Eta_Prior_Type::bernoulli)
                        {
                            rho_post += rho;
                        }
                    }
                }
            }

            // save results of thinned iterations
            if((m+1) % thin == 0)
            {
                xi_mcmc.row(1+nIter_thin_count) = xi.t();
                wSq_mcmc[1+nIter_thin_count] = wSq[0];
                kappa_mcmc[1+nIter_thin_count] = kappa;
                tauSq_mcmc[1+nIter_thin_count] = tauSq[0];

                arma::mat betaMask = betas;
                arma::mat zetaMask = zetas;
                if(BVS)
                {
                    betaMask = betas % arma::join_cols(arma::ones<arma::urowvec>(L), gammas);
                    gamma_mcmc.row(1+nIter_thin_count) = arma::vectorise(gammas).t();

                    if(gammaPrior == Gamma_Prior_Type::bernoulli)
                    {
                        pi_mcmc.row(1+nIter_thin_count) = pi.t();
                    }

                    if(proportion_model)
                    {
                        zetaMask = zetas % arma::join_cols(arma::ones<arma::urowvec>(L), etas);
                        eta_mcmc.row(1+nIter_thin_count) = arma::vectorise(etas).t();

                        if(etaPrior == Eta_Prior_Type::bernoulli)
                        {
                            rho_mcmc.row(1+nIter_thin_count) = rho.t();
                        }
                    }
                }
                else
                {
                    BVS_Sampler::loglikelihood(
                        xi,
                        zetas,
                        betas,
                        etas,
                        gammas,
                        kappa,
                        proportion_model,
                        dataclass,
                        log_likelihood
                    );
                }

                loglikelihood_mcmc.row(1+nIter_thin_count) = log_likelihood.t();

                zeta_mcmc.row(1+nIter_thin_count) = arma::vectorise(zetaMask).t();
                beta_mcmc.row(1+nIter_thin_count) = arma::vectorise(betaMask).t();

                ++nIter_thin_count;
            }
        }
    }
    else
    {
        gammas.ones();
        etas.ones();
        
        for (unsigned int m=0; m<nIter; ++m)
        {
            if ((m+1) % cTotalLength == 0) {
                Rcpp::Rcout << " Running iteration " << m+1 << "\n";
            }

            // update beta variances tau0Sq, tauSq
            tau0Sq = sampleV(hyperpar->tau0A, hyperpar->tau0B, betas.row(0).t());

            for (unsigned int l=0; l<L; ++l)
            {
                tauSq[l] = sampleV(hyperpar->tauA, hyperpar->tauB, betas.submat(1,l,p,l));
            }

            // update xi variance vSq
            v0Sq = sampleV0(hyperpar->v0A, hyperpar->v0B, xi[0]);
            vSq = sampleV(hyperpar->vA, hyperpar->vB, xi.subvec(1, xi.n_elem - 1));

            // update xi in cure fraction
            ARMS_Gibbs::arms_gibbs_xi
            (
                armsPar,
                xi,
                v0Sq,
                vSq,
                datProportion,
                weibullS,
                dataclass
            );

            // update cure rate based on new xi
            logTheta = dataclass.datX0 * xi;
            logTheta.elem(arma::find(logTheta > upperbound)).fill(upperbound);
            datTheta = arma::exp( logTheta );

            // update parameters in the proportion model
            if(proportion_model)
            {
                if(dirichlet)
                {
                    // update zeta variances w0Sq, wSq
                    w0Sq = sampleV(hyperpar->w0A, hyperpar->w0B, zetas.row(0).t());

                    for (unsigned int l=0; l<L; ++l)
                    {
                        wSq[l] = sampleV(hyperpar->wA, hyperpar->wB, zetas.submat(1,l,p,l));
                    }

                    // One more round update besides sampleEta()
                    ARMS_Gibbs::arms_gibbs_zetaFull(
                        armsPar,
                        zetas,
                        w0Sq,
                        wSq,

                        kappa,
                        dirichlet,
                        datTheta,
                        weibullS,
                        weibullLambda,
                        dataclass
                    );

                    // update Dirichlet concentrations and proportions based on new zetas
                    arma::mat alphas = arma::zeros<arma::mat>(N, L);

                    for(unsigned int l=0; l<L; ++l)
                    {
                        arma::vec zetaMask_l = zetas.submat(1, l, p, l);
                        alphas.col(l) = arma::exp( zetas(0, l) + dataclass.datX.slice(l) * zetaMask_l );
                    }

                    alphas.elem(arma::find(alphas > upperbound3)).fill(upperbound3);
                    alphas.elem(arma::find(alphas < lowerbound)).fill(lowerbound);
                    datProportion = alphas / arma::repmat(arma::sum(alphas, 1), 1, L);
                }
                else
                {
                    Rprintf("Warning: In arms_gibbs_zeta(), Dirichlet modeling with logit/alr-link is not implement!\n");
                    break;
                }
            }

            // update Weibull shape parameter kappa
            ARMS_Gibbs::arms_kappa(
                armsPar,
                kappa,
                hyperpar->kappaA,
                hyperpar->kappaB,
                hyperpar->kappaIGamma,
                datTheta,
                datMu,
                datProportion,
                dataclass
            );

            // update Weibull quantities based on new kappa
            for(unsigned int l=0; l<L; ++l)
            {
                weibullLambda.col(l) = datMu.col(l) / std::tgamma(1.0+1.0/kappa);
                weibullS.col(l) = arma::exp(- arma::pow( dataclass.datTime/weibullLambda.col(l), kappa));
            }

            // update betas in non-cure fraction
            ARMS_Gibbs::arms_gibbs_betaFull(
                armsPar,
                betas,
                tauSq,
                tau0Sq,

                kappa,
                datTheta,
                datMu,
                datProportion,
                weibullS,
                dataclass
            );

            #ifdef _OPENMP
            #pragma omp parallel for
            #endif

            // update Weibull quantities based on new betas
            for(unsigned int l=0; l<L; ++l)
            {
                arma::vec betaMask_l = betas.submat(1, l, p, l);
                arma::vec logMu_l = betas(0, l) + dataclass.datX.slice(l) * betaMask_l;
                logMu_l.elem(arma::find(logMu_l > upperbound)).fill(upperbound);

                datMu.col(l) = arma::exp( logMu_l );
                weibullLambda.col(l) = datMu.col(l) / std::tgamma(1.0+1.0/kappa);
                weibullS.col(l) = arma::exp(- arma::pow( dataclass.datTime/weibullLambda.col(l), kappa));
            }

            // save results for un-thinned posterior mean
            if(m >= burnin)
            {
                xi_post += xi;
                zeta_post += zetas;
                kappa_post += kappa;
                beta_post += betas;
            }

            // save results of thinned iterations
            if((m+1) % thin == 0)
            {
                xi_mcmc.row(1+nIter_thin_count) = xi.t();
                wSq_mcmc[1+nIter_thin_count] = wSq[0];
                kappa_mcmc[1+nIter_thin_count] = kappa;
                tauSq_mcmc[1+nIter_thin_count] = tauSq[0];

                arma::mat betaMask = betas;
                arma::mat zetaMask = zetas;
                
                BVS_Sampler::loglikelihood_noBVS(
                        xi,
                        zetas,
                        betas,
                        kappa,
                        proportion_model,
                        dataclass,
                        log_likelihood
                );
                loglikelihood_mcmc.row(1+nIter_thin_count) = log_likelihood.t();

                zeta_mcmc.row(1+nIter_thin_count) = arma::vectorise(zetaMask).t();
                beta_mcmc.row(1+nIter_thin_count) = arma::vectorise(betaMask).t();

                ++nIter_thin_count;
            }
        }
    }

    free(hyperpar);

    Rcpp::Rcout << "\n";

    // wrap all outputs
    Rcpp::List output_mcmc;

    output_mcmc["xi"] = xi_mcmc;
    output_mcmc["kappa"] = kappa_mcmc;
    output_mcmc["betas"] = beta_mcmc;
    output_mcmc["zetas"] = zeta_mcmc;

    arma::mat gamma_post_mean = arma::zeros<arma::mat>(arma::size(gamma_post));
    arma::mat eta_post_mean = arma::zeros<arma::mat>(arma::size(eta_post));

    if(BVS)
    {
        output_mcmc["gammas"] = gamma_mcmc;
        output_mcmc["gamma_acc_rate"] = ((double)gamma_acc_count) / ((double)nIter);

        gamma_post_mean = arma::conv_to<arma::mat>::from(gamma_post) / ((double)(nIter - burnin));

        if(gammaPrior == Gamma_Prior_Type::bernoulli)
        {
            output_mcmc["pi"] = pi_mcmc;
            pi_post /= ((double)(nIter - burnin));
        }

        if(proportion_model)
        {
            output_mcmc["etas"] = eta_mcmc;
            output_mcmc["eta_acc_rate"] = ((double)eta_acc_count) / ((double)nIter);

            eta_post_mean = arma::conv_to<arma::mat>::from(eta_post) / ((double)(nIter - burnin));

            if(etaPrior == Eta_Prior_Type::bernoulli)
            {
                output_mcmc["rho"] = rho_mcmc;
                rho_post /= ((double)(nIter - burnin));
            }
        }
    }

    output_mcmc["loglikelihood"] = loglikelihood_mcmc;
    output_mcmc["tauSq"] = tauSq_mcmc;
    output_mcmc["wSq"] = wSq_mcmc;
    output_mcmc["vSq"] = vSq_mcmc;

    xi_post /= ((double)(nIter - burnin));
    kappa_post /= ((double)(nIter - burnin));
    beta_post /= ((double)(nIter - burnin));
    zeta_post /= ((double)(nIter - burnin));

    Rcpp::List postList = Rcpp::List::create(
        Rcpp::Named("xi") = xi_post,
        Rcpp::Named("kappa") = kappa_post,
        Rcpp::Named("betas") = beta_post,
        Rcpp::Named("zetas") = zeta_post,
        Rcpp::Named("gammas") = gamma_post_mean,
        Rcpp::Named("etas") = eta_post_mean
    );

    if(BVS && gammaPrior == Gamma_Prior_Type::bernoulli)
    {
        postList["pi"] = pi_post;
    }

    if(BVS && proportion_model && etaPrior == Eta_Prior_Type::bernoulli)
    {
        postList["rho"] = rho_post;
    }

    output_mcmc["post"] = postList;

    return output_mcmc;
}