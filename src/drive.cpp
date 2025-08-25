// Main function for the MCMC loop


#include "simple_gibbs.h"
#include "arms_gibbs.h"
#include "BVS.h"
#include "global.h"

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]


//' Main function for the MCMC loop
//'
//' @name mcmc
//'
//' @param nIter Number of MCMC iterations
//' @param burnin Length of MCMC burn-in period
//' @param thin Number of thinning
//' @param n Number of samples to draw
//' @param nsamp How many samples to draw for generating each sample; only the last draw will be kept
//' @param ninit Number of initials as meshgrid values for envelop search
//' @param metropolis TBA
//' @param simple TBA
//' @param convex Adjustment for convexity (non-negative value, default 1.0)
//' @param npoint Maximum number of envelope points
//' @param dirichlet Not yet implemented
//' @param proportion_model TBA
//' @param BVS TBA
//' @param gamma_prior TBA
//' @param gamma_sampler TBA
//' @param eta_prior TBA
//' @param eta_sampler TBA
//' @param initList TBA
//' @param rangeList TBA
//' @param hyperparList TBA
//' @param datEvent TBA
//' @param datTime TBA
//' @param datX TBA
//' @param datX0 TBA
//' @param datProportionConst TBA
//'
// [[Rcpp::export]]
Rcpp::List run_mcmc(
    unsigned int nIter,
    unsigned int burnin,
    unsigned int thin,

    unsigned int n,
    int nsamp,
    int ninit,
    int metropolis,
    bool simple,
    double convex,
    int npoint,
    bool dirichlet,
    bool proportion_model,
    bool BVS,
    const std::string& gamma_prior,
    const std::string& gamma_sampler,
    const std::string& eta_prior,
    const std::string& eta_sampler,

    Rcpp::List initList,
    Rcpp::List rangeList,
    Rcpp::List hyperparList,

    arma::uvec datEvent,
    arma::vec datTime,
    arma::cube datX,
    arma::mat datX0,
    arma::mat datProportionConst)
{
    // dimensions
    unsigned int N = datX.n_rows;
    unsigned int p = datX.n_cols;
    unsigned int L = datX.n_slices;

    // const arma::uvec datEvent
    // const arma::vec datTime,
    // const arma::cube datX,
    // const arma::mat datX0,
    // const arma::mat datProportionConst

    // input constant data sets in a class
    DataClass dataclass(datEvent, datTime, datX, datX0, datProportionConst);
    // ,datProportion, datTheta, datMu, weibullLambda, weibullS);
    datEvent.clear();
    datTime.clear();
    datX.clear();
    datX0.clear();
    datProportionConst.clear();

    // arms parameters in a class
    armsParmClass armsPar(n, nsamp, ninit, metropolis, simple, convex, npoint,
                          Rcpp::as<double>(rangeList["xiMin"]),
                          Rcpp::as<double>(rangeList["xiMax"]),
                          Rcpp::as<double>(rangeList["zetaMin"]),
                          Rcpp::as<double>(rangeList["zetaMax"]),
                          Rcpp::as<double>(rangeList["kappaMin"]),
                          Rcpp::as<double>(rangeList["kappaMax"]),
                          Rcpp::as<double>(rangeList["betaMin"]),
                          Rcpp::as<double>(rangeList["betaMax"]));
    rangeList = Rcpp::List();  // Clear it by creating a new empty List

    // hyperparameters
    // NOTE: Do not change try to change the struct hyperparS to a C++ class until I become a super expert in C++,
    // because that will change a lot the related code toward the original arms C code in change C++ class does not fit well.
    hyperparS *hyperpar = (hyperparS *)malloc(sizeof (hyperparS));

    arma::umat mrfG; // NOT put this in if-condition, otherwise some memory issue for 'hyperpar->mrfG = mrfG.memptr()', not know exactly why?
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
    double pi0 = Rcpp::as<double>(hyperparList["pi"]);
    double rho0 = Rcpp::as<double>(hyperparList["rho"]);
    // std::cout << "... mrfA=" << hyperpar->mrfA << "; mrfB=" << hyperpar->mrfB <<
    // "; mrfA_prop=" << hyperpar->mrfA_prop << "; mrfB_prop=" << hyperpar->mrfB_prop <<
    // "; gamma_prior=" << gamma_prior << "; eta_prior=" << eta_prior <<
    // "\n";

    double vSq = Rcpp::as<double>(hyperparList["vSq"]);
    // hyperpar->vSq = vSq;
    hyperpar->vA = Rcpp::as<double>(hyperparList["vA"]);
    hyperpar->vB = Rcpp::as<double>(hyperparList["vB"]);
    double v0Sq = Rcpp::as<double>(hyperparList["v0Sq"]);
    // hyperpar->v0Sq = v0Sq;
    hyperpar->v0A = Rcpp::as<double>(hyperparList["v0A"]);
    hyperpar->v0B = Rcpp::as<double>(hyperparList["v0B"]);
    arma::vec tauSq = Rcpp::as<arma::vec>(hyperparList["tauSq"]);
    // hyperpar->tauSq = tauSq;
    hyperpar->tauA = Rcpp::as<double>(hyperparList["tauA"]);
    hyperpar->tauB = Rcpp::as<double>(hyperparList["tauB"]);
    double tau0Sq = Rcpp::as<double>(hyperparList["tau0Sq"]);
    // hyperpar->tau0Sq = tau0Sq;
    hyperpar->tau0A = Rcpp::as<double>(hyperparList["tau0A"]);
    hyperpar->tau0B = Rcpp::as<double>(hyperparList["tau0B"]);
    arma::vec wSq = Rcpp::as<arma::vec>(hyperparList["wSq"]);
    // hyperpar->wSq = wSq;
    hyperpar->wA = Rcpp::as<double>(hyperparList["wA"]);
    hyperpar->wB = Rcpp::as<double>(hyperparList["wB"]);
    hyperpar->w0IGamma = Rcpp::as<bool>(hyperparList["w0IGamma"]);
    double w0Sq = Rcpp::as<double>(hyperparList["w0Sq"]);
    // hyperpar->w0Sq = w0Sq;
    hyperpar->w0A = Rcpp::as<double>(hyperparList["w0A"]);
    hyperpar->w0B = Rcpp::as<double>(hyperparList["w0B"]);
    hyperpar->kappaA = Rcpp::as<double>(hyperparList["kappaA"]);
    hyperpar->kappaB = Rcpp::as<double>(hyperparList["kappaB"]);
    hyperpar->kappaIGamma = Rcpp::as<bool>(hyperparList["kappaIGamma"]);

    hyperparList = Rcpp::List();  // Clear it by creating a new empty List

    // Gamma Sampler
    Gamma_Sampler_Type gammaSampler;
    if ( gamma_sampler == "bandit" )
        gammaSampler = Gamma_Sampler_Type::bandit;
    else if ( gamma_sampler == "mc3" )
        gammaSampler = Gamma_Sampler_Type::mc3 ;
    else
    {
        Rprintf("ERROR: Wrong type of Gamma Sampler given!");
        return 1;
    }
    Gamma_Prior_Type gammaPrior;
    if ( gamma_prior == "bernoulli" )
        gammaPrior = Gamma_Prior_Type::bernoulli;
    else if ( gamma_prior == "mrf" )
        gammaPrior = Gamma_Prior_Type::mrf ;
    else
    {
        Rprintf("ERROR: Wrong type of Gamma Prior given!");
        return 1;
    }
    // ****************************************************
    Eta_Sampler_Type etaSampler;
    if ( eta_sampler == "bandit" )
        etaSampler = Eta_Sampler_Type::bandit;
    else if ( eta_sampler == "mc3" )
        etaSampler = Eta_Sampler_Type::mc3 ;
    else
    {
        Rprintf("ERROR: Wrong type of Eta Sampler given!");
        return 1;
    }
    Eta_Prior_Type etaPrior;
    if ( eta_prior == "bernoulli" )
        etaPrior = Eta_Prior_Type::bernoulli;
    else if ( eta_prior == "mrf" )
        etaPrior = Eta_Prior_Type::mrf ;
    else
    {
        Rprintf("ERROR: Wrong type of Eta Prior given!");
        return 1;
    }

    // initial values of key parameters and save them in a struct object
    arma::vec xi = Rcpp::as<arma::vec>(initList["xi"]);
    arma::mat zetas = Rcpp::as<arma::mat>(initList["zetas"]);
    arma::mat betas = Rcpp::as<arma::mat>(initList["betas"]);
    double kappa = Rcpp::as<double>(initList["kappa"]);
    initList = Rcpp::List();  // Clear it by creating a new empty List

    unsigned int nIter_thin = nIter / thin;
    // initializing mcmc results
    arma::vec vSq_mcmc = arma::zeros<arma::vec>(1+nIter_thin);
    vSq_mcmc[0] = vSq; //hyperpar->vSq;
    arma::mat xi_mcmc = arma::zeros<arma::mat>(1+nIter_thin, xi.n_elem);
    xi_mcmc.row(0) = xi.t();
    arma::vec wSq_mcmc = arma::zeros<arma::vec>(1+nIter_thin);
    wSq_mcmc[0] = wSq[0];//hyperpar->wSq; // TODO: only keep the first one for now 
    arma::mat zeta_mcmc = arma::zeros<arma::mat>(1+nIter_thin, (p+1)*L);
    zeta_mcmc.row(0) = arma::vectorise(zetas).t();
    arma::vec tauSq_mcmc = arma::zeros<arma::vec>(1+nIter_thin);
    tauSq_mcmc[0] = tauSq[0]; // TODO: only keep the first one for now 
    arma::mat beta_mcmc = arma::zeros<arma::mat>(1+nIter_thin, (p+1)*L);
    beta_mcmc.row(0) = arma::vectorise(betas).t();
    arma::vec kappa_mcmc = arma::zeros<arma::vec>(1+nIter_thin);
    kappa_mcmc[0] = kappa;

    // initializing relevant quantities; can be declared like arma::mat&

    // quantity 01
    arma::umat gammas = arma::ones<arma::umat>(p, L);
    arma::umat gamma_mcmc;
    arma::mat logP_gamma;// = arma::zeros<arma::mat>(p, L); // this is declared globally to be updated in the M-H sampler for gammas
    unsigned int gamma_acc_count; // count acceptance of gammas via M-H sampler
    if(BVS)
    {
        // if( gammaPrior == Gamma_Prior_Type::bernoulli )
        // {
        logP_gamma = arma::zeros<arma::mat>(p, L);
        // }
        gamma_acc_count = 0;

        // BVS_Sampler::banditInit(p, L, N); // no need due to initializing them as local static variables
        // gammas = arma::zeros<arma::umat>(p, L);
        for(unsigned int l=0; l<L; ++l)
        {
            // the Bernoulli probability is cell-type-specific
            double pi;// = R::rbeta(hyperpar->piA, hyperpar->piB);
            if(gamma_prior == "mrf")
            {
                pi = 1. / (1. + std::exp(- hyperpar->mrfA));
            }
            else
            {
                pi = R::rbeta(hyperpar->piA, hyperpar->piB);
            }
            for(unsigned int j=0; j<p; ++j)
            {
                gammas(j, l) = R::rbinom(1, pi);

                // if( gammaPrior == Gamma_Prior_Type::bernoulli )
                // {
                logP_gamma(j, l) = BVS_Sampler::logPDFBernoulli(gammas(j, l), pi);
                // }
            }
        }
        gamma_mcmc = arma::zeros<arma::umat>(1+nIter_thin, p*L);
        gamma_mcmc.row(0) = arma::vectorise(gammas).t();
    }
    // else
    // {
    //     gammas = arma::ones<arma::umat>(p, L);
    // }

    // quantity 02
    arma::umat etas;
    arma::umat eta_mcmc;
    arma::mat logP_eta;
    unsigned int eta_acc_count; // count acceptance of etas via M-H sampler
    if(BVS)
    {
        // if( etaPrior == Eta_Prior_Type::bernoulli )
        // {
        logP_eta = arma::zeros<arma::mat>(p, L);
        // }
        eta_acc_count = 0; // count acceptance of etas via M-H sampler

        // BVS_Sampler::banditInitEta(p, L, N);
        etas = arma::zeros<arma::umat>(p, L);
        // etas.row(0).fill(1); // for intercept
        if(proportion_model)
        {
            for(unsigned int l=0; l<L; ++l)
            {
                // the Bernoulli probability is cell-type-specific
                double rho;// = R::rbeta(hyperpar->rhoA, hyperpar->rhoB);
                if(eta_prior == "mrf")
                {
                    rho = 1. / (1. + std::exp(- hyperpar->mrfA_prop));
                }
                else
                {
                    rho = R::rbeta(hyperpar->rhoA, hyperpar->rhoB);
                }
                for(unsigned int j=0; j<p; ++j)
                {
                    etas(j, l) = R::rbinom(1, rho);

                    // if( etaPrior == Eta_Prior_Type::bernoulli )
                    // {
                    logP_eta(j, l) = BVS_Sampler::logPDFBernoulli(etas(j, l), rho);
                    // }
                }
            }
            eta_mcmc = arma::zeros<arma::umat>(1+nIter_thin, p*L);
            // eta_mcmc.row(0) = arma::vectorise(etas.submat(1,0,p,L-1)).t();
            eta_mcmc.row(0) = arma::vectorise(etas).t();
        }
    }
    else
    {
        etas = arma::ones<arma::umat>(p, L);
    }
    // etas.fill(1); // Not yet implement BVS for Dirichlet modeling part
    // quantity 1
    arma::vec datTheta = arma::zeros<arma::vec>(N);
    arma::vec logTheta = dataclass.datX0 * xi;
    logTheta.elem(arma::find(logTheta > upperbound)).fill(upperbound);
    datTheta = arma::exp( logTheta );
    // quantity 2
    arma::mat datMu = arma::zeros<arma::mat>(N, L);
    for(unsigned int l=0; l<L; ++l)
    {
        // arma::vec logMu_l = dataclass.datX.slice(l) * betas.col(l);
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
    // // quantities 4 & 5
    // define immediate quantities as shared pointer in the DataClass to allow modifiable
    // std::shared_ptr<arma::mat> weibullS = std::make_shared<arma::mat>(arma::zeros<arma::mat>(N, L));
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

    /*
    // input must-update data struct
    dataUpdateS *datUpdate = (dataUpdateS *)malloc(sizeof (dataUpdateS));
    datUpdate->Proportion = datProportion.memptr();
    datUpdate->Mu = datMu.memptr();
    datUpdate->Theta = datTheta.memptr();
    datUpdate->weibullS = weibullS.memptr();
    datUpdate->weibullLambda = weibullLambda.memptr();
    */

    arma::mat loglikelihood_mcmc = arma::zeros<arma::mat>(1+nIter_thin, N);
    arma::vec log_likelihood;
    BVS_Sampler::loglikelihood(
        xi,
        zetas,
        betas,
        kappa,

        proportion_model,
        dataclass,
        log_likelihood
    );
    arma::vec log_likelihood0;
    BVS_Sampler::loglikelihood0(
        xi,
        zetas,
        betas,
        kappa,

        proportion_model,
        dataclass,
        log_likelihood0
    );
    // loglikelihood_mcmc.row(0) = log_likelihood.t();
    loglikelihood_mcmc.row(0) = log_likelihood0.t();

    double logPosteriorBeta = 0.; // TODO: If using this, it's better to be computed through likelihood()
    double logPosteriorZeta = 0.;

    // // initial points for ARS approach
    // arma::vec initialPoints = arma::linspace( range->xiMin+0.1, range->xiMin-0.1, ninit );

    // ###########################################################
    // ## MCMC loop
    // ###########################################################


    // BVS_Sampler bvssampler(hyperpar, dataclass);

    // BVS_Sampler::banditInit(p, L, N);

    const unsigned int cTotalLength = 50;
    //std::cout
    Rprintf("Running MCMC iterations ...\n");
    unsigned int nIter_thin_count = 0;
    for (unsigned int m=0; m<nIter; ++m)
    {
        // print progression cursor
        if (m % 10 == 0 || m == (nIter - 1))
            //std::cout
            Rcpp::Rcout << "\r[" <<                                           //'\r' aka carriage return should move printer's cursor back at the beginning of the current line
                        std::string(cTotalLength * (m + 1.) / nIter, '#') <<        // printing filled part
                        std::string(cTotalLength * (1. - (m + 1.) / nIter), '-') << // printing empty part
                        "] " << (int)((m + 1.) / nIter * 100.0) << "%\r";             // printing percentage

        // update \xi's variance vSq
        v0Sq = sampleV0(hyperpar->v0A, hyperpar->v0B, xi[0]);
        // hyperpar->v0Sq = v0Sq;
        vSq = sampleV(hyperpar->vA, hyperpar->vB, xi.subvec(1, xi.n_elem - 1));
        // hyperpar->vSq = vSq;
        // vSq_mcmc[1+m] = hyperpar->vSq;

        // update \xi's in cure fraction
        // the void function below passes the address of \xi and update it
        ARMS_Gibbs::arms_gibbs_xi //all key parameters can be declared as global variables and the arms functions be void
        (
            armsPar,
            xi,
            v0Sq,
            vSq,
            // hyperpar->vA,
            // hyperpar->vB,
            datProportion,
            weibullS,
            dataclass
        );
        /*
        xi = ars_gibbs_xi // This does not work so far, even with positive range
        (
          n,
          ninit,
          range->xiMin,
          range->xiMax,

          xi,
          hyperpar->v0Sq,
          hyperpar->vSq,
          // hyperpar->vA,
          // hyperpar->vB,
          datX0,
          datProportion,
          datEvent,
          weibullS
        ); */
        // xi_mcmc.row(1+m) = xi.t();

        // update cure rate based on the new xi
        logTheta = dataclass.datX0 * xi;
        logTheta.elem(arma::find(logTheta > upperbound)).fill(upperbound);
        datTheta = arma::exp( logTheta );

        // update parameters in the proportion model
        if(proportion_model)
        {
            if(dirichlet)
            {
                // update \eta -- variable selection indicators
                // Here both etas and zetas are updated inside due to passing their addresses
                if(BVS)
                {
                    //// update cell-type-specific Bernoulli probability via Gibbs
                    // arma::vec rho = arma::zeros<arma::vec>(L); 
                    // for (unsigned int l=0; l<L; ++l)
                    // {
                    //     rho[l] = R::rbeta(hyperpar->rhoA + (double)(arma::accu(etas.col(l))),
                    //                         hyperpar->rhoB + (double)(p) - (double)(arma::accu(etas.col(l))));
                    // }
                    // arma::umat etas_previous = etas;
                    BVS_Sampler::sampleEta(
                        etas,//etas_previous,
                        etaPrior,
                        etaSampler,
                        logP_eta,
                        eta_acc_count,
                        log_likelihood,

                        armsPar,
                        hyperpar,

                        zetas,
                        betas,
                        xi,
                        kappa,
                        w0Sq,
                        wSq,
                        rho0,
                        dirichlet,

                        logPosteriorZeta,
                        datTheta,
                        weibullS,
                        weibullLambda,
                        dataclass
                    );
                    // eta_mcmc.row(1+m) = arma::vectorise(etas).t();
                }

                // update \zetas' variance wSq
                // hyperpar->wSq = sampleW(hyperpar->wA, hyperpar->wB, zetas.rows(1, p));
                // wSq_mcmc[1+m] = hyperpar->wSq;
                // if(hyperpar->w0IGamma)
                // {
                //     hyperpar->w0Sq = sampleW0(hyperpar->w0A, hyperpar->w0B, zetas.row(0));
                // }

                // One more round update beside sampleEta(), more accurate
                ARMS_Gibbs::arms_gibbs_zeta(
                    armsPar,
                    zetas,
                    // hyperpar->w0Sq,
                    // hyperpar->wSq,
                    w0Sq,
                    wSq,
                    // hyperpar->w0IGamma,
                    hyperpar->w0A,
                    hyperpar->w0B,
                    hyperpar->wA,
                    hyperpar->wB,
                    etas,

                    kappa,
                    dirichlet,
                    datTheta,
                    weibullS,
                    weibullLambda,
                    dataclass,
                    logPosteriorZeta
                );

                // zeta_mcmc.row(1+m) = arma::vectorise(zetas).t();

                // update Dirichlet's concentrations and proportions based on the new zetas
                arma::mat alphas = arma::zeros<arma::mat>(N, L);
                for(unsigned int l=0; l<L; ++l)
                {
                    alphas.col(l) = arma::exp( zetas(0, l) + dataclass.datX.slice(l) * zetas.submat(1, l, p, l) );
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

        // update Weibull's shape parameter kappa
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
        ); // if n>1, here it will be an average
        // kappa_mcmc[1+m] = kappa;

        // update Weibull's quantities based on the new kappa
        for(unsigned int l=0; l<L; ++l)
        {
            weibullLambda.col(l) = datMu.col(l) / std::tgamma(1.0+1.0/kappa);
            weibullS.col(l) = arma::exp(- arma::pow( dataclass.datTime/weibullLambda.col(l), kappa));
        }

        // update \gammas -- variable selection indicators
        if(BVS)
        {
            //// update cell-type-specific Bernoulli probability via Gibbs
            // arma::vec pi = arma::zeros<arma::vec>(L); 
            // for (unsigned int l=0; l<L; ++l)
            // {
            //     pi[l] = R::rbeta(hyperpar->piA + (double)(arma::accu(gammas.col(l))),
            //                             hyperpar->piB + (double)(p) - (double)(arma::accu(gammas.col(l))));
            // }
            // Here both gammas and betas are updated inside due to passing their addresses
            // arma::umat gammas_previous = gammas;
            BVS_Sampler::sampleGamma(
                gammas,//gammas_previous,
                gammaPrior,
                gammaSampler,
                logP_gamma,
                gamma_acc_count,
                log_likelihood,

                armsPar,
                hyperpar,

                xi,
                zetas,
                betas,
                kappa,
                tau0Sq,
                tauSq,
                pi0,

                proportion_model,

                logPosteriorBeta,
                datProportion,
                datTheta,
                datMu,
                weibullS,
                dataclass
            );
            // gamma_mcmc.row(1+m) = arma::vectorise(gammas).t();
        }

        // else
        // {
        // update \betas in non-cure fraction (one more round update besides in sampleGamma())
        ARMS_Gibbs::arms_gibbs_beta(
            armsPar,
            betas,
            // hyperpar->tauSq,
            tauSq,
            tau0Sq,
            hyperpar->tauA,
            hyperpar->tauB,
            hyperpar->tau0A,
            hyperpar->tau0B,

            gammas,

            kappa,
            datTheta,
            datMu,
            datProportion,
            weibullS,
            dataclass,
            logPosteriorBeta
        );

        // }
        // beta_mcmc.row(1+m) = arma::vectorise(betas).t();

        // update \betas' variance tauSq
        // hyperpar->tauSq = sampleTau(hyperpar->tauA, hyperpar->tauB, betas);
        // tauSq_mcmc[1+m] = hyperpar->tauSq;

        // update Weibull's quantities based on the new betas
        for(unsigned int l=0; l<L; ++l)
        {
            // arma::vec logMu_l = dataclass.datX.slice(l) * betas.col(l);
            arma::vec logMu_l = betas(0, l) + dataclass.datX.slice(l) * betas.submat(1, l, p, l);
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
                if(proportion_model)
                {
                    eta_post += etas;
                }
            }
        }

        // save results of thinned iterations
        if((m+1) % thin == 0)
        {
            xi_mcmc.row(1+nIter_thin_count) = xi.t();
            wSq_mcmc[1+nIter_thin_count] = wSq[0];//hyperpar->wSq;// TODO: only keep the firs one for now
            zeta_mcmc.row(1+nIter_thin_count) = arma::vectorise(zetas).t();
            kappa_mcmc[1+nIter_thin_count] = kappa;
            beta_mcmc.row(1+nIter_thin_count) = arma::vectorise(betas).t();
            tauSq_mcmc[1+nIter_thin_count] = tauSq[0];//hyperpar->tauSq; // TODO: only keep the firs one for now

            // update likelihood0
            BVS_Sampler::loglikelihood0(
                xi,
                zetas,
                betas,
                kappa,
                proportion_model,
                dataclass,
                log_likelihood0
            );
            loglikelihood_mcmc.row(1+nIter_thin_count) = log_likelihood0.t();
            if(BVS)
            {
                gamma_mcmc.row(1+nIter_thin_count) = arma::vectorise(gammas).t();
                if(proportion_model)
                {
                    eta_mcmc.row(1+nIter_thin_count) = arma::vectorise(etas).t();
                }
            }
            ++nIter_thin_count;
        }
        if(!BVS)
        {
            // update likelihood
            BVS_Sampler::loglikelihood(
                xi,
                zetas,
                betas,
                kappa,
                proportion_model,
                dataclass,
                log_likelihood
            );
        }


    }

    free(hyperpar);

    Rcpp::Rcout << "\n";

    // wrap all outputs
    Rcpp::List output_mcmc;
    output_mcmc["xi"] = xi_mcmc;
    output_mcmc["kappa"] = kappa_mcmc;
    //output["phi"] = phi_mcmc;
    output_mcmc["betas"] = beta_mcmc;
    output_mcmc["zetas"] = zeta_mcmc;
    arma::mat gamma_post_mean = arma::zeros<arma::mat>(arma::size(gamma_post));
    arma::mat eta_post_mean = arma::zeros<arma::mat>(arma::size(eta_post));
    if(BVS)
    {
        output_mcmc["gammas"] = gamma_mcmc;
        output_mcmc["gamma_acc_rate"] = ((double)gamma_acc_count) / ((double)nIter);
        gamma_post_mean = arma::conv_to<arma::mat>::from(gamma_post) / ((double)(nIter - burnin));
        if(proportion_model)
        {
            output_mcmc["etas"] = eta_mcmc;
            output_mcmc["eta_acc_rate"] = ((double)eta_acc_count) / ((double)nIter);
            eta_post_mean = arma::conv_to<arma::mat>::from(eta_post) / ((double)(nIter - burnin));
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
    output_mcmc["post"] = Rcpp::List::create(Rcpp::Named("xi") = xi_post,
                          Rcpp::Named("kappa") = kappa_post,
                          Rcpp::Named("betas") = beta_post,
                          Rcpp::Named("zetas") = zeta_post,
                          Rcpp::Named("gammas") = gamma_post_mean,
                          Rcpp::Named("etas") = eta_post_mean);

    return output_mcmc;
}

