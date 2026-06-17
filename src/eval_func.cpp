/* Evaluation functions (i.e. log densities) for ARS and ARMS*/

#include <memory> // Include for smart pointers

#include "eval_func.h"
#include "global.h"

// log-density for coefficient xis
double EvalFunction::log_dens_xis(
    double par,
    void *abc_data)
{
    // If myfunc() has to be defined in C code, all vec/mat elements in 'struct common_data{}' and 'void create_mydata()'
    //   need to be defined as 'double *', since array in C will copy big data and occupy too much memory

    double h = 0.;

    // dataS * mydata_parm = *(dataS *)mydata; // error: cannot initialize a variable of type 'dataS *' with an rvalue of type 'void *'
    // *mydata_parm = &abc_data; // error: no viable overloaded '='

    // Allocation of a zero-initialized memory block of (num*size) bytes
    // 'calloc' returns a void* (generic pointer) on success of memory allocation; type-cast it due to illegal in C++ but illegal in C
    // dataS *mydata_parm = (dataS *)calloc(sizeof(dataS), sizeof(dataS));
    // std::unique_ptr<dataS> mydata_parm = std::make_unique<dataS>();
    // *mydata_parm = *(dataS *)abc_data;
    auto mydata_parm = static_cast<dataS*>(abc_data);

    arma::vec xis(mydata_parm->currentPars, mydata_parm->p, true);

    xis[mydata_parm->jj] = par;
    double vSq = mydata_parm->vSq;
    if (mydata_parm->jj == 0)
    {
        vSq = mydata_parm->v0Sq;
    }

    // compute the log density
    double logprior = - par * par / vSq / 2.;

    //arma::mat X(mydata_parm->datX, n, p, false);  // use auxiliary memory
    //std::cout << "...X:\n" << X << "\n";

    arma::mat datX(const_cast<double*>(mydata_parm->datX), mydata_parm->N, mydata_parm->p, false);
    arma::uvec datEvent(const_cast<unsigned int*>(mydata_parm->datEvent), mydata_parm->N, false);

    arma::vec logTheta = datX * xis;
    logTheta.elem(arma::find(logTheta > upperbound)).fill(upperbound);
    arma::vec thetas = arma::exp( logTheta );

    double logpost_first = arma::sum( logTheta.elem(arma::find(datEvent)) );
    arma::vec logpost_second = arma::zeros<arma::vec>(mydata_parm->N);
    arma::mat datProportion(mydata_parm->datProportion, mydata_parm->N, mydata_parm->L, false);
    arma::mat weibullS(mydata_parm->weibullS, mydata_parm->N, mydata_parm->L, false);
    logpost_second = arma::sum(datProportion % weibullS, 1); // Sum over rows



    double logpost_second_sum = - arma::sum(thetas % (1. - logpost_second));

    h = logpost_first + logpost_second_sum + logprior;

    return h;
}


// log-density for coefficient betas
double EvalFunction::log_dens_betas(
    double par,
    void *abc_data)
{
    double h = 0.;

    // Allocation of a zero-initialized memory block of (num*size) bytes
    // 'calloc' returns a void* (generic pointer) on success of memory allocation; type-cast it due to illegal in C++ but illegal in C
    // dataS *mydata_parm = (dataS *)calloc(sizeof(dataS), sizeof(dataS));
    // std::unique_ptr<dataS> mydata_parm = std::make_unique<dataS>();
    // *mydata_parm = *(dataS *)abc_data;
    auto mydata_parm = static_cast<dataS*>(abc_data);

    arma::uvec datEvent(const_cast<unsigned int*>(mydata_parm->datEvent), mydata_parm->N, false);
    arma::vec datTime(const_cast<double*>(mydata_parm->datTime), mydata_parm->N, false);
    arma::mat datX(const_cast<double*>(mydata_parm->datX), mydata_parm->N, mydata_parm->p, false);
    arma::umat gammaIndicator(const_cast<unsigned int*>(mydata_parm->gammaIndicator), mydata_parm->p+1, mydata_parm->L, false);

    arma::mat pars(mydata_parm->currentPars, mydata_parm->p+1, mydata_parm->L, true);
    // arma::mat pars_original(mydata_parm->currentPars, mydata_parm->p, mydata_parm->L, false);
    // arma::mat pars = pars_original; // might be no need to make an extra copy; changing original pointed memory should be fine, since the value of this coordinate will be updated after ARMS
    pars(mydata_parm->jj, mydata_parm->l) = par;

    arma::mat mu_tmp(mydata_parm->datMu, mydata_parm->N, mydata_parm->L, true);
    
    arma::vec pars_l = pars.submat(1, mydata_parm->l, mydata_parm->p, mydata_parm->l) % 
        gammaIndicator.submat(1, mydata_parm->l, mydata_parm->p, mydata_parm->l);
    arma::vec logMu_l = pars(0, mydata_parm->l) + datX * pars_l;
    logMu_l.elem(arma::find(logMu_l > upperbound)).fill(upperbound);
    mu_tmp.col(mydata_parm->l) = arma::exp(logMu_l);

    arma::mat weibullS_tmp(mydata_parm->weibullS, mydata_parm->N, mydata_parm->L, true);
    arma::mat weibull_lambdas = mu_tmp / std::tgamma(1. + 1./mydata_parm->kappa);
    weibullS_tmp.col(mydata_parm->l) = arma::exp( - arma::pow(
                                           datTime / weibull_lambdas.col(mydata_parm->l),
                                           mydata_parm->kappa) );

    // compute log density
    double tau = mydata_parm->tauSq;
    if(mydata_parm->jj == 0)
    {
        tau = mydata_parm->tau0Sq;
    }
    double logprior = - par * par / tau / 2.;

    arma::vec logpost_first = arma::zeros<arma::vec>(mydata_parm->N);
    arma::mat datProportion(mydata_parm->datProportion, mydata_parm->N, mydata_parm->L, false);
    for(unsigned int ll=0; ll<(mydata_parm->L); ++ll)
    {
        logpost_first += datProportion.col(ll) % (mydata_parm->kappa / weibull_lambdas.col(ll)) %
                         arma::pow(datTime/weibull_lambdas.col(ll), mydata_parm->kappa - 1.0) % weibullS_tmp.col(ll);
    }

    double logpost_first_sum = arma::sum( arma::log( logpost_first.elem(arma::find(datEvent)) ) );

    double logpost_second_sum = arma::sum(arma::vec(mydata_parm->datTheta, mydata_parm->N, false) %
                                           datProportion.col(mydata_parm->l) % weibullS_tmp.col(mydata_parm->l));

    h = logpost_first_sum +
        logpost_second_sum +
        logprior;

    return h;
}


// log-density for coefficient zetas
double EvalFunction::log_dens_zetas(
    double par,
    void *abc_data)
{
    double h = 0.;

    // dataS *mydata_parm = (dataS *)calloc(sizeof(dataS), sizeof(dataS));
    // std::unique_ptr<dataS> mydata_parm = std::make_unique<dataS>();
    // *mydata_parm = *(dataS *)abc_data;
    auto mydata_parm = static_cast<dataS*>(abc_data);

    arma::cube datX(const_cast<double*>(mydata_parm->datX), mydata_parm->N, mydata_parm->p, mydata_parm->L, false);
    arma::uvec datEvent(const_cast<unsigned int*>(mydata_parm->datEvent), mydata_parm->N, false);
    arma::mat datProportionConst(const_cast<double*>(mydata_parm->datProportionConst), mydata_parm->N, mydata_parm->L, false);
    arma::umat gammaIndicator(const_cast<unsigned int*>(mydata_parm->gammaIndicator), mydata_parm->p+1, mydata_parm->L, false);

    arma::mat pars(mydata_parm->currentPars, mydata_parm->p+1, mydata_parm->L, true);
    pars(mydata_parm->jj, mydata_parm->l) = par;

    // update proportions based on proposal
    //arma::mat datProportionTmp(mydata_parm->datProportion, mydata_parm->N, mydata_parm->L, true);
    arma::mat alphas = arma::zeros<arma::mat>(mydata_parm->N, mydata_parm->L);

    for(unsigned int ll=0; ll<(mydata_parm->L); ++ll)
    {
        arma::vec pars_l = pars.submat(1, ll, mydata_parm->p, ll) % gammaIndicator.submat(1, ll, mydata_parm->p, ll);
        alphas.col(ll) = arma::exp( pars(0, ll) + datX.slice(ll) * pars_l );
    }
    alphas.elem(arma::find(alphas > upperbound3)).fill(upperbound3);
    alphas.elem(arma::find(alphas < lowerbound)).fill(lowerbound);
    arma::vec alphaRowsum_tmp = arma::sum(alphas, 1);

    // compute log prior
    double w = mydata_parm->wSq;
    if(mydata_parm->jj == 0)
    {
        w = mydata_parm->w0Sq;
    }
    double logprior = - par * par / w / 2.;

    // non-cured density related censored part
    arma::vec logpost_first = arma::zeros<arma::vec>(mydata_parm->N);
    arma::vec logpost_second = arma::zeros<arma::vec>(mydata_parm->N);
    arma::mat weibullS(mydata_parm->weibullS, mydata_parm->N, mydata_parm->L, false);
    arma::mat weibull_lambdas(mydata_parm->weibullLambda, mydata_parm->N, mydata_parm->L, false);
    //arma::mat weibull_lambdas = arma::mat(mydata_parm->datMu, mydata_parm->N, mydata_parm->L, false) / std::tgamma(1. + 1./mydata_parm->kappa);
    //weibullS.elem(arma::find(weibullS < lowerbound)).fill(lowerbound);
    for(unsigned int ll=0; ll<(mydata_parm->L); ++ll)
    {
        //arma::vec tmp = datProportionTmp.col(ll) / alphaRowsum_tmp %  weibullS.col(ll);
        arma::vec tmp = alphas.col(ll) / alphaRowsum_tmp %  weibullS.col(ll);
        logpost_first += arma::pow(weibull_lambdas.col(ll), - mydata_parm->kappa) % tmp;
        logpost_second += tmp;
    }

    double logpost_first_sum = 0.;
    logpost_first_sum = arma::sum( arma::log( logpost_first.elem(arma::find(datEvent)) ) );

    double logpost_second_sum = 0.;
    logpost_second_sum = arma::sum(arma::vec(mydata_parm->datTheta, mydata_parm->N, false) % logpost_second);

    // Dirichlet density
    double log_dirichlet_sum = 0.;
    log_dirichlet_sum = arma::sum(
                            arma::lgamma(alphaRowsum_tmp) - arma::sum(arma::lgamma(alphas), 1) +
                            arma::sum( (alphas - 1.0) % arma::log(datProportionConst), 1 )
                        );

    h = logprior + logpost_first_sum + logpost_second_sum + log_dirichlet_sum;

    return h;
}


// log-density for coefficient betas without BVS
double EvalFunction::log_dens_betasFull(
    double par,
    void* abc_data
)
{
    auto mydata_parm = static_cast<dataS*>(abc_data);

    arma::uvec datEvent(
        const_cast<unsigned int*>(mydata_parm->datEvent),
        mydata_parm->N,
        false
    );

    arma::vec datTime(
        const_cast<double*>(mydata_parm->datTime),
        mydata_parm->N,
        false
    );

    arma::mat datX(
        const_cast<double*>(mydata_parm->datX),
        mydata_parm->N,
        mydata_parm->p,
        false
    );

    // Copy baseline logMu_l. Do not modify the MCMC state inside density.
    arma::vec logMu_l_tmp(
        const_cast<double*>(mydata_parm->logMu_l),
        mydata_parm->N,
        true
    );

    double delta = par - mydata_parm->old_par;

    double tau = mydata_parm->tauSq;

    if (mydata_parm->jj == 0)
    {
        tau = mydata_parm->tau0Sq;
        logMu_l_tmp += delta;
    }
    else
    {
        logMu_l_tmp += datX.col(mydata_parm->jj - 1) * delta;
    }

    logMu_l_tmp.elem(arma::find(logMu_l_tmp > upperbound)).fill(upperbound);

    double logprior = -par * par / tau / 2.0;

    arma::mat mu_tmp(
        mydata_parm->datMu,
        mydata_parm->N,
        mydata_parm->L,
        true
    );

    mu_tmp.col(mydata_parm->l) = arma::exp(logMu_l_tmp);

    arma::mat weibullS_tmp(
        mydata_parm->weibullS,
        mydata_parm->N,
        mydata_parm->L,
        true
    );

    arma::mat weibull_lambdas =
        mu_tmp / std::tgamma(1.0 + 1.0 / mydata_parm->kappa);

    weibullS_tmp.col(mydata_parm->l) = arma::exp(
        -arma::pow(
            datTime / weibull_lambdas.col(mydata_parm->l),
            mydata_parm->kappa
        )
    );

    arma::mat datProportion(
        mydata_parm->datProportion,
        mydata_parm->N,
        mydata_parm->L,
        false
    );

    arma::vec logpost_first(mydata_parm->N, arma::fill::zeros);

    for (unsigned int ll = 0; ll < mydata_parm->L; ++ll)
    {
        logpost_first +=
            datProportion.col(ll) %
            (mydata_parm->kappa / weibull_lambdas.col(ll)) %
            arma::pow(
                datTime / weibull_lambdas.col(ll),
                mydata_parm->kappa - 1.0
            ) %
            weibullS_tmp.col(ll);
    }

    double logpost_first_sum = arma::sum(arma::log(logpost_first.elem(arma::find(datEvent))));

    double logpost_second_sum =
        arma::sum(
            arma::vec(mydata_parm->datTheta, mydata_parm->N, false) %
            datProportion.col(mydata_parm->l) %
            weibullS_tmp.col(mydata_parm->l)
    );

    double h = logpost_first_sum + logpost_second_sum + logprior;

    return h;
}


// log-density for coefficient zetas without BVS
double EvalFunction::log_dens_zetasFull(
    double par,
    void* abc_data
)
{
    auto mydata_parm = static_cast<dataS*>(abc_data);

    const unsigned int N = mydata_parm->N;
    const unsigned int p = mydata_parm->p;
    const unsigned int L = mydata_parm->L;
    const unsigned int l = mydata_parm->l;
    const unsigned int jj = mydata_parm->jj;

    arma::mat datX(
        const_cast<double*>(mydata_parm->datX),
        N,
        p,
        false
    );

    arma::uvec datEvent(
        const_cast<unsigned int*>(mydata_parm->datEvent),
        N,
        false
    );

    arma::mat datProportionConst(
        const_cast<double*>(mydata_parm->datProportionConst),
        N,
        L,
        false
    );

    arma::mat alphas_base(
        const_cast<double*>(mydata_parm->alphas),
        N,
        L,
        false
    );

    arma::vec alphaRowsum_base(
        const_cast<double*>(mydata_parm->alphaRowsum),
        N,
        false
    );

    arma::vec logAlpha_l_base(
        const_cast<double*>(mydata_parm->logAlpha_l),
        N,
        false
    );

    arma::vec alpha_l_base(
        const_cast<double*>(mydata_parm->alpha_l),
        N,
        false
    );

    arma::mat weibullS(
        mydata_parm->weibullS,
        N,
        L,
        false
    );

    arma::mat weibull_lambdas(
        mydata_parm->weibullLambda,
        N,
        L,
        false
    );

    arma::vec datTheta(
        mydata_parm->datTheta,
        N,
        false
    );

    // ------------------------------------------------------------------
    // Candidate alpha_l based on proposed par.
    // Do not mutate current cached state inside density.
    // ------------------------------------------------------------------
    arma::vec logAlpha_l_candidate = logAlpha_l_base;

    double delta = par - mydata_parm->old_par;

    if (jj == 0)
    {
        logAlpha_l_candidate += delta;
    }
    else
    {
        logAlpha_l_candidate += datX.col(jj - 1) * delta;
    }

    logAlpha_l_candidate.elem(
        arma::find(logAlpha_l_candidate > upperbound3)
    ).fill(upperbound3);

    logAlpha_l_candidate.elem(
        arma::find(logAlpha_l_candidate < std::log(lowerbound))
    ).fill(std::log(lowerbound));

    arma::vec alpha_l_candidate = arma::exp(logAlpha_l_candidate);

    alpha_l_candidate.elem(
        arma::find(alpha_l_candidate > upperbound3)
    ).fill(upperbound3);

    alpha_l_candidate.elem(
        arma::find(alpha_l_candidate < lowerbound)
    ).fill(lowerbound);

    arma::vec alphaRowsum_candidate =
        alphaRowsum_base - alpha_l_base + alpha_l_candidate;

    alphaRowsum_candidate.elem(
        arma::find(alphaRowsum_candidate < lowerbound)
    ).fill(lowerbound);

    // ------------------------------------------------------------------
    // Prior.
    // ------------------------------------------------------------------
    double w = mydata_parm->wSq;

    if (jj == 0)
    {
        w = mydata_parm->w0Sq;
    }

    double logprior = -par * par / w / 2.0;

    // ------------------------------------------------------------------
    // Survival part through candidate proportions.
    // ------------------------------------------------------------------
    arma::vec logpost_first(N, arma::fill::zeros);
    arma::vec logpost_second(N, arma::fill::zeros);

    for (unsigned int ll = 0; ll < L; ++ll)
    {
        arma::vec prop_ll;

        if (ll == l)
        {
            prop_ll = alpha_l_candidate / alphaRowsum_candidate;
        }
        else
        {
            prop_ll = alphas_base.col(ll) / alphaRowsum_candidate;
        }

        prop_ll.elem(arma::find(prop_ll < lowerbound)).fill(lowerbound);
        prop_ll %= weibullS.col(ll);

        logpost_first +=
            arma::pow(weibull_lambdas.col(ll), -mydata_parm->kappa) %
            prop_ll;

        logpost_second += prop_ll;
    }

    double logpost_first_sum =
        arma::sum(
            arma::log(logpost_first.elem(arma::find(datEvent)))
        );

    double logpost_second_sum =
        arma::sum(datTheta % logpost_second);

    // ------------------------------------------------------------------
    // Dirichlet part with candidate alpha_l and candidate row sums.
    // ------------------------------------------------------------------

    arma::vec log_dirichlet_vec = arma::lgamma(alphaRowsum_candidate);

    for (unsigned int ll = 0; ll < L; ++ll)
    {
        if (ll == l)
        {
            log_dirichlet_vec -= arma::lgamma(alpha_l_candidate);
            log_dirichlet_vec +=
                (alpha_l_candidate - 1.0) % arma::log(datProportionConst.col(ll));
        }
        else
        {
            log_dirichlet_vec -= arma::lgamma(alphas_base.col(ll));
            log_dirichlet_vec +=
                (alphas_base.col(ll) - 1.0) % arma::log(datProportionConst.col(ll));
        }
    }

    double log_dirichlet_sum = arma::sum(log_dirichlet_vec);

    double h =
        logprior +
        logpost_first_sum +
        logpost_second_sum +
        log_dirichlet_sum;

    return h;
}

// log-density for kappa
double EvalFunction::log_dens_kappa(
    double par,
    void *abc_data)
{
    double h = 0.;

    // dataS *mydata_parm = (dataS *)calloc(sizeof(dataS), sizeof(dataS));
    // std::unique_ptr<dataS> mydata_parm = std::make_unique<dataS>();
    // *mydata_parm = *(dataS *)abc_data;
    auto mydata_parm = static_cast<dataS*>(abc_data);

    double logprior = 0.;
    double logpost_first_sum = 0.;
    double logpost_second_sum = 0.;

    if (mydata_parm->invGamma)
    {
        logprior = - R::dgamma(par, mydata_parm->kappaA, 1.0/mydata_parm->kappaA, true); // equivalent to log(1/dgamma(par, a, b))
    }
    else
    {
        logprior = R::dgamma(par, mydata_parm->kappaA, 1.0/mydata_parm->kappaA, true);
    }

    arma::vec logpost_first = arma::zeros<arma::vec>(mydata_parm->N);
    arma::vec logpost_second = arma::zeros<arma::vec>(mydata_parm->N);
    arma::mat datMu(mydata_parm->datMu, mydata_parm->N, mydata_parm->L, false);
    arma::mat datProportion(mydata_parm->datProportion, mydata_parm->N, mydata_parm->L, false);
    arma::vec datTime(const_cast<double*>(mydata_parm->datTime), mydata_parm->N, false);
    arma::uvec datEvent(const_cast<unsigned int*>(mydata_parm->datEvent), mydata_parm->N, false);
    for(unsigned int ll=0; ll<(mydata_parm->L); ++ll)
    {
        arma::vec weibull_lambdas_tmp = datMu.col(ll) / std::tgamma(1.0+1.0/par);
        arma::vec lambdas_tmp = arma::pow( datTime/weibull_lambdas_tmp, par);
        lambdas_tmp.elem(arma::find(lambdas_tmp > upperbound)).fill(upperbound);
        arma::vec weibullS_tmp = arma::exp(- lambdas_tmp);

        logpost_first += datProportion.col(ll) % (par/weibull_lambdas_tmp) %
                         arma::pow(datTime/weibull_lambdas_tmp, par-1.0) % weibullS_tmp;
        logpost_second += datProportion.col(ll) % weibullS_tmp;
    }

    logpost_first_sum = arma::sum( arma::log( logpost_first.elem(arma::find(datEvent)) ) );

    logpost_second_sum = arma::sum(arma::vec(mydata_parm->datTheta, mydata_parm->N, false) % logpost_second);

    h = logprior + logpost_first_sum + logpost_second_sum;

    return h;
}
