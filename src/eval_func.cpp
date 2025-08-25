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

    //if (par >= *(mydata_parm->xl) && par <= *(mydata_parm->xr)) // this is judged in ARMS::initial()
    //{
    arma::vec xis(mydata_parm->currentPars, mydata_parm->p, false);
    // arma::vec xis_original(mydata_parm->currentPars, mydata_parm->p, false);
    // arma::vec xis = xis_original;

    xis[mydata_parm->jj] = par;
    double vSq = mydata_parm->vSq;
    if (mydata_parm->jj == 0)
    {
        vSq = mydata_parm->v0Sq;
        // // stdvec xis0 = arma::conv_to<stdvec>::from(mydata_parm->currentPars);
        // // stdvec xis0 = arma::conv_to<stdvec>::from(xis);
        // // xis0.erase(xis0.begin());
        // // int ans = std::count(xis0.begin(), xis0.end(), 0.);
        // // double vA_tmp = mydata_parm->vA + 0.5 * (double)ans;
        // arma::vec xis0 = xis;
        // xis0.shed_row(0);
        // double vA_tmp = mydata_parm->vA + 0.5 * arma::accu(xis0 != 0.);
        // double vB_tmp = mydata_parm->vB + 0.5 * arma::as_scalar(xis0.t() * xis0);
        //   // std::inner_product( xis0.begin(), xis0.end(), xis0.begin(), 0. );
        // vSq = 1. / R::rgamma(vA_tmp, 1. / vB_tmp);
    }

    // compute the log density
    double logprior = - par * par / vSq / 2.;

    //arma::mat X(mydata_parm->datX, n, p, false);  // use auxiliary memory
    //std::cout << "...X:\n" << X << "\n";

    arma::mat datX(const_cast<double*>(mydata_parm->datX), mydata_parm->N, mydata_parm->p, false);
    arma::uvec datEvent(const_cast<unsigned int*>(mydata_parm->datEvent), mydata_parm->N, false);

    //arma::vec logTheta = mydata_parm->datX * xis;
    arma::vec logTheta = datX * xis;
    logTheta.elem(arma::find(logTheta > upperbound)).fill(upperbound);
    arma::vec thetas = arma::exp( logTheta );

    //double logpost_first = arma::accu( logTheta.elem(arma::find(mydata_parm->datEvent)) );
    double logpost_first = arma::accu( logTheta.elem(arma::find(datEvent)) );
    arma::vec logpost_second = arma::zeros<arma::vec>(mydata_parm->N);
    arma::mat datProportion(mydata_parm->datProportion, mydata_parm->N, mydata_parm->L, false);
    arma::mat weibullS(mydata_parm->weibullS, mydata_parm->N, mydata_parm->L, false);
    // for(int ll=0; ll<(mydata_parm->L); ++ll)
    // {
    //     //logpost_second += mydata_parm->datProportion.col(ll) % mydata_parm->weibullS.col(ll);
    //     logpost_second += datProportion.col(ll) % weibullS.col(ll);
    // }
    logpost_second = arma::sum(datProportion % weibullS, 1); // Sum over rows



    double logpost_second_sum = - arma::accu(thetas % (1. - logpost_second));

    h = logpost_first + logpost_second_sum + logprior;
    //} else {
    //  h = -1.0e100;
    //}

    // std::cout << "...debug log_dens_xis h=" << h << "\n";

    // free(mydata_parm);

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

    arma::mat pars(mydata_parm->currentPars, mydata_parm->p+1, mydata_parm->L, false);
    // arma::mat pars_original(mydata_parm->currentPars, mydata_parm->p, mydata_parm->L, false);
    // arma::mat pars = pars_original; // might be no need to make an extra copy; changing original pointed memory should be fine, since the value of this coordinate will be updated after ARMS
    pars(mydata_parm->jj, mydata_parm->l) = par;

    arma::mat mu_tmp(mydata_parm->datMu, mydata_parm->N, mydata_parm->L, false);
    // arma::mat mu_original(mydata_parm->datMu, mydata_parm->N, mydata_parm->L, false);
    // arma::mat mu_tmp = mu_original;
    
    // arma::vec logMu_l = datX * pars.col(mydata_parm->l);
    arma::vec logMu_l = pars(0, mydata_parm->l) + datX * pars.submat(1, mydata_parm->l, mydata_parm->p, mydata_parm->l);
    logMu_l.elem(arma::find(logMu_l > upperbound)).fill(upperbound);
    mu_tmp.col(mydata_parm->l) = arma::exp(logMu_l);

    arma::mat weibullS_tmp(mydata_parm->weibullS, mydata_parm->N, mydata_parm->L, false);
    // arma::mat weibullS_original(mydata_parm->weibullS, mydata_parm->N, mydata_parm->L, false);
    // arma::mat weibullS_tmp = weibullS_original;
    //arma::mat lambdas = mu_tmp / std::tgamma(1. + 1./mydata_parm->kappa);
    //weibullS_tmp.col(mydata_parm->l) = arma::exp( -arma::pow(mydata_parm->datTime / lambdas.col(mydata_parm->l), mydata_parm->kappa) );
    arma::mat weibull_lambdas = mu_tmp / std::tgamma(1. + 1./mydata_parm->kappa);
    weibullS_tmp.col(mydata_parm->l) = arma::exp( - arma::pow(
                                           datTime / weibull_lambdas.col(mydata_parm->l),
                                           mydata_parm->kappa) );

    // compute log density
    // double logprior = - par * par / mydata_parm->tauSq / 2.;
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
        // logpost_first += datProportion.col(ll) %
        //   arma::pow(weibull_lambdas.col(ll), - mydata_parm->kappa) % weibullS_tmp.col(ll);
        logpost_first += datProportion.col(ll) % (mydata_parm->kappa / weibull_lambdas.col(ll)) %
                         arma::pow(datTime/weibull_lambdas.col(ll), mydata_parm->kappa - 1.0) % weibullS_tmp.col(ll);
    }

    //double logpost_first_sum = arma::accu( arma::log( logpost_first.elem(arma::find(mydata_parm->datEvent)) ) );
    double logpost_first_sum = arma::accu( arma::log(
            logpost_first.elem(arma::find(datEvent))
                                           ) );

    //double logpost_second_sum = arma::accu(mydata_parm->datTheta %
    //  mydata_parm->datProportion.col(mydata_parm->l) % weibullS_tmp.col(mydata_parm->l));
    double logpost_second_sum = arma::accu(arma::vec(mydata_parm->datTheta, mydata_parm->N, false) %
                                           datProportion.col(mydata_parm->l) % weibullS_tmp.col(mydata_parm->l));

    h = logpost_first_sum +
        logpost_second_sum +
        logprior;

    // std::cout << "...debug log_dens_betas h=" << h << "\n";

    // std::cout << "...h=" << h <<
    // "; l=" << mydata_parm->l <<
    // "; logpost_first_sum=" << logpost_first_sum <<
    // "; max(mu_tmp)=" << arma::max(mu_tmp.col(mydata_parm->l)) <<
    // "; max(weibull_lambdas)=" << arma::max(weibull_lambdas.col(mydata_parm->l)) <<
    // "; logpost_second_sum=" << logpost_second_sum <<
    // "; logprior=" << logprior <<
    // "\n";

    // free(mydata_parm);
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
    arma::mat datProportionConst_tmp(const_cast<double*>(mydata_parm->datProportionConst), mydata_parm->N, mydata_parm->L, false);

    arma::mat pars(mydata_parm->currentPars, mydata_parm->p+1, mydata_parm->L, false);
    // arma::mat pars_original(mydata_parm->currentPars, mydata_parm->p+1, mydata_parm->L, false);
    // arma::mat pars = pars_original;
    pars(mydata_parm->jj, mydata_parm->l) = par;

    // update proportions based on proposal
    //arma::mat datProportionTmp(mydata_parm->datProportion, mydata_parm->N, mydata_parm->L, true);
    arma::mat alphas = arma::zeros<arma::mat>(mydata_parm->N, mydata_parm->L);

    for(unsigned int ll=0; ll<(mydata_parm->L); ++ll)
    {
        alphas.col(ll) = arma::exp( pars(0, ll) + datX.slice(ll) * pars.submat(1, ll, mydata_parm->p, ll) );
    }
    alphas.elem(arma::find(alphas > upperbound3)).fill(upperbound3);
    alphas.elem(arma::find(alphas < lowerbound)).fill(lowerbound);
    arma::vec alphas_Rowsum = arma::sum(alphas, 1);
// /*
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
        //arma::vec tmp = datProportionTmp.col(ll) / alphas_Rowsum %  weibullS.col(ll);
        arma::vec tmp = alphas.col(ll) / alphas_Rowsum %  weibullS.col(ll);
        logpost_first += arma::pow(weibull_lambdas.col(ll), - mydata_parm->kappa) % tmp;
        logpost_second += tmp;
    }

    double logpost_first_sum = 0.;
    logpost_first_sum = arma::accu( arma::log( logpost_first.elem(arma::find(datEvent)) ) );

    double logpost_second_sum = 0.;
    logpost_second_sum = arma::accu(arma::vec(mydata_parm->datTheta, mydata_parm->N, false) % logpost_second);

    // Dirichlet density
    /*
    arma::vec log_dirichlet = arma::zeros<arma::vec>(mydata_parm->N);
    for(int i=0; i<(mydata_parm->N); ++i)
    {
      double rowSum_lgamma_alphas = 0;
      for(int ll=0; ll<(mydata_parm->L); ++l)
      {
        rowSum_lgamma_alphas += std::lgamma(alphas(i,ll));
      }
      log_dirichlet[i] = std::lgamma(alphas_Rowsum[i]) - rowSum_lgamma_alphas +
        arma::accu((alphas.row(i)-1.0) % arma::log(datProportionConst_tmp.row(i)));
    }
    double log_dirichlet_sum = arma::accu(log_dirichlet)
    */
    double log_dirichlet_sum = 0.;
    log_dirichlet_sum = arma::accu(
                            arma::lgamma(alphas_Rowsum) - arma::sum(arma::lgamma(alphas), 1) +
                            arma::sum( (alphas - 1.0) % arma::log(datProportionConst_tmp), 1 )
                        );

    h = logprior + logpost_first_sum + logpost_second_sum + log_dirichlet_sum;

    // std::cout << "...debug log_dens_zetas: h=" << h <<
    // "; logprior=" << logprior <<
    // "; logpost_first_sum=" << logpost_first_sum <<
    // "; logpost_second_sum=" << logpost_second_sum <<
    // "; log_dirichlet_sum=" << log_dirichlet_sum <<
    // "\n";

    // free(mydata_parm);
    return h;
}

// density of truncated normal distribution
double EvalFunction::pdfTruncNorm(double x, double m, double sd, double lower, double upper)
{
    // find quantiles that correspond the the given low and high levels
    double xi = (x - m) / sd;
    double alpha = (lower - m) / sd;
    double beta = (upper - m) / sd;
    double Z = R::qnorm( beta, m, sd, true, false ) - R::qnorm( alpha, m, sd, true, false );

    double ret = 1./std::sqrt(2.0*M_PI) * std::exp(-0.5*xi*xi) / sd / Z;

    return ret;
}

// log-density for Dirichlet dispersion phi if the alternative parametrization is used
double EvalFunction::log_dens_phi(
    double par,
    void *abc_data)
{
    double h = 0.;

    // dataS *mydata_parm = (dataS *)calloc(sizeof(dataS), sizeof(dataS));
    // std::unique_ptr<dataS> mydata_parm = std::make_unique<dataS>();
    // *mydata_parm = *(dataS *)abc_data;
    auto mydata_parm = static_cast<dataS*>(abc_data);

    arma::mat datProportionConst_tmp(const_cast<double*>(mydata_parm->datProportionConst), mydata_parm->N, mydata_parm->L, false);

    arma::mat concentrations = par *
                               arma::mat(mydata_parm->datProportion, mydata_parm->N, mydata_parm->L, false);

    arma::vec normalizingConst = arma::log(arma::tgamma(arma::sum(concentrations, 1))) -
                                 arma::sum(arma::log(arma::tgamma(concentrations)), 1);

    arma::vec geometricTerm = arma::sum((concentrations - 1.) % arma::log(datProportionConst_tmp), 1);

    double sd = std::sqrt(mydata_parm->Delta / 3.);
    double logprior = std::log(pdfTruncNorm(par, 0., sd, 0., 1.0e+6));

    h = logprior + arma::accu(normalizingConst + geometricTerm);

    // std::cout << "...debug log_dens_phi h=" << h << "\n";

    // free(mydata_parm);
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
        // weibull_lambdas_tmp.elem(arma::find(weibull_lambdas_tmp < lowerbound)).fill(lowerbound);
        // arma::vec weibullS_tmp = arma::exp(- arma::pow(datTime/weibull_lambdas_tmp, par));
        arma::vec lambdas_tmp = arma::pow( datTime/weibull_lambdas_tmp, par);
        lambdas_tmp.elem(arma::find(lambdas_tmp > upperbound)).fill(upperbound);
        arma::vec weibullS_tmp = arma::exp(- lambdas_tmp);

        //logpost_first += datProportion.col(ll) % arma::pow(weibull_lambdas_tmp, -par) % weibullS_tmp;
        logpost_first += datProportion.col(ll) % (par/weibull_lambdas_tmp) %
                         arma::pow(datTime/weibull_lambdas_tmp, par-1.0) % weibullS_tmp;
        logpost_second += datProportion.col(ll) % weibullS_tmp;
    }

    //logpost_first %= arma::pow(datTime, par - 1.0) * par;
    logpost_first_sum = arma::accu( arma::log( logpost_first.elem(arma::find(datEvent)) ) );

    logpost_second_sum = arma::accu(arma::vec(mydata_parm->datTheta, mydata_parm->N, false) % logpost_second);

    h = logprior + logpost_first_sum + logpost_second_sum;

    // std::cout << "...debug log_dens_kappa h=" << h << "\n";
    // free(mydata_parm);
    return h;
}
