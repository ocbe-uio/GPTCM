// Gibbs sampling for univariate and multivariate ARMS

#include "arms_gibbs.h"
#include "simple_gibbs.h"


//' Multivariate ARMS via Gibbs sampler for xi
//'
//' @param n Number of samples to draw
//' @param nsamp How many samples to draw for generating each sample; only the last draw will be kept
//' @param ninit Number of initials as meshgrid values for envelop search
//' @param convex Adjustment for convexity (non-negative value, default 1.0)
//' @param npoint Maximum number of envelope points
//'
void ARMS_Gibbs::arms_gibbs_xi(
    const armsParmClass& armsPar,
    arma::vec& currentPars,
    double v0Sq,
    double vSq,
    arma::mat& datProportion,
    arma::mat& weibullS,
    const DataClass &dataclass)
{
    // number of parameters to be updated
    unsigned int p = currentPars.n_elem; 
    unsigned int L = datProportion.n_cols;
    unsigned int N = datProportion.n_rows;

    const double minD = armsPar.xiMin;
    const double maxD = armsPar.xiMax;

    // reallocate struct variables

    // // double xinit[armsPar.ninit];
    std::vector<double> xinit(armsPar.ninit); // Use std::vector instead of VLA to avoid warning about varying length of array
    arma::vec xinit0 = arma::linspace( minD+1.0e-10, maxD-1.0e-10, armsPar.ninit );
    for (unsigned int i = 0; i < armsPar.ninit; ++i)
        xinit[i] = xinit0[i];

    // dataS *mydata = (dataS *)malloc(sizeof (dataS));
    auto mydata = std::make_unique<dataS>(); // modern and safe memory allocation
    mydata->currentPars = currentPars.memptr();
    mydata->p = p;
    mydata->L = L;
    mydata->N = N;
    mydata->vSq = vSq;
    mydata->v0Sq = v0Sq;
    mydata->datProportion = datProportion.memptr();
    mydata->weibullS = weibullS.memptr();
    mydata->datX = dataclass.datX0.memptr();
    mydata->datEvent = dataclass.datEvent.memptr();

    for (unsigned int j = 0; j < p; ++j)
    {
        mydata->jj = j;

        double xsamp = currentPars[j];
        slice_sample(
            EvalFunction::log_dens_xis,
            mydata.get(),
            xsamp,
            10,
            1.0,
            minD,
            maxD
        );
        currentPars[j] = xsamp;
        // free(xsamp);
    }

    // free(mydata);
}



//' Multivariate ARMS via Gibbs sampler for beta
//'
//' @param n Number of samples to draw
//' @param nsamp How many samples to draw for generating each sample; only the last draw will be kept
//' @param ninit Number of initials as meshgrid values for envelop search
//' @param convex Adjustment for convexity (non-negative value, default 1.0)
//' @param npoint Maximum number of envelope points
//'
void ARMS_Gibbs::arms_gibbs_beta(
    const armsParmClass& armsPar,
    arma::mat& currentPars,
    arma::vec& tauSq,
    double tau0Sq,
    const arma::mat& pseudoMean,
    const arma::mat& pseudoVar,
    arma::umat gammas,
    double kappa,
    arma::vec& datTheta,
    arma::mat& datMu,
    arma::mat& datProportion,
    arma::mat& weibullS,
    arma::mat& weibullLambda,
    const DataClass &dataclass
)
{
    /* make a subfunction arms_gibbs for only vector betas that can be used for (varying-length) variable selected vector*/

    // dimensions
    unsigned int N = dataclass.datX.n_rows;
    unsigned int p = dataclass.datX.n_cols;
    unsigned int L = dataclass.datX.n_slices;

    // objects for arms()
    double minD = armsPar.betaMin;
    double maxD = armsPar.betaMax;

    std::vector<double> xinit(armsPar.ninit); // Use std::vector instead of VLA
    arma::vec xinit0 = arma::linspace( minD+1.0e-10, maxD-1.0e-10, armsPar.ninit );
    for (unsigned int i = 0; i < armsPar.ninit; ++i)
        xinit[i] = xinit0[i];  

    // dataS *mydata = (dataS *)malloc(sizeof (dataS));
    auto mydata = std::make_unique<dataS>(); // modern and safe memory allocation

    gammas = arma::join_cols(arma::ones<arma::urowvec>(L), gammas);
    mydata->gammaIndicator = gammas.memptr();
    // currentPars.elem(arma::find(gammas == 0)).fill(0.);
    mydata->currentPars = currentPars.memptr();
    mydata->p = p;
    mydata->L = L;
    mydata->N = N;
    mydata->tau0Sq = tau0Sq;
    mydata->kappa = kappa;
    mydata->datTheta = datTheta.memptr();
    mydata->datMu = datMu.memptr();
    mydata->datProportion = datProportion.memptr();
    mydata->weibullS = weibullS.memptr();
    mydata->datEvent = dataclass.datEvent.memptr();
    mydata->datTime = dataclass.datTime.memptr();

    arma::vec logMu_l = arma::zeros<arma::vec>(N);
    mydata->logMu_l = logMu_l.memptr();
    arma::vec mu_tmp = arma::zeros<arma::vec>(N);
    double GammaFuncKappa = std::tgamma(1. + 1./kappa);

    // bool inactive_var_zero = pseudoVar.is_zero();

    // not easy to parallize the following for-loop due to data dependencies
    for (unsigned int l = 0; l < L; ++l)
    {
        // Gibbs sampling
        mydata->tauSq = tauSq[l];

        logMu_l = currentPars(0, l) + dataclass.datX.slice(l) * 
            (currentPars.submat(1, l, p, l) % gammas.submat(1, l, p, l)) ;

        for (unsigned int j = 0; j < p+1; ++j)
        {
            if (!gammas(j, l))
            {
                if (j > 0) 
                {
                    // currentPars(j, l) = R::rnorm(0., std::sqrt(tauSq[l]));
                    // if (inactive_var_zero) 
                    //     currentPars(j, l) = R::rnorm(0.0, std::sqrt(tauSq[l]));
                    // else
                    currentPars(j, l) = R::rnorm(pseudoMean(j-1,l), std::sqrt(pseudoVar(j-1,l)));
                }
            }
            else
            {
                mydata->jj = j;
                mydata->l = l;
                mydata->datX = dataclass.datX.slice(l).memptr();


                // parameters for ARMS
                double old_par = currentPars(j, l);
                double xprev = old_par;
                // double *xsamp = (double*)malloc(armsPar.nsamp * sizeof(double));
                std::vector<double> xsamp(armsPar.nsamp);

                double qcent[1], xcent[1];
                int neval, ncent = 0;
                int err;
                double convex = armsPar.convex;
                    err = ARMS::arms (
                              xinit.data(), armsPar.ninit, &minD, &maxD,
                              EvalFunction::log_dens_betas, mydata.get(),
                              &convex, armsPar.npoint,
                              armsPar.metropolis, &xprev, xsamp.data(),
                              armsPar.nsamp, qcent, xcent, ncent, &neval);

                // check ARMS validity
                if (err > 0)
                    Rprintf("In arms_gibbs_beta(): error code in ARMS = %d.\n", err);
                if (std::isnan(xsamp[armsPar.nsamp-1]))
                    Rprintf("In arms_gibbs_beta(): NaN generated, possibly due to overflow in (log-)density (e.g. with densities involving exp(exp(...))).\n");
                if (xsamp[armsPar.nsamp-1] < minD || xsamp[armsPar.nsamp-1] > maxD)
                    Rprintf("In arms_gibbs_beta(): %d-th sample out of range [%f, %f] (fused domain). Got %f.\n", armsPar.nsamp, minD, maxD, xsamp[armsPar.nsamp-1]);

                double new_par = xsamp[armsPar.nsamp - 1];
                currentPars(j, l) = new_par;


                // update quantities needed for ARMS updates

                double accepted_delta = new_par - old_par;
                if (j == 0) {        
                    logMu_l += accepted_delta;    
                } else {        
                    logMu_l += dataclass.datX.slice(l).col(j - 1) * accepted_delta;
                }    

                // logMu_l.elem(arma::find(logMu_l > upperbound)).fill(upperbound);
                // logMu_l = arma::min(logMu_l, arma::vec(N).fill(upperbound)); 
                mu_tmp = arma::exp( logMu_l );
                mu_tmp = arma::min(mu_tmp, arma::vec(N).fill(upperbound)); 
                mu_tmp = arma::max(mu_tmp, arma::vec(N).fill(lowerbound)); 
                datMu.col(l) = mu_tmp; 
                // arma::vec lambdas = datMu.col(l) / std::tgamma(1. + 1./kappa);
                // weibullS.col(l) = arma::exp( -arma::pow(datTime / lambdas, kappa) );
                // weibullLambda.col(l) = arma::pow( dataclass.datTime / (datMu.col(l) / std::tgamma(1. + 1./kappa)), kappa);
                // weibullLambda.elem(arma::find(lambdas > upperbound)).fill(upperbound);
                mu_tmp /= GammaFuncKappa;
                // weibullLambda.col(l) = mu_tmp / GammaFuncKappa;
                weibullS.col(l) = arma::exp(        
                    -arma::pow(            
                        dataclass.datTime / mu_tmp, //weibullLambda.col(l),           
                        kappa        
                    )    
                );
                // free(xsamp);
            }
        }
    }

    // free(mydata);
}


// Multivariate ARMS via Gibbs sampler for betaK; used for M-H sampling for gammas update
// NOTE: ARMS_Gibbs::arms_gibbs_betaK() is not used!!!
/*
void ARMS_Gibbs::arms_gibbs_betaK(
    const unsigned int k,
    const armsParmClass& armsPar,
    arma::mat& currentPars,
    double tau0Sq,
    double tauSqK,
    arma::umat gammas,
    double kappa,
    arma::vec& datTheta,
    arma::mat& datMu,
    arma::mat& datProportion,
    arma::mat& weibullS,
    const DataClass &dataclass
)
{
    // make a subfunction arms_gibbs for only vector betas that can be used for (varying-length) variable selected vector

    // dimensions
    unsigned int N = dataclass.datX.n_rows;
    unsigned int p = dataclass.datX.n_cols;
    unsigned int L = dataclass.datX.n_slices;

    // objects for arms()
    double minD = armsPar.betaMin;
    double maxD = armsPar.betaMax;

    std::vector<double> xinit(armsPar.ninit); // Use std::vector instead of VLA
    arma::vec xinit0 = arma::linspace( minD+1.0e-10, maxD-1.0e-10, armsPar.ninit );
    for (unsigned int i = 0; i < armsPar.ninit; ++i)
        xinit[i] = xinit0[i];

    dataS *mydata = (dataS *)malloc(sizeof (dataS));

    gammas = arma::join_cols(arma::ones<arma::urowvec>(L), gammas);
    mydata->gammaIndicator = gammas.memptr();
    // currentPars.elem(arma::find(gammas == 0)).fill(0.);
    mydata->currentPars = currentPars.memptr();
    mydata->p = p;
    mydata->L = L;
    mydata->N = N;
    mydata->tau0Sq = tau0Sq;
    mydata->tauSq = tauSqK;
    mydata->kappa = kappa;
    mydata->datTheta = datTheta.memptr();
    mydata->datMu = datMu.memptr();
    mydata->datProportion = datProportion.memptr();
    mydata->weibullS = weibullS.memptr();
    mydata->datEvent = dataclass.datEvent.memptr();
    mydata->datTime = dataclass.datTime.memptr();

    unsigned int l = k;
    // Gibbs sampling
    // here only for proposal betas conditional on proposal gammas, no need to update tauSq

    for (unsigned int j = 0; j < p+1; ++j)
    {
        if (!gammas(j, l))
        {
            if (j > 0)
                currentPars(j, l) = R::rnorm(0., std::sqrt(tauSqK));
        }
        else
        {
            mydata->jj = j;
            mydata->l = l;
            mydata->datX = dataclass.datX.slice(l).memptr();

            double xprev = currentPars(j, l);
            std::vector<double> xsamp(armsPar.nsamp);

            double qcent[1], xcent[1];
            int neval, ncent = 0;
            int err;
            double convex = armsPar.convex;
            err = ARMS::arms (
                          xinit.data(), armsPar.ninit, &minD, &maxD,
                          EvalFunction::log_dens_betas, mydata,
                          &convex, armsPar.npoint,
                          armsPar.metropolis, &xprev, xsamp.data(),
                          armsPar.nsamp, qcent, xcent, ncent, &neval);

            // check ARMS validity
            if (err > 0)
                Rprintf("In arms_gibbs_beta(): error code in ARMS = %d.\n", err);
            if (std::isnan(xsamp[armsPar.nsamp-1]))
                Rprintf("In arms_gibbs_beta(): NaN generated, possibly due to overflow in (log-)density (e.g. with densities involving exp(exp(...))).\n");
            if (xsamp[armsPar.nsamp-1] < minD || xsamp[armsPar.nsamp-1] > maxD)
                Rprintf("In arms_gibbs_beta(): %d-th sample out of range [%f, %f] (fused domain). Got %f.\n", armsPar.nsamp, minD, maxD, xsamp[armsPar.nsamp-1]);

            currentPars(j, l) = xsamp[armsPar.nsamp - 1];

            // Update relevant quantities based on updated 'currentPars'

            arma::vec logMu_l = currentPars(0, l) + dataclass.datX.slice(l) * (currentPars.submat(1, l, p, l) % gammas.submat(1, l, p, l));
            logMu_l.elem(arma::find(logMu_l > upperbound)).fill(upperbound);
            datMu.col(l) = arma::exp( logMu_l );
            arma::vec lambdas = arma::pow( dataclass.datTime / (datMu.col(l) / std::tgamma(1. + 1./kappa)), kappa);
            lambdas.elem(arma::find(lambdas > upperbound)).fill(upperbound);
            weibullS.col(l) = arma::exp( -lambdas );

        }
    }

    // free(mydata);
}
*/

// Gibbs for betas without BVS
void ARMS_Gibbs::arms_gibbs_betaFull(
    const armsParmClass& armsPar,
    arma::mat& currentPars,
    arma::vec& tauSq,
    double tau0Sq,
    double kappa,
    arma::vec& datTheta,
    arma::mat& datMu,
    arma::mat& datProportion,
    arma::mat& weibullS,
    arma::mat& weibullLambda,
    const DataClass &dataclass
)
{
    /* make a subfunction arms_gibbs for only vector betas that can be used for (varying-length) variable selected vector*/

    // dimensions
    unsigned int N = dataclass.datX.n_rows;
    unsigned int p = dataclass.datX.n_cols;
    unsigned int L = dataclass.datX.n_slices;

    // objects for arms()
    double minD = armsPar.betaMin;
    double maxD = armsPar.betaMax;

    std::vector<double> xinit(armsPar.ninit); // Use std::vector instead of VLA
    arma::vec xinit0 = arma::linspace( minD+1.0e-10, maxD-1.0e-10, armsPar.ninit );
    for (unsigned int i = 0; i < armsPar.ninit; ++i)
        xinit[i] = xinit0[i];  

    // reallocate struct variables
    // dataS *mydata = (dataS *)malloc(sizeof (dataS));
    auto mydata = std::make_unique<dataS>(); // modern and safe memory allocation

    mydata->currentPars = currentPars.memptr();
    mydata->p = p;
    mydata->L = L;
    mydata->N = N;
    mydata->tau0Sq = tau0Sq;
    mydata->kappa = kappa;
    mydata->datTheta = datTheta.memptr();
    mydata->datMu = datMu.memptr();
    mydata->datProportion = datProportion.memptr();
    mydata->weibullS = weibullS.memptr();
    mydata->datEvent = dataclass.datEvent.memptr();
    mydata->datTime = dataclass.datTime.memptr();

    arma::vec logMu_l = arma::zeros<arma::vec>(N);
    mydata->logMu_l = logMu_l.memptr();
    arma::vec mu_tmp(N);
    double GammaFuncKappa = std::tgamma(1. + 1./kappa);

    // not easy to parallize the following for-loop due to data dependencies
    for (unsigned int l = 0; l < L; ++l)
    {
        // Gibbs sampling
        mydata->tauSq = tauSq[l];

        logMu_l = currentPars(0, l) + dataclass.datX.slice(l) * currentPars.submat(1, l, p, l);
        for (unsigned int j = 0; j < p+1; ++j)
        {
            mydata->jj = j;
            mydata->l = l;
            mydata->datX = dataclass.datX.slice(l).memptr();
            double old_par = currentPars(j, l);
            mydata->old_par = old_par;
            
            double xprev = old_par; //currentPars(j, l);
            std::vector<double> xsamp(armsPar.nsamp);

            double qcent[1], xcent[1];
            int neval, ncent = 0;

            int err;
            double convex = armsPar.convex;
                err = ARMS::arms (
                            xinit.data(), armsPar.ninit, &minD, &maxD,
                            EvalFunction::log_dens_betasFull, mydata.get(),
                            &convex, armsPar.npoint,
                            armsPar.metropolis, &xprev, xsamp.data(),
                            armsPar.nsamp, qcent, xcent, ncent, &neval);

            // check ARMS validity
            if (err > 0)
                Rprintf("In arms_gibbs_beta(): error code in ARMS = %d.\n", err);
            if (std::isnan(xsamp[armsPar.nsamp-1]))
                Rprintf("In arms_gibbs_beta(): NaN generated, possibly due to overflow in (log-)density (e.g. with densities involving exp(exp(...))).\n");
            if (xsamp[armsPar.nsamp-1] < minD || xsamp[armsPar.nsamp-1] > maxD)
                Rprintf("In arms_gibbs_beta(): %d-th sample out of range [%f, %f] (fused domain). Got %f.\n", armsPar.nsamp, minD, maxD, xsamp[armsPar.nsamp-1]);

            double new_par = xsamp[armsPar.nsamp - 1];
            currentPars(j, l) = new_par;

            // update quantities needed for ARMS updates
            double accepted_delta = new_par - old_par;
            if (j == 0) {        
                logMu_l += accepted_delta;    
            } else {        
                logMu_l += dataclass.datX.slice(l).col(j - 1) * accepted_delta;
            }    
            // logMu_l.elem(arma::find(logMu_l > upperbound)).fill(upperbound);    
            // logMu_l = arma::min(logMu_l, arma::vec(N).fill(upperbound)); 
            // datMu.col(l) = mu_tmp; 
            // datMu.col(l) = arma::exp(logMu_l);   
            mu_tmp = arma::exp( logMu_l );
            mu_tmp = arma::min(mu_tmp, arma::vec(N).fill(upperbound)); 
            mu_tmp = arma::max(mu_tmp, arma::vec(N).fill(lowerbound)); 
            datMu.col(l) = mu_tmp;   
            weibullLambda.col(l) = datMu.col(l) / GammaFuncKappa;
            weibullS.col(l) = arma::exp(        
                -arma::pow(            
                    dataclass.datTime / weibullLambda.col(l),           
                    kappa        
                )    
            );
        }
    }

    // free(mydata);
}


//' Multivariate ARMS via Gibbs sampler for zeta
//'
//' @param n Number of samples to draw
//' @param nsamp How many samples to draw for generating each sample; only the last draw will be kept
//' @param ninit Number of initials as meshgrid values for envelop search
//' @param convex Adjustment for convexity (non-negative value, default 1.0)
//' @param npoint Maximum number of envelope points
//' @param dirichlet Not yet implemented
//'
void ARMS_Gibbs::arms_gibbs_zeta(
    const armsParmClass& armsPar,
    arma::mat& currentPars,
    double w0Sq,
    arma::vec& wSq,
    const arma::mat& pseudoMean,
    const arma::mat& pseudoVar,

    arma::umat etas,

    double kappa,
    bool dirichlet,
    arma::vec& datTheta,
    arma::mat& weibullS,
    arma::mat& weibullLambda,
    const DataClass &dataclass
    // double& logPosteriorZeta
)
{
    /* make a subfunction arms_gibbs for only vector betas that can be used for (varying-length) variable selected vector*/

    // dimensions
    unsigned int N = dataclass.datX.n_rows;
    unsigned int p = dataclass.datX.n_cols;
    unsigned int L = dataclass.datX.n_slices;

    // logPosteriorZeta = 0.;
    // int armsPar.metropolis = metropolis;

    // objects for arms()
    double minD = armsPar.zetaMin;
    double maxD = armsPar.zetaMax;

    // reallocate struct variables

    std::vector<double> xinit(armsPar.ninit); // Use std::vector instead of VLA
    arma::vec xinit0 = arma::linspace( minD+1.0e-10, maxD-1.0e-10, armsPar.ninit );
    for (unsigned int i = 0; i < armsPar.ninit; ++i)
        xinit[i] = xinit0[i];

    if (!dirichlet)
        Rprintf("Warning: In arms_gibbs_zeta(), Dirichlet modeling with logit/alr-link is not implement!\n");

    // dataS *mydata = (dataS *)malloc(sizeof (dataS));
    auto mydata = std::make_unique<dataS>(); // modern and safe memory allocation

    etas = arma::join_cols(arma::ones<arma::urowvec>(L), etas);
    mydata->gammaIndicator = etas.memptr();
    // currentPars.elem(arma::find(etas == 0)).fill(0.);
    mydata->currentPars = currentPars.memptr();
    mydata->p = p;
    mydata->L = L;
    mydata->N = N;
    mydata->w0Sq = w0Sq;
    mydata->kappa = kappa;
    mydata->datTheta = datTheta.memptr();
    mydata->weibullS = weibullS.memptr();
    mydata->weibullLambda = weibullLambda.memptr();
    mydata->datX = dataclass.datX.memptr();
    mydata->datProportionConst = dataclass.datProportionConst.memptr();
    mydata->datEvent = dataclass.datEvent.memptr();


    for (unsigned int l = 0; l < L; ++l)
    {
        // Gibbs sampling
        
        mydata->wSq = wSq[l];
        
        for (unsigned int j = 0; j < p+1; ++j)
        {
            if (!etas(j, l))
            {
                if (j > 0)
                {
                    // currentPars(j, l) = R::rnorm(0., std::sqrt(wSq[l]));
                    // double inactive_var = (pseudoVar == 0.0) ? wSq[l] : pseudoVar;
                    currentPars(j, l) = R::rnorm(pseudoMean(j-1,l), std::sqrt(pseudoVar(j-1,l)));
                }
            }
            else
            {
                mydata->jj = j;
                mydata->l = l;
                
                double xprev = currentPars(j, l);
                // double *xsamp = (double*)malloc(armsPar.nsamp * sizeof(double));
                std::vector<double> xsamp(armsPar.nsamp);

                double qcent[1], xcent[1];
                int neval, ncent = 0;
                int err;
                double convex = armsPar.convex;
                err = ARMS::arms (
                              xinit.data(), armsPar.ninit, &minD, &maxD,
                              EvalFunction::log_dens_zetas, mydata.get(),
                              &convex, armsPar.npoint,
                              armsPar.metropolis, &xprev, xsamp.data(),
                              armsPar.nsamp, qcent, xcent, ncent, &neval);

                // check ARMS validity
                if (err > 0)
                    Rprintf("In arms_gibbs_zeta(): error code in ARMS = %d.\n", err);
                if (std::isnan(xsamp[armsPar.nsamp-1]))
                    Rprintf("In arms_gibbs_zeta(): NaN generated, possibly due to overflow in (log-)density (e.g. with densities involving exp(exp(...))).\n");
                if (xsamp[armsPar.nsamp-1] < minD || xsamp[armsPar.nsamp-1] > maxD)
                    Rprintf("In arms_gibbs_zeta(): %d-th sample out of range [%f, %f] (fused domain). Got %f.\n", armsPar.nsamp, minD, maxD, xsamp[armsPar.nsamp-1]);

                currentPars(j, l) = xsamp[armsPar.nsamp - 1];
                
                // free(xsamp);
            }
        }
    }

    // free(mydata);
}

// Multivariate ARMS via Gibbs sampler for betaK; used for M-H sampling for gammas update
// NOTE: ARMS_Gibbs::arms_gibbs_zetaK() is not used!!!
/*
void ARMS_Gibbs::arms_gibbs_zetaK(
    const unsigned int k,
    const armsParmClass& armsPar,
    arma::mat& currentPars,
    double w0Sq,
    double wSqK,
    arma::umat etas,
    double kappa,
    bool dirichlet,
    arma::vec& datTheta,
    arma::mat& weibullS,
    arma::mat& weibullLambda,
    const DataClass &dataclass
)
{

    // dimensions
    unsigned int N = dataclass.datX.n_rows;
    unsigned int p = dataclass.datX.n_cols;
    unsigned int L = dataclass.datX.n_slices;

    // objects for arms()
    double minD = armsPar.zetaMin;
    double maxD = armsPar.zetaMax;

    std::vector<double> xinit(armsPar.ninit); // Use std::vector instead of VLA
    arma::vec xinit0 = arma::linspace( minD+1.0e-10, maxD-1.0e-10, armsPar.ninit );
    for (unsigned int i = 0; i < armsPar.ninit; ++i)
        xinit[i] = xinit0[i];

    if (!dirichlet)
        Rprintf("Warning: In arms_gibbs_zeta(), Dirichlet modeling with logit/alr-link is not implement!\n");

    dataS *mydata = (dataS *)malloc(sizeof (dataS));

    etas = arma::join_cols(arma::ones<arma::urowvec>(L), etas);
    mydata->gammaIndicator = etas.memptr();
    mydata->currentPars = currentPars.memptr();
    mydata->p = p;
    mydata->L = L;
    mydata->N = N;
    mydata->w0Sq = w0Sq;
    mydata->wSq = wSqK;
    mydata->kappa = kappa;
    mydata->datTheta = datTheta.memptr();
    mydata->weibullS = weibullS.memptr();
    mydata->weibullLambda = weibullLambda.memptr();
    mydata->datX = dataclass.datX.memptr();
    mydata->datProportionConst = dataclass.datProportionConst.memptr();
    mydata->datEvent = dataclass.datEvent.memptr();

    unsigned int l = k;
    // Gibbs sampling

    for (unsigned int j = 0; j < p+1; ++j)
    {
        if (!etas(j, l))
        {
            if (j > 0)
                currentPars(j, l) = R::rnorm(0., std::sqrt(wSqK));
        }
        else
        {
            mydata->jj = j;
            mydata->l = l;

            double xprev = currentPars(j, l);
            std::vector<double> xsamp(armsPar.nsamp);

            double qcent[1], xcent[1];
            int neval, ncent = 0;

            int err;
            double convex = armsPar.convex;
            err = ARMS::arms (
                          xinit.data(), armsPar.ninit, &minD, &maxD,
                          EvalFunction::log_dens_zetas, mydata,
                          &convex, armsPar.npoint,
                          armsPar.metropolis, &xprev, xsamp.data(),
                          armsPar.nsamp, qcent, xcent, ncent, &neval);

            // check ARMS validity
            if (err > 0)
                Rprintf("In arms_gibbs_zeta(): error code in ARMS = %d.\n", err);
            if (std::isnan(xsamp[armsPar.nsamp-1]))
                Rprintf("In arms_gibbs_zeta(): NaN generated, possibly due to overflow in (log-)density (e.g. with densities involving exp(exp(...))).\n");
            if (xsamp[armsPar.nsamp-1] < minD || xsamp[armsPar.nsamp-1] > maxD)
                Rprintf("In arms_gibbs_zeta(): %d-th sample out of range [%f, %f] (fused domain). Got %f.\n", armsPar.nsamp, minD, maxD, xsamp[armsPar.nsamp-1]);

            currentPars(j, l) = xsamp[armsPar.nsamp - 1];
        }
    }

    // free(mydata);
}
*/

// Gibbs sampling for zetas without BVS
void ARMS_Gibbs::arms_gibbs_zetaFull(
    const armsParmClass& armsPar,
    arma::mat& currentPars,
    double w0Sq,
    arma::vec& wSq,

    double kappa,
    bool dirichlet,
    arma::vec& datTheta,
    arma::mat& weibullS,
    arma::mat& weibullLambda,
    arma::mat& alphas,
    const DataClass& dataclass
)
{
    // dimensions
    unsigned int N = dataclass.datX.n_rows;
    unsigned int p = dataclass.datX.n_cols;
    unsigned int L = dataclass.datX.n_slices;

    // objects for arms()
    double minD = armsPar.zetaMin;
    double maxD = armsPar.zetaMax;

    std::vector<double> xinit(armsPar.ninit);
    arma::vec xinit0 = arma::linspace(
        minD + 1.0e-10,
        maxD - 1.0e-10,
        armsPar.ninit
    );

    for (unsigned int i = 0; i < armsPar.ninit; ++i)
    {
        xinit[i] = xinit0[i];
    }

    if (!dirichlet)
    {
        Rprintf("Warning: In arms_gibbs_zetaFull(), Dirichlet modeling with logit/alr-link is not implemented!\n");
    }

    // dataS* mydata = static_cast<dataS*>(malloc(sizeof(dataS)));
    auto mydata = std::make_unique<dataS>(); // modern and safe memory allocation

    mydata->currentPars = currentPars.memptr();
    mydata->p = p;
    mydata->L = L;
    mydata->N = N;
    mydata->w0Sq = w0Sq;
    mydata->kappa = kappa;
    mydata->datTheta = datTheta.memptr();
    mydata->weibullS = weibullS.memptr();
    mydata->weibullLambda = weibullLambda.memptr();
    mydata->datProportionConst = dataclass.datProportionConst.memptr();
    mydata->datEvent = dataclass.datEvent.memptr();

    // ------------------------------------------------------------------
    // Compute current full alpha matrix once.
    // ------------------------------------------------------------------
    // arma::mat alphas(N, L, arma::fill::zeros);
    arma::vec logAlpha_ll(N);
    for (unsigned int ll = 0; ll < L; ++ll)
    {
        logAlpha_ll = currentPars(0, ll) +
            dataclass.datX.slice(ll) * currentPars.submat(1, ll, p, ll);

        // logAlpha_ll.elem(arma::find(logAlpha_ll > upperbound3)).fill(upperbound3);
        // logAlpha_ll = arma::min(logAlpha_ll, arma::vec(N).fill(upperbound)); // faster alternative
        // logAlpha_ll.elem(arma::find(logAlpha_ll < std::log(lowerbound))).fill(std::log(lowerbound));

        alphas.col(ll) = arma::exp(logAlpha_ll);
    }

    // alphas.elem(arma::find(alphas > upperbound3)).fill(upperbound3);
    // alphas.elem(arma::find(alphas < lowerbound)).fill(lowerbound);
    alphas = arma::min(alphas, arma::mat(N,L).fill(upperbound)); // faster alternative
    alphas = arma::max(alphas, arma::mat(N,L).fill(lowerbound)); 

    arma::vec alphaRowsum = arma::sum(alphas, 1);
    // alphaRowsum.elem(arma::find(alphaRowsum < lowerbound)).fill(lowerbound);
    alphaRowsum = arma::max(alphaRowsum, arma::vec(N).fill(lowerbound)); // faster alternative

    mydata->alphas = alphas.memptr();
    mydata->alphaRowsum = alphaRowsum.memptr();

    // Work vectors for current cell type l.
    arma::vec logAlpha_l(N);
    arma::vec alpha_l(N);
    arma::vec old_alpha_l(N);

    for (unsigned int l = 0; l < L; ++l)
    {
        // Gibbs sampling
        mydata->l = l;
        mydata->wSq = wSq[l];
        mydata->datX = dataclass.datX.slice(l).memptr();

        // Current baseline log-alpha and alpha for this l.
        logAlpha_l =
            currentPars(0, l) +
            dataclass.datX.slice(l) * currentPars.submat(1, l, p, l);

        // logAlpha_l.elem(arma::find(logAlpha_l > upperbound3)).fill(upperbound3);
        // logAlpha_l = arma::min(logAlpha_l, arma::vec(N).fill(upperbound3)); // faster alternative
        // logAlpha_l.elem(arma::find(logAlpha_l < std::log(lowerbound))).fill(std::log(lowerbound));

        alpha_l = arma::exp(logAlpha_l);
        // alpha_l.elem(arma::find(alpha_l > upperbound3)).fill(upperbound3);
        // alpha_l.elem(arma::find(alpha_l < lowerbound)).fill(lowerbound);
        alpha_l = arma::min(alpha_l, arma::vec(N).fill(upperbound)); // faster alternative
        alpha_l = arma::max(alpha_l, arma::vec(N).fill(lowerbound)); 

        // Keep full alpha matrix consistent.
        alphas.col(l) = alpha_l;
        alphaRowsum = arma::sum(alphas, 1);
        // alphaRowsum.elem(arma::find(alphaRowsum < lowerbound)).fill(lowerbound);
        // alphaRowsum = arma::max(alphaRowsum, arma::vec(N).fill(lowerbound)); // faster alternative

        mydata->logAlpha_l = logAlpha_l.memptr();
        mydata->alpha_l = alpha_l.memptr();
        mydata->alphas = alphas.memptr();
        mydata->alphaRowsum = alphaRowsum.memptr();

        for (unsigned int j = 0; j < p + 1; ++j)
        {
            mydata->jj = j;

            double old_par = currentPars(j, l);
            mydata->old_par = old_par;

            double xprev = old_par;
            std::vector<double> xsamp(armsPar.nsamp);

            double qcent[1], xcent[1];
            int neval, ncent = 0;
            double convex = armsPar.convex;

            int err = ARMS::arms(
                xinit.data(),
                armsPar.ninit,
                &minD,
                &maxD,
                EvalFunction::log_dens_zetasFull,
                mydata.get(),
                &convex,
                armsPar.npoint,
                armsPar.metropolis,
                &xprev,
                xsamp.data(),
                armsPar.nsamp,
                qcent,
                xcent,
                ncent,
                &neval
            );

            if (err > 0)
            {
                Rprintf("In arms_gibbs_zetaFull(): error code in ARMS = %d.\n", err);
            }

            if (std::isnan(xsamp[armsPar.nsamp - 1]))
            {
                Rprintf("In arms_gibbs_zetaFull(): NaN generated, possibly due to overflow.\n");
            }

            if (xsamp[armsPar.nsamp - 1] < minD || xsamp[armsPar.nsamp - 1] > maxD)
            {
                Rprintf(
                    "In arms_gibbs_zetaFull(): %d-th sample out of range [%f, %f]. Got %f.\n",
                    armsPar.nsamp,
                    minD,
                    maxD,
                    xsamp[armsPar.nsamp - 1]
                );
            }

            double new_par = xsamp[armsPar.nsamp - 1];
            currentPars(j, l) = new_par;

            // ----------------------------------------------------------
            // Accepted update: update cached logAlpha_l, alpha_l,
            // alphas.col(l), and alphaRowsum.
            // ----------------------------------------------------------
            double accepted_delta = new_par - old_par;

            old_alpha_l = alpha_l;

            if (j == 0)
            {
                logAlpha_l += accepted_delta;
            }
            else
            {
                logAlpha_l += dataclass.datX.slice(l).col(j - 1) * accepted_delta;
            }

            // logAlpha_l.elem(arma::find(logAlpha_l > upperbound3)).fill(upperbound3);
            // logAlpha_l = arma::min(logAlpha_l, arma::vec(N).fill(upperbound3)); // faster alternative
            // logAlpha_l.elem(arma::find(logAlpha_l < std::log(lowerbound))).fill(std::log(lowerbound));

            alpha_l = arma::exp(logAlpha_l);
            // alpha_l.elem(arma::find(alpha_l > upperbound3)).fill(upperbound3);
            // alpha_l.elem(arma::find(alpha_l < lowerbound)).fill(lowerbound);
            alpha_l = arma::min(alpha_l, arma::vec(N).fill(upperbound)); // faster alternative
            alpha_l = arma::max(alpha_l, arma::vec(N).fill(lowerbound)); 

            alphaRowsum = alphaRowsum - old_alpha_l + alpha_l;
            // alphaRowsum.elem(arma::find(alphaRowsum < lowerbound)).fill(lowerbound);
            // alphaRowsum = arma::max(alphaRowsum, arma::vec(N).fill(lowerbound)); // faster alternative

            alphas.col(l) = alpha_l;

            // // Refresh pointers explicitly.
            // mydata->logAlpha_l = logAlpha_l.memptr();
            // mydata->alpha_l = alpha_l.memptr();
            // mydata->alphas = alphas.memptr();
            // mydata->alphaRowsum = alphaRowsum.memptr();
        }
    }

    // free(mydata);
}

//' Univariate ARMS for kappa
//'
//' @param n Number of samples to draw
//' @param nsamp How many samples to draw for generating each sample; only the last draw will be kept
//' @param ninit Number of initials as meshgrid values for envelop search
//' @param convex Adjustment for convexity (non-negative value, default 1.0)
//' @param npoint Maximum number of envelope points
//' @param dirichlet Not yet implemented
//'
void ARMS_Gibbs::arms_kappa(
    const armsParmClass& armsPar,
    double& currentPars,
    double kappaA,
    double kappaB,
    bool invGamma,
    arma::vec& datTheta,
    arma::mat& datMu,
    arma::mat& datProportion,
    const DataClass &dataclass)
{
    // dimensions
    unsigned int N = datProportion.n_rows;
    unsigned int L = datProportion.n_cols;

    // objects for arms()
    const double minD = armsPar.kappaMin;
    const double maxD = armsPar.kappaMax;
    // reallocate struct variables

    std::vector<double> xinit(armsPar.ninit); // Use std::vector instead of VLA
    arma::vec xinit0 = arma::linspace( minD+1.0e-10, maxD-1.0e-10, armsPar.ninit );
    for (unsigned int i = 0; i < armsPar.ninit; ++i)
        xinit[i] = xinit0[i];

    // dataS *mydata = (dataS *)malloc(sizeof (dataS)); // Rcpp::stop() may case memory leakage with malloc()
    auto mydata = std::make_unique<dataS>(); // modern and safe memory allocation

    mydata->L = L;
    mydata->N = N;
    mydata->kappaA = kappaA;
    mydata->kappaB = kappaB;
    mydata->invGamma = invGamma;
    mydata->datTheta = datTheta.memptr();
    mydata->datMu = datMu.memptr();
    mydata->datProportion = datProportion.memptr();
    mydata->datEvent = dataclass.datEvent.memptr();
    mydata->datTime = dataclass.datTime.memptr();

    slice_sample (
        EvalFunction::log_dens_kappa,
        mydata.get(),
        currentPars,
        10,
        1.0,
        minD,
        maxD
    );
    
    // free(mydata);
}

void ARMS_Gibbs::slice_sample(
    double (*logfn)(double par, void *mydata),
    void *mydata,
    double& x,
    const unsigned int steps,
    const double w,
    const double lower,
    const double upper)
{
    double L_bound = 0.;
    double R_bound = 0.;
    double logy = logfn(x, mydata);

    // we can add omp parallelisation here
    for (unsigned int i = 0; i < steps; ++i)
    {
        // draw uniformly from [0, y]
        double logz = logy - R::rexp(1);

        // expand search range
        double u = R::runif(0.0, 1.0) * w;
        L_bound = x - u;
        R_bound = x + (w - u);
        while ( L_bound > lower && logfn(L_bound, mydata) > logz )
        {
            L_bound -= w;
        }
        while ( R_bound < upper && logfn(R_bound, mydata) > logz )
        {
            R_bound += w;
        }

        // sample until draw is within valid range
        double r0 = std::max(L_bound, lower);
        double r1 = std::min(R_bound, upper);

        double xs = x;
        double logys = 0.;
        int cnt = 0;
        do
        {
            cnt++;
            xs = R::runif(r0, r1);
            logys = logfn(xs, mydata);
            if ( logys > logz )
                break;
            if ( xs < x )
            {
                r0 = xs;
            }
            else
            {
                r1 = xs;
            }
        }
        while (cnt < 1e4);

        if (cnt == 1e4) Rcpp::stop("slice_sample_cpp loop did not finish");

        x = xs;
        logy = logys;
    }

}
