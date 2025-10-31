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
    arma::mat datProportion,
    arma::mat weibullS,
    const DataClass &dataclass)
{
    // number of parameters to be updated
    unsigned int p = currentPars.n_elem; // = 1;
    unsigned int L = datProportion.n_cols;
    unsigned int N = datProportion.n_rows;

    // int armsPar.metropolis = metropolis;

    const double minD = armsPar.xiMin;
    const double maxD = armsPar.xiMax;
    // minD = minRange[0]; // [j]
    // maxD = maxRange[0]; // [j]
    //double *xl; xl = &minD;
    //double *xr; xr = &maxD;

    // reallocate struct variables

    // // double xinit[armsPar.ninit];
    std::vector<double> xinit(armsPar.ninit); // Use std::vector instead of VLA to avoid warning about varying length of array
    // if (!armsPar.simple)
    // {
    // }
    arma::vec xinit0 = arma::linspace( minD+1.0e-10, maxD-1.0e-10, armsPar.ninit );
    for (unsigned int i = 0; i < armsPar.ninit; ++i)
        xinit[i] = xinit0[i];

    //dataS *mydata = create_mydata(currentPars, j, vA, vB, datX0, datProportion, datEvent, weibullS);
    dataS *mydata = (dataS *)malloc(sizeof (dataS));
    //create_mydata(currentPars, j, xl, xr, vA, vB, datX0, datProportion, datEvent, weibullS, mydata);
    mydata->currentPars = currentPars.memptr();
    mydata->p = p;
    mydata->L = L;
    mydata->N = N;
    // mydata->xl = xl;
    // mydata->xr = xr;
    // mydata->vA = vA;
    // mydata->vB = vB;
    mydata->vSq = vSq;
    mydata->v0Sq = v0Sq;
    mydata->datProportion = datProportion.memptr();
    mydata->weibullS = weibullS.memptr();
    mydata->datX = dataclass.datX0.memptr();
    mydata->datEvent = dataclass.datEvent.memptr();

    // define how many ARMS samples to draw for currentPars
    // arma::mat samp = arma::zeros<arma::mat>(p, armsPar.n + 1);
    // samp.col(0) = currentPars;

    // for (int i = 0; i < armsPar.n; ++i)
    // {
    // Gibbs sampling

    #ifdef _OPENMP
    #pragma omp parallel for 
    #endif

    for (unsigned int j = 0; j < p; ++j)
    {
        mydata->jj = j;
        // // update \xi's (not intercept) variance vSq
        // mydata->vSq = 10.;
        // if (j > 0)
        // {
        //   // double vA_tmp = vA + 0.5 * (arma::accu(currentPars != 0.) - 1.0);
        //   // double vB_tmp = vB + 0.5 * (arma::as_scalar(currentPars.t() * currentPars) - currentPars[0] * currentPars[0]);
        //   // vSq = 1. / R::rgamma(vA_tmp, 1. / vB_tmp);
        //   mydata->vSq = sampleV(1, vA, vB, currentPars);
        // }

        //double initi = samp(j, i);  //samp(j, i)
        //double *xprev; xprev = &initi;
        // double xprev = samp(j, i);
        /*
            double xprev = currentPars[j];
            // double *xsamp = (double*)malloc(armsPar.nsamp * sizeof(double));
            std::vector<double> xsamp(armsPar.nsamp);

            double qcent[1], xcent[1];
            int neval, ncent = 0;

            int err;
            if (armsPar.simple)
            {
                err = ARMS::arms_simple (
                          armsPar.ninit, &minD, &maxD,
                          EvalFunction::log_dens_xis, mydata,
                          armsPar.metropolis, &xprev, xsamp.data());
            }
            else
            {
                double convex = armsPar.convex;
                // The .data() member function returns a pointer to the underlying array managed by the vector
                err = ARMS::arms (
                          xinit.data(), armsPar.ninit, &minD, &maxD,
                          EvalFunction::log_dens_xis, mydata,
                          &convex, armsPar.npoint,
                          armsPar.metropolis, &xprev, xsamp.data(),
                          armsPar.nsamp, qcent, xcent, ncent, &neval);
            }

            // check ARMS validity
            if (err > 0)
                Rprintf("In arms_gibbs_xi(): error code in ARMS = %d.\n", err);
            if (std::isnan(xsamp[armsPar.nsamp-1]))
                Rprintf("In arms_gibbs_xi(): NaN generated, possibly due to overflow in (log-)density (e.g. with densities involving exp(exp(...))).\n");
            if (xsamp[armsPar.nsamp-1] < minD || xsamp[armsPar.nsamp-1] > maxD)
                Rprintf("In arms_gibbs_xi(): %d-th sample out of range [%f, %f] (fused domain). Got %f.\n", armsPar.nsamp, minD, maxD, xsamp[armsPar.nsamp-1]);

            //xprev = xsamp[nsamp - 1];
            //for (int i = 0; i < n; ++i) samp(j, i + 1) = xsamp[nsamp - n + i];
            currentPars[j] = xsamp[armsPar.nsamp - 1];

        */
        double xsamp = currentPars[j];
        slice_sample (
            EvalFunction::log_dens_xis,
            mydata,
            xsamp,
            10,
            1.0,
            minD,
            maxD
        );
        currentPars[j] = xsamp;
        // samp(j, i + 1) = xsamp[armsPar.nsamp - 1];

        //mydata->currentPars = currentPars.memptr(); // notnded, since 'currentPars[j] = xsamp[nsamp - 1]' above changed the memory

        // free(xsamp);
    }
    // }

    free(mydata);
    // remove the inital values in the first column of samp
    // samp.shed_col(0);

    //std::cout << "...debug sample=" << samp.col(0).t() << "\n";

    // return samp;
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
    double& tau0Sq,

    // const arma::umat& gammas,
    arma::umat gammas,

    double kappa,
    arma::vec& datTheta,
    arma::mat datMu,
    arma::mat& datProportion,
    arma::mat weibullS,
    // const arma::cube datX,
    // const arma::uvec datEvent,
    // const arma::vec datTime,
    const DataClass &dataclass
    // double& logPosteriorBeta
)
{
    /* make a subfunction arms_gibbs for only vector betas that can be used for (varying-length) variable selected vector*/

    // dimensions
    unsigned int N = dataclass.datX.n_rows;
    unsigned int p = dataclass.datX.n_cols;
    unsigned int L = dataclass.datX.n_slices;

    // logPosteriorBeta = 0.; // reset value 0

    // objects for arms()
    double minD = armsPar.betaMin;
    double maxD = armsPar.betaMax;
    // minD = minRange[0]; // [j]
    // maxD = maxRange[0]; // [j]
    //double *xl; xl = &minD;
    //double *xr; xr = &maxD;

    // reallocate struct variables

    // int dometrop = armsPar.metropolis;

    // double xinit[armsPar.ninit];
    std::vector<double> xinit(armsPar.ninit); // Use std::vector instead of VLA
    // if (!armsPar.simple)
    // {
    // }
    arma::vec xinit0 = arma::linspace( minD+1.0e-10, maxD-1.0e-10, armsPar.ninit );
    for (unsigned int i = 0; i < armsPar.ninit; ++i)
        xinit[i] = xinit0[i];  

    dataS *mydata = (dataS *)malloc(sizeof (dataS));

    gammas = arma::join_cols(arma::ones<arma::urowvec>(L), gammas);
    currentPars.elem(arma::find(gammas == 0)).fill(0.);
    mydata->currentPars = currentPars.memptr();
    mydata->p = p;
    mydata->L = L;
    mydata->N = N;
    // mydata->xl = xl;
    // mydata->xr = xr;
    // mydata->vA = vA;
    // mydata->vB = vB;
    mydata->tau0Sq = tau0Sq;
    mydata->kappa = kappa;
    mydata->datTheta = datTheta.memptr();
    mydata->datMu = datMu.memptr();
    mydata->datProportion = datProportion.memptr();
    mydata->weibullS = weibullS.memptr();
    mydata->datEvent = dataclass.datEvent.memptr();
    mydata->datTime = dataclass.datTime.memptr();

    // arma::vec tauSq_tmp = arma::ones<arma::vec>(L);

    // not easy to parallize the following for-loop due to data dependencies
    for (unsigned int l = 0; l < L; ++l)
    {
        // Gibbs sampling
        // mydata->tauSq = sampleW(tauA, tauB, currentPars.col(l));
        // mydata->tau0Sq = sampleW0(tauA, tauB, currentPars(0,l));

        mydata->tauSq = tauSq[l];
        // tauSq_tmp[l] = sampleW(tauA, tauB, currentPars.col(l));
        // mydata->tauSq = tauSq_tmp[l];
        for (unsigned int j = 0; j < p+1; ++j)
        {
            if (gammas(j, l))
            {
                mydata->jj = j;
                mydata->l = l;
                // // update \beta's variance tauSq
                // mydata->tauSq = sampleTau(tauA, tauB, currentPars);

                mydata->datX = dataclass.datX.slice(l).memptr();

                // update quantities needed for ARMS updates
                // if put 'create_mydata' out of for-loop, the following updates can for elements of pointer *mydata
                
                // arma::vec logMu_l = dataclass.datX.slice(l) * currentPars.col(l);
                arma::vec logMu_l = currentPars(0, l) + dataclass.datX.slice(l) * currentPars.submat(1, l, p, l);
                logMu_l.elem(arma::find(logMu_l > upperbound)).fill(upperbound);
                datMu.col(l) = arma::exp( logMu_l );
                // arma::vec lambdas = datMu.col(l) / std::tgamma(1. + 1./kappa);
                // weibullS.col(l) = arma::exp( -arma::pow(datTime / lambdas, kappa) );
                arma::vec lambdas = arma::pow( dataclass.datTime / (datMu.col(l) / std::tgamma(1. + 1./kappa)), kappa);
                lambdas.elem(arma::find(lambdas > upperbound)).fill(upperbound);
                weibullS.col(l) = arma::exp( -lambdas );
                //weibullS.elem(arma::find(weibullS < lowerbound)).fill(lowerbound); // remove later on if using smaller upperbound for lambdas

                // mydata->currentPars = currentPars.memptr();
                // mydata->datMu = datMu.memptr(); // update this due to its change with updated coefficients
                // mydata->weibullS = weibullS.memptr(); // update this due to its change with updated coefficients

                // parameters for ARMS
                //double initi = currentPars(j, l);  //samp(j, i)
                //double *xprev; xprev = &initi;
                double xprev = currentPars(j, l);
                // double *xsamp = (double*)malloc(armsPar.nsamp * sizeof(double));
                std::vector<double> xsamp(armsPar.nsamp);

                double qcent[1], xcent[1];
                int neval, ncent = 0;

                int err;
                // if (armsPar.simple)
                // {
                //     err = ARMS::arms_simple (
                //               armsPar.ninit, &minD, &maxD,
                //               EvalFunction::log_dens_betas, mydata,
                //               armsPar.metropolis, &xprev, xsamp.data());
                // }
                // else
                // {
                // }
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

                // std::cout << "...debug xsamp[1:nsamp]=";
                // for( int i=0; i<nsamp; ++i) {
                //   std::cout << ", " << xsamp[i] ;
                // }
                // std::cout <<  "/n";

                currentPars(j, l) = xsamp[armsPar.nsamp - 1];

                // free(xsamp);
            }
        }
    }

    // TODO: no need here if we confirm to use 'logPbetaK()'
    // logPosteriorBeta = logPbetas(currentPars, tauSq, kappa, datTheta, datProportion, dataclass);

    free(mydata);

    // Assembling output
    /*Rcpp::List out = Rcpp::List::create(
      Rcpp::Named("betas") = currentPars,
      Rcpp::Named("mus") = datMu,
      Rcpp::Named("weibullS") = weibullS
    );
    */
    // return currentPars;
}


// Multivariate ARMS via Gibbs sampler for betaK; used for M-H sampling for gammas update
void ARMS_Gibbs::arms_gibbs_betaK(
    const unsigned int k,
    const armsParmClass& armsPar,
    arma::mat& currentPars,
    double tau0Sq,
    double tauSqK,

    // const arma::umat& gammas,
    arma::umat gammas,

    double kappa,
    arma::vec& datTheta,
    arma::mat datMu,
    arma::mat& datProportion,
    arma::mat weibullS,
    const DataClass &dataclass
    // double& logPosteriorBeta
)
{
    /* make a subfunction arms_gibbs for only vector betas that can be used for (varying-length) variable selected vector*/

    // dimensions
    unsigned int N = dataclass.datX.n_rows;
    unsigned int p = dataclass.datX.n_cols;
    unsigned int L = dataclass.datX.n_slices;

    // logPosteriorBeta = 0.; // reset value 0

    // objects for arms()
    double minD = armsPar.betaMin;
    double maxD = armsPar.betaMax;

    std::vector<double> xinit(armsPar.ninit); // Use std::vector instead of VLA
    // if (!armsPar.simple)
    // {
    // }
    arma::vec xinit0 = arma::linspace( minD+1.0e-10, maxD-1.0e-10, armsPar.ninit );
    for (unsigned int i = 0; i < armsPar.ninit; ++i)
        xinit[i] = xinit0[i];

    dataS *mydata = (dataS *)malloc(sizeof (dataS));

    gammas = arma::join_cols(arma::ones<arma::urowvec>(L), gammas);
    currentPars.elem(arma::find(gammas == 0)).fill(0.);
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
    // mydata->tauSq = sampleW(tauA, tauB, currentPars.col(l));
    for (unsigned int j = 0; j < p+1; ++j)
    {
        if (gammas(j, l))
        {
            mydata->jj = j;
            mydata->l = l;

            mydata->datX = dataclass.datX.slice(l).memptr();

            // arma::vec logMu_l = dataclass.datX.slice(l) * currentPars.col(l);
            arma::vec logMu_l = currentPars(0, l) + dataclass.datX.slice(l) * currentPars.submat(1, l, p, l);
            logMu_l.elem(arma::find(logMu_l > upperbound)).fill(upperbound);
            datMu.col(l) = arma::exp( logMu_l );
            arma::vec lambdas = arma::pow( dataclass.datTime / (datMu.col(l) / std::tgamma(1. + 1./kappa)), kappa);
            lambdas.elem(arma::find(lambdas > upperbound)).fill(upperbound);
            weibullS.col(l) = arma::exp( -lambdas );
            double xprev = currentPars(j, l);
            std::vector<double> xsamp(armsPar.nsamp);

            double qcent[1], xcent[1];
            int neval, ncent = 0;

            int err;
            // if (armsPar.simple)
            // {
            //     err = ARMS::arms_simple (
            //               armsPar.ninit, &minD, &maxD,
            //               EvalFunction::log_dens_betas, mydata,
            //               armsPar.metropolis, &xprev, xsamp.data());
            // }
            // else
            // {
            // }
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

        }
    }

    // logPosteriorBeta = logPbetaK(k, currentPars, mydata->tauSq, kappa, datTheta, datProportion, dataclass);

    free(mydata);
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
    double& w0Sq,
    arma::vec& wSq,

    arma::umat etas,

    double kappa,
    bool dirichlet,
    arma::vec& datTheta,
    arma::mat weibullS,
    arma::mat& weibullLambda,
    const DataClass &dataclass
    // double& logPosteriorZeta
)
{
    // arma::mat datProportion, // remove this argument
    // double phi,
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
    // minD = minRange[0]; // [j]
    // maxD = maxRange[0]; // [j]
    //double *xl; xl = &minD;
    //double *xr; xr = &maxD;

    // reallocate struct variables

    // double xinit[armsPar.ninit];
    std::vector<double> xinit(armsPar.ninit); // Use std::vector instead of VLA
    // if (!armsPar.simple)
    // {
    // }
    arma::vec xinit0 = arma::linspace( minD+1.0e-10, maxD-1.0e-10, armsPar.ninit );
    for (unsigned int i = 0; i < armsPar.ninit; ++i)
        xinit[i] = xinit0[i];

    if (!dirichlet)
        Rprintf("Warning: In arms_gibbs_zeta(), Dirichlet modeling with logit/alr-link is not implement!\n");

    dataS *mydata = (dataS *)malloc(sizeof (dataS));

    etas = arma::join_cols(arma::ones<arma::urowvec>(L), etas);
    currentPars.elem(arma::find(etas == 0)).fill(0.);
    mydata->currentPars = currentPars.memptr();
    mydata->p = p;
    mydata->L = L;
    mydata->N = N;
    mydata->w0Sq = w0Sq;
    // mydata->wSq = wSq;
    //mydata->phi = phi;
    //mydata->dirichlet = dirichlet,
    mydata->kappa = kappa,
    mydata->datTheta = datTheta.memptr();
    //mydata->datMu = datMu.memptr();
    mydata->weibullS = weibullS.memptr();
    mydata->weibullLambda = weibullLambda.memptr();
    mydata->datX = dataclass.datX.memptr();
    mydata->datProportionConst = dataclass.datProportionConst.memptr();
    //mydata->datProportion = datProportion.memptr();
    mydata->datEvent = dataclass.datEvent.memptr();

    // arma::vec wSq_tmp = arma::ones<arma::vec>(L);

    for (unsigned int l = 0; l < L; ++l)
    {
        // Gibbs sampling
        // mydata->w0Sq = sampleW0(w0A, w0B, currentPars(0,l));
        mydata->wSq = wSq[l];
        // wSq_tmp[l] = sampleW(wA, wB, currentPars.submat(1,l,p,l));
        // mydata->wSq = wSq_tmp[l];
        for (unsigned int j = 0; j < p+1; ++j)
        {
            if (etas(j, l))
            {
                mydata->jj = j;
                mydata->l = l;
                // // update \zetas' variance wSq
                // mydata->wSq = sampleW(wA, wB, currentPars.rows(1, p));
                // if(w0IGamma)
                // {
                //   mydata->w0Sq = sampleW0(w0A, w0B, currentPars.row(0));
                // }

                //double initi = currentPars(j, l);  //samp(j, i)
                //double *xprev; xprev = &initi;
                double xprev = currentPars(j, l);
                // double *xsamp = (double*)malloc(armsPar.nsamp * sizeof(double));
                std::vector<double> xsamp(armsPar.nsamp);

                double qcent[1], xcent[1];
                int neval, ncent = 0;

                int err;
                // if (armsPar.simple)
                // {
                //     err = ARMS::arms_simple (
                //               armsPar.ninit, &minD, &maxD,
                //               EvalFunction::log_dens_zetas, mydata,
                //               armsPar.metropolis, &xprev, xsamp.data());
                // }
                // else
                // {
                // }
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

                // update proportions based on currentPars
                /*
                arma::mat alphas = arma::zeros<arma::mat>(N, L);
                for(int ll=0; ll<L; ++ll)
                {
                  alphas.col(ll) = arma::exp( currentPars(0, ll) + datX.slice(ll) * currentPars.submat(1, ll, p, ll) );
                }
                alphas.elem(arma::find(alphas > upperbound3)).fill(upperbound3);
                datProportion = alphas / arma::repmat(arma::sum(alphas, 1), 1, L);
                */

                // mydata->currentPars = currentPars.memptr();
                // mydata->datProportion = datProportion.memptr();

                // free(xsamp);
            }
        }
    }

    // logPosteriorZeta = logPzetas(currentPars, wSq_tmp, kappa, datTheta, weibullS, weibullLambda, dataclass);
    free(mydata);

    // return currentPars;
}

// Multivariate ARMS via Gibbs sampler for betaK; used for M-H sampling for gammas update
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
    arma::mat weibullS,
    arma::mat& weibullLambda,
    const DataClass &dataclass
    // double& logPosteriorZeta
)
{

    // dimensions
    unsigned int N = dataclass.datX.n_rows;
    unsigned int p = dataclass.datX.n_cols;
    unsigned int L = dataclass.datX.n_slices;

    // logPosteriorZeta = 0.;

    // objects for arms()
    double minD = armsPar.zetaMin;
    double maxD = armsPar.zetaMax;

    std::vector<double> xinit(armsPar.ninit); // Use std::vector instead of VLA
    // if (!armsPar.simple)
    // {
    // }
    arma::vec xinit0 = arma::linspace( minD+1.0e-10, maxD-1.0e-10, armsPar.ninit );
    for (unsigned int i = 0; i < armsPar.ninit; ++i)
        xinit[i] = xinit0[i];

    if (!dirichlet)
        Rprintf("Warning: In arms_gibbs_zeta(), Dirichlet modeling with logit/alr-link is not implement!\n");

    dataS *mydata = (dataS *)malloc(sizeof (dataS));

    etas = arma::join_cols(arma::ones<arma::urowvec>(L), etas);
    currentPars.elem(arma::find(etas == 0)).fill(0.);
    mydata->currentPars = currentPars.memptr();
    mydata->p = p;
    mydata->L = L;
    mydata->N = N;
    mydata->w0Sq = w0Sq;
    mydata->wSq = wSqK;
    mydata->kappa = kappa,
            mydata->datTheta = datTheta.memptr();
    mydata->weibullS = weibullS.memptr();
    mydata->weibullLambda = weibullLambda.memptr();
    mydata->datX = dataclass.datX.memptr();
    mydata->datProportionConst = dataclass.datProportionConst.memptr();
    mydata->datEvent = dataclass.datEvent.memptr();

    unsigned int l = k;
    // Gibbs sampling. No need to update variance, since this is only used for updating zetas conditional on proposal etas in M-H sampler
    // mydata->w0Sq = sampleW0(w0A, w0B, currentPars(0,l));
    // mydata->wSq = sampleW(wA, wB, currentPars.submat(1,l,p,l));
    for (unsigned int j = 0; j < p+1; ++j)
    {
        if (etas(j, l))
        {
            mydata->jj = j;
            mydata->l = l;
            double xprev = currentPars(j, l);
            std::vector<double> xsamp(armsPar.nsamp);

            double qcent[1], xcent[1];
            int neval, ncent = 0;

            int err;
            // if (armsPar.simple)
            // {
            //     err = ARMS::arms_simple (
            //               armsPar.ninit, &minD, &maxD,
            //               EvalFunction::log_dens_zetas, mydata,
            //               armsPar.metropolis, &xprev, xsamp.data());
            // }
            // else
            // {
            // }
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

    // logPosteriorZeta = logPzetaK(k, currentPars, mydata->wSq, kappa, datTheta, weibullS, weibullLambda, dataclass);
    free(mydata);

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
    arma::vec datTheta,
    arma::mat datMu,
    arma::mat datProportion,
    const DataClass &dataclass)
{
    // dimensions
    unsigned int N = datProportion.n_rows;
    unsigned int L = datProportion.n_cols;

    //int armsPar.metropolis = metropolis;

    // objects for arms()
    const double minD = armsPar.kappaMin;
    const double maxD = armsPar.kappaMax;
    //double *xl; xl = &minD;
    //double *xr; xr = &maxD;

    // reallocate struct variables

    // double xinit[armsPar.ninit];
    std::vector<double> xinit(armsPar.ninit); // Use std::vector instead of VLA
    // if (!armsPar.simple)
    // {
    // }
    arma::vec xinit0 = arma::linspace( minD+1.0e-10, maxD-1.0e-10, armsPar.ninit );
    for (unsigned int i = 0; i < armsPar.ninit; ++i)
        xinit[i] = xinit0[i];

    dataS *mydata = (dataS *)malloc(sizeof (dataS));
    //mydata->currentPars = &currentPars;
    //mydata->p = p;
    mydata->L = L;
    mydata->N = N;
    // mydata->xl = xl;
    // mydata->xr = xr;
    mydata->kappaA = kappaA;
    mydata->kappaB = kappaB;
    mydata->invGamma = invGamma;
    mydata->datTheta = datTheta.memptr();
    mydata->datMu = datMu.memptr();
    mydata->datProportion = datProportion.memptr();
    mydata->datEvent = dataclass.datEvent.memptr();
    mydata->datTime = dataclass.datTime.memptr();

    // define how many ARMS samples to draw for currentPars; first one as intial value
    // std::vector<double> samp(armsPar.n + 1);
    // samp[0] = currentPars;
    /*
        double xprev = currentPars;
        // double *xsamp = (double*)malloc(armsPar.nsamp * sizeof(double));
        std::vector<double> xsamp(armsPar.nsamp);

        double qcent[1], xcent[1];
        int neval, ncent = 0;

        int err;
        if (armsPar.simple)
        {
            err = ARMS::arms_simple (
                      armsPar.ninit, &minD, &maxD,
                      EvalFunction::log_dens_kappa, mydata,
                      armsPar.metropolis, &xprev, xsamp.data());
        }
        else
        {
            double convex = armsPar.convex;
            err = ARMS::arms (
                      xinit.data(), armsPar.ninit, &minD, &maxD,
                      EvalFunction::log_dens_kappa, mydata,
                      &convex, armsPar.npoint,
                      armsPar.metropolis, &xprev, xsamp.data(),
                      armsPar.nsamp, qcent, xcent, ncent, &neval);
        }

        // check ARMS validity
        if (err > 0)
            Rprintf("In arms_kappa(): error code in ARMS = %d.\n", err);
        if (std::isnan(xsamp[armsPar.nsamp-1]))
            Rprintf("In arms_kappa(): NaN generated, possibly due to overflow in (log-)density (e.g. with densities involving exp(exp(...))).\n");
        if (xsamp[armsPar.nsamp-1] < minD || xsamp[armsPar.nsamp-1] > maxD)
            Rprintf("In arms_kappa(): %d-th sample out of range [%f, %f] (fused domain). Got %f.\n", armsPar.nsamp, minD, maxD, xsamp[armsPar.nsamp-1]);

        currentPars = xsamp[armsPar.nsamp - 1];
        */
    slice_sample (
        EvalFunction::log_dens_kappa,
        mydata,
        currentPars,
        10,
        1.0,
        minD,
        maxD
    );
    // for (int i = 0; i < armsPar.n; ++i)
    //     samp[i + 1] = xsamp[armsPar.nsamp - armsPar.n + i];

    // remove the initial value
    // samp.erase(samp.begin());

    // free(xsamp);
    free(mydata);

    // return samp;
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

        if (cnt == 1e4) ::Rf_error("slice_sample_cpp loop did not finish");

        x = xs;
        logy = logys;
    }

}