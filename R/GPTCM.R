#' @title Fit Bayesian GPTCM Models
#'
#' @description
#' This is the main function to fit the Bayesian GPTCMs (Zhao et al. 2025) with
#' multiscale data for sparse identification of high-dimensional covariates
#'
#' @name GPTCM
#'
#' @importFrom Rcpp evalCpp
#'
#' @param dat input data as a list containing survival data sub-list
#' \code{survObj} with two vectors (\code{event} and \code{time}), clinical
#' variable matrix \code{x0}, cluster-specific covariates \code{X}, and
#' proportions data matrix \code{proportion}
#' @param nIter the number of iterations of the chain
#' @param burnin number of iterations to discard at the start of the chain
#' @param thin thinning MCMC intermediate results to be stored
#' @param tick an integer used for printing the iteration index and some updated
#' parameters every tick-th iteration. Default is 1
#' @param proportion.model logical value; should the proportions be modeled or
#' not. If (\code{proportion.model = FALSE}), the argument \code{dirichlet} will
#' be invalid
#' @param dirichlet logical value; should the proportions be modeled via the
#' common (\code{dirichlet = TRUE}) or alternative (\code{dirichlet = FALSE})
#' parametrization of the Dirichlet regression model
#' @param hyperpar a list of relevant hyperparameters
#' @param BVS logical value for implementing Bayesian variable selection
#' @param kappaIGamma logical value for using inverse-gamma prior (\code{TRUE})
#' or gamma prior (\code{FALSE}) for Weibull's shape parameter
#' shape parameter
#' @param kappaSampler one of \code{"arms", "slice"} (slice not yet implemented)
#' @param gammaPrior one of \code{c("bernoulli", "MRF")}
#' @param gammaSampler one of \code{c("mc3", "bandit")}
#' @param etaPrior one of \code{c("bernoulli", "MRF")}
#' @param etaSampler one of \code{c("mc3", "bandit")}
#' @param w0IGamma logical value; if \code{FALSE}, a common parameter is used
#' for the intercept's prior variance and the coefficient's prior variance
#' @param initial a list of initial values for parameters "kappa", "xi",
#' "betas", and "zetas"
#' @param arms.list a list of parameters for the ARMS method
#'
#' @return An object of a list including the following components:
#' \itemize{
#' \item input - a list of all input parameters by the user
#' \item output - a list of the all mcmc output estimates:
#' \itemize{
#' \item "\code{xi}" - a matrix with MCMC intermediate estimates of effects on clinical variables
#' \item "\code{kappa}" - a vector with MCMC intermediate estimates of the Weibull's shape parameter
#' \item "\code{betas}" - a matrix with MCMC intermediate estimates of effects on cluster-specific survival
#' \item "\code{zetas}" - a matrix with MCMC intermediate estimates of effects on cluster-specific proportions
#' \item "\code{gammas}" - a matrix with MCMC intermediate estimates of inclusion indicators of variables for cluster-specific survival
#' \item "\code{gamma_acc_rate}" - acceptance rate of the M-H sampling for gammas
#' \item "\code{etas}" - a matrix with MCMC intermediate estimates of inclusion indicators of variables for cluster-specific proportions
#' \item "\code{eta_acc_rate}" - acceptance rate of the M-H sampling for etas
#' \item "\code{loglikelihood}" - a matrix with MCMC intermediate estimates of individuals' likelihoods
#' \item "\code{tauSq}" - a vector with MCMC intermediate estimates of tauSq
#' \item "\code{wSq}" - a matrix with MCMC intermediate estimates of wSq
#' \item "\code{vSq}" - a matrix with MCMC intermediate estimates of vSq
#' \item "\code{post}" - a list with posterior means of "xi", "kappa", "betas", "zetas", "gammas", "etas"
#' }
#' \item call - the matched call
#' }
#'
#' @references Zhao Z, Kızılaslan F, Wang S, Zucknick M (2025). \emph{Generalized promotion time cure model: A new modeling framework to identify cell-type-specific genes and improve survival prognosis}. arXiv:2509.01001
#'
#' @examples
#'
#' # simulate data
#' set.seed(123)
#' n <- 200 # subjects
#' p <- 10 # variable selection predictors
#' L <- 3 # cell types
#' dat <- simData(n, p, L)
#'
#' # run a Bayesian GPTCM model: GPTCM-Ber2
#' fit <- GPTCM(dat, nIter = 10, burnin = 0)
#'
#'
#' @export
GPTCM <- function(dat,
                  nIter = 500,
                  burnin = 200,
                  thin = 1,
                  tick = 100,
                  proportion.model = TRUE,
                  dirichlet = TRUE,
                  hyperpar = NULL,
                  BVS = TRUE,
                  kappaIGamma = TRUE,
                  kappaSampler = "arms",
                  gammaPrior = "bernoulli",
                  gammaSampler = "MC3",
                  etaPrior = "bernoulli",
                  etaSampler = "MC3",
                  w0IGamma = TRUE,
                  initial = NULL,
                  arms.list = NULL) {
  # Validation
  stopifnot(burnin < nIter)
  stopifnot(burnin >= 0)

  n <- dim(dat$X)[1]
  p <- dim(dat$X)[2]
  L <- dim(dat$X)[3]

  if (is.null(arms.list)) {
    arms.list <- list(
      n = 1, # This should always be n=1 with the current code
      nsamp = 1,
      ninit = 10,
      metropolis = 1,
      arms.simple = FALSE,
      convex = 1,
      npoint = 100
    )
    # arms.simple: logical value; should the "adaptive rejection metropolis
    # sampling" or the "derivative-free adaptive rejection sampling with
    # metropolis step" (default) is used
  }

  if (arms.list$n != 1) {
    stop("Need to modify 'arms_gibbs.cpp' if arms.list$n > 1!")
  }

  # check the formula
  cl <- match.call()

  gammaSampler <- tolower(gammaSampler)
  if (!gammaSampler %in% c("mc3", "bandit")) {
    stop('Argument "gammaSampler" must be one of c("mc3", "bandit")!')
  }

  gammaPrior <- tolower(gammaPrior)
  if (!gammaPrior %in% c("bernoulli", "mrf")) {
    stop('Argument "gammaSampler" must be one of c("bernoulli", "MRF")!')
  }

  etaSampler <- tolower(etaSampler)
  if (!etaSampler %in% c("mc3", "bandit")) {
    stop('Argument "etaSampler" must be one of c("mc3", "bandit")!')
  }

  etaPrior <- tolower(etaPrior)
  if (!etaPrior %in% c("bernoulli", "mrf")) {
    stop('Argument "etaSampler" must be one of c("bernoulli", "MRF")!')
  }

  # set hyperparamters of all piors
  # if (is.null(hyperpar)) {
  if (is.null(hyperpar)) {
    hyperpar <- list()
  }

  # MRF prior related hyperparameters
  if (gammaPrior == "mrf") { # Maybe not use if-condition, otherwise some issues
    if (is.null(hyperpar$mrfG)) {
      hyperpar$mrfA <- -3
      hyperpar$mrfB <- 0 # 0.01
      hyperpar$mrfG <- matrix(0, nrow = 2, ncol = 3)
    }
    hyperpar$mrfG.weights <- hyperpar$mrfG[, 3]
    hyperpar$mrfG <- hyperpar$mrfG[, 1:2]
  }

  if (etaPrior == "mrf") {
    if (is.null(hyperpar$mrfG.prop)) {
      hyperpar$mrfA.prop <- -3
      hyperpar$mrfB.prop <- 0 # 0.01
      hyperpar$mrfG.prop <- matrix(0, nrow = 2, ncol = 3)
    }
    hyperpar$mrfG.prop.weights <- hyperpar$mrfG.prop[, 3]
    hyperpar$mrfG.prop <- hyperpar$mrfG.prop[, 1:2]
  }

  # spike-and-slab's gammas hyperparameters of beta prior
  if (!"pi" %in% names(hyperpar)) {
    hyperpar$pi <- 0
  }
  if (!"piA" %in% names(hyperpar)) {
    hyperpar$piA <- 2
    hyperpar$piB <- 20
  }

  # spike-and-slab's etas hyperparameters
  if (!"rho" %in% names(hyperpar)) {
    hyperpar$rho <- 0
  }
  if (!"rhoA" %in% names(hyperpar)) {
    hyperpar$rhoA <- 2
    hyperpar$rhoB <- 20
  }

  # hyperpar$tauA <- 20; hyperpar$tauB <- 50
  if (!"tauA" %in% names(hyperpar)) {
    hyperpar$tauA <- 5
    hyperpar$tauB <- 20
  }
  if (!"tau0A" %in% names(hyperpar)) {
    hyperpar$tau0A <- hyperpar$tauA
    hyperpar$tau0B <- hyperpar$tauB
  }
  # hist(1/rgamma(100, 20, 50))
  # hyperpar$wA <- 20; hyperpar$wB <- 50; wSq <- 1
  if (!"wA" %in% names(hyperpar)) {
    hyperpar$wA <- 5
    hyperpar$wB <- 20
  }
  if (!"vA" %in% names(hyperpar)) {
    hyperpar$vA <- 5 # 10
    hyperpar$vB <- 20
  }
  if (kappaIGamma) {
    if (!"kappaA" %in% names(hyperpar)) {
      hyperpar$kappaA <- 5
      hyperpar$kappaB <- 20 # 5
    }
  } else {
    if (!"kappaA" %in% names(hyperpar)) {
      hyperpar$kappaA <- 1 # 3
      hyperpar$kappaB <- 1 # This is for Gamma prior
    }
  }
  hyperpar$kappaIGamma <- kappaIGamma

  if (w0IGamma) {
    hyperpar$w0A <- hyperpar$wA
    hyperpar$w0B <- hyperpar$wB
  }
  hyperpar$w0IGamma <- w0IGamma

  hyperpar$Delta <- 20
  # }
  hyperpar$tauSq <- rep(1, L)
  hyperpar$tau0Sq <- 1
  hyperpar$wSq <- rep(1, L)
  hyperpar$w0Sq <- 1
  # hyperpar$vSq <- c(10, 1, 1)
  hyperpar$vSq <- 1

  if (!"v0Sq" %in% names(hyperpar)) {
    hyperpar$v0Sq <- 10
  }
  hyperpar$v0A <- hyperpar$vA
  hyperpar$v0B <- hyperpar$vB

  # transform proportions data if including values very close to 0 or 1
  # This is the same as in DirichletReg::DR_data
  if (any(dat$proportion < 1e-10) || any(dat$proportion > 1 - 1e-10)) {
    dat$proportion <- (dat$proportion * (n - 1) + 1 / L) / n
  }

  # initialization of parameters
  if (is.null(initial)) {
    initList <- list()

    initList$xi <- rep(0, NCOL(dat$x0))

    initList$kappa <- 0.9
    initList$betas <- matrix(0, nrow = dim(dat$X)[2] + 1, ncol = NCOL(dat$proportion)) # include intercept
    initList$gammas <- matrix(as.numeric(initList$betas[-1, ] != 0),
      nrow = dim(dat$X)[2], ncol = NCOL(dat$proportion)
    )

    ## proportion Dirichlet part
    initList$phi <- 1
    initList$zetas <- matrix(0, nrow = dim(dat$X)[2] + 1, ncol = NCOL(dat$proportion)) # include intercept
  }

  if (!"bound.pos" %in% names(hyperpar)) {
    hyperpar$bound.neg <- -10
    hyperpar$bound.pos <- 10
  }
  hyperpar$bound.kappa <- 1e-2
  rangeList <- list(
    xiMin = hyperpar$bound.neg, xiMax = hyperpar$bound.pos,
    zetaMin = hyperpar$bound.neg, zetaMax = hyperpar$bound.pos,
    betaMin = hyperpar$bound.neg, betaMax = hyperpar$bound.pos,
    kappaMin = hyperpar$bound.kappa, kappaMax = hyperpar$bound.pos
  )


  #################
  ## Output objects
  #################

  ret <- list(input = list(), output = list(), call = cl)
  class(ret) <- "GPTCM"

  ret$input$n <- n
  ret$input$p <- p
  ret$input$L <- L
  ret$input$BVS <- BVS
  ret$input$proportion.model <- proportion.model
  ret$input$dirichlet <- dirichlet
  ret$input$nIter <- nIter
  ret$input$burnin <- burnin
  ret$input$thin <- thin
  ret$input$hyperpar <- hyperpar


  #################
  ## Main steps for Bayesian inference
  #################

  ## MCMC iterations
  ret$output <- run_mcmc(
    nIter,
    burnin,
    thin,
    arms.list$n, # n: number of samples to draw, now only 1
    arms.list$nsamp, # nsamp: number of MCMC for generating each ARMS sample, only keeping the last one
    arms.list$ninit, # ninit: number of initials as meshgrid values for envelop search
    arms.list$metropolis, # metropolis: 0/1 metropolis step or not
    arms.list$arms.simple,
    arms.list$convex, # convex: adjustment for convexity
    arms.list$npoint, # npoint: maximum number of envelope points
    dirichlet,
    proportion.model,
    BVS,
    gammaPrior,
    gammaSampler,
    etaPrior,
    etaSampler,
    initList,
    rangeList,
    hyperpar,
    dat$survObj$event,
    dat$survObj$time,
    dat$X,
    dat$x0,
    dat$proportion
  )

  # survival predictions based on posterior mean
  # ret$output$posterior <- list(
  #   xi = colMeans(ret$output$xi[-c(1:burnin), ]),
  #   kappa = mean(ret$output$kappa[-c(1:burnin)]),
  #   betas = matrix(colMeans(ret$output$betas[-c(1:burnin), ]), ncol = L)
  # )
  # ret$output$posterior <- ret$output$post

  # ret$output$mcmc <- list(
  #   xi = xi.mcmc,
  #   kappa = kappa.mcmc,
  #   phi = phi.mcmc,
  #   betas = betas.mcmc,
  #   zetas = zetas.mcmc,
  #   tauSq = tauSq.mcmc,
  #   wSq = wSq.mcmc,
  #   vSq = vSq.mcmc
  # )

  return(ret)
}
