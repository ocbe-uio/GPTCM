#' @title Extract the posterior estimate of parameters
#' @description
#' Extract the posterior estimate of the parameters of a \code{GPTCM} class object.
#' @name getEstimator
#'
#' @param object an object of class \code{GPTCM}
#' @param estimator the name of one estimator. Default is the latent indicator
#' estimator "\code{gamma}". Other options are among
#' "\code{c('beta', 'zeta', 'eta', 'xi', 'elpd', 'logP')}"
#' @param Pmax threshold that truncate the estimator "\code{gamma}" or
#' "\code{eta}". Default is \code{0}. If \code{Pmax=0.5} and
#' \code{type="conditional"}, it gives median probability model betas
#' @param type the type of output beta. Default is \code{marginal}, giving
#' marginal beta estimation. If \code{type="conditional"}, it gives beta
#' estimation conditional on gamma=1
#'
#' @return Return the estimator from an object of class \code{GPTCM}. It is
#' a matrix or vector
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
#' gamma.hat <- getEstimator(fit, estimator = "gamma")
#'
#' @export
getEstimator <- function(object, estimator = "gamma", Pmax = 0,
                         type = "marginal") {
  if (!estimator %in% c("gamma", "beta", "eta", "zeta", "xi", "elpd", "logP")) {
    stop("Please specify correct 'estimator'!")
  }

  if ((estimator %in% c("gamma", "eta")) && !object$input$BVS) {
    stop("The fitted object from GPTCM does not have variable selection'!")
  }

  if (Pmax < 0 || Pmax > 1) {
    stop("Please specify correct argument 'Pmax' in [0,1]!")
  }

  if (!object$input$BVS) {
    if (type == "conditional") {
      message("NOTE: The argument type is invalid!")
    }
    if (Pmax > 0) {
      message("NOTE: The argument Pmax is invalid!")
    }
  }

  if ((estimator %in% c("gamma", "beta")) && object$input$BVS) {
    # estimate <- matrix(colMeans(object$output$gammas[-c(1:burnin), ]), ncol = L)
    estimate <- object$output$post$gammas
    estimate <- rbind(1, estimate)
    gammas <- estimate
    if (Pmax > 0) {
      estimate[estimate <= Pmax] <- 0
      estimate[estimate > Pmax] <- 1
    }
  }

  if (estimator == "beta") {
    # estimate <- matrix(colMeans(object$output$betas[-c(1:burnin), ]), ncol = L)
    estimate <- object$output$post$betas

    if (type %in% c("marginal", "conditional")) {
      if ((type == "conditional") && object$input$BVS) {
        estimate <- (gammas >= Pmax) * estimate / gammas
        estimate[is.na(estimate)] <- 0
      }
    } else {
      stop("Please specify correct type!")
    }
  }

  if ((estimator %in% c("eta", "zeta")) && object$input$BVS) {
    estimate <- object$output$post$etas
    estimate <- rbind(1, estimate)
    etas <- estimate
    if (Pmax > 0) {
      estimate[estimate <= Pmax] <- 0
      estimate[estimate > Pmax] <- 1
    }
  }

  if (estimator == "zeta") {
    estimate <- object$output$post$zetas

    if (type %in% c("marginal", "conditional")) {
      if ((type == "conditional") && object$input$BVS) {
        estimate <- (etas >= Pmax) * estimate / etas
        estimate[is.na(estimate)] <- 0
      }
    } else {
      stop("Please specify correct type!")
    }
  }

  if (estimator == "xi") {
    estimate <- as.vector(object$output$post$xi)
  }

  if (estimator == "elpd") {
    estimate <- loo::loo(object$output$loglikelihood[
      1 + (object$input$burnin:object$input$nIter) / object$input$thin,
    ])
    estimate
  }

  if (estimator == "logP") {
    estimate <- rowSums(object$output$loglikelihood)
  }

  return(estimate)
}
