#' @title Prediction of survival probability
#'
#' @description
#' Compute predicted survival probability for a GPTCM
#'
#' @name predict.GPTCM
#'
#' @param object the results of a \code{GPTCM} fit
#' @param dat the dataset used in \code{GPTCM()}
#' @param newdata optional new data at which to do predictions. If missing, the
#' prediction will be based on the training data
#' @param type the type of predicted value. Currently it is only valid with
#' \code{'survival'}
#' @param times evaluation time points for survival prediction. Default
#' \code{NULL} for predicting all time points in the \code{newdata} set
#' @param ... for future methods
#'
#' @return A matrix object
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
#' pred.survival <- predict(fit, dat, newdata = dat, times = c(1, 3, 5))
#'
#' @export
predict.GPTCM <- function(object, dat,
                          newdata = NULL,
                          type = "survival",
                          times = NULL, ...) {
  n <- dim(dat$X)[1]
  # p <- dim(dat$X)[2]
  L <- dim(dat$X)[3]

  if (is.null(newdata)) {
    survObj.new <- dat$survObj
  } else {
    survObj.new <- newdata$survObj
  }

  # nIter <- object$input$nIter
  burnin <- object$input$burnin / object$input$thin

  # survival predictions based on posterior mean
  xi.hat <- colMeans(object$output$xi[-c(1:burnin), ])
  betas.hat <- matrix(colMeans(object$output$betas[-c(1:burnin), ]), ncol = L)
  if (object$input$proportion.model) {
    zetas.hat <- matrix(colMeans(object$output$zetas[-c(1:burnin), ]), ncol = L)
  }
  if (object$input$BVS) {
    gammas.hat <- matrix(colMeans(object$output$gammas[-c(1:burnin), ]), ncol = L)
    gammas.hat <- rbind(1, gammas.hat)
    betas.hat <- (gammas.hat >= 0.5) * betas.hat / gammas.hat
    betas.hat[is.na(betas.hat)] <- 0

    if (object$input$proportion.model) {
      etas.hat <- rbind(1, matrix(colMeans(object$output$etas[-c(1:burnin), ]), ncol = L))
      zetas.hat <- (etas.hat >= 0.5) * zetas.hat / etas.hat
      zetas.hat[is.na(zetas.hat)] <- 0
    }
  }
  kappa.hat <- mean(object$output$kappa[-c(1:burnin)])
  thetas.hat <- exp(newdata$x0 %*% xi.hat)

  # predict survival probabilities based on GPTCM
  time_eval <- times
  if (is.null(time_eval)) {
    time_eval <- sort(survObj.new$time)
  }
  Surv.prob <- matrix(nrow = n, ncol = length(time_eval))
  if (object$input$proportion.model) {
    alphas <- sapply(1:L, function(ll) {
      exp(cbind(1, newdata$X[, , ll]) %*% zetas.hat[, ll])
    })
    proportion.hat <- alphas / rowSums(alphas)
  } else {
    proportion.hat <- dat$proportion
  }
  for (j in seq_along(time_eval)) {
    tmp <- 0
    for (l in 1:L) {
      mu <- exp(cbind(1, newdata$X[, , l]) %*% betas.hat[, l])
      lambdas <- mu / gamma(1 + 1 / kappa.hat)
      weibull.S <- exp(-(time_eval[j] / lambdas)^kappa.hat)
      tmp <- tmp + proportion.hat[, l] * weibull.S
    }
    Surv.prob[, j] <- exp(-thetas.hat * (1 - tmp))
  }

  return(Surv.prob)
}
