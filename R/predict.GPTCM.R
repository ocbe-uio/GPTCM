#' @title Predictions for a GPTCM
#'
#' @description
#' Compute fitted values for a generalized promotion time cure model (GPTCM)
#'
#' @name predict.GPTCM
#'
#' @param object the results of a \code{GPTCM} fit
#' @param dat the dataset used in \code{GPTCM()}
#' @param newdata pptional new data at which to do predictions. If missing, the
#' prediction will be based on the training data
#' @param type the type of predicted value. Currently it is only valid with
#' 'survival'
#' @param ... for future methods
#'
#' @return A matrix object
#'
#' @examples
#'
#' x <- 1
#'
#' @export
predict.GPTCM <- function(object, dat, newdata = NULL,
                          type = "survival", ...) {
  n <- dim(dat$XX)[1]
  p <- dim(dat$XX)[2]
  L <- dim(dat$XX)[3]

  if (is.null(newdata)) {
    survObj.new <- dat$survObj
  } else {
    survObj.new <- newdata$survObj
  }

  # nIter <- object$input$nIter
  burnin <- object$input$burnin / object$input$thin

  # survival predictions based on posterior mean
  xi.hat <- colMeans(object$output$mcmc$xi[-c(1:burnin), ])
  betas.hat <- matrix(colMeans(object$output$mcmc$betas[-c(1:burnin), ]), ncol = L)
  if (object$input$proportion.model) {
    zetas.hat <- matrix(colMeans(object$output$mcmc$zetas[-c(1:burnin), ]), ncol = L)
  }
  if (object$input$BVS) {
    gammas.hat <- matrix(colMeans(object$output$mcmc$gammas[-c(1:burnin), ]), ncol = L)
    gammas.hat <- rbind(1, gammas.hat)
    betas.hat <- (gammas.hat >= 0.5) * betas.hat / gammas.hat
    betas.hat[is.na(betas.hat)] <- 0

    if (object$input$proportion.model) {
      etas.hat <- rbind(1, matrix(colMeans(object$output$mcmc$etas[-c(1:burnin), ]), ncol = L))
      zetas.hat <- (etas.hat >= 0.5) * zetas.hat / etas.hat
      zetas.hat[is.na(zetas.hat)] <- 0
    }
  }
  kappa.hat <- mean(object$output$mcmc$kappa[-c(1:burnin)])
  thetas.hat <- exp(newdata$x0 %*% xi.hat)

  # predict survival probabilities based on GPTCM
  time_eval <- sort(newdata$survObj$time)
  Surv.prob <- matrix(nrow = n, ncol = length(time_eval))
  if (object$input$proportion.model) {
    alphas <- sapply(1:L, function(ll) {
      exp(cbind(1, newdata$XX[, , ll]) %*% zetas.hat[, ll])
    })
    proportion.hat <- alphas / rowSums(alphas)
  } else {
    proportion.hat <- dat$proportion
  }
  for (j in 1:length(time_eval)) {
    tmp <- 0
    for (l in 1:L) {
      mu <- exp(cbind(1, newdata$XX[, , l]) %*% betas.hat[, l])
      lambdas <- mu / gamma(1 + 1 / kappa.hat)
      weibull.S <- exp(-(time_eval[j] / lambdas)^kappa.hat)
      tmp <- tmp + proportion.hat[, l] * weibull.S
    }
    Surv.prob[, j] <- exp(-thetas.hat * (1 - tmp))
  }

  return(Surv.prob)
}
