#' @title Target density
#'
#' @description
#' Predefined target density corresponding to the population survival function
#' of GPTCM
#'
#' @name target
#'
#' @param x value generated from the proposal distribution
#' @param theta cure rate parameter (log scale)
#' @param proportion proportions data
#' @param mu mean survival time
#' @param kappas Weibull's true shape parameter
#'
#' @return value of the targeted (improper) probability density function
#'
#'
#' @examples
#'
#' time1 <- target(1.2, 0.1, c(0.2, 0.3, 0.5), c(0.2, 0.1, 0.4), 2)
#'
#' @export
target <- function(x, theta, proportion, mu, kappas) {
  ## Weibull 3
  lambdas <- mu / gamma(1 + 1 / kappas)
  survival.function <- exp(-(x / lambdas)^kappas)
  # improper pdf
  pdf <- exp(-theta * (1 - sum(proportion * survival.function))) *
    theta *
    sum(proportion * kappas / lambdas * 
          (x / lambdas)^(kappas - 1) * 
          exp(-(x / lambdas)^kappas))

  return(pdf)
}
