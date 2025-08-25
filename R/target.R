#' @title Target density
#'
#' @description
#' Predefined target density for S_pop in GTPCM
#'
#' @name target
#'
#' @param x TBA
#' @param theta TBA
#' @param proportion TBA
#' @param mu TBA
#' @param kappas TBA
#'
#' @return An object of ...
#'
#'
#' @examples
#'
#' x <- 1
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
