#' @title Metropolis sampler for a target density
#'
#' @description
#' Random number generator via MH algorithm.
#'
#' @name metropolis_sampler
#'
#' @importFrom stats rweibull runif
#'
#' @param initial_value TBA
#' @param n TBA
#' @param proposal_shape TBA
#' @param proposal_scale TBA
#' @param theta TBA
#' @param proportion TBA
#' @param mu TBA
#' @param kappas TBA
#' @param burnin TBA
#' @param lag TBA
#'
#' @return An object of ...
#'
#'
#' @examples
#'
#' x <- 1
#'
#' @export
metropolis_sampler <- function(initial_value,
                               n = n,
                               proposal_shape = 1,
                               proposal_scale = 1,
                               theta = 1,
                               proportion = 0.5,
                               mu = 1,
                               kappas = 0.9,
                               burnin = 0,
                               lag = 1) {
  results <- list()
  current_state <- initial_value
  for (i in 1:burnin) {
    out <- metropolis_step(
      current_state, proposal_shape, proposal_scale,
      theta, proportion, mu, kappas
    )
    if (!is.na(out$value)) {
      current_state <- out$value
    }
  }
  for (i in 1:n) {
    for (j in 1:lag) {
      out <- metropolis_step(
        current_state, proposal_shape, proposal_scale,
        theta, proportion, mu, kappas
      )
      if (!is.na(out$value)) {
        current_state <- out$value
      }
    }
    results[[i]] <- out
  }
  results <- do.call(rbind, results)
  results
}


metropolis_step <- function(x, proposal_shape, proposal_scale,
                            theta, proportion, mu, kappas) {
  proposed_x <- rweibull(1, shape = proposal_shape, scale = proposal_scale)
  accept_prob <- min(1, target(proposed_x, theta, proportion, mu, kappas) /
    target(x, theta, proportion, mu, kappas))
  u <- runif(1)
  if (is.na(u <= accept_prob)) {
    value <- NA
    accepted <- FALSE
  } else {
    if (u <= accept_prob) {
      value <- proposed_x
      accepted <- TRUE
    } else {
      value <- x
      accepted <- FALSE
    }
  }
  out <- data.frame(value = value, accepted = accepted)
  out
}
