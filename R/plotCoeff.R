#' @title Plot posterior estimates of regression coefficients
#'
#' @description
#' create nice plots for estimated coefficients and 95% credible intervals
#'
#' @name plotCoeff
#'
#' @importFrom graphics plot.default axis points arrows legend segments box
#' @importFrom ggplot2 .data guides guide_legend facet_wrap geom_point geom_segment element_text xlim unit
#' @importFrom ggridges geom_density_ridges
#' @importFrom utils capture.output
#' @importFrom scales alpha
#'
#' @param dat input data as a list containing survival data sub-list
#' \code{survObj} with two vectors (\code{event} and \code{time}), clinical
#' variable matrix \code{x0}, cluster-specific covariates \code{X}, and
#' proportions data matrix \code{proportion}
#' @param datMCMC returned object from the main function \code{GPTCM()}
#' @param estimator print estimators, one of
#' \code{c("beta", "zeta", "gamma", "eta")}
#' @param intercept logical value to print intercepts
#' @param bandwidth a value of bandwidth used for the ridgeplot
#' @param xlim numeric vectors of length 2, giving the x-coordinate range.
#' @param xlab a title for the x axis
#' @param label.y a title for the y axis
#' @param first.coef number of the first variables. Default \code{NULL} for
#' all variables
#' @param y.axis.size text size in pts
#' @param ... others
#'
#' @return A \code{ggplot2::ggplot} object. See \code{?ggplot2::ggplot} for more
#' details of the object.
#'
#' @references Zhao Z, Kızılaslan F, Wang S, Zucknick M (2025). \emph{Generalized promotion time cure model: A new modeling framework to identify cell-type-specific genes and improve survival prognosis}. arXiv:2509.01001
#'
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
#' plotCoeff(dat, datMCMC = fit, estimator = "beta")
#'
#' @export
plotCoeff <- function(dat,
                      datMCMC,
                      estimator = "beta",
                      intercept = FALSE,
                      bandwidth = NULL,
                      xlim = NULL,
                      xlab = NULL,
                      label.y = NULL,
                      first.coef = NULL,
                      y.axis.size = 8, ...) {
  # n <- dim(dat$X)[1]
  p <- dim(dat$X)[2]
  L <- dim(dat$X)[3]
  thin <- datMCMC$input$thin
  # nIter <- datMCMC$input$nIter / thin
  burnin <- datMCMC$input$burnin / thin + 1

  xlab0 <- xlab
  if (is.null(xlab0)) {
    xlab0 <- "Effect"
  }
  if (!is.null(first.coef)) {
    p0 <- first.coef
  }
  if (!is.null(xlim)) {
    xlim.lower <- xlim[1]
    xlim.upper <- xlim[2]
  }

  # legend0 <- ifelse(is.null(legend.labs), 0, 1)

  if (estimator == "beta") {
    betas.mcmc <- datMCMC$output$betas[-c(1:burnin), ]
    betas.true <- dat$betas
    if (!intercept) {
      betas.mcmc <- betas.mcmc[, -seq(1, ((p + 1) * L), by = p + 1)]
    }
    if (!is.null(first.coef)) {
      idx <- rep(1:p0, L) + rep(seq(0, p * (L - 1), by = p), each = p0)
      betas.mcmc <- betas.mcmc[, idx]
      betas.true <- betas.true[1:p0, ]
      p <- p0
    }
    # Final estimates
    if (is.null(xlim)) {
      xlim.lower <- min(betas.mcmc, dat$betas)
      xlim.upper <- max(betas.mcmc, dat$betas)
    }

    if (is.null(label.y)) {
      label.y <- paste0("x", 1:p)
    }

    dat.estimate <- data.frame(
      cluster = rep(paste("Cell type", 1:L), each = length(betas.mcmc) / L),
      covariate = rep(rep(label.y, each = nrow(betas.mcmc)), L),
      value = as.vector(betas.mcmc),
      coef.true = rep(betas.true, each = nrow(betas.mcmc))
    )
    dat.estimate$covariate <- factor(dat.estimate$covariate,
      levels = label.y[p:1]
    )
    if (abs(xlim.lower - xlim.upper) < 0.1) {
      xlim.lower <- xlim.lower - 1
      xlim.upper <- xlim.upper + 1
      dat.estimate$value <- rnorm(length(betas.mcmc), betas.mcmc[1], 1e-4)
    }
    pp <-
      ggplot(dat.estimate,
        mapping = aes(
          x = .data$value,
          y = .data$covariate,
          group = .data$covariate
        )
      ) +
      geom_density_ridges(aes(fill = .data$covariate),
        scale = 0.95, col = NA, bandwidth = bandwidth
      ) +
      facet_wrap(~cluster, nrow = 1, scales = "fixed") +
      geom_point(
        shape = 18, size = 2, data = dat.estimate,
        aes(x = .data$coef.true, y = .data$covariate, fill = "black")
      ) +
      xlim(xlim.lower, xlim.upper) +
      xlab(xlab0) +
      ylab("") +
      theme_bw() +
      theme(
        panel.grid.major.x = element_blank(),
        panel.grid.minor = element_blank()
      ) +
      guides(fill = "none")
    # return(pp)
  }

  if (estimator %in% c("gamma", "eta")) {
    if (estimator == "gamma") {
      bvs.mcmc <- datMCMC$output$gammas[-c(1:burnin), ]
    } else {
      bvs.mcmc <- datMCMC$output$etas[-c(1:burnin), ]
    }
    mPIP <- matrix(colMeans(bvs.mcmc), nrow = p, ncol = L)
    if (any(mPIP < 5e-3)) {
      mPIP[mPIP < 5e-3] <- 5e-3
    }
    # Final estimates

    if (is.null(label.y)) {
      label.y <- paste0("x", 1:p)
    }

    dat.estimate <- data.frame(
      cluster = rep(paste("Cell type", 1:L), each = p),
      covariate = rep(label.y, L),
      value = as.vector(mPIP)
    )
    dat.estimate$covariate <- factor(dat.estimate$covariate,
      levels = label.y[p:1]
    )
    pp <-
      ggplot(dat.estimate,
        mapping = aes(
          x = .data$value,
          y = .data$covariate,
          group = .data$covariate
        )
      ) +
      facet_wrap(~cluster, nrow = 1, scales = "fixed") +
      geom_segment(aes(x = 0, xend = .data$value),
        linewidth = 1, lineend = "butt", col = "blue"
      ) +
      xlim(0, 1) +
      ylab("") +
      xlab(eval(parse(text = paste0("expression(", xlab, "~mPIP~", estimator, ")")))) +
      theme_bw() +
      theme(
        axis.text.y = element_text(size = y.axis.size),
        panel.grid.major.y = element_blank(),
        panel.grid.minor = element_blank(),
        panel.spacing = unit(1, "lines")
      ) +
      guides(fill = "none")
    # return(pp)
  }


  if (estimator == "zeta") {
    # Final estimates
    if (!datMCMC$input$dirichlet) {
      L <- L - 1
    }
    zetas.mcmc <- datMCMC$output$zetas[-c(1:burnin), 1:(L * (p + 1))]
    zetas.true <- dat$zetas[, 1:L]
    if (!intercept) {
      zetas.mcmc <- zetas.mcmc[, -seq(1, ((p + 1) * L), by = p + 1)]
      zetas.true <- zetas.true[-1, ]
    }
    if (!is.null(first.coef)) {
      idx <- rep(1:p0, L) + rep(seq(0, p * (L - 1), by = p), each = p0)
      zetas.mcmc <- zetas.mcmc[, idx]
      zetas.true <- zetas.true[1:p0, ]
      p <- p0
    }
    if (is.null(xlim)) {
      xlim.lower <- min(zetas.mcmc, zetas.true)
      xlim.upper <- max(zetas.mcmc, zetas.true)
    }

    if (is.null(label.y)) {
      label.y <- paste0("x", 1:p)
      if (intercept) {
        label.y <- c("intecept", label.y)
      }
    }

    dat.estimate <- data.frame(
      cluster = rep(paste("Cell type", 1:L), each = length(label.y) * nrow(zetas.mcmc)),
      covariate = rep(rep(label.y, each = nrow(zetas.mcmc)), L),
      value = as.vector(zetas.mcmc),
      coef.true = rep(zetas.true, each = nrow(zetas.mcmc))
    )
    dat.estimate$covariate <- factor(dat.estimate$covariate,
      levels = label.y[rev(seq_along(label.y))]
    )
    if (abs(xlim.lower - xlim.upper) < 0.1) {
      xlim.lower <- xlim.lower - 1
      xlim.upper <- xlim.upper + 1
      dat.estimate$value <- rnorm(length(zetas.mcmc), zetas.mcmc[1], 1e-4)
    }
    pp <-
      ggplot(dat.estimate,
        mapping = aes(
          x = .data$value,
          y = .data$covariate,
          group = .data$covariate
        )
      ) +
      geom_density_ridges(aes(fill = .data$covariate),
        scale = 0.95, col = NA, bandwidth = bandwidth
      ) +
      facet_wrap(~cluster, nrow = 1, scales = "fixed") +
      geom_point(
        shape = 18, size = 2, data = dat.estimate,
        aes(x = .data$coef.true, y = .data$covariate, fill = "black")
      ) +
      xlim(xlim.lower, xlim.upper) +
      xlab(xlab0) +
      ylab("") +
      theme_bw() +
      theme(
        panel.grid.major.x = element_blank(),
        panel.grid.minor = element_blank()
      ) +
      guides(fill = "none")
    # return(pp)
  }

  return(pp)

  # if (!is.null(legend.labs)) {
  #   par(mar = c(5.1, 0, 4.1, 0))
  #   plot(0, xaxt = "n", yaxt = "n", bty = "n", pch = "", ylab = "", xlab = "")
  #   legend("topleft",
  #     bty = "n",
  #     legend = legend.labs,
  #     pch = c(4, 5, NA),
  #     lty = c(NA, NA, 1),
  #     col = scales::alpha(c("red", "green", "green"), 0.7)
  #   )
  #   par(mar = c(5.1, 4.1, 4.1, 2.1))
  # }
}
