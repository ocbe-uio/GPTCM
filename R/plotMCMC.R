#' @title MCMC trace-plots
#'
#' @description
#' Trace-plots of regression coefficients over MCMC iterations
#'
#' @name plotMCMC
#'
#' @importFrom ggplot2 ggplot aes geom_step theme element_blank
#' @importFrom graphics segments
#'
#' @param dat input data as a list containing survival data sub-list
#' \code{survObj} with two vectors (\code{event} and \code{time}), clinical
#' variable matrix \code{x0}, cluster-specific covariates \code{X}, and
#' proportions data matrix \code{proportion}
#' @param datMCMC returned object from the main function \code{GPTCM()}
#' @param estimator print estimators, one of
#' \code{c("beta", "zeta", "gamma", "eta")}
#'
#' @return A \code{ggplot2::ggplot} object. See \code{?ggplot2::ggplot} for more
#' details of the object.
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
#' plotMCMC(dat, datMCMC = fit, estimator = "xi")
#'
#' @export
plotMCMC <- function(dat, datMCMC, estimator = "xi") {
  # n <- dim(dat$X)[1]
  p <- dim(dat$X)[2]
  L <- dim(dat$X)[3]
  # nIter <- datMCMC$input$nIter
  # burnin <- datMCMC$input$burnin

  # p.idx <- 1:(p * L)
  p.idx <- 1:((p + 1) * L)
  # p.DirTrue.idx <- 1:((p + 1) * L)
  p.DirFalse.idx <- 1:((p + 1) * (L - 1))
  if (p > 10) {
    # p.idx <- rep(1:10, L) + p * rep(0:(L - 1), each = 10)
    p.idx <- rep(1:11, L) + (p + 1) * rep(0:(L - 1), each = 11)
    # p.DirTrue.idx <- rep(1:11, L) + (p + 1) * rep(0:(L - 1), each = 11)
    p.DirFalse.idx <- rep(1:11, L - 1) + (p + 1) * rep(0:(L - 2), each = 11)
    p <- 10
  }
  if ("beta" %in% estimator) {
    betas.mcmc <- datMCMC$output$betas[, p.idx]
    dat$betas <- rbind(dat$beta0, dat$betas)
    ylabel <- paste0(
      "expression(beta['", rep(0:p, L), ",",
      rep(1:L, each = p + 1), "'])"
    )
    layout(matrix(seq_len(NCOL(betas.mcmc)), ncol = L))
    par(mar = c(2, 4.1, 2, 2))
    for (j in seq_len(NCOL(betas.mcmc))) {
      plot(betas.mcmc[, j],
        type = "l", lty = 1, ylab = eval(parse(text = ylabel[j])),
        ylim = summary(c(betas.mcmc, dat$betas))[c(1, 6)]
      )
      abline(h = dat$betas[p.idx[j]], col = "red")
    }
  }

  if ("zeta" %in% estimator) {
    dirichlet <- datMCMC$input$dirichlet
    if (dirichlet) {
      zetas.mcmc <- datMCMC$output$zetas[, p.idx]
      ylabel <- paste0(
        "expression(zeta['", rep(0:p, L), ",",
        rep(1:L, each = p + 1), "'])"
      )
    } else {
      zetas.mcmc <- datMCMC$output$zetas[, p.DirFalse.idx] # 1:((p + 1) * (L - 1))]
      ylabel <- paste0(
        "expression(zeta['", rep(0:p, L - 1), ",",
        rep(1:(L - 1), each = p + 1), "'])"
      )
    }

    ylabel <- paste0(
      "expression(zeta['", rep(0:p, ifelse(dirichlet, L, L - 1)), ",",
      rep(1:ifelse(dirichlet, L, L - 1), each = p + 1), "'])"
    )
    layout(matrix(seq_len(NCOL(zetas.mcmc)), nrow = p + 1))
    par(mar = c(2, 4.1, 2, 2))
    for (j in seq_len(NCOL(zetas.mcmc))) {
      plot(zetas.mcmc[, j],
        type = "l", lty = 1, ylab = eval(parse(text = ylabel[j])),
        ylim = summary(c(zetas.mcmc, dat$zetas))[c(1, 6)]
      )
      abline(h = dat$zetas[p.idx[j]], col = "red")
    }
  }

  if (any(estimator %in% c("gamma", "eta"))) {
    bvs.mcmc <- eval(parse(text = paste0("datMCMC$output$", estimator, "s")))
    nIter <- nrow(bvs.mcmc)
    p <- dim(dat$X)[2]

    layout(matrix(1:L, nrow = 1))
    par(mar = c(2, 4.1, 2, 2))

    for (l in 1:L) {
      plot(c(1:p) ~ 1,
        type = "n",
        xlim = c(1, nIter),
        yaxt = "n",
        ylab = "",
        xlab = "Iteration",
        main = paste("Cell type", l)
      )
      axis(2, at = 1:p, labels = paste0("x", p:1), tick = FALSE, las = 1)
      for (j in 1:p) {
        for (i in 1:nIter) {
          if (bvs.mcmc[i, p * (l - 1) + j] == 1) {
            segments(x0 = i - 0.4, y0 = p - j + 1, x1 = i + 0.4, y1 = p - j + 1, lwd = 1)
          } # Draw line
        }
      }
    }
  }

  if ("xi" %in% estimator) {
    p <- dim(datMCMC$output$xi)[2]
    xi.mcmc <- datMCMC$output$xi

    ylabel <- paste0("expression(xi[", 0:(p - 1), "])")
    layout(matrix(1:p, ncol = 1))
    par(mar = c(2, 4.1, 2, 2))
    for (j in 1:p) {
      plot(xi.mcmc[, j],
        type = "l", lty = 1, ylab = eval(parse(text = ylabel[j])),
        ylim = summary(c(xi.mcmc, dat$xi))[c(1, 6)]
      )
      abline(h = dat$xi[j], col = "red")
    }
  }


  if (any(estimator %in% c("kappa", "tau", "w", "v", "phi"))) {
    layout(matrix(seq_along(estimator), ncol = 1))
    par(mar = c(2, 4.1, 2, 2))
    if ("kappa" %in% estimator) {
      kappa.mcmc <- datMCMC$output$kappa
      plot(kappa.mcmc,
        type = "l", lty = 1,
        ylab = expression(kappa), xlab = "MCMC iteration",
        ylim = summary(c(kappa.mcmc, dat$kappa))[c(1, 6)]
      )
      abline(h = dat$kappa, col = "red")
    }

    if ("tau" %in% estimator) {
      tauSq.mcmc <- datMCMC$output$tauSq
      plot(tauSq.mcmc,
        type = "l", lty = 1,
        ylab = expression(tau^2), xlab = "MCMC iteration",
        ylim = summary(as.vector(tauSq.mcmc))[c(1, 6)]
      )
    }

    if ("w" %in% estimator) {
      wSq.mcmc <- datMCMC$output$wSq
      plot(wSq.mcmc,
        type = "l", lty = 1,
        ylab = expression(w^2), xlab = "MCMC iteration",
        ylim = summary(as.vector(wSq.mcmc))[c(1, 6)]
      )
    }

    if ("v" %in% estimator) {
      vSq.mcmc <- datMCMC$output$vSq
      plot(vSq.mcmc,
        type = "l", lty = 1,
        ylab = expression(v^2), xlab = "MCMC iteration",
        ylim = summary(as.vector(vSq.mcmc))[c(1, 6)]
      )
    }

    if ("phi" %in% estimator) {
      phi.mcmc <- datMCMC$output$phi
      plot(phi.mcmc,
        type = "l", lty = 1,
        ylab = expression(phi), xlab = "MCMC iteration",
        ylim = summary(as.vector(phi.mcmc))[c(1, 6)]
      )
    }
  }
}
