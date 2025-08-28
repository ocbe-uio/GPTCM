#' @title Time-dependent Brier scores
#'
#' @description
#' Predict time-dependent Brier scores based on different survival models
#'
#' @name plotBrier
#'
#' @importFrom survival Surv coxph
#' @importFrom riskRegression Score
#' @importFrom stats median as.formula
#' @importFrom ggplot2 ggplot aes .data geom_step theme element_blank xlab ylab
#' @importFrom ggplot2 theme_bw guides guide_legend
#' @importFrom utils globalVariables
#' @importFrom graphics layout par abline
#'
#' @param dat TBA
#' @param datMCMC TBA
#' @param datMCMC2 TBA
#' @param dat.new TBA
#' @param time.star TBA
#' @param xlab TBA
#' @param ylab TBA
#' @param ... TBA
#'
#' @return A \code{ggplot2::ggplot} object. See \code{?ggplot2::ggplot} for more
#' details of the object.
#'
#' @examples
#'
#' x <- 1
#'
#' @export
plotBrier <- function(dat, datMCMC,
                      datMCMC2 = NULL,
                      dat.new = NULL,
                      time.star = NULL,
                      xlab = "Time",
                      ylab = "Brier score", ...) {
  n <- dim(dat$XX)[1]
  p <- dim(dat$XX)[2]
  L <- dim(dat$XX)[3]

  # re-organize clinical variables for classical Cox and PTCM models
  x <- apply(dat$XX, c(1, 2), mean)
  colnames(x) <- paste0("x", 1:p)
  x.median <- apply(dat$XX, c(1, 2), median)
  colnames(x.median) <- paste0("x.median", 1:p)
  p.orig <- p
  if (p > 10) {
    p <- 7
    x <- x[, 1:p]
    x.median <- x.median[, 1:p]
    message("Warning: For classical survival models, only the first 7 covariates in X are used!")
  }
  survObj <- data.frame(dat$survObj, dat$x0[, -1], x, x.median)
  x0.names <- paste0("x0", 1:(NCOL(dat$x0) - 1))
  names(survObj)[3:(NCOL(dat$x0) - 1 + 2)] <- x0.names

  if (is.null(dat.new)) {
    dat.new.flag <- FALSE
    dat.new <- dat
    survObj.new <- survObj
  } else {
    dat.new.flag <- TRUE
    # re-organize clinical variables for classical Cox and PTCM models
    x.new <- apply(dat.new$XX, c(1, 2), mean)
    colnames(x.new) <- paste0("x", 1:p.orig)
    x.median.new <- apply(dat.new$XX, c(1, 2), median)
    colnames(x.median.new) <- paste0("x.median", 1:p.orig)
    if (p > 10) {
      p <- 10
      x <- x.new[, 1:p]
      x.median.new <- x.median.new[, 1:p]
      message("Warning: For classical survival models, only the first 10 covariates in X.new are used!")
    }
    survObj.new <- data.frame(dat.new$survObj, dat.new$x0[, -1], x.new, x.median.new)
    names(survObj.new)[3:(NCOL(dat.new$x0) - 1 + 2)] <- x0.names
  }

  # nIter <- datMCMC$input$nIter
  burnin <- datMCMC$input$burnin / datMCMC$input$thin

  # survival predictions based on posterior mean
  xi.hat <- colMeans(datMCMC$output$mcmc$xi[-c(1:burnin), ])
  betas.hat <- matrix(colMeans(datMCMC$output$mcmc$betas[-c(1:burnin), ]), ncol = L)
  if (datMCMC$input$proportion.model) {
    zetas.hat <- matrix(colMeans(datMCMC$output$mcmc$zetas[-c(1:burnin), ]), ncol = L)
  }
  if (datMCMC$input$BVS) {
    gammas.hat <- matrix(colMeans(datMCMC$output$mcmc$gammas[-c(1:burnin), ]), ncol = L)
    gammas.hat <- rbind(1, gammas.hat)
    betas.hat <- (gammas.hat >= 0.5) * betas.hat / gammas.hat
    betas.hat[is.na(betas.hat)] <- 0

    if (datMCMC$input$proportion.model) {
      etas.hat <- rbind(1, matrix(colMeans(datMCMC$output$mcmc$etas[-c(1:burnin), ]), ncol = L))
      zetas.hat <- (etas.hat >= 0.5) * zetas.hat / etas.hat
      zetas.hat[is.na(zetas.hat)] <- 0
    }
  }
  kappa.hat <- mean(datMCMC$output$mcmc$kappa[-c(1:burnin)])
  thetas.hat <- exp(dat.new$x0 %*% xi.hat)

  # predict survival probabilities based on GPTCM
  time_eval <- sort(dat.new$survObj$time)
  Surv.prob <- matrix(nrow = n, ncol = length(time_eval))
  if (datMCMC$input$proportion.model) {
    alphas <- sapply(1:L, function(ll) {
      exp(cbind(1, dat.new$XX[, , ll]) %*% zetas.hat[, ll])
    })
    proportion.hat <- alphas / rowSums(alphas)
  } else {
    proportion.hat <- dat$proportion
  }
  for (j in 1:length(time_eval)) {
    tmp <- 0
    for (l in 1:L) {
      mu <- exp(cbind(1, dat.new$XX[, , l]) %*% betas.hat[, l])
      lambdas <- mu / gamma(1 + 1 / kappa.hat)
      weibull.S <- exp(-(time_eval[j] / lambdas)^kappa.hat)
      tmp <- tmp + proportion.hat[, l] * weibull.S
    }
    Surv.prob[, j] <- exp(-thetas.hat * (1 - tmp))
  }
  pred.prob <- 1 - Surv.prob

  # predict survival probabilities for dat.new based on GPTCM
  if (!is.null(datMCMC2)) {
    burnin <- datMCMC2$input$burnin / datMCMC2$input$thin

    # survival predictions based on posterior mean
    xi.hat2 <- colMeans(datMCMC2$output$mcmc$xi[-c(1:burnin), ])
    betas.hat2 <- matrix(colMeans(datMCMC2$output$mcmc$betas[-c(1:burnin), ]), ncol = L)
    if (datMCMC2$input$proportion.model) {
      zetas.hat2 <- matrix(colMeans(datMCMC2$output$mcmc$zetas[-c(1:burnin), ]), ncol = L)
    }
    if (datMCMC2$input$BVS) {
      gammas.hat2 <- matrix(colMeans(datMCMC2$output$mcmc$gammas[-c(1:burnin), ]), ncol = L)
      gammas.hat2 <- rbind(1, gammas.hat2)
      betas.hat2 <- (gammas.hat2 >= 0.5) * betas.hat2 / gammas.hat2
      betas.hat2[is.na(betas.hat2)] <- 0

      if (datMCMC2$input$proportion.model) {
        etas.hat2 <- rbind(1, matrix(colMeans(datMCMC2$output$mcmc$etas[-c(1:burnin), ]), ncol = L))
        zetas.hat2 <- (etas.hat2 >= 0.5) * zetas.hat2 / etas.hat2
        zetas.hat2[is.na(zetas.hat2)] <- 0
      }
    }
    kappa.hat2 <- mean(datMCMC2$output$mcmc$kappa[-c(1:burnin)])
    thetas.hat2 <- exp(dat.new$x0 %*% xi.hat2)

    # predict survival probabilities based on GPTCM
    Surv.prob2 <- matrix(nrow = n, ncol = length(time_eval))
    if (datMCMC2$input$proportion.model) {
      alphas <- sapply(1:L, function(ll) {
        exp(cbind(1, dat.new$XX[, , ll]) %*% zetas.hat2[, ll])
      })
      proportion.hat2 <- alphas / rowSums(alphas)
    } else {
      proportion.hat2 <- dat$proportion
    }
    for (j in 1:length(time_eval)) {
      tmp <- 0
      for (l in 1:L) {
        mu <- exp(cbind(1, dat.new$XX[, , l]) %*% betas.hat2[, l])
        lambdas <- mu / gamma(1 + 1 / kappa.hat2)
        weibull.S <- exp(-(time_eval[j] / lambdas)^kappa.hat2)
        tmp <- tmp + proportion.hat2[, l] * weibull.S
      }
      Surv.prob2[, j] <- exp(-thetas.hat2 * (1 - tmp))
    }
    pred.prob2 <- 1 - Surv.prob2
  }

  # other competing survival models
  formula.tmp <- as.formula(paste0("Surv(time, event) ~ ", paste0(x0.names, collapse = "+")))
  fitCox.clin <- survival::coxph(formula.tmp, data = survObj, y = TRUE, x = TRUE)
  survfit0 <- survival::survfit(fitCox.clin, survObj.new) # data.frame(x01=survObj.new$x01,x02=survObj.new$x02))
  pred.fitCox.clin <- t(1 - summary(survfit0, times = time_eval, extend = TRUE)$surv)

  formula.tmp <- as.formula(paste0("Surv(time, event) ~ ", paste0(colnames(x.median), collapse = "+")))
  fitCox.X.median <- survival::coxph(formula.tmp, data = survObj, y = TRUE, x = TRUE)
  survfit0 <- survival::survfit(fitCox.X.median, survObj.new)
  pred.fitCox.X.median <- t(1 - summary(survfit0, times = time_eval, extend = TRUE)$surv)

  formula.tmp <- as.formula(paste0("Surv(time, event) ~ ", paste0(paste0("x", 1:p), collapse = "+")))
  fitCox.X.mean <- survival::coxph(formula.tmp, data = survObj, y = TRUE, x = TRUE)
  # formula.tmp <- as.formula(paste0("Surv(time, event) ~ x01+x02+", paste0(colnames(x.median), collapse = "+")))
  # fitCox.clin.X.median <- survival::coxph(formula.tmp, data = survObj, y=TRUE, x = TRUE)
  survfit0 <- survival::survfit(fitCox.X.mean, survObj.new)
  pred.fitCox.X.mean <- t(1 - summary(survfit0, times = time_eval, extend = TRUE)$surv)

  formula.tmp <- as.formula(paste0("Surv(time, event) ~ ", paste0(c(x0.names, paste0("x", 1:p)), collapse = "+")))
  fitCox.clin.X.mean <- survival::coxph(formula.tmp, data = survObj, y = TRUE, x = TRUE)
  survfit0 <- survival::survfit(fitCox.clin.X.mean, survObj.new)
  pred.fitCox.clin.X.mean <- t(1 - summary(survfit0, times = time_eval, extend = TRUE)$surv)

  # library(miCoPTCM) # good estimation for cure fraction; same BS as Cox.clin
  formula.tmp <- as.formula(paste0("Surv(time, event) ~ ", paste0(x0.names, collapse = "+")))
  p0 <- 1 + length(x0.names)
  suppressWarnings(
    resMY <- miCoPTCM::PTCMestimBF(formula.tmp,
      data = survObj,
      varCov = matrix(0, nrow = p0, ncol = p0),
      init = rep(0, p0)
    )
  )
  # use interpolation to resMY$estimCDF for testing validation time points
  if (dat.new.flag) {
    n.new <- length(dat.new$survObj$time)
    estimCDF.new <- rep(NA, n.new)

    time.old.sort <- sort(survObj$time)
    time.old.min <- min(survObj$time)
    time.old.max <- max(survObj$time)
    time.old.max2 <- survObj$time[n - 1]
    for (i in 1:n.new) {
      if (dat.new$survObj$time[i] %in% survObj$time) {
        estimCDF.new[i] <- resMY$estimCDF[which(time.old.sort ==
          dat.new$survObj$time[i])[1]]
      } else {
        if (dat.new$survObj$time[i] < time.old.min) {
          # use linear interpolation
          estimCDF.new[i] <- resMY$estimCDF[1] *
            dat.new$survObj$time[i] / time.old.min
        } else {
          if (dat.new$survObj$time[i] < time.old.max) {
            # use linear interpolation
            time.idxU <- which(dat.new$survObj$time[i] < time.old.sort)[1]
            time.idxL <- time.idxU - 1
            estimCDF.new[i] <- resMY$estimCDF[time.idxL] +
              (resMY$estimCDF[time.idxU] - resMY$estimCDF[time.idxL]) *
                (dat.new$survObj$time[i] - time.old.sort[time.idxL])
          } else {
            # use linear extrapolation
            estimCDF.new[i] <- resMY$estimCDF[n] +
              (resMY$estimCDF[n] - resMY$estimCDF[n - 1]) *
                (dat.new$survObj$time[i] - time.old.max) /
                (time.old.max - time.old.min)
          }
        }
      }
    }
  }
  Surv.PTCM <- exp(-exp(dat.new$x0 %*% resMY$coefficients) %*% t(resMY$estimCDF))
  predPTCM.prob <- 1 - Surv.PTCM

  list.models <- list(
    "Cox.clin" = pred.fitCox.clin,
    "Cox.X.mean" = pred.fitCox.X.mean,
    "Cox.X.median" = pred.fitCox.X.median,
    # "Cox.clin.X.median"=fitCox.clin.X.median,
    "Cox.clin.X.mean" = pred.fitCox.clin.X.mean,
    "PTCM.clin" = predPTCM.prob,
    # "GPTCM-BetaBin" = pred.prob2,
    "GPTCM" = pred.prob
  )
  if (!is.null(datMCMC2)) {
    list.models <- c(list.models, list("GPTCM2" = pred.prob2))
  }
  g <- riskRegression::Score(
    list.models,
    formula = Surv(time, event) ~ 1,
    metrics = "brier", summary = "ibs",
    data = survObj.new,
    conf.int = FALSE, times = time_eval
  )
  data <- g$Brier$score
  if (!is.null(time.star)) {
    data <- data[data$times <= time.star, ]
  }
  levels(data$model)[1] <- "Kaplan-Meier"
  # utils::globalVariables(c("times", "Brier", "model"))
  # NOTE: `aes_string()` was deprecated in ggplot2 3.0.0.
  g2 <- ggplot2::ggplot(data, aes(
    # x = "times", y = "Brier", group = "model", color = "model"
    x = .data$times, y = .data$Brier, group = .data$model, color = .data$model
  )) +
    xlab(xlab) +
    ylab(ylab) +
    geom_step(direction = "vh") + # , alpha=0.4) +
    theme_bw() +
    guides(color = guide_legend(title = "Models"))
  # theme(
  #   legend.position = "inside",
  #   legend.position.inside = c(0.4, 0.25),
  #   legend.title = element_blank()
  # )

  g2
}
