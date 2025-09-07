#' @title Plot curves of time-dependent Brier score
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
#' @param dat input data as a list containing survival data sub-list
#' \code{survObj} with two vectors (\code{event} and \code{time}), clinical
#' variable matrix \code{x0}, cluster-specific covariates \code{X}, and
#' proportions data matrix \code{proportion}
#' @param datMCMC returned object from the main function \code{GPTCM()}
#' @param dat.new input data for out-sample prediction, with the same format
#' as \code{dat}
#' @param time.star largest time for survival prediction
#' @param xlab a title for the x axis
#' @param ylab a title for the y axis
#' @param PTCM logical value for adding survival prediction by the PTCM
#' @param ... other parameters
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
#' fit <- GPTCM(dat, nIter = 5, burnin = 0)
#'
#' #plotBrier(dat, datMCMC = fit, PTCM = FALSE)
#'
#' @export
plotBrier <- function(dat, datMCMC,
                      dat.new = NULL,
                      time.star = NULL,
                      xlab = "Time",
                      ylab = "Brier score",
                      PTCM = TRUE, ...) {
  n <- dim(dat$X)[1]
  p <- dim(dat$X)[2]
  L <- dim(dat$X)[3]

  # re-organize clinical variables for classical Cox and PTCM models
  x <- apply(dat$X, c(1, 2), mean)
  colnames(x) <- paste0("x", 1:p)
  x.median <- apply(dat$X, c(1, 2), median)
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
    x.new <- apply(dat.new$X, c(1, 2), mean)
    colnames(x.new) <- paste0("x", 1:p.orig)
    x.median.new <- apply(dat.new$X, c(1, 2), median)
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
  xi.hat <- colMeans(datMCMC$output$xi[-c(1:burnin), ])
  betas.hat <- matrix(colMeans(datMCMC$output$betas[-c(1:burnin), ]), ncol = L)
  if (datMCMC$input$proportion.model) {
    zetas.hat <- matrix(colMeans(datMCMC$output$zetas[-c(1:burnin), ]), ncol = L)
  }
  if (datMCMC$input$BVS) {
    gammas.hat <- matrix(colMeans(datMCMC$output$gammas[-c(1:burnin), ]), ncol = L)
    gammas.hat <- rbind(1, gammas.hat)
    betas.hat <- (gammas.hat >= 0.5) * betas.hat / gammas.hat
    betas.hat[is.na(betas.hat)] <- 0

    if (datMCMC$input$proportion.model) {
      etas.hat <- rbind(1, matrix(colMeans(datMCMC$output$etas[-c(1:burnin), ]), ncol = L))
      zetas.hat <- (etas.hat >= 0.5) * zetas.hat / etas.hat
      zetas.hat[is.na(zetas.hat)] <- 0
    }
  }
  kappa.hat <- mean(datMCMC$output$kappa[-c(1:burnin)])
  thetas.hat <- exp(dat.new$x0 %*% xi.hat)

  # predict survival probabilities based on GPTCM
  time_eval <- sort(dat.new$survObj$time)
  Surv.prob <- matrix(nrow = n, ncol = length(time_eval))
  if (datMCMC$input$proportion.model) {
    alphas <- sapply(1:L, function(ll) {
      exp(cbind(1, dat.new$X[, , ll]) %*% zetas.hat[, ll])
    })
    proportion.hat <- alphas / rowSums(alphas)
  } else {
    proportion.hat <- dat$proportion
  }
  for (j in seq_along(time_eval)) {
    tmp <- 0
    for (l in 1:L) {
      mu <- exp(cbind(1, dat.new$X[, , l]) %*% betas.hat[, l])
      lambdas <- mu / gamma(1 + 1 / kappa.hat)
      weibull.S <- exp(-(time_eval[j] / lambdas)^kappa.hat)
      tmp <- tmp + proportion.hat[, l] * weibull.S
    }
    Surv.prob[, j] <- exp(-thetas.hat * (1 - tmp))
  }
  pred.prob <- 1 - Surv.prob

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

  if (PTCM) {
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
      # time.old.max2 <- survObj$time[n - 1]
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
  }

  list.models <- list(
    "Cox.clin" = pred.fitCox.clin,
    "Cox.X.mean" = pred.fitCox.X.mean,
    "Cox.X.median" = pred.fitCox.X.median,
    # "Cox.clin.X.median"=fitCox.clin.X.median,
    "Cox.clin.X.mean" = pred.fitCox.clin.X.mean,
    # "PTCM.clin" = predPTCM.prob,
    # "GPTCM-BetaBin" = pred.prob2,
    "GPTCM" = pred.prob
  )
  if (PTCM) {
    list.models <- c(list.models, list("GPTCM-PTCM.clin" = predPTCM.prob))
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
