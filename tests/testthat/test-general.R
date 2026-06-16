# simulate data
n <- 200 # subjects
p <- 10 # variable selection predictors
L <- 3 # cell types
library(GPTCM)
set.seed(1234)
dat <- simData(n, p, L)

### Run GPTCM-Ber2 model

## set beta priors for Bernoulli's probability
hyperpar <- list(
  piA = 1, piB = 2, 
  rhoA = 1, rhoB = 2
)

set.seed(123)
fit <- GPTCM(dat, hyperpar = hyperpar, nIter = 2000, burnin = 1000)

betas_hat <- fit$output$post$betas[-1, ]
betas_hat <- betas_hat * (fit$output$post$gammas>=0.5)

test_that("fit has properly class and length", {
  expect_s3_class(fit, "GPTCM")
  expect_length(fit, 3L)
  expect_length(fit$input, 10L)
  expect_length(fit$input$hyperpar, 34L)
  expect_length(fit$output, 15L)
})

BVS_acc <- sum(as.numeric(fit$output$post$gammas >=0.5) == as.numeric(dat$betas != 0)) / (p * L)

test_that("fit has expected values", {
  tol <- 5e-1
  with(fit$output, {
    expect_equal(as.vector(post$xi), c(0.9, 0.6, -1.0), tolerance = tol)
    expect_equal(as.vector(post$betas[1, ]), rep(0, L), tolerance = tol)
  })
  expect_equal(BVS_acc, 1.0, 0.2)
})
