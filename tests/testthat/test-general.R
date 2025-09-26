## Generate the simulated dataset
library(GPTCM)
n <- 200 # subjects
p <- 10 # variable selection predictors
L <- 3 # cell types
set.seed(123)
dat <- simData(n, p, L)

test_that("dat has properly class and dimension", {
  expect_length(dat, 13L)
  expect_length(dat$survObj, 2L)
  expect_vector(dat$survObj$event, ptype = double(), size = n)
  expect_vector(dat$survObj$time, ptype = double(), size = n)
  expect_equal(dim(dat$proportion), c(n, L))
  expect_equal(dim(dat$X), c(n, p, L))
})

test_that("dat has expected values", {
  tol <- 1e-3
  expect_equal(head(dat$survObj$event), c(0, 1, 0, 1, 1, 0), tolerance = tol)
  expect_equal(head(dat$survObj$time), 
               c(3.6864904, 2.7431144, 3.3184946, 0.3633143, 0.1224004, 3.1462821), 
               tolerance = tol)
  expect_equal(as.vector(dat$proportion[1, ]), 
               c(0.000573022, 0.051430838, 0.947996140), 
               tolerance = tol)
  expect_equal(dat$X[1, 1, ], 
               c(-1.9061346,  0.5466742, -0.7721637), 
               tolerance = tol)
})

## Run a Bayesian GPTCM with spike-and-slab priors

set.seed(123)
fit <- GPTCM(dat, nIter = 10, burnin = 2)


test_that("fit has properly class and length", {
  expect_s3_class(fit, "GPTCM")
  expect_length(fit, 3L)
  expect_length(fit$input, 10L)
  expect_length(fit$input$hyperpar, 32L)
  expect_length(fit$output, 13L)
  expect_length(fit$output$betas, 3663L)
})

test_that("fit has expected values", {
  tol <- 1e-3
  with(fit$output$post, {
    expect_equal(kappa, 0.9262288, tolerance = tol)
    expect_equal(as.vector(xi), 
                 c(0.2682705, 0.2761795, -0.6375256), 
                 tolerance = tol)
    expect_equal(as.vector(betas[1:2, ]), 
                 c(0.99975518,  0,  1.62017963,  0, -0.01426152,  0), 
                 tolerance = tol)
    expect_equal(as.vector(zetas[1:2, ]), 
                 c(-0.976766309,  0, -1.132659925,  0,  0.193755512, -0.008727163), 
                 tolerance = tol)
  })
})

