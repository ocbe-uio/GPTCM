## Generate the simulated dataset
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
               c(3.6865, 2.7431, 3.3185, 0.3633, 0.1224, 3.1463), 
               tolerance = tol)
  expect_equal(as.vector(dat$proportion[1, ]), 
               c(0.000573022, 0.051430838, 0.947996140), 
               tolerance = tol)
  expect_equal(dat$X[1, 1, ], c(-1.9061,  0.5467, -0.7722), tolerance = tol)
})

## Run a Bayesian GPTCM with spike-and-slab priors

set.seed(123)
fit <- GPTCM(dat, nIter = 110, burnin = 10)


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
    expect_equal(kappa, 1.632145, tolerance = tol)
    expect_equal(as.vector(xi), 
                 c(0.4326006, 0.214969, -0.7831737), 
                 tolerance = tol)
    expect_equal(as.vector(betas[1:2, ]), 
                 c(-0.6971, -0.9087, -0.0346, -0.0025, -0.1471,  0.8426), 
                 tolerance = tol)
    expect_equal(as.vector(zetas[1:2, ]), 
                 c(-0.4728,  0.4160, -0.6378, -0.2879,  0.8331,  0), 
                 tolerance = tol)
  })
})

