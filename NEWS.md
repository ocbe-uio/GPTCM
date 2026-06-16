<div style="text-align: left;">

### GPTCM 2.0.1 (2026-06-13)

* Separate MCMC loop for `BVS = FALSE` to speed up the running without variable selection

### GPTCM 2.0.0 (2026-06-03)

* Fix CRAN check warnings
* Update the M-H sampling for variable selection by using the Carlin-Chib approach (Carlin and Chib, 1995; JRSSB)
* Clean up C++ functions


### GPTCM 1.1.3 (2025-10-31)

* Remove redundant variables `logPosteriorZeta` and `logPosteriorBeta` in cpp files
* Remove partial likelihood `loglikelihood0()`, and always use joint likelihood
* Remove useless func `arms_simple()`
* Pass more references and using more `const` keyword in functions
* Update `simData()` by changing xi0 to simulate survival data with censoring rate 0.2
* Use `openmp` to parallelize some for-loop
* Update vignette


### GPTCM 1.1.2 (2025-09-26)

* Update simulation examples in vignette due to the change of `mvnfast::rmvn()`
* Convert line endings in `configure.ac` to LF
* Improve help documentation
* Update function `simData()` for any number of clusters 
* Simplify `BVS.cpp`


### GPTCM 1.1.1 (2025-09-16)

* First CRAN version

</div>