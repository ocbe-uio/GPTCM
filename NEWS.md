<div style="text-align: left;">

## GPTCM 1.1.3 (2025-10-01) (GitHub Only)

* TODO: update results in vignette

* Remove redundant variables `logPosteriorZeta` and `logPosteriorBeta` in cpp files
* Remove partial likelihood `loglikelihood0()`, and always use joint likelihood
* Pass more references and using more `const` keyword in functions
* Update `simData()` by changing xi0 to simulate survival data with censoring rate 0.2

## GPTCM 1.1.2 (2025-09-26)

* Update simulation examples in vignette due to the change of `mvnfast::rmvn()`
* Convert line endings in `configure.ac` to LF
* Improve help documentation
* Update function `simData()` for any number of clusters 
* Simplify `BVS.cpp`

## GPTCM 1.1.1 (2025-09-16)

* Fix a warning about fallback compilation with Armadillo 14.6.3 from CRAN check on machine `r-devel-linux-x86_64-debian-gcc`
* Fix issues raised from CRAN reviewer
* Change `MASS::mvrnorm()` to `mvnfast::rmvn()`

## GPTCM 1.1.0 (2025-09-07) (GitHub Only)

* First CRAN version
* Add vignette

## GPTCM 1.0.3 (2025-09-05) (GitHub Only)

* Fix NOTE about `elapsed time` from Winbuilder checks 

## GPTCM 1.0.2 (2025-09-05) (GitHub Only)

* Improve help files

## GPTCM 1.0.1 (2025-08-28) (GitHub Only)

* In `simData()`, suppress intercept when simulating data based on the Cox model

## GPTCM 1.0.0 (2025-08-25) (GitHub Only)

* First released version at `ocbe-uio` github

</div>