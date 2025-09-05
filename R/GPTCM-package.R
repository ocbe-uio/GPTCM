#' @keywords internal
#' @aliases GPTCM-package NULL
#'
#'
"_PACKAGE"

## usethis namespace: start
#' @useDynLib GPTCM, .registration = TRUE
## usethis namespace: end
NULL

## nocov start

.onLoad <- function(libname, pkgname) {
  # CRAN OMP THREAD LIMIT
  Sys.setenv("OMP_THREAD_LIMIT" = 1)
}

## nocov end

