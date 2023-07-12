### Helper functions for mixed models

#' Flatten a block-diagonal sparse matrix
#'
#' Take a n x d block diagonal matrix with blocks of dimension m x p
#' and flatten to a (dense) matrix of dimension n x p. Taken from R package
#' \code{mam} by Alex Stringer.
#'
#' @param Z Block diagonal sparse matrix. Can be non-square and have different number
#' of rows. Must have equal number of columns.
#' @param n Number of rows in each block, vector
#' @param d Number of columns in each block, scalar
#'
#' @return A dense matrix with the appropriate structure
#'
#' @export
#'
flatten_bdmat <- function(Z,n,d) {
  # Z: sparse block diagonal matrix to flatten
  # n: block row dimension
  # d: block column dimension
  p <- ncol(Z)/d
  ZT <- methods::as(Z,'TsparseMatrix')
  jvec <- list()
  # previous: as.integer(rep(rep(0:(d-1),each=n),times=p))
  # ZT@j <- as.integer(rep(rep(0:(d-1),each=n),times=p))
  for (i in 1:length(n))
    jvec[[i]] <- rep(0:(d-1),each=n[i])
  ZT@j <- as.integer(Reduce(c,jvec))
  as.matrix(ZT[ ,1:d,drop=FALSE])
}