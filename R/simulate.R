#' Simulate data from a binary mixed model
#' 
#' Simulate data from a binary mixed model with specified random
#' effects (co)variance. See details.
#' 
#' @details When \code{S} is a scalar, it is the variance of a random intercept,
#' and this simulates data from a Bernoulli model with \code{lme4} formula
#' \code{y ~ x + (1|id)}; \code{beta} should be length 2. When \code{S} is a
#' \code{2x2} matrix, it is the covariance matrix of a random intercept and slope,
#' and this simulates data from a Bernoulli model with \code{lme4} formula
#' \code{y ~ x*t + (t|id)}; \code{beta} should be length 4.
#' The covariates \code{x} and \code{t} are a group-level
#' indicator which is 0 for haldf the groups and 1 for the other half (\code{x}),
#' and an even spacing of the interval \code{(-3,3)} with \code{n} elements (\code{t}).
#' 
#' @return A \code{data.frame} containing the covariates, group indicators, and simulated
#' response.
#' 
#' @param m Integer, number of groups
#' @param n Integer number of observations per group
#' @param beta Vector of regression coefficients for the linear predictor
#' @param S Either (a) variance of single random slope, or (b) 
#' covariance matrix of random intercept and slope
#' 
#' @examples 
#' 
#' simulate_data(10,2,c(0,0),1)
#' simulate_data(10,2,c(0,0,0,0),diag(2))
#' 
#' 
#' @export
simulate_data <- function(m,n,beta,S) {
  # generate the random effects
  if (!is.matrix(S)) {
    u <- stats::rnorm(m,0,sqrt(S))
  } else {
    u <- mvtnorm::rmvnorm(m,sigma = S)
    u <- as.numeric(t(u))
  }
  
  # covariates
  id <- rep(1:m,each=n)
  # x_i = 1 for half the subjects and 0 for the other half
  x <- c(rep(0,floor(m/2)),rep(1,m-floor(m/2)))[sample.int(m)]
  x <- rep(x,each=n)
  t <- rep(seq(-3,3,length.out=n),m)

  # This part is needed for simulation but will be repeated within the model fitting function too
  if (!is.matrix(S)) {
    ff <- y ~ x + (1|id)
    df <- data.frame(id=id,x=x,y=0)
  } else{
    ff <- y ~ x*t + (t|id)
    df <- data.frame(id=id,x=x,t=t,y=0)
  }
  reterms <- lme4::glFormula(ff,data=df,family=stats::binomial) # TODO: update this for other families
  X <- reterms$X
  Z <- t(reterms$reTrms$Zt)
  eta <- as.numeric(X %*% beta + Z %*% u)
  pp <- 1 / (1 + exp(-eta))
  df$y <- stats::rbinom(m*n,1,pp)
  
  df
}

# Simulate data with a covariate that varies across groups
simulate_data_nonconstant <- function(m,n,beta,S) {
  # generate the random effects
  if (!is.matrix(S)) {
    u <- stats::rnorm(m,0,sqrt(S))
  } else {
    u <- mvtnorm::rmvnorm(m,sigma = S)
    u <- as.numeric(t(u))
  }
  
  # covariates
  id <- rep(1:m,each=n)
  x <- rnorm(length(id))
  t <- rep(seq(-3,3,length.out=n),m)

  # This part is needed for simulation but will be repeated within the model fitting function too
  if (!is.matrix(S)) {
    ff <- y ~ x + (1|id)
    df <- data.frame(id=id,x=x,y=0)
  } else{
    ff <- y ~ x*t + (t|id)
    df <- data.frame(id=id,x=x,t=t,y=0)
  }
  reterms <- lme4::glFormula(ff,data=df,family=stats::binomial) # TODO: update this for other families
  X <- reterms$X
  Z <- t(reterms$reTrms$Zt)
  eta <- as.numeric(X %*% beta + Z %*% u)
  pp <- 1 / (1 + exp(-eta))
  df$y <- stats::rbinom(m*n,1,pp)
  
  df
}

