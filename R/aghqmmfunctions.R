#' Control options for aghqmm
#' 
#' Control options for the \code{aghqmm} function, see that function.
#'
#' @param ... Used to override default arguments, see Details.
#' 
#' @details You can provide options for the following control arguments
#' to override their defaults:
#' 
#' TODO
#'
#' @family aghqmm
#' 
#' @export
aghqmm_control <- function(...) {
  # default control arguments
  out <- list(
    tol = 1e-08,
    maxitr = 100,
    inner_tol = 1e-08,
    inner_maxitr = 10,
    h = 1e-08,
    verbose = FALSE,
    bfgshist = 4,
    bfgsdelta = 1e-06,
    past = 3,
    max_linesearch = 100,
    method = "lbfgs",
    iter_EM = 0, # for GLMMadaptive only
    onlynllgrad = FALSE
  )
  userargs <- list(...)
  for (arg in names(userargs)) out[arg] <- userargs[arg]
  out
}

#' Binary Mixed Model via AGHQ with Exact Gradients
#' 
#' Fit a binary mixed model using AGHQ with exact gradients of the
#' approximate log-marginal likelihood.
#' 
#' @param formula An R mixed model formula pf the form \code{y ~ x*t + (t|id)},
#' compatible with \code{lme4::glmer}; see that function for details.
#' @param data A \code{data.frame} containing the variables found in \code{formula}.
#' @param k Order of the adaptive quadrature. In 1 dimension, \code{k} is the number of
#' points; in d-dimensions there are \code{k^d} total points.
#' @param family \code{R glm} family; currently only \code{binomial()} is supported
#' and this assumes a binary response.
#' @param method Method to use to fit the model. \code{lbfgs} implements the novel
#' exact gradient-based quasi-Newton method from the paper; \code{newton} uses the
#' exact gradient to build a finite-differenced Hessian for use in Newton optimization.
#' The other options call others' software to facilitate comparison.
#' @param control Output of \code{aghqmm_control()}, see that function.
#' 
#' @details TODO
#' 
#' @return A list containing elements: TODO
#'
#' @examples 
#' library(mlmRev)
#' data("Contraception")
#' aghqmm::aghqmm(use ~ age + urban + (age|district),data=Contraception)
#' 
#' \dontrun{
#' ## Not run
#' library(lme4)
#' glmer(use ~ age + urban + (age|district),data=Contraception,family = binomial)
#' library(GLMMadaptive)
#' mixed_model(use ~ age + urban,random = ~age|district,data=Contraception,family=binomial)
#' library(glmmEP)
#' reterms <- lme4::glFormula(formula,data=data,family=stats::binomial)
#' X <- reterms$X
#' Z <- X[ ,c(1,2)]
#' idvec <- reterms$reTrms$flist$district
#' glmmEP(as.numeric(Contraception$use)-1,X,Z,idvec) # note: singular
#' }
#' 
#' 
#' @family aghqmm
#' 
#' @export
#'
aghqmm <- function(
  formula,
  data,
  k=5,
  family=stats::binomial(),
  method = c("lbfgs","newton","both","GLMMadaptive","lme4","glmmEP"),
  control = aghqmm_control()) {
  
  method <- method[1]
  response <- all.vars(formula)[1]
  # formula = y ~ x*t + (t|id)
  # generate the data
  reterms <- lme4::glFormula(formula,data=data,family=stats::binomial) # TODO: update this for other families
  idvar <- names(reterms$reTrms$cnms)[1]
  X <- reterms$X
  Z <- t(reterms$reTrms$Zt)
  
  # check if response is 0/1 coded
  if(!all.equal(sort(unique(data[[response]])),c(0,1))) 
    stop(paste0("Response should be coded as numeric 0/1, yours is coded as",sort(unique(data[[response]])),". I don't know how to automatically convert it, sorry!"))
  
  
  modeldata <- with(reterms,list(
    X = X,
    Z = t(reTrms$Zt),
    y = data[[response]],
    group = data[[idvar]],
    id = as.numeric(table(data[[idvar]]))
  ))
  
  # Prepare the data for the likelihood functions
  yy <- with(modeldata,split(y,group))
  XX <- with(modeldata,lapply(split(X,group),matrix,ncol=ncol(X)))
  ZZ <- vector(mode = 'list',length = length(modeldata$id))
  d <- length(Reduce(c,reterms$reTrms$cnms)) # NOTE: not really tested
  for (i in 1:length(modeldata$id)) {
    row_id <- with(modeldata,sum(id[1:i]) + 1 - id[i]):with(modeldata,sum(id[1:i]))
    col_id <- (1 + (i-1)*d):(i*d)
    ZZ[[i]] <- as.matrix(modeldata$Z[row_id,col_id])
    colnames(ZZ[[i]]) <- NULL
  }
  
  # Quadrature
  gg <- mvQuad::createNIGrid(d,'GHe',k)
  nn <- mvQuad::getNodes(gg)
  ww <- mvQuad::getWeights(gg)
  
  # Optimize
  pardim <- ncol(modeldata$X) + d*(d+1)/2
  linmod <- stats::glm(lme4::nobars(formula),data,family=family)
  betastart <- stats::coef(linmod)
  betadim <- length(betastart)
  thetastart <- c(betastart,c(0,0,0))
  ustart <- rep(0,d*length(modeldata$id))
  if (method %in% c("lbfgs","newton","both")) {
    control$method <- method
    opt <- optimizeaghq(thetastart,ustart,yy,XX,ZZ,nn,ww,control)  
  }
  
  if (is.null(opt$nll)) {
    # add on nll and grad
    control$onlynllgrad <- TRUE
    nllandgrad <- optimizeaghq(opt$theta,ustart,yy,XX,ZZ,nn,ww,control)  
    opt$nll <- nllandgrad$nll
    opt$grad <- nllandgrad$grad
  }
  opt$normgrad <- max(abs(opt$grad)) # infinity norm
  
  
  opt
}